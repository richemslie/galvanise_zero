'''
todo
====
* bcolz.set_nthreads(nthreads)
* call samples, samples -> call self play data - observations
* evaluation speed tests


Train Pipeline Overview:
========================

db is a bcolz table (and bcolz is awesome!)
gendata_ZZZ_YY.json.gz - are json data files produced from self play (one server, n workers).
 * ZZZ is game
 * YY is network generation step played with
gendata_summary.json - summary of the json data files, and what is in db


init
----
* read summary file (gendata_summary.json), and validate against db
* validate existing gendata files (number of samples & md5sum)
* if invalid - delete db or summary file (if the exist).  Create new db & summary.


sync
----
* check directory for any recent gendata files:
  * read new data, and preprocess into numpy arrays (for training data to keras/tf)
  * add the numpy data to db
  * update the summary file


create an indexer
-----------------
* specified from buckets
* create a ChunkIndexer - which will create train/validation batches for one epoch
* XXX what about weightings from training data?  Future step.

callbacks
---------
* before each epoch.  Idea is to keep epochs small (1 million) (XXX todo)

'''

# std imports
import gc
import os
import sys
import gzip
import math
import random
import hashlib
import datetime

# 3rd party imports
import bcolz
import numpy as np

# ggplib imports
from ggplib.util import log

# ggpzero imports
from ggpzero.util import attrutil

from ggpzero.defs import datadesc
from ggpzero.nn.manager import get_manager

from ggpzero.util.state import decode_state


class Check(Exception):
    pass


def timestamp():
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M")


def reshape(d):
    # reshape list of to a numpy array of -1, x, x, x
    new_shape = [-1] + list(d.shape)
    return np.concatenate([d], axis=0).reshape(new_shape)


def fake_columns(transformer):
    # fake some data. Note that we could use the gendata_X instead of doing this, but this at least
    # gives us a warm fuzzy that something isn't badly configured.
    t = transformer
    sm = t.game_info.get_sm()
    basestate = sm.get_initial_state()
    channels = t.state_to_channels(basestate.to_list())

    cols = [reshape(channels)]

    # create a fake policy for each role
    for role_index in range(len(sm.get_roles())):
        ls = sm.get_legal_state(role_index)

        # create a uniform policy
        policy = [(ls.get_legal(c), 1 / float(ls.get_count())) for c in range(ls.get_count())]

        policy_array = t.policy_to_array(policy, role_index)
        cols.append(reshape(policy_array))

    value_head = transformer.value_to_array([0, 1])
    cols.append(reshape(value_head))

    assert len(cols) == 4
    return cols


class Buckets(object):
    def __init__(self, bucket_def):
        self.bucket_def = bucket_def

    def get(self, depth):
        if not self.bucket_def:
            return 1.0

        for idx, (cut_off, pct) in enumerate(self.bucket_def):
            assert cut_off != 0

            if cut_off < 0:
                # only allow one cut_off @ -1
                assert cut_off == -1
                assert len(self.bucket_def[idx:]) == 1
                return pct

            if depth < cut_off:
                return pct


class ChunkIndexer(object):

    def __init__(self, buckets, step_summaries):
        self.buckets = buckets
        self.step_summaries = step_summaries

    def find_levels(self, starting_step=None,
                    ignore_after_step=None,
                    validation_split=0.9):

        ''' finds train_levels, validation_level
        where levels is a range is a list of (start, end].

        starting_step: says ignore the first n levels
        ignore_after_step: says ignore after a step is reached
        validation_split: the percentage to allocate to training
        '''

        self.train_levels = []
        self.validation_levels = []

        index = 0
        for step, summary in enumerate(self.step_summaries):
            if starting_step is not None and step > starting_step:
                break

            if ignore_after_step is not None and step < ignore_after_step:
                continue

            assert step == summary.step

            index_end = index + summary.num_samples
            validation_start = index + int(summary.num_samples * validation_split)

            self.train_levels.append((index, validation_start))
            self.validation_levels.append((validation_start, index_end))

            index = index_end

        # want most recent first
        self.train_levels.reverse()
        self.validation_levels.reverse()
        assert len(self.train_levels) == len(self.validation_levels)

    def create_indices_for_level(self, level_index, validation=False, max_size=-1):
        ''' returns a shuffled list of indices '''
        start, end = self.validation_levels[level_index] if validation else self.train_levels[level_index]
        indices = range(start, end)
        random.shuffle(indices)
        if max_size > 0:
            indices = indices[:max_size]
        return indices

    def get_indices(self, max_size=None, validation=False):
        '''
        same bucket algorithm as old way, but with indexing:

        figure out the sizes required from each generation based on buckets (any rounding issues,
        drop from oldest generation) [also works for scaling down if we add max_number_of_samples]

        create a range(n) where n is the size of a generation.  shuffle.  remove head or tail until
        size.  [old version removed tail, but it doesn't matter]

        combine all (need to offset start index of each generation data]

        shuffle.
        '''

        levels = self.validation_levels if validation else self.train_levels
        sizes = [end - start for start, end in levels]

        # apply buckets
        bucket_sizes = []
        for depth, sz in enumerate(sizes):
            percent = self.buckets.get(depth)
            if percent < 0:
                continue

            sz *= percent
            bucket_sizes.append(int(sz))

        # do we have more data than needed for epoch?
        sizes = bucket_sizes
        total_size = sum(sizes)

        if max_size is not None or max_size > 0:
            if total_size > max_size:
                scale = max_size / float(total_size)

                # round up, but then we remove from last level
                new_sizes = [int(math.ceil(s * scale)) for s in sizes]
                if sum(new_sizes) > max_size:
                    new_sizes[-1] -= sum(new_sizes) - max_size
                sizes = new_sizes

            assert sum(sizes) <= max_size

        all_indices = []
        for ii, s in enumerate(sizes):
            all_indices += self.create_indices_for_level(ii, validation=validation, max_size=s)

        random.shuffle(all_indices)
        return all_indices

    def training_epoch(self, epoch_size=None):
        return self.get_indices(max_size=epoch_size, validation=False)

    def validation_epoch(self, epoch_size=None):
        # XXX maybe add a trim mode - so always taking most recent data???  Maybe better to just
        # have seperate buckets?
        return self.get_indices(max_size=epoch_size, validation=True)


class StatsAccumulator(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples

        self.unique_matches_set = set()
        self.total_draws = 0

        # when all policies lengths = 1
        self.bare_policies = 0

        # match_identifier -> [depths]
        self.match_depths = {}

        self.total_resigns = 0
        self.total_false_positives = 0
        self.total_puct_visits = 0

        # XXX only 2 for now
        self.total_ratio_of_roles = [0, 0]

        # XXX only 2 for now
        self.total_final_scores_per_roles = [0, 0]

        d = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        self.total_puct_score_dist = [[x, 0] for x in d]

    @property
    def unique_matches(self):
        return len(self.match_depths)

    @property
    def draw_ratio(self):
        return self.total_draws / float(self.num_samples)

    @property
    def bare_policies_ratio(self):
        return self.bare_policies / float(self.num_samples)

    @property
    def av_starting_depth(self):
        return sum(m[0] for m in self.match_depths.values()) / float(self.unique_matches)

    @property
    def av_ending_depth(self):
        return sum(m[-1] for m in self.match_depths.values()) / float(self.unique_matches)

    @property
    def av_resigns(self):
        return self.total_resigns / float(self.num_samples)

    @property
    def av_resign_false_positive(self):
        return self.total_false_positives / float(self.num_samples)

    @property
    def av_puct_visits(self):
        return self.total_puct_visits / float(self.num_samples)

    @property
    def av_final_scores(self):
        return [(s / float(self.num_samples)) for s in self.total_final_scores_per_roles]

    @property
    def ratio_of_roles(self):
        return [(x / float(self.num_samples)) for x in self.total_ratio_of_roles]

    @property
    def av_puct_score_dist(self):
        return str([(p, total / float(self.num_samples)) for p, total in self.total_puct_score_dist])

    def add(self, sample):
        self.total_draws += 1 if abs(sample.final_score[0] - 0.5) < 0.01 else 0
        self.bare_policies += 1 if all(len(p) == 1 for p in sample.policies) else 0
        self.match_depths.setdefault(sample.match_identifier, []).append(sample.depth)

        self.total_resigns += 1 if sample.has_resigned else 0
        self.total_false_positives += 1 if sample.resign_false_positive else 0
        self.total_puct_visits += sample.resultant_puct_visits

        for ri in range(2):
            if len(sample.policies[ri]) > 1:
                self.total_ratio_of_roles[ri] += 1

        for ii, score in enumerate(sample.final_score):
            self.total_final_scores_per_roles[ii] += score

        indx = 0
        score = sample.resultant_puct_score[0]
        for ii, (upper_limit, _) in enumerate(self.total_puct_score_dist):
            if (score - 0.0001) < upper_limit:
                continue

            indx = ii
        self.total_puct_score_dist[indx][1] += 1


class DataCache(object):
    def __init__(self, transformer, gen_prefix):
        self.transformer = transformer
        self.gen_prefix = gen_prefix

        man = get_manager()
        self.data_path = man.samples_path(self.transformer.game, gen_prefix)
        self.summary_path = os.path.join(self.data_path, "gendata_summary.json")

        self.summary = self.get_summary()
        self.save_summary_file()
        bcolz.set_nthreads(4)

    def get_summary(self, create=False):
        if create or not os.path.exists(self.summary_path):
            summary = datadesc.GenDataSummary(game=self.transformer.game,
                                              gen_prefix=self.gen_prefix,
                                              last_updated=timestamp(),
                                              total_samples=0)
        else:
            summary = attrutil.json_to_attr(open(self.summary_path).read())

        return summary

    def list_files(self):
        match_start_with = "gendata_%s_" % self.transformer.game
        match_ends_with = ".json.gz"
        # fn -> (fn, msd5sum, num_samples
        files = {}
        for fn in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, fn)
            if fn.startswith(match_start_with) and fn.endswith(match_ends_with):
                step = int(fn.replace(match_start_with, "").replace(match_ends_with, ""))

                m = hashlib.md5()
                m.update(open(file_path).read())
                md5sum = m.hexdigest()

                assert step not in files

                files[step] = file_path, md5sum

        for step in sorted(files):
            file_path, md5sum = files[step]
            yield step, file_path, md5sum

    def check_summary(self):
        ' checks summary against existing files '
        total_samples = 0

        try:
            if self.summary.game != self.transformer.game:
                raise Check("Game not same %s/%s" % (self.summary.game, self.transformer.game))

            expect = 0
            for step_sum, (step, file_path, md5sum) in zip(self.summary.step_summaries, self.list_files()):

                # special case exception, this should never happen!
                if step_sum.step != expect:
                    raise Exception("Weird - step_sum.step != expect, please check %s/%s" % (step_sum.step,
                                                                                             expect))

                if step_sum.step != step:
                    raise Check("step_sum(%d) != step(%d)" % (step_sum.step, step))

                if step_sum.md5sum != md5sum:
                    msg = "Summary check: for file %s, md5sum(%s) != md5sum(%s)" % (file_path,
                                                                                    step_sum.md5sum,
                                                                                    md5sum)
                    log.warning(msg)
                    # raise Check(msg)

                total_samples += step_sum.num_samples
                expect += 1

            if self.summary.total_samples != total_samples:
                raise Check("Total samples mismatch %s != %s" % (self.summary.total_samples,
                                                                 total_samples))

        except Check as exc:
            log.error("Summary check failed: %s" % exc)
            return False

        return True

    def verify_db(self):
        ' checks summary against existing files '
        db_path = os.path.join(self.data_path, "__db__")
        if not os.path.exists(db_path):
            return False

        try:
            self.db = bcolz.open(db_path, mode='a')

            # XXX check columns are correct types

            if self.summary.total_samples != self.db.size:
                m = "db and summary file different sizes summary = %s != %s"
                raise Check(m % (self.summary.total_samples, self.db.size))

        except Exception as exc:
            log.error("error accessing db directory: %s" % exc)
            return False

        return True

    def create_db(self):
        ' delete existing bcolz db (warn) and then create a fresh  '
        db_path = os.path.join(self.data_path, "__db__")

        if os.path.exists(db_path):
            log.warning("Please delete old db")
            sys.exit(1)

        # these are columns for bcolz table
        cols = fake_columns(self.transformer)

        # and create a table
        self.db = bcolz.ctable(cols, names=["channels", "policy0", "policy1", "value"], rootdir=db_path)

        # remove the single row
        self.db.resize(0)
        self.db.flush()

        log.info("Created new db")

    def files_to_process(self):
        ' generate files to process '
        if self.summary.step_summaries:
            last_processed = self.summary.step_summaries[-1].step
        else:
            last_processed = -1

        for step, file_path, md5sum in self.list_files():
            if step <= last_processed:
                continue
            yield step, file_path, md5sum

    def sync(self):
        # check summary matches current set of files
        if not self.check_summary() or not self.verify_db():
            self.get_summary(create=True)
            self.create_db()

        for step, file_path, md5sum in self.files_to_process():
            # lets delete any spurious memory
            gc.collect()

            log.debug("Proccesing %s" % file_path)
            data = attrutil.json_to_attr(gzip.open(file_path).read())

            if len(data.samples) != data.num_samples:
                # pretty inconsequential, but we should at least notify
                msg = "num_samples (%d) versus actual samples (%s) differ... trimming"
                log.warning(msg % (data.num_samples, len(data.samples)))

                data.num_samples = min(len(data.samples), data.num_samples)
                data.samples = data.samples[:data.num_samples]

            log.debug("Game %s, with gen: %s and sample count %s" % (data.game,
                                                                     data.with_generation,
                                                                     data.num_samples))

            indx = self.db.size
            self.db.resize(indx + data.num_samples)

            stats = StatsAccumulator(data.num_samples)
            t = self.transformer
            for sample in data.samples:
                t.check_sample(sample)
                stats.add(sample)

                # add channels
                state = decode_state(sample.state)
                prev_states = [decode_state(s) for s in sample.prev_states]
                cols = [t.state_to_channels(state, prev_states)]

                for ri, policy in enumerate(sample.policies):
                    cols.append(t.policy_to_array(policy, ri))

                cols.append(t.value_to_array(sample.final_score))

                # XXX this seems not an efficient way to do things
                for ii, name in enumerate(self.db.names):
                    self.db[name][indx] = cols[ii]
                indx += 1

            self.db.flush()
            log.debug("Added samples to db")

            # add to the summary and save it
            step_sum = datadesc.StepSummary(step=step,
                                            filename=file_path,
                                            with_generation=data.with_generation,
                                            num_samples=data.num_samples,
                                            md5sum=md5sum,
                                            stats_unique_matches=stats.unique_matches,
                                            stats_draw_ratio=stats.draw_ratio,
                                            stats_bare_policies_ratio=stats.bare_policies_ratio,
                                            stats_av_starting_depth=stats.av_starting_depth,
                                            stats_av_ending_depth=stats.av_ending_depth,
                                            stats_av_resigns=stats.av_resigns,
                                            stats_av_resign_false_positive=stats.av_resign_false_positive,
                                            stats_av_puct_visits=stats.av_puct_visits,
                                            stats_ratio_of_roles=stats.ratio_of_roles,
                                            stats_av_final_scores=stats.av_final_scores,
                                            stats_av_puct_score_dist=stats.av_puct_score_dist)

            print attrutil.attr_to_json(step_sum, pretty=True)

            self.summary.last_updated = timestamp()
            self.summary.total_samples = self.db.size
            self.summary.step_summaries.append(step_sum)

            self.save_summary_file()
            log.debug("Saved summary file")

        # lets delete any spurious memory
        gc.collect()
        self.save_summary_file()
        log.info("Data cache synced, saved summary file.")

    def save_summary_file(self):
        with open(self.summary_path, 'w') as open_file:
            open_file.write(attrutil.attr_to_json(self.summary, pretty=True))

    def create_chunk_indexer(self, buckets, **kwds):
        assert isinstance(buckets, Buckets)
        indexer = ChunkIndexer(buckets, self.summary.step_summaries)
        indexer.find_levels(**kwds)
        return indexer

    def generate(self, indices, batch_size):
        for ii in range(0, len(indices), batch_size):
            next_indices = indices[ii:ii + batch_size]

            record = self.db[next_indices]

            inputs = record["channels"]
            outputs = [record[name] for name in self.db.names[1:]]
            yield inputs, outputs
