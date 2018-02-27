import os
import gzip

from collections import Counter

import numpy as np

from ggplib.util import log
from ggplib.db import lookup

from ggpzero.util import attrutil

from ggpzero.defs import confs

from ggpzero.nn.manager import get_manager

from ggpzero.nn import network


###############################################################################

def unison_shuffle(*arrays):
    assert arrays
    length_of_array0 = len(arrays[0])
    for a in arrays[1:]:
        assert len(a) == length_of_array0

    perms = np.random.permutation(length_of_array0)

    return [a[perms] for a in arrays]


class TrainException(Exception):
    pass


def reshape(d):
    new_shape = [-1] + list(d[0].shape)
    return np.concatenate(d, axis=0).reshape(new_shape)


class Buckets(object):
    def __init__(self, bucket_def):
        self.bucket_def = bucket_def

    def get(self, depth, max_depth):
        if not self.bucket_def:
            return 1.0

        for idx, (cut_off, pct) in enumerate(self.bucket_def):
            if cut_off <= 0:
                return self.get2(depth, max_depth, self.bucket_def[idx:])

            if depth < cut_off:
                return pct

    def get2(self, depth, max_depth, stripped_def):
        assert len(stripped_def) == 1
        return stripped_def[0][1]


class SamplesData(object):
    def __init__(self, game, with_generation, num_samples):
        self.game = game
        self.with_generation = with_generation
        self.num_samples = num_samples

        self.samples = []

        self.state_identifiers = []
        self.inputs = []
        self.outputs = []
        self.transformed = False

    def add_sample(self, sample):
        self.samples.append(sample)

    def verify_samples(self, sm):
        # create a basestate
        basestate = sm.new_base_state()

        from collections import Counter
        counters = [Counter(), Counter()]
        max_values = [{}, {}]
        min_values = [{}, {}]
        for s in self.samples:
            basestate.from_list(s.state)
            sm.update_bases(basestate)

            # get legals...
            for ri in range(2):
                ls = sm.get_legal_state(ri)
                policy = s.policies[ri]
                for legal in ls.to_list():
                    found = False
                    for ll, pp in policy:
                        if ll == legal:
                            max_values[ri][legal] = max(max_values[ri].get(legal, -1), pp)
                            min_values[ri][legal] = min(max_values[ri].get(legal, 2), pp)
                            found = True
                            break
                    assert found
                    counters[ri][legal] += 1
        for ri in range(2):
            print sm.legal_to_move(ri, 57)
            print "count", counters[ri][57]
            print "min/max", min_values[ri][57], max_values[ri][57]

    def transform_all(self, transformer):
        for sample in self.samples:
            # ask the transformer to come up with a unique identifier for sample
            self.state_identifiers.append(transformer.identifier(sample.state,
                                                                 sample.prev_states))

            sample = transformer.check_sample(sample)
            transformer.sample_to_nn(sample, self.inputs, self.outputs)

        self.transformed = True

        # free up memory
        self.samples = []

    def __iter__(self):
        assert self.transformed
        for s, ins, outs in zip(self.state_identifiers, self.inputs, self.outputs):
            yield s, ins, outs


class SamplesBuffer(object):

    def __init__(self):
        self.sample_data_cache = {}

    def files_to_sample_data(self, conf):
        man = get_manager()

        assert isinstance(conf, confs.TrainNNConfig)

        step = conf.next_step - 1

        starting_step = conf.starting_step
        if starting_step < 0:
            starting_step = max(step + starting_step, 0)

        while step >= starting_step:
            store_path = man.samples_path(conf.game, conf.generation_prefix)
            fn = os.path.join(store_path, "gendata_%s_%s.json.gz" % (conf.game, step))
            if fn not in self.sample_data_cache:
                raw_data = attrutil.json_to_attr(gzip.open(fn).read())
                data = SamplesData(raw_data.game,
                                   raw_data.with_generation,
                                   raw_data.num_samples)

                for s in raw_data.samples:
                    data.add_sample(s)

                if len(data.samples) != data.num_samples:
                    # pretty inconsequential, but we should at least notify
                    msg = "num_samples (%d) versus actual samples (%s) differ... trimming"
                    log.warning(msg % (data.num_samples, len(data.samples)))

                    data.num_samples = min(len(data.samples), data.num_samples)
                    data.samples = data.samples[:data.num_samples]

                self.sample_data_cache[fn] = data

            yield fn, self.sample_data_cache[fn]
            step -= 1


class LevelData(object):
    def __init__(self, level):
        self.level = level
        self.inputs = []
        self.outputs = None

        # set in validation_split
        self.validation_inputs = self.validation_outputs = None

    def add(self, ins, outs):
        self.inputs.append(ins)
        if self.outputs is None:
            # create a list for each output
            self.outputs = [[] for _ in outs]

        for o, outputs in zip(outs, self.outputs):
            outputs.append(o)

    def shuffle(self):
        # faster np version
        perms = np.random.permutation(len(self.inputs))
        self.inputs = self.inputs[perms]
        self.outputs = [o[perms] for o in self.outputs]

    def validation_split(self, validation_split):
        ' split input/outputs, DONT shuffle'

        self.inputs = reshape(self.inputs)
        self.outputs = [reshape(o) for o in self.outputs]

        train_count = int(len(self.inputs) * validation_split)

        self.inputs, self.validation_inputs = self.inputs[:train_count], self.inputs[train_count:]

        # go through each output.  We have to do it here, as we want nicely shuffled data
        outputs, validation_outputs = [], []
        for o in self.outputs:
            outputs.append(o[:train_count])
            validation_outputs.append(o[train_count:])
        self.outputs = outputs
        self.validation_outputs = validation_outputs

    def get_number_outputs(self):
        return len(self.outputs)

    def validation(self, percent):
        count = int(len(self.inputs) * percent)
        outputs = [o[:count] for o in self.validation_outputs]
        return self.validation_inputs[:count], outputs

    def resample(self, percent):
        self.shuffle()
        count = int(len(self.inputs) * percent)
        outputs = [o[:count] for o in self.outputs]
        return self.inputs[:count], outputs

    def debug(self):
        # good to see some outputs
        for x in (10, 420, 42):
            log.info('train input, shape: %s.  Example: %s' % (self.inputs.shape, self.inputs[x]))
            for o in self.outputs:
                log.info('train output, shape: %s.  Example: %s' % (o.shape, o[x]))

    def __len__(self):
        return len(self.inputs)


def validation_data(leveled_data, buckets):
    assert leveled_data

    number_outputs = leveled_data[0].get_number_outputs()

    # will concatenate at end
    v_inputs = []
    v_outputs = [[] for _ in range(number_outputs)]

    for depth, level_data in enumerate(leveled_data):
        percent = buckets.get(depth, len(leveled_data))
        ins, outs = level_data.validation(percent)
        v_inputs.append(ins)
        for o, o2 in zip(v_outputs, outs):
            o.append(o2)

    inputs = np.concatenate(v_inputs)
    outputs = []
    for o in v_outputs:
        outputs.append(np.concatenate(o))

    # finally shuffle
    r = unison_shuffle(inputs, *outputs)
    inputs, outputs = r[0], r[1:]
    return inputs, outputs


def resample_data(leveled_data, buckets):
    assert leveled_data

    number_outputs = leveled_data[0].get_number_outputs()

    # will concatenate at end
    v_inputs = []
    v_outputs = [[] for _ in range(number_outputs)]

    for depth, level_data in enumerate(leveled_data):
        percent = buckets.get(depth, len(leveled_data))
        ins, outs = level_data.resample(percent)
        v_inputs.append(ins)
        for o, o2 in zip(v_outputs, outs):
            o.append(o2)

    inputs = np.concatenate(v_inputs)
    outputs = []
    for o in v_outputs:
        outputs.append(np.concatenate(o))

    # finally shuffle
    r = unison_shuffle(inputs, *outputs)
    inputs, outputs = r[0], r[1:]
    return inputs, outputs


class TrainManager(object):
    def __init__(self, train_config, transformer):
        # take a copy of the initial train_config
        assert isinstance(train_config, confs.TrainNNConfig)
        self.train_config = attrutil.clone(train_config)

        attrutil.pprint(train_config)

        self.transformer = transformer

        # lookup via game_name (this gets statemachine & statemachine model)
        self.game_info = lookup.by_name(train_config.game)

        # will be created in gather_data
        self.buckets = None
        self.samples_buffer = None
        self.next_generation = "%s_%s" % (train_config.generation_prefix, train_config.next_step)

    def update_config(self, train_config, next_generation_prefix=None):
        assert train_config.game == self.train_config.game

        # simply update these
        for attr_name in ("use_previous next_step overwrite_existing "
                          "validation_split batch_size epochs "
                          "compile_strategy learning_rate".split()):
            value = getattr(train_config, attr_name)
            setattr(self.train_config, attr_name, value)

        rebuild_cache = False
        if train_config.drop_dupes_count < self.train_config.drop_dupes_count:
            rebuild_cache = True

        if train_config.starting_step < self.train_config.starting_step:
            rebuild_cache = True

        self.train_config.starting_step = train_config.starting_step
        self.train_config.drop_dupes_count = train_config.drop_dupes_count
        if rebuild_cache:
            self.buckets = None
            self.samples_buffer = None

        if next_generation_prefix is not None:
            self.next_generation = "%s_%s" % (next_generation_prefix, train_config.next_step)
        else:
            self.next_generation = "%s_%s" % (train_config.generation_prefix, train_config.next_step)

    def get_network(self, nn_model_config, generation_descr):
        # abbreviate, easier on the eyes
        conf = self.train_config

        attrutil.pprint(nn_model_config)

        man = get_manager()
        if man.can_load(conf.game, self.next_generation):
            msg = "Generation already exists %s / %s" % (conf.game, self.next_generation)
            log.error(msg)
            if not conf.overwrite_existing:
                raise TrainException("Generation already exists %s / %s" % (conf.game,
                                                                            self.next_generation))
        nn = None
        retraining = False
        if conf.use_previous:
            prev_generation = "%s_%s_prev" % (conf.generation_prefix,
                                              conf.next_step - 1)

            if man.can_load(conf.game, prev_generation):
                log.info("Previous generation found: %s" % prev_generation)
                nn = man.load_network(conf.game, prev_generation)
                retraining = True

            else:
                log.warning("No previous generation to use...")

        if nn is None:
            nn = man.create_new_network(conf.game, nn_model_config, generation_descr)

        nn.summary()

        self.nn = nn
        self.retraining = retraining
        log.info("Network %s, retraining: %s" % (self.nn, self.retraining))

    def gather_data(self):
        # abbreviate, easier on the eyes
        conf = self.train_config

        if self.samples_buffer is None:
            print "Recreating samples buffer"
            self.samples_buffer = SamplesBuffer()
            self.buckets = Buckets(conf.resample_buckets)

        total_samples = 0
        count = Counter()
        leveled_data = []
        for fn, sample_data in self.samples_buffer.files_to_sample_data(conf):
            assert sample_data.game == conf.game

            log.debug("Proccesing %s" % fn)
            log.debug("Game %s, with gen: %s and sample count %s" % (sample_data.game,
                                                                     sample_data.with_generation,
                                                                     sample_data.num_samples))

            if not sample_data.transformed:
                # sample_data.verify_samples(self.game_info.get_sm())
                sample_data.transform_all(self.transformer)

            level_data = LevelData(len(leveled_data))

            # XXX is this deduping a good idea?  Once we start using prev_states, then there will
            # be a lot less deduping?  I guess it is likely not too different.
            for state, ins, outs in sample_data:
                # keep the top n only?
                if conf.drop_dupes_count > 0 and count[state] == conf.drop_dupes_count:
                    continue

                count[state] += 1
                level_data.add(ins, outs)

            num_samples = len(level_data)
            log.verbose("DROPPED DUPES %s" % (sample_data.num_samples - num_samples))
            log.verbose("Left over samples %d" % num_samples)

            log.verbose("Validation split")
            level_data.validation_split(conf.validation_split)

            leveled_data.append(level_data)

            total_samples += num_samples

        log.info("total samples: %s" % total_samples)

        return leveled_data

    def do_epochs(self, leveled_data):
        conf = self.train_config

        # first get validation data, then we can forget about it as it doesn't need reshuffled
        validation_inputs, validation_outputs = validation_data(leveled_data, self.buckets)

        num_epochs = conf.epochs if self.retraining else conf.epochs * 2

        training_logger = network.TrainingLoggerCb(num_epochs)
        controller = network.TrainingController(self.retraining,
                                                len(self.transformer.policy_dist_count))

        # XXX add hyper parameters

        XX_value_weight_reduction = 0.333
        XX_value_weight_min = 0.05

        value_weight = 1.0

        self.nn.compile(self.train_config.compile_strategy,
                        self.train_config.learning_rate,
                        value_weight=value_weight)

        # num_samples = len(train_data.inputs)

        for i in range(num_epochs):

            if controller.stop_training:
                log.warning("Stop training early via controller")
                break

            # resample the samples!
            inputs, outputs = resample_data(leveled_data, self.buckets)

            if i > 0:
                log.info("controller.value_loss_diff %.3f" % controller.value_loss_diff)

                orig_weight = value_weight
                if controller.value_loss_diff > 0.004:
                    value_weight *= XX_value_weight_reduction
                elif controller.value_loss_diff > 0.001:
                    value_weight *= (XX_value_weight_reduction * 2)
                else:
                    # increase it again???
                    if controller.value_loss_diff < 0:
                        value_weight /= XX_value_weight_reduction

                    elif orig_weight < 0.5 and controller.value_loss_diff < 0.002:
                        value_weight /= (XX_value_weight_reduction * 2)

                value_weight = min(max(XX_value_weight_min, value_weight), 1.0)
                if abs(value_weight - orig_weight) > 0.0001:
                    self.nn.compile(self.train_config.compile_strategy,
                                    self.train_config.learning_rate,
                                    value_weight=value_weight)

            self.nn.fit(inputs,
                        outputs,
                        validation_inputs,
                        validation_outputs,
                        conf.batch_size,
                        shuffle=False,
                        callbacks=[training_logger, controller])

        self.controller = controller

    def save(self, generation_test_name=None):
        man = get_manager()
        if generation_test_name:
            man.save_network(self.nn, generation_name=generation_test_name)
            return

        man.save_network(self.nn, generation_name=self.next_generation)

        ###############################################################################
        # save a previous model for next time
        if self.controller.retrain_best is None:
            log.warning("No retraining network")
            return

        log.info("Saving retraining network with val_policy_acc: %.4f" % (
            self.controller.retrain_best_val_policy_acc))

        # there is an undocumented keras clone function, but this is sure to work (albeit slow and evil)
        from ggpzero.util.keras import keras_models

        for_next_generation = "%s_prev" % self.next_generation

        prev_model = keras_models.model_from_json(self.nn.keras_model.to_json())
        prev_model.set_weights(self.controller.retrain_best)

        prev_generation_descr = attrutil.clone(self.nn.generation_descr)
        prev_generation_descr.name = for_next_generation
        prev_nn = network.NeuralNetwork(self.nn.gdl_bases_transformer,
                                        prev_model, prev_generation_descr)
        man.save_network(prev_nn, for_next_generation)
