import os

from ggplib.util import log
from ggplib.db import lookup

from ggpzero.util import attrutil

from ggpzero.defs import msgs, confs, templates

from ggpzero.nn.manager import get_manager


# XXX general jumping through hoops to cache samples.  The right thing to do is store samples in an
# efficient and preprocessed format.


class TrainException(Exception):
    pass


class GenerationSamples(object):
    def __init__(self, game, with_generation, num_samples):
        self.game = game
        self.with_generation = with_generation
        self.num_samples = num_samples

        self.samples = []

        self.states = []
        self.input_channels = []
        self.output_policies = []
        self.output_final_scores = []
        self.transformed = False

    def add_sample(self, sample):
        self.samples.append(sample)

    def transform_all(self, transformer):
        for sample in self.samples:
            self.states.append(tuple(sample.state))
            transformer.check_sample(sample)
            transformer.sample_to_nn(sample,
                                     self.input_channels,
                                     self.output_policies,
                                     self.output_final_scores)
        self.transformed = True

        # free up memory
        self.samples = []

    def __iter__(self):
        assert self.transformed
        for s, ins, pol_head, val_head in zip(self.states, self.input_channels,
                                              self.output_policies, self.output_final_scores):
            yield s, ins, [pol_head, val_head]


class SamplesBuffer(object):
    ''' this is suppose to be our in memory buffer.  But we throw it away everytime we train.  XXX
    ie should we keep it in memory, or is it too big?  '''

    def __init__(self):
        self.conf = confs.TrainData()

    gen_data_cache = {}

    @classmethod
    def files_to_gen_samples(cls, conf):
        assert isinstance(conf, msgs.TrainNNRequest)

        step = conf.next_step - 1
        while step >= conf.starting_step:
            fn = os.path.join(conf.store_path, "gendata_%s_%s.json" % (conf.game, step))
            if fn not in cls.gen_data_cache:

                raw_data = attrutil.json_to_attr(open(fn).read())
                gen_samples = GenerationSamples(raw_data.game,
                                                raw_data.with_generation,
                                                raw_data.num_samples)

                for s in raw_data.samples:
                    gen_samples.add_sample(s)

                cls.gen_data_cache[fn] = gen_samples

            yield fn, cls.gen_data_cache[fn]
            step -= 1

    def add(self, ins, outs, validation=False):
        if validation:
            self.conf.validation_input_channels.append(ins)
            self.conf.validation_output_policies.append(outs[0])
            self.conf.validation_output_final_scores.append(outs[1])
        else:
            self.conf.input_channels.append(ins)
            self.conf.output_policies.append(outs[0])
            self.conf.output_final_scores.append(outs[1])

    def strip(self, train_count, validate_count):
        ''' Needs to throw away the tail of the list (since that is the oldest data) '''

        for name in "input_channels output_policies output_final_scores".split():
            data = getattr(self.conf, name)
            data = data[:train_count]
            setattr(self.conf, name, data)

        for name in "validation_input_channels validation_output_policies validation_output_final_scores".split():
            data = getattr(self.conf, name)
            data = data[:validate_count]
            setattr(self.conf, name, data)

    def check(self):
        c = self.conf
        assert len(c.input_channels) == len(c.output_policies)
        assert len(c.input_channels) == len(c.output_final_scores)

        assert len(c.validation_input_channels) == len(c.validation_output_policies)
        assert len(c.validation_input_channels) == len(c.validation_output_final_scores)

    def summary(self):
        log.info("SamplesBuffer - train size %d, validate size %d" % (len(self.conf.input_channels),
                                                                      len(self.conf.validation_input_channels)))


def parse(conf, game_info, transformer):
    assert isinstance(conf, msgs.TrainNNRequest)

    samples_buffer = SamplesBuffer()

    total_samples = 0
    from collections import Counter
    count = Counter()

    for fn, gen_data in samples_buffer.files_to_gen_samples(conf):
        assert gen_data.game == conf.game

        log.debug("Proccesing %s" % fn)
        log.debug("Game %s, with gen: %s and sample count %s" % (gen_data.game,
                                                                 gen_data.with_generation,
                                                                 gen_data.num_samples))

        if not gen_data.transformed:
            gen_data.transform_all(transformer)

        data = []

        # XXX is this deduping a good idea?  Once we start using prev_states, then there will be a
        # lot less deduping?
        for state, ins, outs in gen_data:
            # keep the top n only?
            if conf.drop_dupes_count > 0 and count[state] == conf.drop_dupes_count:
                continue

            count[state] += 1
            data.append((ins, outs))

        log.verbose("DROPPED DUPES %s" % (gen_data.num_samples - len(data)))

        num_samples = len(data)
        log.verbose("Left over samples %d" % num_samples)

        train_count = int(num_samples * conf.validation_split)

        for ins, outs in data[:train_count]:
            samples_buffer.add(ins, outs)

        for ins, outs in data[train_count:]:
            samples_buffer.add(ins, outs, validation=True)

        total_samples += num_samples
        if total_samples > conf.max_sample_count:
            break

    samples_buffer.check()
    samples_buffer.summary()

    if conf.max_sample_count < total_samples:
        train_count = int(conf.max_sample_count * conf.validation_split)
        validate_count = conf.max_sample_count - train_count
        log.info("Stripping %s samples from data set" % (total_samples -
                                                         conf.max_sample_count))
        samples_buffer.strip(train_count, validate_count)

    samples_buffer.check()
    samples_buffer.summary()
    return samples_buffer.conf


def parse_and_train(conf):
    assert isinstance(conf, msgs.TrainNNRequest)
    attrutil.pprint(conf)

    # lookup via game_name (this gets statemachine & statemachine model)
    game_info = lookup.by_name(conf.game)

    next_generation = "%s_%s" % (conf.generation_prefix, conf.next_step)

    man = get_manager()

    nn = None
    # check the generation does not already exist

    retraining = False
    if man.can_load(conf.game, next_generation):
        msg = "Generation already exists %s / %s" % (conf.game, next_generation)
        log.error(msg)
        if not conf.overwrite_existing:
            raise TrainException("Generation already exists %s / %s" % (conf.game, next_generation))

    if conf.use_previous:
        prev_generation = "%s_%s_prev" % (conf.generation_prefix,
                                          conf.next_step - 1)

        if man.can_load(conf.game, prev_generation):
            log.info("Previous generation found: %s" % prev_generation)
            nn = man.load_network(conf.game, prev_generation)
            retraining = True
        else:
            log.warning("No previous generation to use...")

    # XXX nn_model_conf should be passed in
    nn_model_conf = templates.nn_model_config_template(conf.game, conf.network_size)

    if nn is None:
        nn = man.create_new_network(conf.game, nn_model_conf)

    print attrutil.pprint(nn_model_conf)
    nn.summary()

    train_conf = parse(conf, game_info, man.get_transformer(conf.game))
    train_conf.epochs = conf.epochs
    train_conf.batch_size = conf.batch_size

    res = nn.train(train_conf, retraining=retraining)
    man.save_network(nn, conf.game, next_generation)

    ###############################################################################
    # save a previous model for next time
    if res.retrain_best is None:
        log.warning("No retraining network")
        return

    log.info("Saving retraining network with val_policy_acc: %.4f" % res.retrain_best_val_policy_acc)

    prev_nn = man.create_new_network(conf.game, nn_model_conf)
    prev_nn.get_model().set_weights(res.retrain_best)

    for_next_generation = "%s_%s_prev" % (conf.generation_prefix,
                                          conf.next_step)

    man.save_network(prev_nn, conf.game, for_next_generation)
