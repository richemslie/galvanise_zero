import os

from ggplib.util import log
from ggplib.db import lookup

from ggpzero.util import attrutil

from ggpzero.defs import msgs, confs, templates

from ggpzero.nn.manager import get_manager


class TrainException(Exception):
    pass


def check_sample(sample, game_model):
    assert len(sample.state) == len(game_model.bases)
    assert len(sample.final_score) == len(game_model.roles)

    num_actions = sum(len(actions) for actions in game_model.actions)
    for legal, p in sample.policy:
        assert 0 <= legal < num_actions
        assert -0.01 < p < 1.01

    assert 0 <= sample.lead_role_index <= len(game_model.roles)


class SamplesBuffer(object):
    ''' this is suppose to be our in memory buffer.  But we throw it away everytime we train.  XXX
    ie should we keep it in memory, or is it too big?  '''

    def __init__(self, transformer):
        self.transformer = transformer
        self.train_samples = []
        self.validation_samples = []

    def add(self, sample, validation=False):
        assert isinstance(sample, confs.Sample)

        if validation:
            self.validation_samples.append(sample)
        else:
            self.train_samples.append(sample)

    def strip(self, train_count, validate_count):
        ''' Needs to throw away the tail of the list '''
        self.train_samples = self.train_samples[:train_count]
        self.validation_samples = self.validation_samples[:validate_count]

    def create_training_conf(self):
        conf = confs.TrainData()

        log.debug("transforming training samples: %s" % len(self.train_samples))
        for sample in self.train_samples:
            self.transformer.sample_to_nn(sample,
                                          conf.input_channels,
                                          conf.output_policies,
                                          conf.output_final_scores)

        log.debug("transforming validation samples: %s" % len(self.validation_samples))
        for sample in self.validation_samples:
            self.transformer.sample_to_nn(sample,
                                          conf.validation_input_channels,
                                          conf.validation_output_policies,
                                          conf.validation_output_final_scores)

        return conf


def get_data(conf):
    assert isinstance(conf, msgs.TrainNNRequest)

    step = conf.next_step - 1
    while step >= conf.starting_step:
        fn = os.path.join(conf.store_path, "gendata_%s_%s.json" % (conf.game, step))
        yield fn, attrutil.json_to_attr(open(fn).read())
        step -= 1


def parse(conf, game_info, transformer):
    assert isinstance(conf, msgs.TrainNNRequest)

    samples_buffer = SamplesBuffer(transformer)

    total_samples = 0
    from collections import Counter
    count = Counter()

    for fn, gen_data in get_data(conf):
        log.debug("Proccesing %s" % fn)
        log.debug("Game %s, with gen: %s and sample count %s" % (gen_data.game,
                                                                 gen_data.with_generation,
                                                                 gen_data.num_samples))

        samples = []
        # XXX is this deduping a good idea?  Once we start using prev_states, then there will be a
        # lot less deduping?
        for g in gen_data.samples:
            s = tuple(g.state)
            # keep the top n only?
            if conf.drop_dupes_count > 0 and count[s] == conf.drop_dupes_count:
                continue

            count[s] += 1
            samples.append(g)

        log.verbose("DROPPED DUPES %s" % (gen_data.num_samples - len(samples)))

        assert gen_data.game == conf.game
        num_samples = len(samples)
        train_count = int(num_samples * conf.validation_split)

        for s in samples[:train_count]:
            check_sample(s, game_info.model)
            samples_buffer.add(s)

        for s in samples[train_count:]:
            check_sample(s, game_info.model)
            samples_buffer.add(s, validation=True)

        total_samples += num_samples
        if total_samples > conf.max_sample_count:
            break

    log.info("Number of samples %s" % total_samples)

    if conf.max_sample_count < total_samples:
        train_count = int(conf.max_sample_count * conf.validation_split)
        validate_count = conf.max_sample_count - train_count
        log.info("Stripping %s samples from data set" % (total_samples -
                                                         conf.max_sample_count))
        samples_buffer.strip(train_count, validate_count)

        log.info("Number of samples %s" % total_samples)

    return samples_buffer.create_training_conf()


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
