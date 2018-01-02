import os
import numpy as np

from ggplib.util import log
from ggplib.db import lookup

from ggplearn.util import attrutil

from ggplearn import msgdefs, templates

from ggplearn.nn.manager import get_manager


def check_sample(sample, game_model):
    assert len(sample.state) == len(game_model.bases)
    assert len(sample.final_score) == len(game_model.roles)

    num_actions = sum(len(actions) for actions in game_model.actions)
    for legal, p in sample.policy:
        assert 0 <= legal < num_actions
        assert -0.01 < p < 1.01

    assert 0 <= sample.lead_role_index <= len(game_model.roles)


class SamplesHolder(object):
    def __init__(self, game_info, transformer):
        assert len(game_info.model.roles) == 2, "only 2 roles supported for now"

        self.game_info = game_info
        self.transformer = transformer
        self.train_samples = []
        self.validation_samples = []

        self.policy_1_index_start = len(game_info.model.actions[0])
        self.expected_policy_len = sum(len(actions) for actions in game_info.model.actions)

    def add(self, sample, validation=False):
        assert isinstance(sample, msgdefs.Sample)

        if validation:
            self.validation_samples.append(sample)
        else:
            self.train_samples.append(sample)

    def strip(self, train_count, validate_count):
        # we can just cleverly use -1 * train_count, but this is clearer
        train_index = len(self.train_samples) - train_count
        validate_index = len(self.validation_samples) - validate_count
        self.train_samples = self.train_samples[train_index:]
        self.validation_samples = self.validation_samples[validate_index:]

    def policy_as_array(self, sample):
        index_start = 0 if sample.lead_role_index == 0 else self.policy_1_index_start
        policy_outputs = np.zeros(self.expected_policy_len)
        for idx, prob in sample.policy:
            policy_outputs[index_start + idx] = prob

        return policy_outputs

    def sample_to_nn_style(self, sample, data):
        check_sample(sample, self.game_info.model)

        # transform samples -> numpy arrays as inputs/outputs to nn

        # input 1
        planes = self.transformer.state_to_channels(sample.state, sample.lead_role_index)

        # input - planes
        data[0].append(planes)

        # input - non planes
        non_planes = self.transformer.get_non_cord_input(sample.state)
        data[1].append(non_planes)

        # output - policy
        data[2].append(self.policy_as_array(sample))

        # output - best/final scores
        data[3].append(sample.final_score)

    def massage_data(self):
        training_data = [[] for _ in range(4)]
        validation_data = [[] for _ in range(4)]

        log.debug("massaging training samples: %s" % len(self.train_samples))
        for sample in self.train_samples:
            self.sample_to_nn_style(sample, training_data)

        log.debug("massaging validation samples: %s" % len(self.validation_samples))
        for sample in self.validation_samples:
            self.sample_to_nn_style(sample, validation_data)

        for ii, data in enumerate(training_data):
            arr = np.array(data)
            arr.astype('float32')
            training_data[ii] = arr
            log.info("Shape of training data %d: %s" % (ii, arr.shape))

        for ii, data in enumerate(validation_data):
            arr = np.array(data)
            arr.astype('float32')
            validation_data[ii] = arr
            log.info("Shape of validation data %d: %s" % (ii, arr.shape))

        # good always a good idea to print some outputs
        print training_data[0][-120]
        print training_data[1][-120]
        print training_data[2][-120]
        print training_data[3][-120]

        return msgdefs.TrainData(inputs=training_data[:2],
                                 outputs=training_data[2:],
                                 validation_inputs=validation_data[:2],
                                 validation_outputs=validation_data[2:])


def get_data(conf):
    assert isinstance(conf, msgdefs.TrainNNRequest)

    step = conf.next_step - 1
    while step >= conf.starting_step:
        fn = os.path.join(conf.store_path, "gendata_%s_%s.json" % (conf.game, step))
        yield fn, attrutil.json_to_attr(open(fn).read())
        step -= 1


def parse(conf, game_info, transformer):
    assert isinstance(conf, msgdefs.TrainNNRequest)

    samples_holder = SamplesHolder(game_info, transformer)

    total_samples = 0
    from collections import Counter
    count = Counter()

    for fn, gen_data in get_data(conf):
        log.debug("Proccesing %s" % fn)
        log.debug("Game %s, with gen: %s and sample count %s" % (gen_data.game,
                                                                 gen_data.with_generation,
                                                                 gen_data.num_samples))

        samples = []
        # XXX should we even support this deduping?
        for g in gen_data.samples:
            s = tuple(g.state)
            # keep the top 3 only
            if count[s] == 4:
                pass  # XXX continue

            count[s] += 1
            samples.append(g)

        print "DROPPED DUPES", gen_data.num_samples - len(samples)

        # assert gen_data.num_samples == len(gen_data.samples)

        assert gen_data.game == conf.game
        num_samples = len(samples)
        train_count = int(num_samples * conf.validation_split)

        for s in samples[:train_count]:
            check_sample(s, game_info.model)
            samples_holder.add(s)

        for s in samples[train_count:]:
            check_sample(s, game_info.model)
            samples_holder.add(s, validation=True)

        total_samples += num_samples
        if total_samples > conf.max_sample_count:
            break

    log.info("Total samples %s" % total_samples)

    if conf.max_sample_count < total_samples:
        train_count = int(conf.max_sample_count * conf.validation_split)
        validate_count = conf.max_sample_count - train_count
        log.info("Stripping %s samples from data set" % (total_samples - conf.max_sample_count))
        samples_holder.strip(train_count, validate_count)

    train_conf = samples_holder.massage_data()
    return train_conf


def get_nn_model_conf(conf):
    # temp here
    nn_model_conf = templates.nn_model_config_template(conf.game, conf.network_size)

    # slightly bigger than small
    nn_model_conf.residual_layers = 8
    nn_model_conf.cnn_filter_size = 112

    nn_model_conf.dropout_rate_policy = 0.33
    nn_model_conf.dropout_rate_value = 0.5

    # ensure no regularisation
    nn_model_conf.alphazero_regularisation = False

    # this is learning rate for adam
    nn_model_conf.learning_rate = 0.001

    # nn_model_conf.alphazero_regularisation = True
    attrutil.pprint(nn_model_conf)

    return nn_model_conf


def parse_and_train(conf):
    assert isinstance(conf, msgdefs.TrainNNRequest)
    attrutil.pprint(conf)

    # lookup via game_name (this gets statemachine & statemachine model)
    game_info = lookup.by_name(conf.game)

    next_generation = "%sgen_%s_%s" % (conf.generation_prefix,
                                       conf.network_size,
                                       conf.next_step)

    man = get_manager()

    nn = None

    if conf.use_previous:
        prev_generation = "%sgen_%s_%s_prev" % (conf.generation_prefix,
                                                conf.network_size,
                                                conf.next_step - 1)

        if man.can_load(conf.game, prev_generation):
            log.info("Previous generation found: %s" % prev_generation)
            nn = man.load_network(conf.game, prev_generation)
        else:
            log.warning("No previous generation to use...")

    # XXX should be passed in
    nn_model_conf = get_nn_model_conf(conf)

    if nn is None:
        nn = man.create_new_network(conf.game, nn_model_conf)

    nn.summary()

    train_conf = parse(conf, game_info, man.get_transformer(conf.game))
    train_conf.epochs = conf.epochs
    train_conf.batch_size = conf.batch_size

    nn.compile(alphazero_regularisation=nn_model_conf.alphazero_regularisation,
               learning_rate=nn_model_conf.learning_rate)
    res = nn.train(train_conf)
    man.save_network(nn, conf.game, next_generation)

    ###############################################################################
    # save a previous model for next time
    if res.retrain_best is None:
        log.warning("No retraining network")
        return

    log.info("Saving retraining network with val_policy_acc: %.4f" % res.retrain_best_val_policy_acc)

    prev_nn = man.create_new_network(conf.game, nn_model_conf)
    prev_nn.get_model().set_weights(res.retrain_best)

    for_next_generation = "%sgen_%s_%s_prev" % (conf.generation_prefix,
                                                conf.network_size,
                                                conf.next_step)

    man.save_network(prev_nn, conf.game, for_next_generation)
