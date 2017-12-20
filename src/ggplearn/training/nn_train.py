import os
import numpy as np

from ggplib.util import log

from ggplib.db import lookup

from ggplearn.util import attrutil

from ggplearn.nn import bases

from ggplearn import msgdefs


def check_sample(sample, game_model):
    assert len(sample.state) == len(game_model.bases)
    assert len(sample.final_score) == len(game_model.roles)

    num_actions = sum(len(actions) for actions in game_model.actions)
    for legal, p in sample.policy:
        assert 0 <= legal < num_actions
        assert -0.01 < p < 1.01

    assert 0 <= sample.lead_role_index <= len(game_model.roles)


class SamplesHolder(object):
    def __init__(self, game_info, bases_config):
        assert len(game_info.model.roles) == 2, "only 2 roles supported for now"

        self.game_info = game_info
        self.bases_config = bases_config
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
        planes = self.bases_config.state_to_channels(sample.state, sample.lead_role_index)

        # input - planes
        data[0].append(planes)

        # input - non planes
        non_planes = self.bases_config.get_non_cord_input(sample.state)
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
    last_step = conf.next_step - 1
    step = 0
    while step <= last_step:
        fn = os.path.join(conf.store_path, "gendata_%s_%s.json" % (conf.game, step))
        yield fn, attrutil.json_to_attr(open(fn).read())
        step += 1


def parse_and_train(conf):
    assert isinstance(conf, msgdefs.TrainNNRequest)

    attrutil.pprint(conf)

    use_previous = conf.use_previous
    if use_previous:
        if conf.next_step % 5 == 0:
            log.warning("Not using previous since time to cycle...")
            use_previous = False

    # lookup via game_name (this gets statemachine & statemachine model)
    game_info = lookup.by_name(conf.game)

    prev_generation = "%sgen_%s_%s_prev" % (conf.generation_prefix,
                                            conf.network_size,
                                            conf.next_step - 1)

    next_generation = "%sgen_%s_%s" % (conf.generation_prefix,
                                       conf.network_size,
                                       conf.next_step)

    nn = None
    if use_previous:
        bases_config = bases.get_config(conf.game,
                                        game_info.model,
                                        prev_generation)
        nn = bases_config.create_network()
        if nn.can_load():
            log.info("Previous generation found.")

            nn.load()
            bases_config.update_generation(next_generation)

        else:
            log.warning("No previous generation to use...")
            nn = None

    if nn is None:
        bases_config = bases.get_config(conf.game,
                                        game_info.model,
                                        next_generation)

        # more parameters passthrough?  XXX
        nn = bases_config.create_network(network_size=conf.network_size)

    nn.summary()

    samples_holder = SamplesHolder(game_info, bases_config)

    total_samples = 0
    for fn, gen_data in get_data(conf):
        log.debug("Proccesing %s" % fn)
        log.debug("Game %s, with policy gen: %s, with score gen: %s" % (gen_data.game,
                                                                        gen_data.with_policy_generation,
                                                                        gen_data.with_score_generation))

        assert gen_data.num_samples == len(gen_data.samples)

        assert gen_data.game == conf.game
        train_count = int(gen_data.num_samples * conf.validation_split)

        for s in gen_data.samples[:train_count]:
            check_sample(s, game_info.model)
            samples_holder.add(s)

        for s in gen_data.samples[train_count:]:
            check_sample(s, game_info.model)
            samples_holder.add(s, validation=True)

        total_samples += gen_data.num_samples

    if conf.max_sample_count < total_samples:
        train_count = int(conf.max_sample_count * conf.validation_split)
        validate_count = conf.max_sample_count - train_count
        log.info("Stripping %s samples from data set" % (total_samples - conf.max_sample_count))
        samples_holder.strip(train_count, validate_count)

    train_conf = samples_holder.massage_data()
    train_conf.epochs = conf.epochs
    train_conf.batch_size = conf.batch_size

    nn.compile()
    res = nn.train(train_conf)
    nn.save()

    ###############################################################################
    # save a previous model for next time
    for_next_generation = "%sgen_%s_%s_prev" % (conf.generation_prefix,
                                                conf.network_size,
                                                conf.next_step)
    bases_config.update_generation(for_next_generation)

    log.info("Saving retraining network with val_policy_acc: %.4f" % res.retrain_best_val_policy_acc)
    nn.get_model().set_weights(res.retrain_best)
    nn.save()
