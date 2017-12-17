import os
import numpy as np

from ggplib.util import log

from ggplib.db import lookup

from ggplearn.util import attrutil

from ggplearn.nn import bases, network

from ggplearn.distributed import msgs


class Sample(object):
    def __init__(self, state, policy_dist, final_score, lead_role_index):
        state = tuple(state)

        # XXX ALL GAME SPECIFIC
        # This is for breakthough
        assert len(state) == 130

        # this is also 2 player specfic
        assert len(final_score) == 2
        assert isinstance(final_score, dict)
        final_score = final_score['white'] / 100.0, final_score['black'] / 100.0

        for pair in policy_dist:
            assert len(pair) == 2

        self.state = state
        self.policy_dist = policy_dist
        self.final_score = final_score
        self.lead_role_index = lead_role_index

    def policy_as_array(self, sm_model):
        index_start = 0 if self.lead_role_index == 0 else len(sm_model.actions[0])
        expected_actions_len = sum(len(actions) for actions in sm_model.actions)
        policy_outputs = np.zeros(expected_actions_len)
        for idx, prob in self.policy_dist:
            policy_outputs[idx + index_start] = prob
        return policy_outputs


class SamplesHolder(object):
    def __init__(self, game_info, base_config):
        self.game_info = game_info
        self.base_config = base_config
        self.train_samples = []
        self.validation_samples = []

    def add(self, sample, validation=False):
        assert isinstance(sample, msgs.Sample)

        # convert to a local sample
        s = Sample(sample.state,
                   sample.policy,
                   sample.final_score,
                   sample.lead_role_index)

        if validation:
            self.validation_samples.append(s)
        else:
            self.train_samples.append(s)

    def sample_to_nn_style(self, sample, data):

        # transform samples -> numpy arrays as inputs/outputs to nn

        # input 1
        planes = self.base_config.state_to_channels(sample.state, sample.lead_role_index)

        # input - planes
        data[0].append(planes)

        # input - non planes
        non_planes = self.base_config.get_non_cord_input(sample.state)
        data[1].append(non_planes)

        # output - policy
        data[2].append(sample.policy_as_array(self.game_info.model))

        # output - best/final scores
        data[3].append(sample.final_score)

    def massage_data(self):
        training_data = [[] for _ in range(4)]
        validation_data = [[] for _ in range(4)]

        for sample in self.train_samples:
            self.sample_to_nn_style(sample, training_data)

        for sample in self.validation_samples:
            self.sample_to_nn_style(sample, validation_data)

        for ii, data in enumerate(training_data):
            arr = np.array(data)
            arr.astype('float32')
            training_data[ii] = arr
            log.debug("Shape of training data %d: %s" % (ii, arr.shape))

        for ii, data in enumerate(validation_data):
            arr = np.array(data)
            arr.astype('float32')
            validation_data[ii] = arr
            log.debug("Shape of validation data %d: %s" % (ii, arr.shape))

        # good always a good idea to print some outputs
        print training_data[0][-120]
        print training_data[1][-120]
        print training_data[2][-120]
        print training_data[3][-120]

        return network.TrainData(inputs=training_data[:2],
                                 outputs=training_data[2:],
                                 validation_inputs=validation_data[:2],
                                 validation_outputs=validation_data[2:])


def get_data(store_path, last_step):
    step = 0
    while step <= last_step:
        f = open(os.path.join(store_path, "gendata_%s.json" % step))
        yield attrutil.json_to_attr(f.read())
        step += 1


def parse_and_train(conf):
    assert isinstance(conf, msgs.TrainNNRequest)

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
        base_config = bases.get_config(conf.game,
                                       game_info.model,
                                       prev_generation)
        nn = base_config.create_network()
        if nn.can_load():
            log.info("Previous generation found.")

            nn.load()
            base_config.update_generation(next_generation)

        else:
            log.warning("No previous generation to use...")
            nn = None

    if nn is None:
        base_config = bases.get_config(conf.game,
                                       game_info.model,
                                       next_generation)

        # more parameters passthrough?  XXX
        nn = base_config.create_network(network_size=conf.network_size)

    samples_holder = SamplesHolder(game_info, base_config)

    for gen_data in get_data(conf.store_path, conf.next_step - 1):
        print "with policy gen", gen_data.with_policy_generation
        print "with score gen", gen_data.with_score_generation
        print "number of samples", gen_data.num_samples
        assert gen_data.num_samples == len(gen_data.samples)

        assert gen_data.game == conf.game
        train_count = int(gen_data.num_samples * conf.validation_split)

        for s in gen_data.samples[:train_count]:
            samples_holder.add(s)

        for s in gen_data.samples[train_count:]:
            samples_holder.add(s, validation=True)

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
    base_config.update_generation(for_next_generation)

    log.info("Saving retraining network with val_policy_acc: %.4f" % res.retrain_best_val_policy_acc)
    nn.get_model().set_weights(res.retrain_best)
    nn.save()


def go():
    from ggplib.util.init import setup_once
    setup_once()

    conf = msgs.TrainNNRequest()
    conf.game = "breakthrough"

    conf.network_size = "smaller"
    conf.generation_prefix = "v2_"
    conf.store_path = os.path.join(os.environ["GGPLEARN_PATH"], "data", "breakthrough", "v2")

    # uses previous network
    conf.use_previous = True
    conf.next_step = 52

    conf.validation_split = 0.8
    conf.batch_size = 64
    conf.epochs = 16
    conf.max_sample_count = 100000

    parse_and_train(conf)


def main_wrap(fn):
    import pdb
    import sys
    import traceback

    try:
        fn()

    except Exception as exc:
        print exc
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main_wrap(go)
