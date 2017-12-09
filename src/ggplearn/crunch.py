''' no idea what to call this file.  Doing data crunching... so there you go. '''
import os

from collections import OrderedDict

import json

import numpy as np

from ggplib.util import log

from ggplib.db import lookup

from ggplearn import net
from ggplearn import net_config

json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)


def get_from_json(path, includes=None, excludes=None):
    # XXX fn is a bit too generic to be here '

    includes = includes or []
    excludes = excludes or []
    files = os.listdir(path)
    for f in files:
        if not f.endswith(".json"):
            continue

        if len([n for n in includes if n not in f]):
            continue

        if len([n for n in excludes if n in f]):
            continue

        for n in excludes:
            if n in f:
                continue

        filename = os.path.join(path, f)
        buf = open(filename).read()
        yield json.loads(buf), filename


class Sample(object):
    def __init__(self, state, policy_dist, final_scores):
        assert isinstance(state, tuple)

        # This is for breakthough
        assert len(state) == 130

        # this is also 2 player specfic
        assert len(final_scores) == 2

        for pair in policy_dist:
            assert len(pair) == 2

        self.state = state
        self.policy_dist = policy_dist
        self.final_scores = final_scores

        # XXX needing populated
        self.depth = -1
        self.generation = ""
        self.lead_role_index = -1

    def policy_as_array(self, base_infos, sm_model):
        # XXX tmp - breakthrough specific
        if self.lead_role_index == -1:
            for idx, binfo in enumerate(base_infos):
                if not self.state[idx]:
                    continue

                if binfo.terms[0] == "control":
                    if binfo.terms[1] == "white":
                        self.lead_role_index = 0
                    else:
                        assert binfo.terms[1] == "black"
                        self.lead_role_index = 1

        assert self.lead_role_index >= 0
        index_start = 0 if self.lead_role_index == 0 else len(sm_model.actions[0])
        expected_actions_len = sum(len(actions) for actions in sm_model.actions)
        policy_outputs = np.zeros(expected_actions_len)
        for idx, prob in self.policy_dist:
            policy_outputs[idx + index_start] = prob
        return policy_outputs

    def to_desc(self):
        d = OrderedDict()
        d["state"] = self.state
        d["depth"] = self.depth
        d["generation"] = self.generation
        d["lead_role_index"] = self.lead_role_index
        d["policy_dist"] = self.policy_dist
        d["final_scores"] = self.final_scores
        return d


class TrainerBase(object):
    BATCH_SIZE = 128
    EPOCHS = 24

    def __init__(self, game_name, generation):
        self.game_name = game_name
        self.generation = generation

        # lookup via game_name (this gets statemachine & statemachine model)
        self.game_info = lookup.by_name(game_name)

        # the bases config (XXX idea is not to use hard coded stuff)
        self.nn_config = net_config.get_bases_config(game_name)

        # update the base_infos and config
        self.base_infos = net_config.create_base_infos(self.nn_config,
                                                       self.game_info.model)

        self.number_of_outputs = sum(len(l) for l in self.game_info.model.actions)

        # keras model
        self.nn_model = net.get_network_model(self.nn_config,
                                              self.number_of_outputs)

        self.training_inputs, self.training_outputs = [], []
        self.training_inputs, self.training_outputs = [], []

    def sample_to_nn_style(self, sample, data):
        # transform samples -> numpy arrays as inputs/outputs to nn

        # input 1
        x = net_config.state_to_channels(sample.state,
                                         sample.lead_role_index,
                                         self.nn_config,
                                         self.base_infos)
        # input - planes
        data[0].append(x)

        # input - non planes
        x = [v for v, base_info in zip(sample.state, self.base_infos)
             if base_info.channel is None]
        data[1].append(x)

        # output - policy
        data[2].append(sample.policy_as_array(self.base_infos,
                                              self.game_info.model))

        # output - best/final scores
        data[3].append(sample.final_scores)

    def massage_data(self, training_samples, validation_samples):
        training_data = [[] for _ in range(4)]
        validation_data = [[] for _ in range(4)]

        for sample in training_samples:
            self.sample_to_nn_style(sample, training_data)

        for sample in validation_samples:
            self.sample_to_nn_style(sample, validation_data)

        for i, data in enumerate(training_data):
            a = np.array(data)
            a.astype('float32')
            training_data[i] = a
            log.debug("Shape of training data %d: %s" % (i, a.shape))

        for i, data in enumerate(validation_data):
            a = np.array(data)
            a.astype('float32')
            validation_data[i] = a
            log.debug("Shape of validation data %d: %s" % (i, a.shape))

        self.training_inputs = training_data[:2]
        self.training_outputs = training_data[2:]

        self.validation_inputs = validation_data[:2]
        self.validation_outputs = validation_data[2:]

    def nn_summary(self):
        ' log keras nn summary '

        # one way to get print_summary to output string!
        lines = []
        self.nn_model.summary(print_fn=lines.append)
        for l in lines:
            log.verbose(l)

    def train(self):
        self.nn_model.fit(self.training_inputs,
                          self.training_outputs,
                          verbose=0,
                          batch_size=self.BATCH_SIZE,
                          epochs=self.EPOCHS,
                          validation_data=[self.validation_inputs, self.validation_outputs],
                          callbacks=[net.MyProgbarLogger(), net.MyCallback()],
                          shuffle=True)

    def save(self):
        # save model / weights
        with open("models/model_nn_%s_%s.json" % (self.game_name, self.generation), "w") as f:
            f.write(self.nn_model.to_json())

        self.nn_model.save_weights("models/weights_nn_%s_%s.h5" % (self.game_name,
                                                                   self.generation),
                                   overwrite=True)
