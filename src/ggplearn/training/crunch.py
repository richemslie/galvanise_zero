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
    for the_file in files:
        if not the_file.endswith(".json"):
            continue

        if len([ii for ii in includes if ii not in the_file]):
            continue

        if len([ii for ii in excludes if ii in the_file]):
            continue

        for ii in excludes:
            if ii in the_file:
                continue

        filename = os.path.join(path, the_file)
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

    def ensure_lead_role_index_set(self, base_infos):
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

    def policy_as_array(self, sm_model):
        index_start = 0 if self.lead_role_index == 0 else len(sm_model.actions[0])
        expected_actions_len = sum(len(actions) for actions in sm_model.actions)
        policy_outputs = np.zeros(expected_actions_len)
        for idx, prob in self.policy_dist:
            policy_outputs[idx + index_start] = prob
        return policy_outputs

    def to_desc(self):
        desc = OrderedDict()
        desc["state"] = self.state
        desc["depth"] = self.depth
        desc["generation"] = self.generation
        desc["lead_role_index"] = self.lead_role_index
        desc["policy_dist"] = self.policy_dist
        desc["final_scores"] = self.final_scores
        return desc


class SampleToNetwork(object):
    def __init__(self, game_name, generation):
        self.game_name = game_name
        self.generation = generation

        # lookup via game_name (this gets statemachine & statemachine model)
        self.game_info = lookup.by_name(game_name)

        # the bases config (XXX idea is not to use hard coded stuff)
        self.base_config = net_config.get_bases_config(game_name,
                                                       self.game_info.model,
                                                       self.generation)

        self.training_inputs, self.training_outputs = [], []
        self.training_inputs, self.training_outputs = [], []

    def create_network(self, **kwds):
        return self.base_config.create_network(**kwds)

    def sample_to_nn_style(self, sample, data):
        # transform samples -> numpy arrays as inputs/outputs to nn

        sample.ensure_lead_role_index_set(self.base_config.base_infos)

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
        data[3].append(sample.final_scores)

    def massage_data(self, training_samples, validation_samples):
        training_data = [[] for _ in range(4)]
        validation_data = [[] for _ in range(4)]

        for sample in training_samples:
            self.sample_to_nn_style(sample, training_data)

        for sample in validation_samples:
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

        return net.TrainData(inputs=training_data[:2],
                             outputs=training_data[2:],
                             validation_inputs=validation_data[:2],
                             validation_outputs=validation_data[2:])
