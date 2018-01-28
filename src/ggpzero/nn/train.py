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
    for a in arrays:
        return a[perms]


class TrainException(Exception):
    pass


def reshape(d):
    new_shape = [-1] + list(d[0].shape)
    return np.concatenate(d, axis=0).reshape(new_shape)


def transpose_and_reshape(outputs):
    new_d = []
    for idx in range(len(outputs[0])):
        row = []
        for o in outputs:
            row.append(o[idx])

        new_d.append(reshape(row))
    return new_d


class TrainData(object):
    def __init__(self):
        self.inputs = []
        self.outputs = []

        self.validation_inputs = []
        self.validation_outputs = []

    def summary(self):
        assert len(self.inputs) == len(self.outputs)
        assert len(self.validation_inputs) == len(self.validation_outputs)
        log.info("SamplesBuffer - train size %d, validate size %d" % (len(self.inputs),
                                                                      len(self.validation_inputs)))

    def reshape_all(self):
        inputs = reshape(self.inputs)
        validation_inputs = reshape(self.validation_inputs)

        outputs = transpose_and_reshape(self.outputs)
        validation_outputs = transpose_and_reshape(self.validation_outputs)

        # good to see some outputs
        for x in (10, 420, 42):
            log.info('train input, shape: %s.  Example: %s' % (inputs.shape, inputs[x]))
            for o in outputs:
                log.info('train output, shape: %s.  Example: %s' % (o.shape, o[x]))

        return inputs, outputs, validation_inputs, validation_outputs

    def strip(self, train_count, validate_count):
        ''' Needs to throw away the tail of the list (since that is the oldest data) '''

        self.inputs = self.inputs[:train_count]
        self.outputs = self.outputs[:train_count]

        self.validation_inputs = self.validation_inputs[:validate_count]
        self.validation_outputs = self.validation_outputs[:validate_count]


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
        self.data = TrainData()
        self.sample_data_cache = {}

    def files_to_sample_data(self, conf):
        man = get_manager()

        assert isinstance(conf, confs.TrainNNConfig)

        step = conf.next_step - 1
        while step >= conf.starting_step:
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


class TrainManager(object):
    def __init__(self, train_config, transformer, next_generation_prefix=None):
        assert isinstance(train_config, confs.TrainNNConfig)
        attrutil.pprint(train_config)
        self.train_config = train_config

        self.transformer = transformer

        # lookup via game_name (this gets statemachine & statemachine model)
        self.game_info = lookup.by_name(train_config.game)

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

        samples_buffer = SamplesBuffer()

        total_samples = 0
        count = Counter()

        train_data = TrainData()

        for fn, sample_data in samples_buffer.files_to_sample_data(conf):
            assert sample_data.game == conf.game

            log.debug("Proccesing %s" % fn)
            log.debug("Game %s, with gen: %s and sample count %s" % (sample_data.game,
                                                                     sample_data.with_generation,
                                                                     sample_data.num_samples))

            if not sample_data.transformed:
                sample_data.transform_all(self.transformer)

            data = []

            # XXX is this deduping a good idea?  Once we start using prev_states, then there will
            # be a lot less deduping?  I guess it is likely not too different.
            for state, ins, outs in sample_data:
                # keep the top n only?
                if conf.drop_dupes_count > 0 and count[state] == conf.drop_dupes_count:
                    continue

                count[state] += 1
                data.append((ins, outs))

            log.verbose("DROPPED DUPES %s" % (sample_data.num_samples - len(data)))

            num_samples = len(data)
            log.verbose("Left over samples %d" % num_samples)

            train_count = int(num_samples * conf.validation_split)

            for ins, outs in data[:train_count]:
                train_data.inputs.append(ins)
                train_data.outputs.append(outs)

            for ins, outs in data[train_count:]:
                train_data.validation_inputs.append(ins)
                train_data.validation_outputs.append(outs)

            total_samples += num_samples
            if total_samples > conf.max_sample_count:
                break

        train_data.summary()

        if conf.max_sample_count < total_samples:
            train_count = int(conf.max_sample_count * conf.validation_split)
            validate_count = conf.max_sample_count - train_count

            log.info("Stripping %s samples from data set" % (total_samples -
                                                             conf.max_sample_count))

            train_data.strip(train_count, validate_count)

        train_data.summary()
        return train_data

    def do_epochs(self, train_data):
        conf = self.train_config

        training_logger = network.TrainingLoggerCb(conf.epochs)
        controller = network.TrainingController(self.retraining,
                                                len(self.transformer.policy_dist_count))

        # XXX add hyper parameters

        XX_value_weight_reduction = 0.333
        XX_value_weight_min = 0.05

        value_weight = 1.0
        # start retraining with reduce weight
        if self.retraining:
            value_weight *= XX_value_weight_reduction

        self.nn.compile(self.train_config.compile_strategy,
                        self.train_config.learning_rate,
                        value_weight=value_weight)

        num_samples = len(train_data.inputs)

        inputs, outputs, validation_inputs, validation_outputs = train_data.reshape_all()

        for i in range(conf.epochs):

            if controller.stop_training:
                log.warning("Stop training early via controller")
                break

            # let's shuffle our data by ourselvs
            unison_shuffle(inputs, *outputs)

            if (conf.max_epoch_samples_count > 0 and
                conf.max_epoch_samples_count < num_samples):
                inputs = inputs[:conf.max_epoch_samples_count]
                outputs = [o[:conf.max_epoch_samples_count] for o in outputs]

            if i > 0:
                log.info("controller.value_loss_diff %.3f" % controller.value_loss_diff)

                orig_weight = value_weight
                if orig_weight > 0.2 and controller.value_loss_diff > 0.001:
                    value_weight *= XX_value_weight_reduction
                elif controller.value_loss_diff > 0.01:
                    value_weight *= XX_value_weight_reduction
                else:
                    # increase it again...
                    value_weight /= XX_value_weight_reduction

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
