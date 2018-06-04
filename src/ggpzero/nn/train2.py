
from ggplib.util import log
from ggplib.db import lookup

from ggpzero.util import attrutil

from ggpzero.defs import confs

from ggpzero.nn.manager import get_manager

from ggpzero.nn import network
from ggpzero.nn import datacache

# XXX tmp:
from ggpzero.nn.train import TrainException


###############################################################################

class TrainManager(object):
    def __init__(self, train_config, transformer):
        # take a copy of the initial train_config
        assert isinstance(train_config, confs.TrainNNConfig)
        self.transformer = transformer

        # lookup via game_name (this gets statemachine & statemachine model)
        self.game_info = lookup.by_name(train_config.game)
        self.update_config(attrutil.clone(train_config))

    def update_config(self, train_config, next_generation_prefix=None):
        # abbreviate, easier on the eyes
        self.train_config = train_config
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

        self.cache = datacache.DataCache(self.transformer, conf.generation_prefix)
        self.cache.sync()
        # XXX fix worker.py and nn_train.py to not even call this

    def do_epochs(self, _):
        # XXX remove leveled_data, since gather_data does not return anything anymore

        conf = self.train_config

        train_at_step = conf.next_step - 1
        ignore_after_step = conf.starting_step
        assert 0 <= ignore_after_step <= train_at_step
        buckets_def = datacache.Buckets(conf.resample_buckets)

        indexer = self.cache.create_chunk_indexer(buckets_def,
                                                  starting_step=train_at_step,
                                                  ignore_after_step=ignore_after_step,
                                                  validation_split=conf.validation_split)

        # first get validation data, then we can forget about it as it doesn't need reshuffled
        validation_indices = indexer.validation_epoch()

        num_epochs = conf.epochs if self.retraining else conf.epochs * 2

        training_logger = network.TrainingLoggerCb(num_epochs)
        controller = network.TrainingController(self.retraining,
                                                len(self.transformer.policy_dist_count))

        # XXX add hyper parameters

        XX_value_weight_reduction = 0.333
        XX_value_weight_min = 0.05

        # all games are zero sum that are trained and because 2 values, x2 the error.  XXX
        value_weight = 1.0

        self.nn.compile(self.train_config.compile_strategy,
                        self.train_config.learning_rate,
                        value_weight=value_weight)

        for i in range(num_epochs):

            if controller.stop_training:
                log.warning("Stop training early via controller")
                break

            # resample the samples!
            training_indices = indexer.training_epoch()

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

            fitter = self.nn.keras_model.fit_generator
            fitter(self.cache.generate(training_indices, conf.batch_size),
                   len(training_indices) / conf.batch_size,
                   epochs=1,
                   verbose=0,
                   validation_data=self.cache.generate(validation_indices, conf.batch_size),
                   validation_steps=len(validation_indices) / conf.batch_size,
                   callbacks=[training_logger, controller],
                   shuffle=False,
                   initial_epoch=0)

        self.controller = controller

    def save(self):
        # XXX set generation attributes

        man = get_manager()

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
