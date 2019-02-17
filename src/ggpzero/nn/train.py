
from builtins import super

from ggplib.util import log
from ggplib.db import lookup

from ggpzero.util import attrutil
from ggpzero.util.keras import keras_callbacks, Progbar

from ggpzero.defs import confs

from ggpzero.nn.manager import get_manager

from ggpzero.nn import datacache


class TrainException(Exception):
    pass


###############################################################################

class TrainingLoggerCb(keras_callbacks.Callback):
    ''' simple progress bar.  default was breaking with too much metrics '''

    def __init__(self, num_epochs, batch_size):
        super().__init__()
        self.at_epoch = 0
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []
            self.progbar.update(self.seen, self.log_values)

    def on_batch_end(self, batch, logs=None):
        self.seen += logs.get('size')

        for k in logs:
            if "loss" in k and "val" not in k:
                self.log_values.append((k, logs[k]))

        self.progbar.update(self.seen, self.log_values)

    def on_epoch_begin(self, epoch, logs=None):
        self.at_epoch += 1
        log.info('Epoch %d/%d' % (self.at_epoch, self.num_epochs))

        # oh man, keras consistency... XXX
        try:
            self.target = self.params['samples']
        except KeyError:
            self.target = self.params['steps'] * self.batch_size

        self.progbar = Progbar(target=self.target)
        self.seen = 0

    def on_epoch_end(self, epoch, logs=None):
        # print so we have a gap between progress bar and logging
        print

        assert logs

        epoch += 1

        def str_by_name(names, dp=3):
            fmt = "%%s = %%.%df" % dp
            strs = [fmt % (k, logs[k]) for k in names]
            return ", ".join(strs)

        loss_names = [n for n in logs.keys() if 'loss' in n and 'val_' not in n]
        val_loss_names = [n for n in logs.keys() if 'loss' in n and 'val_' in n]

        log.info(str_by_name(loss_names, 4))
        log.info(str_by_name(val_loss_names, 4))

        # accuracy:
        for output in "policy value".split():
            acc = []
            val_acc = []
            for k in self.params['metrics']:
                if output not in k or "acc" not in k:
                    continue
                if "value" in output and "top" in k:
                    continue

                if 'val_' in k:
                    val_acc.append(k)
                else:
                    acc.append(k)

            log.info("%s : %s" % (output, str_by_name(acc)))
            log.info("%s : %s" % (output, str_by_name(val_acc)))


class TrainingController(keras_callbacks.Callback):
    ''' custom callback to do nice logging and early stopping '''

    def __init__(self, num_policies):
        self.stop_training = False
        self.at_epoch = 0

        self.best = None
        self.best_val_policy_acc = -1

        self.num_policies = num_policies

        self.epoch_last_set_at = None
        self.value_loss_diff = -1

    def policy_acc(self, logs):
        if self.num_policies == 1:
            acc, val_acc = logs['policy_0_acc'], logs['val_policy_0_acc']

        else:
            acc = val_acc = 0
            for i in range(self.num_policies):
                acc += logs['policy_%s_acc' % i]
                val_acc += logs['val_policy_%s_acc' % i]

            acc -= 0.5 * self.num_policies
            val_acc -= 0.5 * self.num_policies

        return acc, val_acc

    def set_value_overfitting(self, logs):
        loss = logs['value_loss']
        val_loss = logs['val_value_loss']

        # positive loss - *may* mean we are overfitting.
        self.value_loss_diff = val_loss - loss

    def on_epoch_begin(self, epoch, logs=None):
        self.at_epoch += 1

    def on_epoch_end(self, _, logs=None):
        epoch = self.at_epoch

        self.set_value_overfitting(logs)

        # deals with more than one head
        policy_acc, val_policy_acc = self.policy_acc(logs)

        log.debug("combined policy accuracy %.4f/%.4f" % (policy_acc, val_policy_acc))

        # are we overitting?
        overfitting = policy_acc - 0.02 > val_policy_acc

        # store best weights as best val_policy_acc

        # allow small decrease
        allow_policy_acc = self.best_val_policy_acc - 0.01
        if (self.epoch_last_set_at is None or
            (val_policy_acc > allow_policy_acc and not overfitting)):
            log.debug("Setting best to last val_policy_acc %.4f" % val_policy_acc)
            self.best = self.model.get_weights()
            self.best_val_policy_acc = max(val_policy_acc, self.best_val_policy_acc)
            self.epoch_last_set_at = epoch

        # stop training:
        if epoch >= 3:
            if overfitting:
                log.info("Early stopping... since policy accuracy overfitting")
                self.stop_training = True

            # if things havent got better (XXX hardcoded to 5) - STOP.  We can go on forever without improving.
            if self.epoch_last_set_at is not None and epoch > self.epoch_last_set_at + 5:
                log.info("Early stopping... since not improving (disabled)")
                self.stop_training = True

    def do_train_end(self):
        if self.best:
            log.warning("Switching to best weights with val_policy_acc %.4f" % self.best_val_policy_acc)
            self.model.set_weights(self.best)


###############################################################################

class TrainManager(object):
    def __init__(self, train_config, transformer, do_data_augmentation=False):
        # take a copy of the initial train_config
        assert isinstance(train_config, confs.TrainNNConfig)
        self.transformer = transformer
        self.do_data_augmentation = do_data_augmentation

        # lookup via game_name (this gets statemachine & statemachine model)
        self.game_info = lookup.by_name(train_config.game)
        self.update_config(attrutil.clone(train_config))
        self.cb = None

    def set_callback(self, cb):
        self.cb = cb

    def do_callbacks(self):
        ''' so can do other stuff - used to share GPU '''
        if self.cb is not None:
            self.cb()

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

        # ZZZ XXX hack in for now
        self.l2_loss = None
        if nn_model_config.l2_regularisation:
            assert isinstance(nn_model_config.l2_regularisation, float)
            self.l2_loss = nn_model_config.l2_regularisation

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
            prev_generation = "%s_%s" % (conf.generation_prefix,
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

    def update_value_weighting(self, value_weight, force_compile=False):
        ''' dynamic value weighting.  Based off of the value loss, as an approximated of overfitting
        value head '''

        # add these as hyper parameters?  Not sure if ever want to change them.  This function
        # should be a hyperparmeter.
        value_weight_reduction = 0.333
        value_weight_min = 0.05

        log.info("controller.value_loss_diff %.3f" % self.controller.value_loss_diff)

        orig_weight = value_weight
        if self.controller.value_loss_diff > 0.004:
            value_weight *= value_weight_reduction
        elif self.controller.value_loss_diff > 0.001:
            value_weight *= (value_weight_reduction * 2)
        else:
            # increase it again???
            if self.controller.value_loss_diff < 0:
                value_weight /= value_weight_reduction

            elif orig_weight < 0.5 and self.controller.value_loss_diff < 0.002:
                value_weight /= (value_weight_reduction * 2)

        value_weight = min(max(value_weight_min, value_weight), 1.0)
        if force_compile or abs(value_weight - orig_weight) > 0.0001:
            self.compile_nn(value_weight)

        return value_weight

    def compile_nn(self, value_weight):
        self.nn.compile(self.train_config.compile_strategy,
                        self.train_config.learning_rate,
                        value_weight,
                        l2_loss=self.l2_loss)

    def do_epochs(self, num_epochs_include_all=-1):

        # abbreviate, easier on the eyes
        conf = self.train_config

        cache = datacache.DataCache(self.transformer, conf.generation_prefix,
                                    do_augment_data=self.do_data_augmentation)
        cache.sync()

        max_epoch_size = conf.max_epoch_size
        if max_epoch_size is None:
            max_epoch_size = cache.total_samples

        train_at_step = conf.next_step - 1
        ignore_after_step = conf.starting_step
        assert 0 <= ignore_after_step <= train_at_step
        buckets_def = datacache.Buckets(conf.resample_buckets)

        indexer = cache.create_chunk_indexer(buckets_def,
                                             starting_step=train_at_step,
                                             ignore_after_step=ignore_after_step,
                                             validation_split=conf.validation_split)

        # first get validation data, then we can forget about it as it doesn't need reshuffled
        validation_size = int(max_epoch_size * (1 - conf.validation_split))
        validation_indices = indexer.validation_epoch(validation_size)

        training_logger = TrainingLoggerCb(conf.epochs, conf.batch_size)
        self.controller = TrainingController(len(self.transformer.policy_dist_count))

        # starting value weight
        value_weight = conf.initial_value_weight

        self.compile_nn(value_weight)

        for i in range(conf.epochs):
            if self.controller.stop_training:
                log.warning("Stop training early via controller")
                break

            # resample the samples!
            if i < num_epochs_include_all:
                training_indices = indexer.training_epoch(max_epoch_size, include_all=1)
            else:
                training_indices = indexer.training_epoch(max_epoch_size)

            if i > 0:
                value_weight = self.update_value_weighting(value_weight)

            assert len(validation_indices) / conf.batch_size > 0, \
                "validation steps must be more than zero (not enough data)"

            fitter = self.nn.get_model().fit_generator
            fitter(cache.generate(training_indices, conf.batch_size),
                   len(training_indices) / conf.batch_size,
                   epochs=1,
                   verbose=0,
                   validation_data=cache.generate(validation_indices, conf.batch_size),
                   validation_steps=len(validation_indices) / conf.batch_size,
                   callbacks=[training_logger, self.controller],
                   shuffle=False,
                   initial_epoch=0)

            self.do_callbacks()

        self.controller.do_train_end()

    def save(self):
        # XXX set generation attributes

        man = get_manager()

        man.save_network(self.nn, generation_name=self.next_generation)
        self.do_callbacks()
