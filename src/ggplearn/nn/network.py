' Regularisation tricks (tyvm) credit to :https://github.com/mokemokechicken/reversi-alpha-zero '

import os

import numpy as np
from keras import models, metrics
from keras.optimizers import SGD
from keras.utils.generic_utils import Progbar
import keras.callbacks
import keras.backend as K

from ggplib.util import log
from ggplearn.nn import bases


def model_path(game, generation):
    filename = "%s_%s.json" % (game, generation)
    return os.path.join(os.environ["GGPLEARN_PATH"], "data", "models", filename)


def weights_path(game, generation):
    filename = "%s_%s.h5" % (game, generation)
    return os.path.join(os.environ["GGPLEARN_PATH"], "data", "weights", filename)


def top_2_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


def top_3_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def objective_function_for_policy(y_true, y_pred):
    return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)


###############################################################################

class TrainingLoggerCb(keras.callbacks.Callback):
    ''' simple progress bar.  default was breaking with too much metrics '''
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

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
        log.info('Epoch %d/%d' % (epoch + 1, self.epochs))

        self.target = self.params['samples']

        self.progbar = Progbar(target=self.target)
        self.seen = 0

    def on_epoch_end(self, epoch, logs=None):
        # print so we have a gap between progress bar and logging
        print

        assert logs

        epoch += 1
        log.debug("Epoch %s/%s" % (epoch, self.epochs))

        def str_by_name(names, dp=3):
            fmt = "%%s = %%.%df" % dp
            strs = [fmt % (k, logs[k]) for k in names]
            return ", ".join(strs)

        loss_names = "loss policy_loss score_loss".split()
        val_loss_names = "val_loss val_policy_loss val_score_loss".split()

        log.info(str_by_name(loss_names, 4))
        log.info(str_by_name(val_loss_names, 4))

        # accuracy:
        for output in "policy score".split():
            acc = []
            val_acc = []
            for k in self.params['metrics']:
                if output not in k or "acc" not in k:
                    continue
                if "score" in output and "top" in k:
                    continue

                if 'val' in k:
                    val_acc.append(k)
                else:
                    acc.append(k)

            log.info("%s : %s" % (output, str_by_name(acc)))
            log.info("%s : %s" % (output, str_by_name(val_acc)))


class EarlyStoppingCb(keras.callbacks.Callback):
    ''' custom callback to do nice logging and early stopping '''

    def on_train_begin(self, logs=None):
        self.best = None
        self.best_val_policy_acc = -1

        self.retrain_best = None
        self.retrain_best_val_policy_acc = -1
        self.epoch_last_set_at = None

        self.epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        assert logs
        epoch += 1
        policy_acc = logs['policy_acc']
        val_policy_acc = logs['val_policy_acc']

        # store best weights as best val_policy_acc
        if val_policy_acc > self.best_val_policy_acc:
            log.debug("Setting best to last val_policy_acc %.4f" % val_policy_acc)
            self.best = self.model.get_weights()
            self.best_val_policy_acc = val_policy_acc
            self.epoch_last_set_at = epoch

        # always do at 3 epochs before starting
        if epoch <= 3:
            return

        check_retraining_weights = (self.retrain_best is None or
                                    (policy_acc < val_policy_acc and
                                     val_policy_acc > self.retrain_best_val_policy_acc))

        if check_retraining_weights:
            # store retraining weights
            log.debug("Setting retraining_weights to val_policy_acc %.4f" % val_policy_acc)
            self.retrain_best = self.model.get_weights()
            self.retrain_best_val_policy_acc = val_policy_acc

        if epoch >= 5:
            # if we are overfitting
            if policy_acc - 0.02 > val_policy_acc:
                log.info("Early stopping... since policy accuracy overfitting")
                self.model.stop_training = True
                return

            # if things havent got better - STOP.  We can go on forever without improving.
            # if self.epoch_last_set_at is not None and epoch > self.epoch_last_set_at + 3:
            #    log.info("Early stopping... since not improving")
            #    self.model.stop_training = True
            #    return

    def on_train_end(self, logs=None):
        if self.best:
            log.info("Switching to best weights with val_policy_acc %s" % self.best_val_policy_acc)
            self.model.set_weights(self.best)


###############################################################################

class NeuralNetwork(object):

    def __init__(self, bases_config, keras_model=None):
        self.bases_config = bases_config
        self.keras_model = keras_model

    def summary(self):
        ' log keras nn summary '

        # one way to get print_summary to output string!
        lines = []
        self.keras_model.summary(print_fn=lines.append)
        for l in lines:
            log.verbose(l)

    def predict_n(self, states, lead_role_indexes):
        num_states = len(states)

        X_0 = [self.bases_config.state_to_channels(s, ri) for s, ri in zip(states,
                                                                           lead_role_indexes)]

        X_0 = np.array(X_0)
        X_1 = np.array([self.bases_config.get_non_cord_input(s) for s in states])

        Y = self.keras_model.predict([X_0, X_1], batch_size=num_states)
        assert len(Y) == 2

        result = []
        for i in range(num_states):
            policy, scores = Y[0][i], Y[1][i]
            result.append((policy, scores))

        return result

    def predict_1(self, state, lead_role_index):
        return self.predict_n([state], [lead_role_index])[0]

    def compile(self, alphazero_regularisation=False):
        if alphazero_regularisation:
            optimizer = SGD(lr=1e-2, momentum=0.9)
            loss = [objective_function_for_policy, "mean_squared_error"]
        else:
            loss = ['categorical_crossentropy', 'mean_squared_error']
            optimizer = "adam"

        # loss is much less on score.  it overfits really fast.
        self.keras_model.compile(loss=loss, optimizer=optimizer,
                                 loss_weights=[1.0, 0.01],
                                 metrics=["acc", top_2_acc, top_3_acc])

    def train(self, conf):
        validation_data = [conf.validation_inputs, conf.validation_outputs]

        early_stopping_cb = EarlyStoppingCb()
        self.keras_model.fit(conf.inputs,
                             conf.outputs,
                             verbose=0,
                             batch_size=conf.batch_size,
                             epochs=conf.epochs,
                             validation_data=validation_data,
                             callbacks=[TrainingLoggerCb(), early_stopping_cb],
                             shuffle=True)

        return early_stopping_cb

    def get_model(self):
        return self.keras_model

    def save(self):
        # save model / weights
        with open(model_path(self.bases_config.game,
                             self.bases_config.generation), "w") as f:
            f.write(self.keras_model.to_json())

        self.keras_model.save_weights(weights_path(self.bases_config.game,
                                                   self.bases_config.generation),
                                      overwrite=True)

    def load(self):
        # save model / weights
        f = model_path(self.bases_config.game, self.bases_config.generation)

        self.keras_model = models.model_from_json(open(f).read())
        self.keras_model.load_weights(weights_path(self.bases_config.game,
                                                   self.bases_config.generation))

    def can_load(self):
        # save model / weights
        mode_fn = model_path(self.bases_config.game, self.bases_config.generation)
        weights_fn = weights_path(self.bases_config.game,
                                  self.bases_config.generation)

        return os.path.exists(mode_fn) and os.path.exists(weights_fn)


###############################################################################

def create(generation, game_info, load=True, **kwds):
    ''' game_info is got from db.lookup().
    if load is False, and then call save will overwrite existing generation '''

    bases_config = bases.get_config(game_info.game,
                                    game_info.model,
                                    generation=generation)

    nn = bases_config.create_network(**kwds)
    if load:
        nn.load()
    return nn
