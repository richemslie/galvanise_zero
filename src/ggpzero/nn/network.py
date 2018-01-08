import numpy as np
from keras import metrics
from keras.optimizers import SGD, Adam
from keras.utils.generic_utils import Progbar
import keras.callbacks
import keras.backend as K

from ggplib.util import log
from ggpzero.defs import confs


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

        loss_names = "loss policy_loss value_loss".split()
        val_loss_names = "val_loss val_policy_loss val_value_loss".split()

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


class EarlyStoppingCb(keras.callbacks.Callback):
    ''' custom callback to do nice logging and early stopping '''

    def __init__(self, network, retraining):
        self.network = network
        self.retraining = retraining
        self.recompiled = False

    def on_train_begin(self, logs=None):
        self.best = None
        self.best_val_policy_acc = -1

        self.retrain_best = None
        self.retrain_best_val_policy_acc = -1
        self.epoch_last_set_at = None

        self.epochs = self.params['epochs']

    def check_value_overfitting(self, logs):
        if self.recompiled:
            return

        loss = logs['value_loss']
        val_loss = logs['val_value_loss']

        if loss - 0.001 < val_loss:
            # XXX doesn't work
            #self.network.compile(value_loss=0.01)
            self.recompiled = True

    def on_epoch_end(self, epoch, logs=None):
        assert logs
        epoch += 1

        # XXX doesnt work
        # self.check_value_overfitting(logs)

        policy_acc = logs['policy_acc']
        val_policy_acc = logs['val_policy_acc']

        # store best weights as best val_policy_acc
        if val_policy_acc > self.best_val_policy_acc:
            log.debug("Setting best to last val_policy_acc %.4f" % val_policy_acc)
            self.best = self.model.get_weights()
            self.best_val_policy_acc = val_policy_acc
            self.epoch_last_set_at = epoch

        store_retraining_weights = ((policy_acc + 0.01) < val_policy_acc and
                                    val_policy_acc > self.retrain_best_val_policy_acc)

        if store_retraining_weights:
            log.debug("Setting retraining_weights to val_policy_acc %.4f" % val_policy_acc)
            self.retrain_best = self.model.get_weights()
            self.retrain_best_val_policy_acc = val_policy_acc

        # seems the first time around we should it give it chance (for breakthrough we didnt need,
        # with reversi it takes at 10 epochs to stablize in the training).
        if ((not self.retraining and epoch >= 16) or
            (self.retraining and epoch >= 3)):

            # if we are overfitting
            if policy_acc - 0.02 > val_policy_acc:
                log.info("Early stopping... since policy accuracy overfitting")
                self.model.stop_training = True

            # if things havent got better - STOP.  We can go on forever without improving.
            if self.epoch_last_set_at is not None and epoch > self.epoch_last_set_at + 4:
                log.info("Early stopping... since not improving")
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best:
            log.info("Switching to best weights with val_policy_acc %s" % self.best_val_policy_acc)
            self.model.set_weights(self.best)


###############################################################################

class NeuralNetwork(object):
    ''' combines a keras model and gdl bases transformer to give a clean interface to use as a
        network. '''

    def __init__(self, gdl_bases_transformer, keras_model):
        self.gdl_bases_transformer = gdl_bases_transformer
        self.keras_model = keras_model

    def summary(self):
        ' log keras nn summary '

        # one way to get print_summary to output string!
        lines = []
        self.keras_model.summary(print_fn=lines.append)
        for l in lines:
            log.verbose(l)

    def predict_n(self, states, prev_states=None):
        # prev_states -> list of list of states

        to_channels = self.gdl_bases_transformer.state_to_channels
        if prev_states:
            X = np.array([to_channels(s, prevs)
                          for s, prevs in zip(states, prev_states)])
        else:
            X = np.array([to_channels(s) for s in states])

        Y = self.keras_model.predict(X, batch_size=len(states))

        assert len(Y) == 2

        result = []
        for i in range(len(states)):
            policy, values = Y[0][i], Y[1][i]
            result.append((policy, values))

        return result

    def predict_1(self, state, prev_states=None):
        if prev_states:
            return self.predict_n([state], [prev_states])[0]
        else:
            return self.predict_n([state])[0]

    def compile(self, use_sgd=False, learning_rate=None):
        if learning_rate is not None:
            lr = learning_rate
        else:
            if use_sgd:
                lr = 1e-2
            else:
                lr = 1e-3

        if use_sgd:
            optimizer = SGD(lr=lr, momentum=0.9)
            loss = [objective_function_for_policy, "mean_squared_error"]
        else:
            loss = ['categorical_crossentropy', 'mean_squared_error']
            optimizer = Adam(lr=lr)

        log.warning("Compiling with %s" % optimizer)

        # loss is much less on value.  it overfits really fast.
        self.keras_model.compile(loss=loss, optimizer=optimizer,
                                 loss_weights=[1.0, 0.01],
                                 metrics=["acc", top_3_acc])

    def train(self, train_conf, retraining=False):
        assert isinstance(train_conf, confs.TrainData)

        for k in ['input_channels', 'output_policies', 'output_final_scores',
                  'validation_input_channels', 'validation_output_policies',
                  'validation_output_final_scores']:
            v = getattr(train_conf, k)
            new_shape = [-1] + list(v[0].shape)
            log.info('train.%s count: %s.  Example:' % (k, new_shape))
            print v[42]
            setattr(train_conf, k, np.concatenate(v, axis=0).reshape(new_shape))

        outputs = [train_conf.output_policies,
                   train_conf.output_final_scores]

        validation_data = [train_conf.validation_input_channels,
                           [train_conf.validation_output_policies,
                            train_conf.validation_output_final_scores]]

        early_stopping_cb = EarlyStoppingCb(self, retraining)

        # compile the model XXX pass in config options
        self.compile()

        log.warning("self.keras_model.fit()")
        self.keras_model.fit(train_conf.input_channels,
                             outputs,
                             verbose=0,
                             batch_size=train_conf.batch_size,
                             epochs=train_conf.epochs,
                             validation_data=validation_data,
                             callbacks=[TrainingLoggerCb(), early_stopping_cb],
                             shuffle=True)

        return early_stopping_cb

    def get_model(self):
        assert self.keras_model is not None
        return self.keras_model
