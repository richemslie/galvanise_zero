''' mostly just an interface to keras...  hoping to try other frameworks too. '''

from builtins import super

import numpy as np

from ggplib.util import log


from ggpzero.util.keras import SGD, Adam, Progbar, keras_callbacks, keras_metrics


def top_3_acc(y_true, y_pred):
    return keras_metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def objective_function_for_policy(y_true, y_pred):
    # from https://github.com/mokemokechicken/reversi-alpha-zero
    from ggpzero.util.keras import K
    return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)


class HeadResult(object):
    def __init__(self, transformer, policies, values):
        assert len(transformer.policy_dist_count) == len(policies)
        self.policies = policies
        self.scores = values

    def __repr__(self):
        return "HeadResult(policies=%s, scores=%s" % (self.policies, self.scores)


###############################################################################

class TrainingLoggerCb(keras_callbacks.Callback):

    ''' simple progress bar.  default was breaking with too much metrics '''

    def __init__(self, num_epochs):
        super().__init__()
        self.at_epoch = 0
        self.num_epochs = num_epochs

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

        self.target = self.params['samples']

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

    def __init__(self, retraining, num_policies):
        self.retraining = retraining

        self.stop_training = False
        self.at_epoch = 0

        self.best = None
        self.best_val_policy_acc = -1

        self.num_policies = num_policies

        self.retrain_best = None
        self.retrain_best_val_policy_acc = -1
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
        if self.retrain_best is None and self.retraining:
            log.info('Reusing old retraining network for *next* retraining network')
            self.retrain_best = self.model.get_weights()

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
        if (self.epoch_last_set_at is None or
            (val_policy_acc > self.best_val_policy_acc and not overfitting)):
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

        # stop training:
        if (not self.retraining and epoch >= 4 or
            self.retraining and epoch >= 2):
            if overfitting:
                log.info("Early stopping... since policy accuracy overfitting")
                self.stop_training = True

            # if things havent got better - STOP.  We can go on forever without improving.
            if self.epoch_last_set_at is not None and epoch > self.epoch_last_set_at + 3:
                log.info("Early stopping... since not improving")
                self.stop_training = True

    def on_train_end(self, logs=None):
        if self.best:
            log.info("Switching to best weights with val_policy_acc %.4f" % self.best_val_policy_acc)
            self.model.set_weights(self.best)


class NeuralNetwork(object):
    ''' combines a keras model and gdl bases transformer to give a clean interface to use as a
        network. '''

    def __init__(self, gdl_bases_transformer, keras_model, generation_descr):
        self.gdl_bases_transformer = gdl_bases_transformer
        self.keras_model = keras_model
        self.generation_descr = generation_descr

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
            assert len(prev_states) == len(states)
            X = np.array([to_channels(s, prevs)
                          for s, prevs in zip(states, prev_states)])
        else:
            X = np.array([to_channels(s) for s in states])

        Y = self.keras_model.predict(X, batch_size=len(states))

        result = []
        for i in range(len(states)):
            heads = HeadResult(self.gdl_bases_transformer,
                               [Y[k][i] for k in range(len(Y) - 1)],
                               Y[-1][i])
            result.append(heads)

        return result

    def predict_1(self, state, prev_states=None):
        if prev_states:
            return self.predict_n([state], [prev_states])[0]
        else:
            return self.predict_n([state])[0]

    def compile(self, compile_strategy, learning_rate=None, value_weight=1.0):
        value_objective = "mean_squared_error"
        if compile_strategy == "SGD":
            policy_objective = objective_function_for_policy
            if learning_rate:
                optimizer = SGD(lr=learning_rate, momentum=0.9)
            else:
                optimizer = SGD(lr=1e-2, momentum=0.9)

        elif compile_strategy == "adam":
            policy_objective = 'categorical_crossentropy'
            if learning_rate:
                optimizer = Adam(lr=learning_rate)
            else:
                optimizer = Adam()

        elif compile_strategy == "amsgrad":
            policy_objective = 'categorical_crossentropy'
            if learning_rate:
                optimizer = Adam(lr=learning_rate, amsgrad=True)
            else:
                optimizer = Adam(amsgrad=True)
        else:
            log.error("UNKNOWN compile strategy %s" % compile_strategy)
            raise Exception("UNKNOWN compile strategy %s" % compile_strategy)

        num_policies = len(self.gdl_bases_transformer.policy_dist_count)

        loss = [policy_objective] * num_policies
        loss.append(value_objective)
        loss_weights = [1.0] * num_policies
        loss_weights.append(value_weight)

        if learning_rate is not None:
            msg = "Compiling with %s (learning_rate=%.4f, value_weight=%.3f)"
            log.warning(msg % (optimizer, learning_rate, value_weight))
        else:
            log.warning("Compiling with %s (value_weight=%.3f)" % (optimizer, value_weight))

        self.keras_model.compile(loss=loss, optimizer=optimizer,
                                 loss_weights=loss_weights,
                                 metrics=["acc", top_3_acc])

    def fit(self, input_channels, outputs, validation_input_channels,
            validation_outputs, batch_size, callbacks, **kwds):

        self.keras_model.fit(input_channels,
                             outputs,
                             verbose=0,
                             batch_size=batch_size,
                             epochs=1,
                             validation_data=[validation_input_channels,
                                              validation_outputs],
                             callbacks=callbacks,
                             **kwds)

    def get_model(self):
        assert self.keras_model is not None
        return self.keras_model
