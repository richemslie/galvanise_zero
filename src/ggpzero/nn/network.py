''' mostly just an interface to keras...  hoping to try other frameworks too. '''

import numpy as np

from ggplib.util import log


from ggpzero.util.keras import SGD, Adam, keras_metrics


class HeadResult(object):
    def __init__(self, transformer, policies, values):
        assert len(transformer.policy_dist_count) == len(policies)
        self.policies = policies
        self.scores = values

    def __repr__(self):
        return "HeadResult(policies=%s, scores=%s" % (self.policies, self.scores)


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
        policy_objective = 'categorical_crossentropy'
        if compile_strategy == "SGD":
            if learning_rate:
                optimizer = SGD(lr=learning_rate, momentum=0.9)
            else:
                optimizer = SGD(lr=1e-2, momentum=0.9)

        elif compile_strategy == "adam":
            if learning_rate:
                optimizer = Adam(lr=learning_rate)
            else:
                optimizer = Adam()

        elif compile_strategy == "amsgrad":
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

        def top_3_acc(y_true, y_pred):
            return keras_metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

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
