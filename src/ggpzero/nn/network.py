''' mostly just an interface to keras...  hoping to try other frameworks too. '''

import numpy as np

from ggplib.util import log

from ggpzero.util.keras import SGD, Adam, keras_metrics, keras_regularizers, keras_models


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
        ' this is for testing purposes. We use C++ normally to access network '
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
        ' this is for testing purposes. We use C++ normally to access network '
        if prev_states:
            return self.predict_n([state], [prev_states])[0]
        else:
            return self.predict_n([state])[0]

    def compile(self, compile_strategy, learning_rate=None, value_weight=1.0,
                l2_loss=None, l2_non_residual=True):
        # XXX allow l2_loss on final layers.

        value_objective = "mean_squared_error"
        policy_objective = 'categorical_crossentropy'
        if compile_strategy == "SGD":
            if learning_rate is None:
                learning_rate = 0.01
            optimizer = SGD(lr=learning_rate, momentum=0.9)

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

        if l2_loss is not None:
            log.warning("Applying l2 loss (%.5f)" % l2_loss)
            l2_loss = keras_regularizers.l2(l2_loss)

        rebuild_model = False
        for layer in self.keras_model.layers:
            # To get global weight decay in keras regularizers have to be added to every layer
            # in the model.

            if hasattr(layer, 'kernel_regularizer'):

                ignore = False
                if l2_non_residual:
                    ignore = True

                    if "policy" in layer.name or "value" in layer.name:
                        if "flatten" not in layer.name:
                            ignore = False
                else:
                    ignore = "_se_" in layer.name

                if ignore:
                    if layer.kernel_regularizer is not None:
                        log.warning("Ignoring but regularizer was set @ %s/%s.  Unsetting." % (layer.name, layer))
                        layer.kernel_regularizer = None
                        rebuild_model = True

                    continue

                if l2_loss is not None and layer.kernel_regularizer is None:
                    rebuild_model = True
                    log.info("Applying l2 loss to %s/%s" % (layer.name, layer))
                    layer.kernel_regularizer = l2_loss

                if layer.kernel_regularizer is not None and l2_loss is None:
                    log.info("Unsetting l2 loss on %s/%s" % (layer.name, layer))
                    rebuild_model = True
                    layer.kernel_regularizer = l2_loss

        # This ensures a fresh build of the network (there is no API to do this in keras, hence
        # this hacky workaround).  Furthermore, needing to rebuild the network here, before
        # compiling, is somewhat buggy/idiosyncrasy of keras.
        if rebuild_model:
            config = self.keras_model.get_config()
            weights = self.keras_model.get_weights()
            self.keras_model = keras_models.Model.from_config(config)
            self.keras_model.set_weights(weights)

        self.keras_model.compile(loss=loss, optimizer=optimizer,
                                 loss_weights=loss_weights,
                                 metrics=["acc", top_3_acc])

    def get_model(self):
        assert self.keras_model is not None
        return self.keras_model
