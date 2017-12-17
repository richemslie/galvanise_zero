import os

import attr

import numpy as np
from keras import models
from keras.optimizers import SGD

from ggplib.util import log

from ggplearn.nn import model, bases


def model_path(game, generation):
    filename = "%s_%s.json" % (game, generation)
    return os.path.join(os.environ["GGPLEARN_PATH"], "data", "models", filename)


def weights_path(game, generation):
    filename = "%s_%s.h5" % (game, generation)
    return os.path.join(os.environ["GGPLEARN_PATH"], "data", "weights", filename)


@attr.s
class TrainData(object):
    inputs = attr.ib()
    outputs = attr.ib()
    validation_inputs = attr.ib()
    validation_outputs = attr.ib()
    batch_size = attr.ib(512)
    epochs = attr.ib(24)


class NeuralNetwork(object):

    def __init__(self, bases_config, model=None):
        self.bases_config = bases_config
        self.model = model

    def summary(self):
        ' log keras nn summary '

        # one way to get print_summary to output string!
        lines = []
        self.model.summary(print_fn=lines.append)
        for l in lines:
            log.verbose(l)

    def predict_n(self, states, lead_role_indexes):
        num_states = len(states)

        X_0 = [self.bases_config.state_to_channels(s, ri) for s, ri in zip(states,
                                                                           lead_role_indexes)]

        X_0 = np.array(X_0)
        X_1 = np.array([self.bases_config.get_non_cord_input(s) for s in states])

        Y = self.model.predict([X_0, X_1], batch_size=num_states)
        assert len(Y) == 2

        result = []
        for i in range(num_states):
            policy, scores = Y[0][i], Y[1][i]
            result.append((policy, scores))

        return result

    def predict_1(self, state, lead_role_index):
        return self.predict_n([state], [lead_role_index])[0]

    def compile(self):
        # loss is much less on score.  It overfits really fast.
        if False:  # params.ALPHAZERO_REGULARISATION
            optimizer = SGD(lr=1e-2, momentum=0.9)
            loss = [model.objective_function_for_policy, "mean_squared_error"]
        else:
            loss = ['categorical_crossentropy', 'mean_squared_error']
            optimizer = "adam"

            self.model.compile(loss=loss, optimizer=optimizer,
                               loss_weights=[1.0, 0.01],
                               metrics=["acc", model.top_2_acc, model.top_3_acc])

    def train(self, conf):
        validation_data = [conf.validation_inputs, conf.validation_outputs]

        # XXX rename this - refactor - etc
        my_cb = model.MyCallback()
        self.model.fit(conf.inputs,
                       conf.outputs,
                       verbose=0,
                       batch_size=conf.batch_size,
                       epochs=conf.epochs,
                       validation_data=validation_data,
                       callbacks=[model.MyProgbarLogger(), my_cb],
                       shuffle=True)

        return my_cb

    def get_model(self):
        return self.model

    def save(self):
        # save model / weights
        with open(model_path(self.bases_config.game,
                             self.bases_config.generation), "w") as f:
            f.write(self.model.to_json())

        self.model.save_weights(weights_path(self.bases_config.game,
                                             self.bases_config.generation),
                                overwrite=True)

    def load(self):
        # save model / weights
        f = model_path(self.bases_config.game, self.bases_config.generation)
        self.model = models.model_from_json(open(f).read())

        self.model.load_weights(weights_path(self.bases_config.game,
                                             self.bases_config.generation))

    def can_load(self):
        # save model / weights
        mode_fn = model_path(self.bases_config.game, self.bases_config.generation)
        weights_fn = weights_path(self.bases_config.game,
                                  self.bases_config.generation)

        return os.path.exists(mode_fn) and os.path.exists(weights_fn)


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
