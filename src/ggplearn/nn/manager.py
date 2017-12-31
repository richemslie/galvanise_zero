import os

from keras.models import model_from_json

from ggplearn import msgdefs

from ggplib.db import lookup

from ggplearn.nn.network import NeuralNetwork
from ggplearn.nn.model import get_network_model, is_channels_first

the_manager = None


def model_path(game, generation):
    filename = "%s_%s.json" % (game, generation)
    return os.path.join(os.environ["GGPLEARN_PATH"], "data", game, "models", filename)


def weights_path(game, generation):
    filename = "%s_%s.h5" % (game, generation)
    return os.path.join(os.environ["GGPLEARN_PATH"], "data", game, "weights", filename)


class Manager(object):
    def __init__(self):
        self.transformer_classes = {}

        # instantiated transformers, lazy constructed
        self.transformers = {}

    def register_transformer(self, clz):
        self.transformer_classes[clz.game] = clz

    def get_transformer(self, game):
        transformer = self.transformers.get(game)

        if transformer is None:
            clz = self.transformer_classes.get(game)
            assert clz is not None, "Did not find bases transformer for game: %s" % game

            # looks up the game in the ggplib database
            game_info = lookup.by_name(game)

            channel_last = not is_channels_first()
            self.transformers[game] = transformer = clz(game_info, channel_last)

        return transformer

    def create_new_network(self, game, nn_model_conf):
        assert isinstance(nn_model_conf, msgdefs.NNModelConfig)

        transformer = self.get_transformer(game)
        keras_model = get_network_model(nn_model_conf)
        return NeuralNetwork(transformer, keras_model)

    def save_network(self, nn, game, generation):
        # save model / weights
        with open(model_path(game, generation), "w") as f:
            f.write(nn.get_model().to_json())

        nn.get_model().save_weights(weights_path(game, generation),
                                    overwrite=True)

    def load_network(self, game, generation):
        json_str = open(model_path(game, generation)).read()
        keras_model = model_from_json(json_str)
        keras_model.load_weights(weights_path(game, generation))
        return NeuralNetwork(self.get_transformer(game), keras_model)

    def can_load(self, game, generation):
        return (os.path.exists(model_path(game, generation)) and
                os.path.exists(weights_path(game, generation)))


###############################################################################

def get_manager():
    ' singleton for Manager '
    global the_manager
    if the_manager is None:
        the_manager = Manager()

        from ggplearn.nn.bases import init
        init()

    return the_manager
