import os


from ggplib.db import lookup

from ggpzero.util import attrutil
from ggpzero.util.keras import keras_models

from ggpzero.defs import confs, datadesc

from ggpzero.nn.network import NeuralNetwork
from ggpzero.nn.model import get_network_model
from ggpzero.defs import templates

the_manager = None


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Manager(object):
    def __init__(self, data_path=None):

        if data_path is None:
            data_path = os.path.join(os.environ["GGPZERO_PATH"], "data")
        self.data_path = data_path

        # instantiated transformers, lazy constructed
        self.transformers = {}

    def samples_path(self, game, generation_prefix):
        p = os.path.join(self.data_path, game, generation_prefix)
        ensure_directory_exists(p)
        return p

    def generation_path(self, game, generation_name=None):
        p = os.path.join(self.data_path, game, "generations")
        ensure_directory_exists(p)
        if generation_name is not None:
            filename = "%s.json" % generation_name
            p = os.path.join(p, filename)
        return p

    def model_path(self, game, generation_name=None):
        p = os.path.join(self.data_path, game, "models")
        ensure_directory_exists(p)
        if generation_name is not None:
            filename = "%s.json" % generation_name
            p = os.path.join(p, filename)
        return p

    def weights_path(self, game, generation_name=None):
        p = os.path.join(self.data_path, game, "weights")
        ensure_directory_exists(p)
        if generation_name is not None:
            filename = "%s.h5" % generation_name
            p = os.path.join(p, filename)
        return p

    def get_transformer(self, game, generation_descr=None):
        from ggpzero.nn.bases import GdlBasesTransformer, GdlBasesTransformer_Draws

        if generation_descr is None:
            generation_descr = templates.default_generation_desc(game)

        assert isinstance(generation_descr, datadesc.GenerationDescription)

        desc = generation_descr
        key = (game, desc.channel_last, desc.multiple_policy_heads, desc.num_previous_states, desc.draw_head)

        transformer = self.transformers.get(key)

        if transformer is None:
            # looks up the game in the ggplib database
            game_info = lookup.by_name(game)
            transformer_clz = GdlBasesTransformer_Draws if generation_descr.draw_head else GdlBasesTransformer
            transformer = transformer_clz(game_info, generation_descr)
            self.transformers[key] = transformer

        return transformer

    def create_new_network(self, game, nn_model_conf=None, generation_descr=None):
        if generation_descr is None:
            generation_descr = templates.default_generation_desc(game)

        transformer = self.get_transformer(game, generation_descr)

        if isinstance(nn_model_conf, str):
            nn_model_conf = templates.nn_model_config_template(game,
                                                               network_size_hint=nn_model_conf,
                                                               transformer=transformer)

        elif nn_model_conf is None:
            nn_model_conf = templates.nn_model_config_template(game,
                                                               network_size_hint="small",
                                                               transformer=transformer)

        assert isinstance(nn_model_conf, confs.NNModelConfig)
        assert isinstance(generation_descr, datadesc.GenerationDescription)

        keras_model = get_network_model(nn_model_conf, generation_descr)
        return NeuralNetwork(transformer, keras_model, generation_descr)

    def save_network(self, nn, generation_name=None):
        game = nn.generation_descr.game
        if generation_name is None:
            generation_name = nn.generation_descr.name
        else:
            nn.generation_descr.name = generation_name

        # save model / weights
        with open(self.model_path(game, generation_name), "w") as f:
            f.write(nn.get_model().to_json())

        nn.get_model().save_weights(self.weights_path(game, generation_name),
                                    overwrite=True)

        with open(self.generation_path(game, generation_name), "w") as f:
            f.write(attrutil.attr_to_json(nn.generation_descr, pretty=True))

    def load_network(self, game, generation_name):
        json_str = open(self.generation_path(game, generation_name)).read()
        generation_descr = attrutil.json_to_attr(json_str)

        json_str = open(self.model_path(game, generation_name)).read()
        keras_model = keras_models.model_from_json(json_str)

        keras_model.load_weights(self.weights_path(game, generation_name))
        transformer = self.get_transformer(game, generation_descr)
        return NeuralNetwork(transformer, keras_model, generation_descr)

    def can_load(self, game, generation_name):
        exists = os.path.exists
        return (exists(self.model_path(game, generation_name)) and
                exists(self.weights_path(game, generation_name)) and
                exists(self.generation_path(game, generation_name)))


###############################################################################

def get_manager():
    ' singleton for Manager '
    global the_manager
    if the_manager is None:
        the_manager = Manager()

    return the_manager
