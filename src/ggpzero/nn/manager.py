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
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


class Manager(object):
    def __init__(self, data_path=None):

        if data_path is None:
            data_path = os.path.join(os.environ["GGPZERO_PATH"], "data")
        self.data_path = data_path

        self.transformer_classes = {}

        # instantiated transformers, lazy constructed
        self.transformers = {}

    def samples_path(self, game, generation_prefix):
        p = os.path.join(self.data_path, game, generation_prefix)
        ensure_directory_exists(p)
        return p

    def generation_path(self, game, generation):
        filename = "%s.json" % generation
        p = os.path.join(self.data_path, game, "generations", filename)
        ensure_directory_exists(p)
        return p

    def model_path(self, game, generation):
        filename = "%s.json" % generation
        p = os.path.join(self.data_path, game, "models", filename)
        ensure_directory_exists(p)
        return p

    def weights_path(self, game, generation):
        filename = "%s.h5" % generation
        p = os.path.join(self.data_path, game, "weights", filename)
        ensure_directory_exists(p)
        return p

    def register_transformer(self, clz):
        self.transformer_classes[clz.game] = clz

    def get_transformer(self, game, generation_descr=None):
        if generation_descr is None:
            generation_descr = templates.default_generation_desc(game)

        assert isinstance(generation_descr, datadesc.GenerationDescription)

        desc = generation_descr
        key = (game, desc.channel_last, desc.multiple_policy_heads, desc.num_previous_states)

        transformer = self.transformers.get(key)

        if transformer is None:
            clz = self.transformer_classes.get(game)
            assert clz is not None, "Did not find bases transformer for game: %s" % game

            # looks up the game in the ggplib database
            game_info = lookup.by_name(game)
            self.transformers[key] = transformer = clz(game_info, generation_descr)

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
            nn_model_conf = templates.nn_model_config_template(game, transformer=transformer)

        assert isinstance(nn_model_conf, confs.NNModelConfig)
        assert isinstance(generation_descr, datadesc.GenerationDescription)

        keras_model = get_network_model(nn_model_conf)
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

    def load_network_fixme(self, game):
        import glob
        import datetime
        p = os.path.join(self.data_path, game, "models")
        os.chdir(p)
        gens = glob.glob("%s_*" % game)

        for g in gens:
            print "doing", game, g
            generation = os.path.splitext(g)[0]
            new_style_gen = generation.replace(game + "_", "")

            print generation, new_style_gen

            # dummy generation_descr
            generation_descr = templates.default_generation_desc(game)

            json_str = open(self.model_path(game, generation)).read()
            keras_model = keras_models.model_from_json(json_str)

            keras_model.load_weights(self.weights_path(game, generation))
            transformer = self.get_transformer(game, generation_descr)
            print transformer, keras_model, generation_descr
            nn = NeuralNetwork(transformer, keras_model, generation_descr)
            generation_descr.name = new_style_gen
            generation_descr.trained_losses = "unknown"
            generation_descr.trained_validation_losses = "unknown"
            generation_descr.trained_policy_accuracy = "unknown"
            generation_descr.trained_value_accuracy = "unknown"
            ctime = os.stat(self.model_path(game, generation)).st_ctime

            generation_descr.date_created = datetime.datetime.fromtimestamp(ctime).strftime("%Y/%m/%d %H:%M")
            print generation_descr
            self.save_network(nn)

    def load_network(self, game, generation):
        json_str = open(self.generation_path(game, generation)).read()
        generation_descr = attrutil.json_to_attr(json_str)

        json_str = open(self.model_path(game, generation)).read()
        keras_model = keras_models.model_from_json(json_str)

        keras_model.load_weights(self.weights_path(game, generation))
        transformer = self.get_transformer(game, generation_descr)
        return NeuralNetwork(transformer, keras_model, generation_descr)

    def can_load(self, game, generation):
        exists = os.path.exists
        return (exists(self.model_path(game, generation)) and
                exists(self.weights_path(game, generation)) and
                exists(self.generation_path(game, generation)))


###############################################################################

def get_manager():
    ' singleton for Manager '
    global the_manager
    if the_manager is None:
        the_manager = Manager()

        from ggpzero.nn.bases import init
        init()

    return the_manager
