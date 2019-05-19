from ggplib.util.init import setup_once
from ggplib.db import lookup

from ggpzero.util import attrutil

from ggpzero.defs import confs, templates
from ggpzero.nn import train
from ggpzero.nn.manager import get_manager
from ggpzero.nn.model import get_network_model
from ggpzero.nn.network import NeuralNetwork

man = get_manager()


def setup():
    # set up ggplib
    setup_once()

    # ensure we have database with ggplib
    lookup.get_database()


def test_generation_desc():
    game = "breakthrough"
    gen_prefix = "x1"
    prev_states = 1
    gen_desc = templates.default_generation_desc(game,
                                                 gen_prefix,
                                                 multiple_policy_heads=True,
                                                 num_previous_states=prev_states)
    attrutil.pprint(gen_desc)


def test_nn_model_config_template():
    game = "breakthrough"
    gen_prefix = "x1"
    prev_states = 1

    gen_desc = templates.default_generation_desc(game,
                                                 gen_prefix,
                                                 multiple_policy_heads=True,
                                                 num_previous_states=prev_states)
    transformer = man.get_transformer(game, gen_desc)

    model = templates.nn_model_config_template("breakthrough", "small", transformer)
    attrutil.pprint(model)

    keras_model = get_network_model(model, gen_desc)
    network = NeuralNetwork(transformer, keras_model, gen_desc)
    print network
    network.summary()


def test_nn_model_config_template2():
    game = "breakthrough"
    gen_prefix = "x1"
    prev_states = 1

    gen_desc = templates.default_generation_desc(game,
                                                 gen_prefix,
                                                 multiple_policy_heads=True,
                                                 num_previous_states=prev_states)
    transformer = man.get_transformer(game, gen_desc)

    model = templates.nn_model_config_template("breakthrough", "small",
                                               transformer, features=True)
    attrutil.pprint(model)

    keras_model = get_network_model(model, gen_desc)
    network = NeuralNetwork(transformer, keras_model, gen_desc)
    print network
    network.summary()


def test_train_config_template():
    game = "breakthrough"
    gen_prefix = "x1"

    train_config = templates.train_config_template(game, gen_prefix)
    attrutil.pprint(train_config)


def test_base_puct_config():
    config = templates.base_puct_config(dirichlet_noise_pct=0.5)
    attrutil.pprint(config)


def test_selfplay_config():
    config = templates.selfplay_config_template()
    attrutil.pprint(config)


def test_server_config_template():
    game = "breakthrough"
    gen_prefix = "x1"
    prev_states = 1
    config = templates.server_config_template(game, gen_prefix, prev_states)
    attrutil.pprint(config)
