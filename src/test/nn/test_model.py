from ggplib.db import lookup

from ggplearn import templates

from ggplearn.nn.manager import get_manager


def setup():
    from ggplib.util.init import setup_once
    from ggplearn.util import keras

    # set up ggplib
    setup_once()
    # ensure we have database with ggplib
    lookup.get_database()

    # initialise keras/tf
    keras.init()

    # just ensures we have the manager ready
    get_manager()


def test_config():
    man = get_manager()
    for game in "breakthrough connectFour reversi".split():

        # lookup game in manager
        transformer = man.get_transformer(game)
        print transformer.x_term
        print transformer.x_cords

        # look game from database
        game_info = lookup.by_name(game)
        assert game == game_info.game

        sm = game_info.get_sm()
        basestate = sm.get_initial_state()

        print transformer.num_rows
        print transformer.num_cols

        # illegal fishing
        try:
            return_to = transformer.channel_last

            transformer.channel_last = False
            print transformer.state_to_channels(basestate.to_list(), 0)
            print transformer.state_to_channels(basestate.to_list(), 1)

            # illegal fishing
            transformer.channel_last = True
            print transformer.state_to_channels(basestate.to_list(), 0)
            print transformer.state_to_channels(basestate.to_list(), 1)
        finally:
            transformer.channel_last = return_to


def test_net_create():
    man = get_manager()

    for game in "breakthrough reversi".split():
        # create a nn
        model_conf = templates.nn_model_config_template(game)
        nn = man.create_new_network(game, model_conf)

        game_info = lookup.by_name(game)
        sm = game_info.get_sm()
        basestate = sm.get_initial_state()

        policy, scores = nn.predict_1(basestate.to_list(), 0)
        print policy, scores

        res = nn.predict_n([basestate.to_list(), basestate.to_list()], [1, 0])
        assert len(res) == 2 and len(res[0]) == 2 and len(res[1]) == 2
        print policy, scores


def test_net_sizes():
    man = get_manager()

    for game in "breakthrough reversi".split():
        # create a nn
        for size in "tiny smaller small normal large".split():
            print
            print size
            model_conf = templates.nn_model_config_template(game, size)
            nn = man.create_new_network(game, model_conf)
            nn.summary()
            print
            # print "hit return"
            # raw_input()


def test_net_sizes_with_l2():
    man = get_manager()

    for game in "breakthrough reversi".split():
        # create a nn
        for size in "tiny small normal large".split():
            print
            print size
            model_conf = templates.nn_model_config_template(game, size)
            model_conf.alphazero_regularisation = True
            model_conf.dropout_rate_value = -1
            model_conf.dropout_rate_policy = -1
            nn = man.create_new_network(game, model_conf)
            nn.summary()
            print
            # print "hit return to compile"
            # raw_input()
            nn.compile()


def test_save_load_net():
    man = get_manager()
    game = "breakthrough"
    gen = "gen1"

    model_conf = templates.nn_model_config_template("breakthrough", "tiny")
    nn = man.create_new_network(game, model_conf)
    nn.summary()

    man.save_network(nn, game, gen)

    assert man.can_load(game, gen)

    nn2 = man.load_network(game, gen)
    nn2.summary()

    assert nn is not nn2
