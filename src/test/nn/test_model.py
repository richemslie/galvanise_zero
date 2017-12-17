from ggplib.db import lookup

from ggplearn.nn import bases


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    lookup.get_database()

    from ggplearn.util.keras import constrain_resources
    constrain_resources()


def test_config():
    for game in "breakthrough connectFour reversi".split():
        # look game from database

        game_info = lookup.by_name(game)

        assert game == game_info.game
        bases_config = bases.get_config(game_info.game, game_info.model)
        assert bases_config

        print bases_config.x_term
        print bases_config.x_cords

        sm = game_info.get_sm()
        basestate = sm.get_initial_state()

        print bases_config.num_rows
        print bases_config.num_cols

        print bases_config.state_to_channels(basestate.to_list(), 0, channel_last=False)
        print bases_config.state_to_channels(basestate.to_list(), 1, channel_last=False)

        print bases_config.state_to_channels(basestate.to_list(), 0, channel_last=True)
        print bases_config.state_to_channels(basestate.to_list(), 1, channel_last=True)


def test_net_create():
    for game in "breakthrough reversi".split():
        game_info = lookup.by_name(game)

        assert game == game_info.game
        bases_config = bases.get_config(game_info.game, game_info.model)
        nn = bases_config.create_network()

        sm = game_info.get_sm()
        basestate = sm.get_initial_state()

        policy, scores = nn.predict_1(basestate.to_list(), 0)
        print policy, scores

        res = nn.predict_n([basestate.to_list(), basestate.to_list()], [1, 0])
        assert len(res) == 2 and len(res[0]) == 2 and len(res[1]) == 2
        print policy, scores


def test_net_sizes():
    for game in "breakthrough reversi".split():
        game_info = lookup.by_name(game)

        assert game == game_info.game
        bases_config = bases.get_config(game_info.game, game_info.model)
        nn = bases_config.create_network()
        nn.summary()

        print "small"
        nn = bases_config.create_network(small=True)
        nn.summary()

        print "tiny"
        nn = bases_config.create_network(tiny=True)
        nn.summary()


def test_net_sizes_with_l2():
    for game in "breakthrough reversi".split():
        game_info = lookup.by_name(game)

        assert game == game_info.game
        bases_config = bases.get_config(game_info.game, game_info.model)
        nn = bases_config.create_network(a0_reg=True)
        nn.summary()

        print "small"
        nn = bases_config.create_network(small=True, a0_reg=True)
        nn.summary()

        print "tiny"
        nn = bases_config.create_network(tiny=True, a0_reg=True)
        nn.summary()


def test_save_load_net():
    game_info = lookup.by_name("breakthrough")

    bases_config = bases.get_config(game_info.game, game_info.model, "test")
    nn = bases_config.create_network(tiny=True)
    nn.summary()

    nn.save()
    nn.load()
    nn.save()
