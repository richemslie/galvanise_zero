import os
import random

import numpy as np

import tensorflow as tf

from ggplib.util.init import setup_once
from ggplib.db import lookup

from ggpzero.util import keras
from ggpzero.defs import templates

from ggpzero.nn.manager import get_manager

def setup():
    # set up ggplib
    setup_once()

    # ensure we have database with ggplib
    lookup.get_database()

    # initialise keras/tf
    keras.init()

    # just ensures we have the manager ready
    get_manager()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    np.set_printoptions(threshold=100000)


games = ("checkers escortLatch breakthroughSmall breakthrough "
         "connectFour reversi cittaceot speedChess".split())

# games = ["escortLatch"]


def advance_state(sm, basestate):
    sm.update_bases(basestate)

    # leaks, but who cares, it is a test
    joint_move = sm.get_joint_move()
    base_state = sm.new_base_state()

    for role_index in range(len(sm.get_roles())):
        ls = sm.get_legal_state(role_index)
        choice = ls.get_legal(random.randrange(0, ls.get_count()))
        joint_move.set(role_index, choice)

    # play move, the base_state will be new state
    sm.next_state(joint_move, base_state)
    return base_state


def test_basic_config():
    man = get_manager()

    for game in games:
        # lookup game in manager
        transformer = man.get_transformer(game)
        print transformer.x_term
        print transformer.x_cords

        # look game from database
        game_info = lookup.by_name(game)
        assert game == game_info.game

        sm = game_info.get_sm()
        basestate = sm.get_initial_state()

        print "rows x cols", transformer.num_rows, transformer.num_cols

        print
        print transformer.state_to_channels(basestate.to_list())
        basestate = advance_state(game_info.get_sm(), basestate)
        print
        print transformer.state_to_channels(basestate.to_list())


def test_config_channel_last():
    man = get_manager()

    for game in games:
        # create GenerationDescription
        generation_descr = templates.default_generation_desc(game)
        generation_descr.channel_last = True

        # lookup game in manager
        transformer = man.get_transformer(game, generation_descr)
        print transformer.x_term
        print transformer.x_cords

        # look game from database
        game_info = lookup.by_name(game)
        assert game == game_info.game

        sm = game_info.get_sm()
        basestate = sm.get_initial_state()

        print "rows x cols", transformer.num_rows, transformer.num_cols

        print
        print transformer.state_to_channels(basestate.to_list())
        basestate = advance_state(game_info.get_sm(), basestate)
        print
        print transformer.state_to_channels(basestate.to_list())


def test_config_previous_states():
    man = get_manager()

    for game in games:
        # create GenerationMetaAttributes
        generation_descr = templates.default_generation_desc(game)
        generation_descr.num_previous_states = 2

        # lookup game in manager
        transformer = man.get_transformer(game, generation_descr)
        print transformer.x_term
        print transformer.x_cords

        # look game from database
        game_info = lookup.by_name(game)
        assert game == game_info.game

        sm = game_info.get_sm()
        basestate0 = sm.get_initial_state()
        basestate1 = advance_state(game_info.get_sm(), basestate0)
        basestate2 = advance_state(game_info.get_sm(), basestate1)

        print "rows x cols", transformer.num_rows, transformer.num_cols
        print "num_channels", transformer.num_channels

        print "basestate0:"
        print transformer.state_to_channels(basestate0.to_list(), [])

        print "basestate1:"
        print transformer.state_to_channels(basestate1.to_list(), [basestate0.to_list()])

        print "basestate2:"
        print transformer.state_to_channels(basestate2.to_list(), [basestate1.to_list(),
                                                                   basestate0.to_list()])

def test_net_create():
    man = get_manager()

    for game in games:
        # create a nn
        nn = man.create_new_network(game, "tiny")

        game_info = lookup.by_name(game)
        sm = game_info.get_sm()
        basestate0 = sm.get_initial_state()

        prediction = nn.predict_1(basestate0.to_list())
        print prediction.policies, prediction.scores

        basestate1 = advance_state(game_info.get_sm(), basestate0)
        predictions = nn.predict_n([basestate1.to_list(), basestate1.to_list()])
        assert len(predictions) == 2
        assert len(predictions[0].policies) == 2
        assert len(predictions[0].scores) == 2
        print predictions[0].policies, predictions[0].scores


def test_net_multiple_policies():
    man = get_manager()

    for game in games:
        # create GenerationDescription
        generation_descr = templates.default_generation_desc(game, multiple_policy_heads=True)

        nn = man.create_new_network(game, "tiny", generation_descr)
        nn.summary()

        game_info = lookup.by_name(game)
        sm = game_info.get_sm()
        basestate0 = sm.get_initial_state()

        heads = nn.predict_1(basestate0.to_list())
        print heads.policies, heads.scores

        basestate1 = advance_state(game_info.get_sm(), basestate0)
        heads = nn.predict_n([basestate1.to_list()])
        assert len(heads) == 1 and len(heads[0].policies) == 2 and len(heads[0].scores) == 2
        print heads[0].policies, heads[0].scores


def test_net_different_net_sizes():
    man = get_manager()

    for game in games:

        # create a nn for these sizes
        for size in "tiny smaller small medium-small medium medium-large large larger massive".split():
            print
            print size
            nn = man.create_new_network(game, size)
            nn.summary()
            print
            print game, size
            # print "hit return"
            # raw_input()


def test_net_sizes_with_l2():
    man = get_manager()

    for game in games:
        generation_descr = templates.default_generation_desc(game, name="L2_1")
        transformer = man.get_transformer(game, generation_descr)

        # create a nn
        for size in "tiny medium".split():
            print
            print size
            model_conf = templates.nn_model_config_template(game, size, transformer)
            model_conf.l2_regularisation = True
            model_conf.dropout_rate_value = -1
            model_conf.dropout_rate_policy = -1
            nn = man.create_new_network(game, model_conf, generation_descr)
            nn.summary()
            print
            # print "hit return to compile"
            # raw_input()
            nn.compile()


def test_save_load_net():
    man = get_manager()
    game = "breakthrough"
    generation = "gen_1"

    generation_descr = templates.default_generation_desc(game, generation)
    transformer = man.get_transformer(game, generation_descr)
    model_conf = templates.nn_model_config_template(game, "tiny", transformer)
    nn = man.create_new_network(game, model_conf, generation_descr)
    nn.summary()

    man.save_network(nn)

    assert man.can_load(game, generation)

    nn2 = man.load_network(game, generation)
    nn2.summary()

    assert nn is not nn2
    assert nn.generation_descr.name == generation
    assert nn.generation_descr.name == nn2.generation_descr.name
    assert nn.gdl_bases_transformer is nn2.gdl_bases_transformer


def test_cittaceot():
    man = get_manager()

    game = "cittaceot"

    # create a nn
    nn = man.create_new_network(game, "tiny")
    game_info = lookup.by_name(game)
    sm = game_info.get_sm()
    basestate = sm.get_initial_state()

    predictions = nn.predict_1(basestate.to_list())
    print predictions.policies, predictions.scores

    predictions = nn.predict_n([basestate.to_list(), basestate.to_list()])
    assert len(predictions) == 2 and len(predictions[0].policies) == 2 and len(predictions[0].scores) == 2
    print predictions


def test_speed_chess_network_sizes():
    game = "speedChess"

    game_info = lookup.by_name(game)
    sm = game_info.get_sm()

    # create GenerationDescription
    def get_generation_descr():
        generation_descr = templates.default_generation_desc(game)
        generation_descr.num_previous_states = 0
        yield generation_descr

        generation_descr.num_previous_states = 2
        yield generation_descr

        generation_descr.num_previous_states = 30
        yield generation_descr

        generation_descr.multiple_policy_heads = True
        yield generation_descr

    man = get_manager()
    for descr in get_generation_descr():
        transformer = man.get_transformer(game, descr)

        nn = man.create_new_network(game, "small", descr)
        nn.summary()


def test_tron():
    import py.test
    py.test.skip("WIP")
    game = "tron_10x10"
    generation = "test_1"

    man = get_manager()

    # create a nn
    model_conf = templates.nn_model_config_template(game)
    generation_descr = templates.default_generation_desc(game, generation, multiple_policy_heads=True)

    nn = man.create_new_network(game, model_conf, generation_descr)

    game_info = lookup.by_name(game)
    sm = game_info.get_sm()
    basestate = sm.get_initial_state()

    policy, scores = nn.predict_1(basestate.to_list())
    print policy, scores

    res = nn.predict_n([basestate.to_list(), basestate.to_list()])
    assert len(res) == 2 and len(res[0]) == 2 and len(res[1]) == 2
    print policy, scores
