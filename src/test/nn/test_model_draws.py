import os
import random

import numpy as np
import tensorflow as tf

from ggplib.util.init import setup_once
from ggplib.db import lookup
from ggpzero.util import keras

from ggpzero.nn.manager import get_manager


def setup():
    # set up ggplib
    setup_once()

    # ensure we have database with ggplib
    from gzero_games.ggphack import addgame
    lookup.get_database()

    # initialise keras/tf
    keras.init()

    # just ensures we have the manager ready
    get_manager()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    np.set_printoptions(threshold=100000)


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


def test_baduk():
    def show(pred):
        win_0, win_1, draw = list(pred.scores)
        print "wins/draw", win_0, win_1, draw
        win_0 += draw / 2
        win_1 += draw / 2
        print "wins only", win_0, win_1, win_0 + win_1

    man = get_manager()

    game = "baduk_9x9"

    # create a nn
    game_info = lookup.by_name(game)
    sm = game_info.get_sm()

    nn = man.load_network(game, "h3_11")
    nn.summary()

    basestate = sm.get_initial_state()

    predictions = nn.predict_1(basestate.to_list())
    print predictions.policies, predictions.scores

    predictions = nn.predict_n([basestate.to_list(), basestate.to_list()])
    assert len(predictions) == 2 and len(predictions[0].policies) == 2 and len(predictions[0].scores) == 3
    show(predictions[0])

    prevs = []
    for i in range(4):
        prevs.append(basestate)
        basestate = advance_state(game_info.get_sm(), basestate)

    prediction = nn.predict_1(basestate.to_list(), [p.to_list() for p in prevs])
    show(prediction)
