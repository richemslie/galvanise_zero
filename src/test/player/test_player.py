import os
import time

from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game

from ggpzero.defs import confs
from ggpzero.player.puctplayer import PUCTPlayer

import py.test

ITERATIONS = 1

current_gen = "v5_90"

# first in the run, completely random weights
random_gen = "v5_0"

default_puct_config = confs.PUCTPlayerConfig(generation=current_gen,
                                             playouts_per_iteration=42,
                                             playouts_per_iteration_noop=1)


def setup():
    import tensorflow as tf

    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    import numpy as np
    np.set_printoptions(threshold=100000)


def test_puct_tournament():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    # add two players
    pymcs = get.get_player("pymcs")
    pymcs.max_run_time = 0.1

    black = PUCTPlayer(conf=default_puct_config)

    gm.add_player(pymcs, "white")
    gm.add_player(black, "black")

    acc_black_score = 0
    acc_red_score = 0
    for _ in range(ITERATIONS):
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["white"]

    print "white_score", gm.players_map["white"].name, acc_red_score
    print "black_score", gm.players_map["black"].name, acc_black_score


def test_fast_plays():
    ''' very fast rollouts, basically this config of puct player is a policy player '''
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    import attr
    conf = confs.PUCTPlayerConfig(**attr.asdict(default_puct_config))
    conf.verbose = False

    # just checking that we haven't modified default
    assert not conf.verbose and default_puct_config.verbose

    conf.playouts_per_iteration = 1
    conf.playouts_per_iteration_noop = 0
    conf.dirichlet_noise_alpha = -1
    print conf

    # add two players
    white = PUCTPlayer(conf=conf)
    black = PUCTPlayer(conf=conf)

    gm.add_player(white, "white")
    gm.add_player(black, "black")

    acc_black_score = 0
    acc_red_score = 0
    s = time.time()
    for _ in range(ITERATIONS):
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["white"]

        print gm.get_game_depth()

    print "time taken", time.time() - s
    print "white_score", gm.players_map["white"].name, acc_red_score
    print "black_score", gm.players_map["black"].name, acc_black_score


def test_not_taking_win():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    # add two players
    gm.add_player(PUCTPlayer(default_puct_config), "white")
    gm.add_player(PUCTPlayer(default_puct_config), "black")

    str_state = """ (true (control black))
    (true (cellHolds 8 8 black)) (true (cellHolds 8 7 black)) (true (cellHolds 8 2 white))
    (true (cellHolds 8 1 white)) (true (cellHolds 7 8 black)) (true (cellHolds 6 7 white))
    (true (cellHolds 6 2 white)) (true (cellHolds 6 1 white)) (true (cellHolds 5 4 black))
    (true (cellHolds 5 3 black)) (true (cellHolds 5 2 white)) (true (cellHolds 5 1 white))
    (true (cellHolds 4 8 black)) (true (cellHolds 4 7 black)) (true (cellHolds 4 1 white))
    (true (cellHolds 3 8 black)) (true (cellHolds 3 6 black)) (true (cellHolds 3 2 white))
    (true (cellHolds 2 8 black)) (true (cellHolds 2 7 black)) (true (cellHolds 2 2 white))
    (true (cellHolds 2 1 white)) (true (cellHolds 1 8 black)) (true (cellHolds 1 7 black))
    (true (cellHolds 1 2 white)) (true (cellHolds 1 1 white)) """

    gm.start(meta_time=30, move_time=5,
             initial_basestate=gm.convert_to_base_state(str_state))

    last_move = gm.play_single_move(last_move=None)
    assert last_move[1] == "(move 7 8 6 7)"

