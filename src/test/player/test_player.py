import os

from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db import lookup

from ggpzero.util import attrutil
from ggpzero.defs import confs, templates
from ggpzero.nn.manager import get_manager

from ggpzero.player.puctplayer import PUCTPlayer
from ggpzero.battle.bt import pretty_board


BOARD_SIZE = 7
GAME = "bt_7"
RANDOM_GEN = "rand_0"

GOOD_GEN1 = "x1_132"


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

    man = get_manager()
    if not man.can_load(GAME, RANDOM_GEN):
        network = man.create_new_network(GAME)
        man.save_network(network, RANDOM_GEN)


def play(player_white, player_black, move_time=0.5):
    gm = GameMaster(lookup.by_name(GAME), verbose=True)
    gm.add_player(player_white, "white")
    gm.add_player(player_black, "black")

    gm.start(meta_time=15, move_time=move_time)

    move = None
    while not gm.finished():

        # print out the board
        pretty_board(BOARD_SIZE, gm.sm)

        move = gm.play_single_move(last_move=move)

    gm.finalise_match(move)


def test_random():
    # add two players
    # simplemcts vs RANDOM_GEN
    pymcs = get.get_player("simplemcts")
    pymcs.max_run_time = 0.25

    eval_config = templates.base_puct_config(verbose=True,
                                             max_dump_depth=1)
    puct_config = confs.PUCTPlayerConfig("gzero",
                                         True,
                                         100,
                                         0,
                                         RANDOM_GEN,
                                         eval_config)

    attrutil.pprint(puct_config)

    puct_player = PUCTPlayer(puct_config)

    play(pymcs, puct_player)


def test_trained():
    # simplemcts vs GOOD_GEN
    simple = get.get_player("simplemcts")
    simple.max_run_time = 0.5

    eval_config = confs.PUCTEvaluatorConfig(verbose=True,
                                            puct_constant=0.85,
                                            puct_constant_root=3.0,

                                            dirichlet_noise_pct=-1,

                                            fpu_prior_discount=0.25,
                                            fpu_prior_discount_root=0.15,

                                            choose="choose_temperature",
                                            temperature=2.0,
                                            depth_temperature_max=10.0,
                                            depth_temperature_start=0,
                                            depth_temperature_increment=0.75,
                                            depth_temperature_stop=1,
                                            random_scale=1.0,
                                            batch_size=1,
                                            max_dump_depth=1)

    puct_config = confs.PUCTPlayerConfig("gzero",
                                         True,
                                         200,
                                         0,
                                         GOOD_GEN1,
                                         eval_config)
    attrutil.pprint(puct_config)

    puct_player = PUCTPlayer(puct_config)

    play(simple, puct_player)
    #play(puct_player, simple)
