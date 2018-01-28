import os

import tensorflow as tf

from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game

from ggpzero.defs import templates
from ggpzero.nn.manager import get_manager

from ggpzero.player.cpuctplayer import CppPUCTPlayer


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)


def play_game(generation_name, num_previous_states):
    game = "breakthrough"

    # ensure we have a network
    man = get_manager()
    generation_descr = templates.default_generation_desc(game,
                                                         multiple_policy_heads=True,
                                                         num_previous_states=num_previous_states)

    nn = man.create_new_network(game, "tiny", generation_descr)
    man.save_network(nn, generation_name)
    nn.summary()

    conf = templates.puct_config_template(generation_name, "compete")

    gm = GameMaster(get_gdl_for_game(game))

    gm.add_player(CppPUCTPlayer(conf=conf), "white")
    gm.add_player(get.get_player("random"), "black")

    gm.start(meta_time=30, move_time=15)
    gm.play_to_end()


def test_play_game():
    generation_name = "test_play_0"
    play_game(generation_name, num_previous_states=0)


def test_play_game2():
    generation_name = "test_play_3"
    play_game(generation_name, num_previous_states=3)
