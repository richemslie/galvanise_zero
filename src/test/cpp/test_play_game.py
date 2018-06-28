import os

import tensorflow as tf

from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db import lookup

from ggpzero.util import attrutil
from ggpzero.defs import templates
from ggpzero.nn.manager import get_manager

from ggpzero.player.puctplayer import PUCTPlayer, PUCTPlayerV2, get_default_conf


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)


def play_game(generation_name, player_clz, num_previous_states=1):
    game = "breakthrough"

    # ensure we have a network
    man = get_manager()
    generation_descr = templates.default_generation_desc(game,
                                                         multiple_policy_heads=True,
                                                         num_previous_states=num_previous_states)

    nn = man.create_new_network(game, "tiny", generation_descr)
    man.save_network(nn, generation_name)
    nn.summary()

    conf = get_default_conf(generation_name, max_dump_depth=2, playouts_per_iteration=42)

    attrutil.pprint(conf)

    gm = GameMaster(lookup.by_name(game))

    gm.add_player(player_clz(conf), "white")
    gm.add_player(get.get_player("random"), "black")

    gm.start(meta_time=30, move_time=15)

    last_move = None
    while not gm.finished():
        last_move = gm.play_single_move(last_move=last_move)

    gm.finalise_match(last_move)
    return gm


def test_play_game():
    generation_name = "test_play_0"
    play_game(generation_name, PUCTPlayer, num_previous_states=0)


def test_play_game_prev_states():
    generation_name = "test_play_3"
    play_game(generation_name, PUCTPlayer, num_previous_states=3)


def test_play_game_v2():
    generation_name = "test_play_0"
    play_game(generation_name, PUCTPlayerV2, num_previous_states=0)


def test_play_game_v2_prev_states():
    generation_name = "test_play_3"
    play_game(generation_name, PUCTPlayerV2, num_previous_states=3)
