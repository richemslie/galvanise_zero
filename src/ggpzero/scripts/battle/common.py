import os
import re

import tensorflow as tf

from ggplib.util.init import setup_once

from ggplib.player import get

from ggpzero.util.keras import init

from ggpzero.defs import confs


def setup():
    setup_once()
    init()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)


def get_puct_config(gen, **kwds):
    conf = confs.PUCTPlayerConfig(name="",
                                  generation=gen,
                                  verbose=True,

                                  playouts_per_iteration=-1,
                                  playouts_per_iteration_noop=0,

                                  dirichlet_noise_alpha=-1,

                                  root_expansions_preset_visits=-1,
                                  puct_before_expansions=3,
                                  puct_before_root_expansions=5,
                                  puct_constant_before=3.0,
                                  puct_constant_after=0.75,

                                  choose="choose_temperature",
                                  temperature=1.0,
                                  depth_temperature_max=5.0,
                                  depth_temperature_start=0,
                                  depth_temperature_increment=0.5,
                                  depth_temperature_stop=1,
                                  random_scale=1.00,

                                  fpu_prior_discount=0.25,

                                  max_dump_depth=1)
    for k, v in kwds.items():
        setattr(conf, k, v)

    return conf


def parse_sgf(text):
    tokens = re.split(r'([a-zA-Z]+\[[^\]]+\])', text)
    tokens = [t for t in tokens if ']' in t.strip()]

    def game_info():
        for t in tokens:
            key, value = re.search(r'([a-zA-Z]+)\[([^\]]+)\]', t).groups()
            yield key, value

    moves = []
    moves = []
    pb = pw = None
    for k, v in game_info():
        # white/black is wrong on LG (XXX check this)
        if k == "PB":
            pw = v
        if k == "PW":
            pb = v
        if k in "WB":
            moves.append((k, v))

    return pw, pb, moves


def simplemcts_player(move_time):
    player = get.get_player("simplemcts")
    player.max_tree_search_time = move_time
    return player
