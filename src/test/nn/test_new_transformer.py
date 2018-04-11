import os

import numpy as np

import tensorflow as tf

from ggplib.util.init import setup_once

from ggplib.db import lookup

from ggpzero.util import keras
from ggpzero.defs import gamedesc, templates
from ggpzero.nn.bases import GdlBasesTransformer

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

    np.set_printoptions(threshold=100000, precision=2)


def test_game_descriptions():
    game_descs = gamedesc.Games()
    names = [name for name in dir(game_descs) if name[0] != "_"]

    names = ["breakthroughSmall", "breakthrough", "internationalDraughts"]
    names = ["internationalDraughts"]

    for name in names:
        print
        print "=" * 80
        print name
        print "=" * 80

        meth = getattr(game_descs, name)
        game_description = meth()

        print name, game_description.game
        print game_description
        print "-" * 80

        game_info = lookup.by_name(game_description.game)

        # create GenerationDescription
        generation_descr = templates.default_generation_desc(game_description.game)

        transformer = GdlBasesTransformer(game_info, generation_descr, game_description)
        transformer = transformer

        sm = game_info.get_sm()
        basestate = sm.get_initial_state()

        from test_model import advance_state

        for i in range(25):
            print "move made", i
            print sm.basestate_to_str(basestate)
            print transformer.state_to_channels(basestate.to_list())

            sm.update_bases(basestate)
            if sm.is_terminal():
                break

            basestate = advance_state(sm, basestate)
