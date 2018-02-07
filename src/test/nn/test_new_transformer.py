import os
from collections import Counter

import numpy as np

import tensorflow as tf

from ggplib.util.init import setup_once
from ggplib.util.symbols import SymbolFactory
from ggplib.db import lookup

from ggpzero.util import keras
from ggpzero.defs import gamedesc
from ggpzero.nn.bases import BaseInfo

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


class BaseToBoardSpace(object):
    def __init__(self, base_indx, channel_id, x_cord_idx, y_cord_idx):
        # which base it is
        self.base_indx = base_indx

        # index into the board channel (relative to start channel for state [there can be multiple
        # of them with prev_states])
        self.channel_id = channel_id

        # the x/y cords
        self.x_cord_idx = x_cord_idx
        self.y_cord_idx = y_cord_idx

    def __repr__(self):
        return "BaseToBoardSpace(%s, %s, %s, %s)" % (self.base_indx, self.channel_id, self.x_cord_idx, self.y_cord_idx)


class BaseToChannelSpace(object):
    def __init__(self, base_indx, channel_id, value):
        # which base it is
        self.base_indx = base_indx

        # which channel it is (absolute)
        self.channel_id = channel_id

        # the value to set the entire channel (flood fill)
        self.value = value

    def __repr__(self):
        return "BaseToChannelSpace(%s, %s, %s)" % (self.base_indx, self.channel_id, self.value)


def create_base_infos(game_info, desc):
    # ############ XXX
    # base_infos (this part anyway) should probably be done in the StateMachineModel
    symbol_factory = SymbolFactory()
    sm_model = game_info.model
    base_infos = [BaseInfo(idx, s, symbol_factory.symbolize(s)) for idx, s in enumerate(sm_model.bases)]
    # ########### XXX

    all_cords = []
    for y_cord in desc.x_cords:
        for x_cord in desc.y_cords:
            all_cords.append((y_cord, x_cord))

    # first do board space
    board_space = []
    channel_mapping = {}

    count = Counter()
    for b_info in base_infos:
        the_bc = None
        for bc in desc.board_channels:
            if b_info.terms[0] == bc.base_term:
                the_bc = bc
                break

        if not the_bc:
            print b_info.terms, "No board_channels"
            continue

        matched = []
        for bt in bc.board_terms:
            print 'here1', b_info.terms
            print 'here2', b_info.terms[bt.term_idx], bt.terms
            if b_info.terms[bt.term_idx] in bt.terms:
                matched.append(b_info.terms[bt.term_idx])
                continue

        print matched
        if len(matched) != len(bc.board_terms):
            print "Not matched", b_info.terms

        # create a BaseToBoardSpace
        key = tuple([b_info.terms[0]] + matched)
        count[key] += 1

        if count[key] == 1:
            assert key not in channel_mapping
            channel_mapping[key] = len(channel_mapping)

        channel_id = channel_mapping[key]

        x_cord = b_info.terms[the_bc.x_term_idx]
        y_cord = b_info.terms[the_bc.y_term_idx]

        x_idx = desc.x_cords.index(x_cord)
        y_idx = desc.y_cords.index(y_cord)

        board_space.append(BaseToBoardSpace(b_info.index, channel_id, x_idx, y_idx))

    for b in board_space:
        print base_infos[b.base_indx].terms, "->", b

    # now do channel space
    channel_space = []

    for channel_id, cc in enumerate(desc.control_channels):
        look_for = set([tuple(cb.arg_terms) for cb in cc.control_bases])

        for b_info in base_infos:
            the_terms = tuple(b_info.terms)
            if the_terms in look_for:

                done = False
                for cb in cc.control_bases:
                    if the_terms == tuple(cb.arg_terms):
                        channel_space.append(BaseToChannelSpace(b_info.index, channel_id, cb.value))
                        done = True
                        break

                assert done

    for cs in channel_space:
        print base_infos[cs.base_indx].terms, "->", cs


def test_game_descriptions():
    game_descs = gamedesc.Games()
    names = [name for name in dir(game_descs) if name[0] != "_"]

    for name in names:
        print
        print "=" * 80
        print name
        print "=" * 80

        meth = getattr(game_descs, name)
        desc = meth()

        print name, desc.game
        print desc
        print "-" * 80

        game_info = lookup.by_name(desc.game)
        create_base_infos(game_info, desc)
