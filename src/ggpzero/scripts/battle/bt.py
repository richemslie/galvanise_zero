"""
        # moves = "b2-c3 a5-a4 e2-d3 b5-c4 f1-e2 f5-e4 a1-b2 f6-f5 b2-b3 a6-b5 f2-e3 e4-d3 e2-d3
        # f5-e4 e1-e2 e4-d3 e2-d3 c4-b3 a2-b3 a4-a3 d1-e2 e6-f5 c3-c4 a3-b2 c1-b2 e5-d4".split()
        # moves += "".split()

        # wanderer - main variation
        # play(["e2-d3", "e5-d4", "a2-b3", "a5-b4", "f2-f3", "f6-e5", "f1-f2", "f5-f4", "e1-e2",
        # "c5-c4", "a1-a2"], MOVE_TIME)

        # play(["b2-c3"], MOVE_TIME)

        # gzero - variation
        # play(["e2-d3", "e5-d4", "a2-b3", "a5-b4", "f2-f3", "f6-e5", "a1-a2", "e5-e4", "f3-e4",
        # "f5-e4", "f1-e2", "c5-c4", "d3-e4"], MOVE_TIME)

        # game 1 / gen x1_93

        # play(["e2-d3", "c5-d4", "d2-c3", "d4-c3", "b2-c3",
        #      "b5-c4", "a2-b3", "c4-d3", "c2-d3", "e5-e4",
        #      "f1-e2", "f5-f4", "c1-d2"], MOVE_TIME)

        # game 2 / gen x1_93
        # play(["c2-d3", "b5-c4", "d3-c4", "d5-c4", "b2-c3", "e5-e4", "a2-a3",
        #      "a6-b5", "a1-b2", "a5-a4", "f2-e3"], MOVE_TIME)
"""

import os
import pdb
import sys
import traceback

import tensorflow as tf

from ggplib.db.helper import get_gdl_for_game
from ggplib.db import lookup
from ggplib.player.gamemaster import GameMaster

from ggpzero.defs import confs
from ggpzero.player.puctplayer import PUCTPlayer


# BOARD_SIZE = 6
BOARD_SIZE = 8


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)


def get_config(generation):
    return confs.PUCTPlayerConfig(name="bt_config",
                                  generation=generation,
                                  verbose=True,

                                  playouts_per_iteration=-1,
                                  playouts_per_iteration_noop=0,

                                  dirichlet_noise_alpha=-1,

                                  root_expansions_preset_visits=-1,
                                  puct_before_expansions=3,
                                  puct_before_root_expansions=5,
                                  puct_constant_before=3.0,
                                  puct_constant_after=0.75,

                                  fpu_prior_discount=0.25,

                                  choose="choose_top_visits",
                                  temperature=1.5,
                                  depth_temperature_max=3.0,
                                  depth_temperature_start=0,
                                  depth_temperature_increment=0.5,
                                  depth_temperature_stop=4,
                                  random_scale=1.00,

                                  max_dump_depth=2)


def pretty_board(sm):
    ' pretty print board current state of match '
    from ggplib.util.symbols import SymbolFactory

    lines = []
    for i, value in enumerate(sm.get_current_state().to_list()):
        if value:
            lines.append(sm.get_gdl(i))

    sf = SymbolFactory()
    states = sf.to_symbols("\n".join(lines))
    mapping = {}
    control = None
    for s in list(states):
        if s[1][0] == "control":
            control = s[1][1]
        else:
            if BOARD_SIZE == 8:
                assert s[1][0] == "cell"
            else:
                assert s[1][0] == "cellHolds"

            key = int(s[1][1]), int(s[1][2])
            mapping[key] = s[1][3]

    lines = []
    line_len = BOARD_SIZE * 4 + 1
    lines.append("    +" + "-" * (line_len - 2) + "+")
    for i in reversed(range(1, BOARD_SIZE + 1)):
        ll = [" %s  |" % i]
        for j in reversed(range(1, BOARD_SIZE + 1)):
            key = j, i
            if key in mapping:
                if mapping[key] == "black":
                    ll.append(" %s |" % u"\u2659")
                else:
                    assert mapping[key] == "white"
                    ll.append(" %s |" % u"\u265F")
            else:
                ll.append("   |")

        lines.append("".join(ll))
        if i > 1:
            lines.append("    " + "-" * line_len)

    lines.append("    +" + "-" * (line_len - 2) + "+")
    if BOARD_SIZE == 8:
        lines.append("     " + ' '.join(' %s ' % c for c in 'abcdefgh'))
    else:
        lines.append("     " + ' '.join(' %s ' % c for c in 'abcdef'))

    print
    print
    print "\n".join(lines)
    print "Control:", control


def play(moves, generation, move_time):
    # add players
    if BOARD_SIZE == 8:
        game = "breakthrough"
    else:
        game = "breakthroughSmall"

    gm = GameMaster(get_gdl_for_game(game))
    conf = get_config(generation)
    for role in gm.sm.get_roles():
        gm.add_player(PUCTPlayer(conf), role)

    sm = lookup.by_name(game).get_sm()
    sm.reset()

    pretty_board(sm)

    # get some states
    joint_move = sm.get_joint_move()
    base_state = sm.get_initial_state()

    def to_cords(s):
        if BOARD_SIZE == 8:
            mapping_x_cord = {x0 : x1 for x0, x1 in zip('abcdefgh', '87654321')}
        else:
            mapping_x_cord = {x0 : x1 for x0, x1 in zip('abcdef', '654321')}

        return mapping_x_cord[s[0]], s[1]

    def f(ri, i):
        return sm.legal_to_move(ri, ls.get_legal(i))

    lead_role_index = 0
    gdl_moves = []
    for m in moves:
        from_, to_ = map(to_cords, m.split("-"))
        gdl_move = []
        for ri in range(2):
            if ri == lead_role_index:
                gdl_role_move = "(move %s %s %s %s)" % (from_[0], from_[1], to_[0], to_[1])
            else:
                gdl_role_move = "noop"

            ls = sm.get_legal_state(ri)
            the_moves = [f(ri, ii) for ii in range(ls.get_count())]
            choice = the_moves.index(gdl_role_move)
            joint_move.set(ri, ls.get_legal(choice))
            gdl_move.append(str(gdl_role_move))

        # update state machine
        sm.next_state(joint_move, base_state)
        sm.update_bases(base_state)
        lead_role_index ^= 1

        gdl_moves.append(str(gdl_move))
        print "%s -> %s" % (m, gdl_move)
        pretty_board(sm)

    # play move via gamemaster:
    gm.reset()
    gm.start(meta_time=15, move_time=move_time,
             initial_basestate=sm.get_current_state(),
             game_depth=len(moves))

    move = gm.play_single_move(last_move=None)
    player = gm.get_player(lead_role_index)

    move = move[lead_role_index]
    move = move.replace("(move", "").replace(")", "")
    a, b, c, d = move.split()
    if BOARD_SIZE == 8:
        mapping_x_cord = {x0 : x1 for x0, x1 in zip('87654321', 'abcdefgh')}
    else:
        mapping_x_cord = {x0 : x1 for x0, x1 in zip('654321', 'abcdef')}

    next_move = "%s%s-%s%s" % (mapping_x_cord[a], b, mapping_x_cord[c], d)

    sm.update_bases(gm.sm.get_current_state())
    pretty_board(sm)
    print "PLAYED", next_move, player.last_probability
    return next_move, player.last_probability


if __name__ == "__main__":

    try:
        setup()

        MOVE_TIME = 1 * 60

        moves = []
        lines = []
        for i in range(32):
            move, prob = play(moves, MOVE_TIME)
            lines.append("%s : %s / %.4f" % ("white" if i % 2 == 0 else "black", move, prob))
            moves.append(move)

        for l in lines:
            print l

    except Exception as exc:
        print exc
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
