import os
import re

import tensorflow as tf

from ggplib.db import lookup
from ggplib.db.helper import get_gdl_for_game
from ggplib.player.gamemaster import GameMaster

from ggpzero.defs import confs
from ggpzero.player.puctplayer import PUCTPlayer


def lg_to_gdl(m):
    r = "abcdefghij"[::-1]
    return r.index(m[0]) + 1, int(m[1:])


def gdl_to_lg(x, y):
    r = "abcdefghij"[::-1]
    return "%s%s" % (r[x + 1], y)


# print gdl_to_lg(1, 4)
# print gdl_to_lg(4, 7)
# print gdl_to_lg(7, 7)

# print lg_to_gdl("j4")
# print lg_to_gdl("g7")
# print lg_to_gdl("d7")


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)


def config(gen):
    return confs.PUCTPlayerConfig(name=gen,
                                  generation=gen,
                                  verbose=True,

                                  playouts_per_iteration=-1,
                                  playouts_per_iteration_noop=0,

                                  dirichlet_noise_alpha=0.03,

                                  root_expansions_preset_visits=-1,
                                  puct_before_expansions=3,
                                  puct_before_root_expansions=5,
                                  puct_constant_before=3.0,
                                  puct_constant_after=0.75,

                                  choose="choose_temperature",
                                  temperature=1.5,
                                  depth_temperature_max=3.0,
                                  depth_temperature_start=0,
                                  depth_temperature_increment=0.5,
                                  depth_temperature_stop=6,
                                  random_scale=1.00,

                                  fpu_prior_discount=0.3,

                                  max_dump_depth=2)


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
        # white/black is wrong on LG
        if k == "PB":
            pw = v
        if k == "PW":
            pb = v
        if k in "WB":
            moves.append((k, v))

    return pw, pb, moves


def play_moves(moves):
    sm = lookup.by_name("amazons_10x10").get_sm()
    sm.reset()

    # get some objects
    joint_move = sm.get_joint_move()
    base_state = sm.get_initial_state()

    def f(ri, ls, i):
        return sm.legal_to_move(ri, ls.get_legal(i))

    def next_sm(role_index, gdl_str_move):
        for ri in range(2):
            ls = sm.get_legal_state(ri)
            if ls.get_count() == 1 and f(ri, ls, 0) == "noop":
                choice = 0
            else:
                the_moves = [f(ri, ls, ii) for ii in range(ls.get_count())]
                choice = the_moves.index(gdl_str_move)

            joint_move.set(ri, ls.get_legal(choice))

        # update state machine
        sm.next_state(joint_move, base_state)
        sm.update_bases(base_state)

    role_index = 0
    for who, move_str in moves:
        if role_index == 0:
            assert who == "W"
        else:
            assert who == "B"

        # actually 2 moves
        queen, fire = move_str.split("/")
        from_pos, to_pos = queen.split("-")

        gdl_move = "(move %s %s %s %s)" % (lg_to_gdl(from_pos)[0],
                                           lg_to_gdl(from_pos)[1],
                                           lg_to_gdl(to_pos)[0],
                                           lg_to_gdl(to_pos)[1])
        next_sm(role_index, gdl_move)
        print "%s -> %s" % (move_str, gdl_move)

        gdl_move = "(fire %s %s)" % (lg_to_gdl(fire)[0],
                                     lg_to_gdl(fire)[1])
        next_sm(role_index, gdl_move)
        print "%s -> %s" % (move_str, gdl_move)

        role_index = 1 if role_index == 0 else 0

    return base_state


def play_game(config_0, config_1, moves=[], move_time=10.0):
    # add players
    gm = GameMaster(get_gdl_for_game("amazons_10x10"))
    gm.add_player(PUCTPlayer(config_0), "white")
    gm.add_player(PUCTPlayer(config_1), "black")

    # play move via gamemaster:
    gm.reset()

    if moves:
        state = play_moves(moves)
        gm.start(meta_time=15,
                 initial_basestate=state,
                 game_depth=len(moves) * 2,
                 move_time=move_time)
    else:
        gm.start(meta_time=15,
                 move_time=move_time)

    def remove_gdl(m):
        return m.replace("(place ", "").replace(")", "").strip().replace(' ', '')

    move = None
    while not gm.finished():
        move = gm.play_single_move(last_move=move)

        ri = 1 if move[0] == "noop" else 0
        str_move = remove_gdl(move[ri])

def test():
    txt = "(;EV[null]PB[X]PW[Z]SO[K];W[j4-g7/d7];B[g10-h9/g8];W[g7-h8/g9];B[a7-b8/c8];W[d1-i6/j6];B[j7-i7/h7];W[h8-g7/h6])"
    pw, pb, moves = parse_sgf(txt)
    print "players (white) '%s' vs '%s' black " % (pw, pb)

    config_0 = config("h1_27")
    config_1 = config("h1_27")

    move_time = 30.0
    number_of_games = 1

    play_game(config_0, config_1, moves=moves, move_time=move_time)


if __name__ == "__main__":
    setup()
    test()
