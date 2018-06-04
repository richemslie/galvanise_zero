import os
import re
import pdb
import sys
import time
import hashlib
import traceback

import tensorflow as tf

from ggplib.db import lookup
from ggplib.db.helper import get_gdl_for_game

from ggplib.player.gamemaster import GameMaster

from ggpzero.defs import confs
from ggpzero.player.puctplayer import PUCTPlayer


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)


def config(gen, **kwds):
    conf = confs.PUCTPlayerConfig(name=gen,
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


def swapaxis(s):
    mapping_x = {x1 : x0 for x0, x1 in zip('abcdefghi', '123456789')}
    mapping_y = {x0 : x1 for x0, x1 in zip('abcdefghi', '123456789')}
    for x0, x1 in zip(('j', 'k', 'l', 'm'), ('10', '11', '12', '13')):
        mapping_x[x1] = x0
        mapping_y[x0] = x1

    return "%s%s" % (mapping_x[s[1]], mapping_y[s[0]])


def play_moves(moves, game_size):
    sm = lookup.by_name("hexLG%s" % game_size).get_sm()
    sm.reset()

    # get some objects
    joint_move = sm.get_joint_move()
    base_state = sm.get_initial_state()

    gdl_moves = []

    def f(ri, i):
        return sm.legal_to_move(ri, ls.get_legal(i))

    role_index = 0
    for m in moves:
        gdl_move = []
        for ri in range(2):
            ls = sm.get_legal_state(ri)
            if ls.get_count() == 1 and f(ri, 0) == "noop":
                gdl_role_move = "noop"

            else:
                if m == "swap":
                    gdl_role_move = m
                else:
                    low, high = m[0].lower(), int(m[1:])
                    assert 'a' <= low <= 'z'
                    gdl_role_move = "(place %s %s)" % (low, high)

            the_moves = [f(ri, ii) for ii in range(ls.get_count())]
            choice = the_moves.index(gdl_role_move)
            joint_move.set(ri, ls.get_legal(choice))
            gdl_move.append(str(gdl_role_move))

        gdl_moves.append(str(gdl_move))
        print "%s -> %s" % (m, gdl_move)

        # update state machine
        sm.next_state(joint_move, base_state)
        sm.update_bases(base_state)
        print sm.basestate_to_str(base_state)

        role_index = 1 if role_index == 0 else 0

    return role_index, base_state


def simplemcts_player(move_time):
    from ggplib.player import get
    player = get.get_player("simplemcts")
    player.max_tree_search_time = move_time
    return player


def play_game(player_black, player_white, moves=[], game_size=11, move_time=10.0):
    # create gm and add players
    gm = GameMaster(get_gdl_for_game("hexLG%s" % game_size), verbose=True)
    gm.add_player(player_black, "black")
    gm.add_player(player_white, "white")

    # play move via gamemaster:
    gm.reset()

    if moves:
        _, state = play_moves(moves, game_size)
        gm.start(meta_time=15,
                 initial_basestate=state,
                 game_depth=len(moves),
                 move_time=move_time)
    else:
        gm.start(meta_time=15,
                 move_time=move_time)

    def remove_gdl(m):
        return m.replace("(place ", "").replace(")", "").strip().replace(' ', '')

    sgf_moves = []

    def add_sgf_move(ri, m):
        sgf_moves.append((ri, m))
        if m == "swap":
            assert ri == 1
            assert len(sgf_moves) == 1

            # hexgui does do swap like LG.   THis is a (double) hack.
            moved_move = swapaxis(sgf_moves[0][1])
            sgf_moves[0] = (0, moved_move)
            sgf_moves.append((1, moved_move))

    for i, move in enumerate(moves):
        add_sgf_move(i % 2, move)

    move = None
    while not gm.finished():
        move = gm.play_single_move(last_move=move)
        ri = 1 if move[0] == "noop" else 0
        str_move = remove_gdl(move[ri])
        add_sgf_move(ri, str_move)
        write_hexgui_sgf(player_black.name, player_white.name, sgf_moves, gm.match_id)

    write_hexgui_sgf(player_black.name, player_white.name, sgf_moves, gm.match_id)
    return sgf_moves


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


def hex_get_state(game_size, sgf):
    black_player, white_player, sgf_moves = parse_sgf(sgf)

    expect = 'W'
    moves = []
    for who, move in sgf_moves:
        assert expect == who
        expect = 'B' if who == 'W' else 'W'
        move = move[0] + str("abcdefghijklmnop".index(move[1]) + 1)
        moves.append(move)

    return play_moves(moves, game_size)


def write_hexgui_sgf(black_name, white_name, sgf_moves, game_id=None):
    if game_id is None:
        game_id = hashlib.md5(hashlib.sha1("%.5f" % time.time()).hexdigest()).hexdigest()[:6]

    with open("game_%s_%s_%s.sgf" % (black_name, white_name, game_id), "w") as f:
        f.write("(;FF[4]EV[null]PB[%s]PW[%s]SZ[%s]GC[game#%s];" % (black_name, white_name, game_size, game_id))

        # piece colours are swapped for hexgui from LG
        for ri, m in sgf_moves:
            f.write("%s[%s];" % ("B" if ri == 0 else "W", m))
        f.write(")\n")


if __name__ == "__main__":
    game_size = 13
    move_time = 30.0
    number_of_games = 8

    match = ""
    try:
        setup()

        dirichlet_noise_alpha = 0.03
        gen_black = sys.argv[1]
        gen_white = sys.argv[2]

        if gen_black == "-":
            player_black = simplemcts_player(move_time)
        else:
            player_black = PUCTPlayer(config(gen_black, dirichlet_noise_alpha=dirichlet_noise_alpha))

        if gen_white == "-":
            player_white = simplemcts_player(move_time)
        else:
            player_white = PUCTPlayer(config(gen_white, dirichlet_noise_alpha=dirichlet_noise_alpha))

        print player_white, player_black

        moves = []
        if match:
            black_player, white_player, sgf_moves = parse_sgf(match)
            expect = 'W'
            moves = []
            for who, move in sgf_moves:
                assert expect == who
                expect = 'B' if who == 'W' else 'W'
                if move != "swap":
                    move = move[0] + str("abcdefghijklmnop".index(move[1]) + 1)
                print move
                moves.append(move)

        for i in range(number_of_games):
            sgf_moves = play_game(player_black, player_white, moves=moves[:],
                                  game_size=game_size, move_time=move_time)

            # swap roles for next game
            player_black, player_white = player_white, player_black

    except Exception as exc:
        print exc
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
