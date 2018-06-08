import sys

from ggplib.db import lookup
from ggplib.db.helper import get_gdl_for_game
from ggplib.player.gamemaster import GameMaster

from ggpzero.player.puctplayer import PUCTPlayer

from ggpzero.scripts.battle.common import simplemcts_player, parse_sgf, setup, get_puct_config


def play_moves(moves):
    sm = lookup.by_name("amazons_10x10").get_sm()
    sm.reset()

    # get some objects
    joint_move = sm.get_joint_move()
    base_state = sm.get_initial_state()

    # print sm.basestate_to_str(base_state)

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

    def lg_to_ggp(m):
        return ("jihgfedcba".index(m[0]) + 1), int(m[1:])

    role_index = 0
    for who, move_str in moves:
        if role_index == 0:
            assert who == "W"
        else:
            assert who == "B"

        # actually 2 moves
        queen, fire = move_str.split("/")
        from_pos, to_pos = queen.split("-")
        from_pos, to_pos, fire = map(lg_to_ggp, (from_pos, to_pos, fire))

        gdl_move = "(move %s %s %s %s)" % (from_pos + to_pos)
        next_sm(role_index, gdl_move)

        gdl_move = "(fire %s %s)" % fire
        next_sm(role_index, gdl_move)

        role_index = 1 if role_index == 0 else 0

    return role_index, base_state


def amazons_get_state(sgf):
    _, _, sgf_moves = parse_sgf(sgf)
    return play_moves(sgf_moves)


def play_game(player_black, player_white, sgf_moves, move_time):

    # add players
    gm = GameMaster(get_gdl_for_game("amazons_10x10"))
    gm.add_player(player_white, "white")
    gm.add_player(player_black, "black")

    # play move via gamemaster:
    gm.reset()

    if sgf_moves:
        _, state = play_moves(sgf_moves)
        gm.start(meta_time=15,
                 initial_basestate=state,
                 game_depth=len(sgf_moves) * 2,
                 move_time=move_time)
    else:
        gm.start(meta_time=15,
                 move_time=move_time)

    gm.play_to_end()


def main():
    # command line options, gen v gen

    gen_black = sys.argv[1]
    gen_white = sys.argv[2]

    # modify these
    move_time = 2.0
    number_of_games = 1
    match = "(;EV[null]PB[xxx]PW[xxx]SO[http://www.littlegolem.com];W[j4-g7/d7];B[g10-h9/g8];W[g7-h8/g9];B[a7-b8/c8];W[d1-i6/j6];B[j7-i7/h7];W[h8-g7/h6];B[b8-b7/c7];W[g7-f8/f10];B[h9-h8/i8];W[f8-e9/g7])"

    dirichlet_noise_alpha = 0.03

    if gen_black == "-":
        player_black = simplemcts_player(move_time)
    else:
        player_black = PUCTPlayer(get_puct_config(gen_black,
                                                  dirichlet_noise_alpha=dirichlet_noise_alpha))

    if gen_white == "-":
        player_white = simplemcts_player(move_time)
    else:
        player_white = PUCTPlayer(get_puct_config(gen_white,
                                                  dirichlet_noise_alpha=dirichlet_noise_alpha))

    print player_white, player_black

    sgf_moves = []
    if match:
        _, _, sgf_moves = parse_sgf(match)

    for i in range(number_of_games):
        play_game(player_black, player_white, sgf_moves[:], move_time)

        # swap roles for next game
        player_black, player_white = player_white, player_black


if __name__ == "__main__":
    import pdb
    import traceback

    try:
        setup()
        main()

    except Exception as exc:
        print exc
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
