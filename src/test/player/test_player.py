import time

from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game
from ggplearn.player.simple import NNPlayerOneShot
from ggplearn.player.mc import PUCTPlayer, PUCTPlayerConf
import py.test

def setup():
    from ggplearn.util.keras import constrain_resources
    constrain_resources()


def test_reversi_tournament():
    py.test.skip("no reversi right now")
    gm = GameMaster(get_gdl_for_game("reversi"))

    # add two players
    pymcs = get.get_player("pymcs")
    pymcs.max_run_time = 0.25

    random = get.get_player("pyrandom")
    nn0 = NNPlayerOneShot("no-scores1")

    gm.add_player(nn0, "black")
    gm.add_player(random, "red")

    acc_black_score = 0
    acc_red_score = 0
    for i in range(10):
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["red"]

    print "black_score", gm.players_map["black"].name, acc_black_score
    print "red_score", gm.players_map["red"].name, acc_red_score


def test_bt_tournament():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    # add two players
    pymcs = get.get_player("pymcs")
    pymcs.max_run_time = 0.25

    # random = get.get_player("pyrandom")
    nn0 = NNPlayerOneShot("gen9")

    gm.add_player(pymcs, "white")
    gm.add_player(nn0, "black")

    acc_black_score = 0
    acc_red_score = 0
    for i in range(5):
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["white"]

    print "white_score", gm.players_map["white"].name, acc_red_score
    print "black_score", gm.players_map["black"].name, acc_black_score


def test_montecarlo1():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    # add two players
    white = PUCTPlayer("gen9_small")
    white.conf.NUM_OF_PLAYOUTS_PER_ITERATION = 8
    white.conf.DIRICHLET_NOISE_ALPHA = -1
    white.conf.EXPAND_ROOT = -1

    black = NNPlayerOneShot("gen9_small")

    gm.add_player(black, "white")
    gm.add_player(white, "black")

    acc_black_score = 0
    acc_red_score = 0
    for i in range(5):
        gm.reset()
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["white"]

    print "white_score", gm.players_map["white"].name, acc_red_score
    print "black_score", gm.players_map["black"].name, acc_black_score


def test_montecarlo2():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    # add two players
    white = PUCTPlayer("gen9_small")
    white.conf.NUM_OF_PLAYOUTS_PER_ITERATION = 50

    black = get.get_player("simplemcts")
    black.max_tree_search_time = 0.5

    gm.add_player(white, "white")
    gm.add_player(black, "black")

    acc_black_score = 0
    acc_red_score = 0
    for i in range(5):
        gm.reset()
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["white"]

    print "white_score", gm.players_map["white"].name, acc_red_score
    print "black_score", gm.players_map["black"].name, acc_black_score


def test_fast_plays():
    ''' very fast rollouts '''
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    conf = PUCTPlayerConf()
    conf.verbose = false
    conf.num_of_playouts_per_iteration = 1
    conf.num_of_playouts_per_iteration_noop = 0
    conf.dirichlet_noise_alpha = 1.0
    conf.expand_root = -1

    # add two players
    nn0 = PUCTPlayer(generation="test", conf=conf)
    nn1 = PUCTPlayer(generation="test", conf=conf)

    gm.add_player(nn0, "white")
    gm.add_player(nn1, "black")

    acc_black_score = 0
    acc_red_score = 0
    s = time.time()
    for i in range(10):
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["white"]

    print "time taken", time.time() - s
    print "white_score", gm.players_map["white"].name, acc_red_score
    print "black_score", gm.players_map["black"].name, acc_black_score


def test_not_taking_win():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    conf = PUCTPlayerConf()
    conf.verbose = True
    conf.num_of_playouts_per_iteration = 42
    conf.num_of_playouts_per_iteration_noop = 1
    conf.dirichlet_noise_alpha = 0.1
    conf.expand_root = 100

    # add two c++ players
    nn0 = PUCTPlayer(generation="v2_gen_smaller_47", conf=conf)
    nn1 = PUCTPlayer(generation="v2_gen_smaller_47", conf=conf)
    gm.add_player(nn0, "white")
    gm.add_player(nn1, "black")

    str_state = """(true (control black)) (true (cellHolds 8 8 black)) (true (cellHolds 8 7 black)) (true (cellHolds 8 2 white)) (true (cellHolds 8 1 white)) (true (cellHolds 7 8 black)) (true (cellHolds 6 7 white)) (true (cellHolds 6 2 white)) (true (cellHolds 6 1 white)) (true (cellHolds 5 4 black)) (true (cellHolds 5 3 black)) (true (cellHolds 5 2 white)) (true (cellHolds 5 1 white)) (true (cellHolds 4 8 black)) (true (cellHolds 4 7 black)) (true (cellHolds 4 1 white)) (true (cellHolds 3 8 black)) (true (cellHolds 3 6 black)) (true (cellHolds 3 2 white)) (true (cellHolds 2 8 black)) (true (cellHolds 2 7 black)) (true (cellHolds 2 2 white)) (true (cellHolds 2 1 white)) (true (cellHolds 1 8 black)) (true (cellHolds 1 7 black)) (true (cellHolds 1 2 white)) (true (cellHolds 1 1 white)) """


    gm.start(meta_time=30, move_time=2,
             initial_basestate=gm.convert_to_base_state(str_state))

    last_move = None
    depth = 1
    basestate = gm.sm.new_base_state()
    while not gm.finished():
        last_move = gm.play_single_move(last_move=last_move)

        gm.sm.get_current_state(basestate)
        from ggplearn.util.bt import pretty_print_board

        print
        print "gm state:"
        pretty_print_board(gm.sm, basestate.to_list())

        depth += 1

