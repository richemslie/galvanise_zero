import time

from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game

from ggplearn import msgdefs

from ggplearn.player.puctplayer import PUCTPlayer
from ggplearn.player.policyplayer import PolicyPlayer

current_gen = "v5_gen_small_71"


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggplearn.util.keras import init
    init(data_format='channels_last')


def test_speed_of_one_shot():
    ITERATIONS = 5

    gm = GameMaster(get_gdl_for_game("breakthrough"))

    conf = msgdefs.PolicyPlayerConf(generation=current_gen, verbose=False)

    white = PolicyPlayer(conf)
    black = PolicyPlayer(conf)

    gm.add_player(white, "white")
    gm.add_player(black, "black")

    gm.reset()
    gm.start(meta_time=30, move_time=15)
    gm.play_to_end()

    acc_time = 0
    for _ in range(ITERATIONS):
        gm.reset()
        gm.start(meta_time=30, move_time=15)
        s = time.time()
        gm.play_to_end()
        acc_time += time.time() - s
        print gm.get_game_depth()

    print "average time taken", acc_time / ITERATIONS


def test_speed_of_one_simulation():
    ITERATIONS = 5

    gm = GameMaster(get_gdl_for_game("breakthrough"))

    conf_puct = msgdefs.PUCTPlayerConf(verbose=False,
                                       generation=current_gen,
                                       playouts_per_iteration=800,
                                       playouts_per_iteration_noop=0,
                                       dirichlet_noise_alpha=-1,
                                       expand_root=-1)
    conf_policy = msgdefs.PolicyPlayerConf(generation=current_gen, verbose=False)

    # add two players
    white = PUCTPlayer(conf=conf_puct)
    black = PolicyPlayer(conf=conf_policy)
    black.verbose = False

    gm.add_player(white, "white")
    gm.add_player(black, "black")

    acc_time = 0
    for _ in range(ITERATIONS):
        gm.reset()
        gm.start(meta_time=30, move_time=15)

        last_move = gm.play_single_move(last_move=None)
        last_move = gm.play_single_move(last_move=last_move)

        s = time.time()
        gm.play_single_move(last_move=last_move)
        acc_time += time.time() - s
        assert gm.get_game_depth() == 2

    print "average time taken", acc_time / ITERATIONS
