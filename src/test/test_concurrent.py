import time

from ggplib.db.helper import get_gdl_for_game
from ggplib.player.gamemaster import GameMaster

from ggpzero.defs import confs
from ggpzero.nn.scheduler import create_scheduler

from ggpzero.player.puctplayer import PUCTPlayer


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()


default_generation = "v5_90"

puct_config = confs.PUCTPlayerConfig(name="puct_abc",
                                     generation="noone",
                                     verbose=False,
                                     playouts_per_iteration=42,
                                     playouts_per_iteration_noop=0,
                                     choose="choose_top_visits")


def test_concurrrent_gamemaster():
    scheduler = create_scheduler("breakthrough", default_generation, batch_size=256)

    gamemasters = []

    # create a bunch of gamemaster and players [slow first step]
    CONCURRENT_GAMES = 128
    for _ in range(CONCURRENT_GAMES):
        gm = GameMaster(get_gdl_for_game("breakthrough"), fast_reset=True)
        gamemasters.append(gm)

        for role in ("white", "black"):
            player = PUCTPlayer(conf=puct_config)

            # this is a bit underhand, since we know PUCTPlayer won't reget the nn
            player.puct_evaluator.nn = scheduler

            gm.add_player(player, role)

        # anything as long it is at right resolution (since we are trying to parellise self play -
        # this is the resolution we want.
        gm.start(meta_time=300, move_time=300)
        scheduler.add_runnable(gm.play_single_move, None)

    s = time.time()
    scheduler.run()
    print "time taken in python", scheduler.acc_python_time
    print "time taken in predicting", scheduler.acc_predict_time
    print "total time taken for one move of %s concurrent games: %s" % (CONCURRENT_GAMES, time.time() - s)
