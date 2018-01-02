import time

from ggplib.db.helper import get_gdl_for_game
from ggplib.player.gamemaster import GameMaster

from ggpzero.defs import confs, msgs
from ggpzero.nn.scheduler import create_scheduler

from ggpzero.training import approximate_play

from ggpzero.player.puctplayer import PUCTPlayer


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init(data_format='channels_last')


default_generation = "v5_76"

# create some gamemaster/players
policy_config = confs.PolicyPlayerConfig(verbose=False,
                                         generation="noone")

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


def test_approx_play():
    session = approximate_play.Session()

    def f(runner):
        sample = runner.generate_sample(session)
        runner.sample = sample

    scheduler = create_scheduler("breakthrough", default_generation, batch_size=512)

    conf = msgs.ConfigureApproxTrainer("breakthrough")
    conf.player_select_conf = policy_config
    conf.player_policy_conf = puct_config
    conf.player_score_conf = puct_config

    CONCURRENT_GAMES = 512
    runners = []
    for _ in range(CONCURRENT_GAMES):
        r = approximate_play.Runner(conf)
        r.patch_players(scheduler)
        runners.append(r)
        scheduler.add_runnable(f, r)

    s = time.time()
    scheduler.run()

    # collect samples
    samples = [runner.sample for runner in runners]
    print "samples", len(samples)
    print "time taken in python", scheduler.acc_python_time
    print "time taken in predicting", scheduler.acc_predict_time
    print "total time taken", time.time() - s

    # go again... faster 2nd time
    scheduler.acc_python_time = 0
    scheduler.acc_predict_time = 0
    for r in runners:
        scheduler.add_runnable(f, r)

    s = time.time()
    scheduler.run()

    # collect samples
    samples = [r.sample for r in runners]

    print len(samples)
    print "time taken in python", scheduler.acc_python_time
    print "time taken in predicting", scheduler.acc_predict_time
    print "total time taken", time.time() - s
