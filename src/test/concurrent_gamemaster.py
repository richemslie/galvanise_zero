
import time

from ggplib.db.helper import get_gdl_for_game
from ggplib.player.gamemaster import GameMaster

from ggplearn import msgdefs
from ggplearn.nn.scheduler import create_scheduler

from ggplearn.training.approximate_play import Runner

from ggplearn.player.puctplayer import PUCTPlayer
# from ggplearn.player.policyplayer import PolicyPlayer


def setup():
    from ggplib.util.init import setup_once
    setup_once()


default_generation = "v5_gen_normal_9"

# create some gamemaster/players
policy_config = msgdefs.PolicyPlayerConf(verbose=False,
                                         generation="noone",
                                         choose_exponential_scale=0.25)

puct_config = msgdefs.PUCTPlayerConf(name="rev2-test",
                                     generation="noone",
                                     verbose=False,
                                     playouts_per_iteration=800,
                                     playouts_per_iteration_noop=1,
                                     expand_root=100,
                                     dirichlet_noise_alpha=0.3,
                                     cpuct_constant_first_4=0.75,
                                     cpuct_constant_after_4=0.75,
                                     choose="choose_top_visits")

policy_config2 = msgdefs.PolicyPlayerConf(verbose=False,
                                          generation="noone",
                                          choose_exponential_scale=-1)


def test_concurrrent_gamemaster():
    scheduler = create_scheduler("reversi", batch_size=256)

    gamemasters = []

    # create a bunch of gamemaster and players [slow first step]
    for i in range(128):
        gm = GameMaster(get_gdl_for_game("reversi"), fast_reset=True)
        gamemasters.append(gm)

        for role in ("red", "black"):
            player = PUCTPlayer(conf=puct_config)

            # this is a bit underhand, since we know PUCTPlayer won't reget the nn
            player.nn = scheduler

            gm.add_player(player, role)

        # anything as long it is at right resolution (since we are trying to parellise self play -
        # this is the resolution we want.
        gm.start(meta_time=300, move_time=300)
        scheduler.add_runnable(gm.play_single_move, None)

    s = time.time()
    scheduler.run()
    print "time taken in python", scheduler.acc_python_time
    print "time taken in predicting", scheduler.acc_predict_time
    print "total time taken", time.time() - s


def f(runner):
    sample = runner.generate_sample()
    runner.sample = sample


def test_approx_play():
    scheduler = create_scheduler("reversi", batch_size=256)

    conf = msgdefs.ConfigureApproxTrainer("reversi")
    conf.player_select_conf = policy_config
    conf.player_policy_conf = puct_config
    conf.player_score_conf = policy_config2

    runners = []
    for i in range(256):
        r = Runner(conf)
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
