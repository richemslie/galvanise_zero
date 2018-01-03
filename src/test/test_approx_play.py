import time

from ggpzero.training import approximate_play
from ggpzero.defs import confs, msgs


ITERATIONS = 2
current_gen = "v5_0"


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()


def test_approx():
    conf = msgs.ConfigureApproxTrainer("breakthrough")
    conf.generation = current_gen
    conf.player_select_conf = confs.PolicyPlayerConfig(verbose=False,
                                                       generation=current_gen)

    conf.player_policy_conf = confs.PUCTPlayerConfig(name="puct_abc",
                                                     verbose=False,
                                                     generation=current_gen,
                                                     playouts_per_iteration=42,
                                                     playouts_per_iteration_noop=0,
                                                     choose="choose_top_visits")

    conf.player_score_conf = conf.player_policy_conf

    session = approximate_play.Session()
    runner = approximate_play.Runner(conf)

    # slow first run
    runner.generate_sample(session)
    runner.reset_debug()

    total_time = 0
    for _ in range(ITERATIONS):
        start = time.time()
        print runner.generate_sample(session)
        total_time += (time.time() - start)

    print "average time to generate sample: %.2f" % (total_time / float(ITERATIONS))

