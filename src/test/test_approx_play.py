'''
tiny/smaller


average time to generate sample: 9.41
av time for do_policy: 2.24
av time for play one game: 7.17

smaller/small

average time to generate sample: 12.36
av time for do_policy: 3.54
av time for play one game: 8.82

'''

import time

import attr

from ggplearn.training.approximate_play import RunnerConf, Runner
from ggplearn.player import mc


def go_test():
    conf = RunnerConf()
    conf.policy_generation = "gen9_small"
    conf.score_generation = "gen9_smaller"

    conf.score_puct_player_conf = mc.PUCTPlayerConf(name="score_puct",
                                                    verbose=False,
                                                    num_of_playouts_per_iteration=32,
                                                    num_of_playouts_per_iteration_noop=0,
                                                    expand_root=5,
                                                    dirichlet_noise_alpha=0.1,
                                                    cpuct_constant_first_4=0.75,
                                                    cpuct_constant_after_4=0.75,
                                                    choose="choose_converge")

    conf.policy_puct_player_conf = mc.PUCTPlayerConf(name="policy_puct",
                                                     verbose=False,
                                                     num_of_playouts_per_iteration=800,
                                                     num_of_playouts_per_iteration_noop=0,
                                                     expand_root=5,
                                                     dirichlet_noise_alpha=-1,
                                                     cpuct_constant_first_4=3.0,
                                                     cpuct_constant_after_4=0.75,
                                                     choose="choose_converge")


    runner = Runner(conf)
    number_of_samples = 5

    # slow first run
    runner.generate_sample()
    runner.reset_debug()

    total_time = 0

    for i in range(5):
        start = time.time()
        print runner.generate_sample()
        total_time += (time.time() - start)

    print "average time to generate sample: %.2f" % (total_time / number_of_samples)
    print "av time for do_policy: %.2f" % (runner.acc_time_for_do_policy / number_of_samples)
    print "av time for play one game: %.2f" % (runner.acc_time_for_play_one_game / number_of_samples)


if __name__ == "__main__":
    from ggplib.util.init import setup_once
    setup_once("approximate_play")

    from ggplearn.util.keras import constrain_resources
    constrain_resources()

    # import cProfile
    # cProfile.run('go()')
    go_test()
