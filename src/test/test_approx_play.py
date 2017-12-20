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

from ggplearn.training.approximate_play import Runner
from ggplearn import msgdefs


current_gen = "testgen_normal_1"


def go_test():
    conf = msgdefs.ConfigureApproxTrainer()
    conf.player_select_conf = msgdefs.PolicyPlayerConf(verbose=False,
                                                       generation=current_gen,
                                                       choose_exponential_scale=0.3)

    conf.player_policy_conf = msgdefs.PUCTPlayerConf(name="policy_puct",
                                                     verbose=False,
                                                     generation=current_gen,
                                                     playouts_per_iteration=800,
                                                     playouts_per_iteration_noop=0,
                                                     expand_root=100,
                                                     dirichlet_noise_alpha=-1,
                                                     cpuct_constant_first_4=3.0,
                                                     cpuct_constant_after_4=0.75,
                                                     choose="choose_converge")

    conf.player_score_conf = msgdefs.PolicyPlayerConf(verbose=False,
                                                      generation=current_gen,
                                                      choose_exponential_scale=-1)

    runner = Runner(conf)
    number_of_samples = 10

    # slow first run
    runner.generate_sample()
    runner.reset_debug()

    total_time = 0

    for _ in range(number_of_samples):
        start = time.time()
        print runner.generate_sample()
        total_time += (time.time() - start)

    print "av time for play_one_game: %.2f" % (runner.acc_time_for_play_one_game / number_of_samples)
    print "av time for do_policy: %.2f" % (runner.acc_time_for_do_policy / number_of_samples)
    print "av time for do_score: %.2f" % (runner.acc_time_for_do_score / number_of_samples)

    print "average time to generate sample: %.2f" % (total_time / number_of_samples)


if __name__ == "__main__":
    from ggplearn.util.main import main_wrap
    main_wrap(go_test)
