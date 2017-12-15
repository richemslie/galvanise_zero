import sys
from os.path import join as joinpath

import numpy as np
from ggplib.util import log
from ggplib import interface

from ggplearn.training.crunch import get_from_json, Sample, SampleToNetwork

DEBUG = False


def shuffle_list(a_list):
    # shuffle data
    shuffle = np.arange(len(a_list))
    np.random.shuffle(shuffle)

    return [a_list[i] for i in shuffle]


def gather_new_data(path, game, generation, all_states):
    print "------"
    print "%s : Gathering for %s, gen: %s" % (path, game, generation)

    new_data = []

    path = joinpath(path, generation)

    dupes = 0

    bad_files = []
    for game_data, filename in get_from_json(path, includes=[game, "samples"], excludes=["initial"]):
        assert game_data["game"] == game

        with_generation = None
        if "postfix" in game_data:
            with_generation = game_data["postfix"]

        if "with_generation" in game_data:
            with_generation = game_data["with_generation"]

        if with_generation is not None and with_generation != generation:
            print "Wrong generation", path, filename, with_generation
            bad_files.append(filename)
            continue

        num_samples = game_data["num_samples"]
        assert num_samples == len(game_data["samples"])

        for s_desc in game_data["samples"]:
            state = tuple(s_desc["state"])
            policy_dist = [(d["legal"], d["new_prob"]) for d in s_desc["policy_dists"]]
            final_scores = [s_desc["final_scores"][r] / 100.0 for r in "white black".split()]

            if DEBUG and len(new_data) < 3:
                print s_desc
                print state
                print policy_dist
                print final_scores

            if state in all_states:
                dupes += 1
                continue

            new_data.append(Sample(state, policy_dist, final_scores))
            all_states.add(state)

    if bad_files:
        assert False, "basically start debugger"

    print "DUPES", dupes
    return new_data



class ParseConfig(object):
    GAME = "breakthrough"
    GENERATIONS_SO_FAR = 9

    NETWORK_SIZE = "smaller"
    A0_L2_REG = False

    # NOTE: so far it seems to best with all the data
    MAX_SAMPLE_COUNT = 200000
    VALIDATION_SPLIT = 0.8

    BATCH_SIZE = 512
    EPOCHS = 42


def main(path):
    conf = ParseConfig()
    assert conf.NETWORK_SIZE in "tiny smaller small normal".split()

    # think i've been trying to get to this variable name: :)
    next_generation = "gen%d" % conf.GENERATIONS_SO_FAR

    if conf.NETWORK_SIZE != "normal":
        next_generation += "_" + conf.NETWORK_SIZE
    else:
        next_generation

    log_name_base = "parsex__%s_" % conf.GAME
    interface.initialise_k273(1, log_name_base=log_name_base)
    log.initialise()

    stn = SampleToNetwork(conf.GAME, next_generation)
    if conf.A0_L2_REG:
        net = stn.create_network(network_size=conf.NETWORK_SIZE,
                                 a0_reg=True, dropout=False)
    else:
        net = stn.create_network(network_size=conf.NETWORK_SIZE,)

    net.summary()

    # start gathering data:

    training_data = []
    validation_data = []

    # deprecated XXXX
    # for game_data, _ in get_from_json(path, includes=[GAME, "initial"], excludes=["samples"]):
    #     assert game_data["game"] == GAME

    #     num_samples = game_data["num_samples"]
    #     samples = game_data["samples"]

    #     assert num_samples == len(samples)
    #     assert game_data["num_training_samples"] + game_data["num_validation_samples"] == num_samples

    #     for i in range(game_data["num_training_samples"]):
    #         s = samples[i]

    #         state = tuple(s["state"])
    #         policy_dist = s["policy_dist"]

    #         # XXX I got these reversed... will fix for next time around
    #         final_scores = s["best_scores"]

    #         training_data.append(Sample(state, policy_dist, final_scores))

    #     for i in range(game_data["num_validation_samples"]):
    #         s = samples[i]
    #         state = tuple(s["state"])
    #         policy_dist = s["policy_dist"]

    #         # XXX I got these reversed... will fix for next time around
    #         final_scores = s["best_scores"]

    #         validation_data.append(Sample(state, policy_dist, final_scores))

    # print "#training_data", len(training_data)
    # print "#validation_data", len(validation_data)
    # print

    training_data = []
    validation_data = []
    validation_set = {}
    all_generations = ["gen%d" % i for i in range(conf.GENERATIONS_SO_FAR)]
    for gen in all_generations:
        generation_data = gather_new_data(path, conf.GAME, gen, set())
        print "generation '%s' data : %d" % (gen, len(generation_data))

        validation_count = int(len(generation_data) * (1 - conf.VALIDATION_SPLIT))

        count_v, count_t = 0, 0
        for s in generation_data:
            if s.state in validation_set:
                count_t += 1
                training_data.append(s)
                continue

            if count_v == validation_count:
                count_t += 1
                training_data.append(s)
                continue

            validation_set[s.state] = s
            count_v += 1

        print "#training_data", count_t
        print "#validation_data", count_v
        print

    validation_data += validation_set.values()

    if (conf.MAX_SAMPLE_COUNT and
        len(training_data) + len(validation_data) > conf.MAX_SAMPLE_COUNT):
        print "#training_data", len(training_data)
        print "#validation_data", len(validation_data)
        print

        max_number_training_data = int(conf.MAX_SAMPLE_COUNT * conf.VALIDATION_SPLIT)
        max_number_validation_data = conf.MAX_SAMPLE_COUNT - max_number_training_data

        start_train = len(training_data) - max_number_training_data
        start_validation = len(validation_data) - max_number_validation_data
        pct = start_train / float(len(training_data))
        print 'dropping %d%% of initial data' % (pct * 100)
        training_data = training_data[start_train:]
        validation_data = validation_data[start_validation:]

    print "# final training_data", len(training_data)
    print "# final validation_data", len(validation_data)
    print

    shuffle_list(training_data)
    shuffle_list(validation_data)

    train_data = stn.massage_data(training_data, validation_data)

    # yippee
    net.train(train_data, batch_size=conf.BATCH_SIZE, epochs=conf.EPOCHS)
    net.save()


def main_wrap():
    import pdb
    import traceback

    try:
        path = sys.argv[1]
        main(path)

    except Exception as exc:
        print exc
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


###############################################################################

if __name__ == "__main__":
    main_wrap()
