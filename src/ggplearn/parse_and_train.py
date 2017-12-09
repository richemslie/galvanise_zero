import sys
from os.path import join as joinpath

from ggplib.util import log
from ggplib import interface

from ggplearn.crunch import get_from_json, Sample, TrainerBase

DEBUG = False

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

        for s in game_data["samples"]:
            state = tuple(s["state"])
            policy_dist = [(d["legal"], d["new_prob"]) for d in s["policy_dists"]]
            final_scores = [s["final_scores"][r] / 100.0 for r in "white black".split()]

            if DEBUG and len(new_data) < 3:
                print s
                print state
                print policy_dist
                print final_scores

            if state in all_states:
                dupes += 1
                continue

            new_data.append(Sample(state, policy_dist, final_scores))
            all_states.add(state)

    if bad_files:
        PDB

    print "DUPES", dupes
    return new_data


def main(path):
    GAME = "breakthrough"
    GENERATIONS_SO_FAR = 7
    ALL_GENERATIONS = ["gen%d" % i for i in range(GENERATIONS_SO_FAR)]

    # think i've been trying to get to this variable name: :)
    NEXT_GENERATION = "gen%d" % GENERATIONS_SO_FAR

    MAX_SAMPLE_COUNT = 150000
    VALIDATION_SPLIT = 0.8

    log_name_base = "parsex__%s_" % GAME
    interface.initialise_k273(1, log_name_base=log_name_base)
    log.initialise()

    training_data = []
    validation_data = []

    for game_data, _ in get_from_json(path, includes=[GAME, "initial"], excludes=["samples"]):
        assert game_data["game"] == GAME

        num_samples = game_data["num_samples"]
        samples = game_data["samples"]

        assert num_samples == len(samples)
        assert game_data["num_training_samples"] + game_data["num_validation_samples"] == num_samples

        for i in range(game_data["num_training_samples"]):
            s = samples[i]

            state = tuple(s["state"])
            policy_dist = s["policy_dist"]

            # XXX fekked that up I think... will fix for next gen
            final_scores = s["best_scores"]

            training_data.append(Sample(state, policy_dist, final_scores))

        for i in range(game_data["num_validation_samples"]):
            s = samples[i]
            state = tuple(s["state"])
            policy_dist = s["policy_dist"]

            # XXX fekked that up I think... will fix for next gen
            final_scores = s["best_scores"]

            validation_data.append(Sample(state, policy_dist, final_scores))

    print "#training_data", len(training_data)
    print "#validation_data", len(validation_data)
    print


    all_states = set()
    for gen in ALL_GENERATIONS:
        generation_data = gather_new_data(path, GAME, gen, all_states)

        split_index = int(len(generation_data) * VALIDATION_SPLIT)
        new_training_data = generation_data[:split_index]
        new_validation_data = generation_data[split_index:]

        print "generation '%s' data : %d" % (gen, len(generation_data))
        print "#training new data : %d" % len(new_training_data)
        print "#validation new data : %d" % len(new_validation_data)
        print

        training_data += new_training_data
        validation_data += new_validation_data


    if MAX_SAMPLE_COUNT and training_data + validation_data > MAX_SAMPLE_COUNT:
        print "#training_data", len(training_data)
        print "#validation_data", len(validation_data)
        print

        max_number_training_data = int(MAX_SAMPLE_COUNT * VALIDATION_SPLIT)
        max_number_validation_data = MAX_SAMPLE_COUNT - max_number_training_data

        start_train = len(training_data) - max_number_training_data
        start_validation = len(validation_data) - max_number_validation_data
        pct = start_train / float(len(training_data))
        print 'dropping %d%% of initial data' % (pct * 100)
        training_data = training_data[start_train:]
        validation_data = validation_data[start_validation:]

    print "# final training_data", len(training_data)
    print "# final validation_data", len(validation_data)
    print

    tb = TrainerBase(GAME, NEXT_GENERATION)
    tb.nn_summary()
    tb.massage_data(training_data, validation_data)
    tb.train()
    tb.save()


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
