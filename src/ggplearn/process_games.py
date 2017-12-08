import os
import sys
import random
import tempfile

from collections import OrderedDict

import json
import numpy as np

from ggplib.util import log

from ggplib.db import lookup

from ggplearn import net
from ggplearn import net_config

json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

###############################################################################
# hyperparameters

class TrainConfig:
    BATCH_SIZE = 128
    EPOCHS = 24
    VALIDATION_SPLIT = 0.2
    MAX_GAMES = None
    DROP_MOVES_FROM_START_PCT = 0.9
    DROP_MOVES_AFTER_START_PCT = 0.6
    SAVE_DATA_BEFORE_TRAIN = True

###############################################################################

def get_from_json(path, game_name):
    files = os.listdir(path)
    for f in files:
        if f.endswith(".json"):
            if "model" in f:
                continue
            if game_name not in f:
                continue
            buf = open(os.path.join(path, f)).read()
            data = json.loads(buf)
            for g in data:
                if g["game"] == game_name:
                    yield g
                else:
                    log.warning("%s NOT CORRECT GAME (%s)" % (f, g["game"]))
                    break


class PlayOne(object):
    def __init__(self, state, actions, dist_desc, lead_role_index):
        self.state = state
        self.actions = actions

        # for persistence
        self.dist_desc = dist_desc

        # who's turn it is...
        self.lead_role_index = lead_role_index

    def set_scores(self, best, final):
        self.scores = [0] * 4
        if self.lead_role_index:
            self.scores[0] = 1 - best
            self.scores[1] = best

            self.scores[2] = 1 - final
            self.scores[3] = final
        else:
            self.scores[0] = best
            self.scores[1] = 1 - best

            self.scores[2] = final
            self.scores[3] = 1 - final


class Game:
    def __init__(self, data, model):
        # should we do this everytime... thinking about speed
        self.roles = model.roles
        assert len(self.roles) == 2
        self.expected_state_len = len(model.bases)
        assert len(model.actions) == 2
        self.expected_actions_len = sum(len(l) for l in model.actions)
        self.player_0_actions_len = len(model.actions[0])

        self.plays = []
        depth = 0
        while True:
            key = "depth_%d" % depth
            if key not in data:
                break

            info = data[key]

            # final state does not have move or candidates
            if "move" not in info:
                break

            state = tuple(info['state'])
            assert len(state) == self.expected_state_len

            (actions, best_score,
             lead_role_index, dist) = self.process_actions(info['move'],
                                                           info['candidates'])

            final_score = data["final_scores"][self.roles[lead_role_index]] / 100.0

            play = PlayOne(state, actions, dist, lead_role_index)
            play.set_scores(best_score, final_score)

            self.plays.append(play)

            depth += 1

    def process_actions(self, move, candidates):
        self.move = move
        assert len(move) == 2
        lead_role_index = 0 if move.index("noop") else 1
        lead_role = self.roles[lead_role_index]
        passive_role = self.roles[0 if lead_role_index else 1]

        assert lead_role != passive_role

        # noop, throw away
        assert len(candidates[passive_role]) == 1

        actions = [0 for _ in range(self.expected_actions_len)]

        index_start = 0 if lead_role_index == 0 else self.player_0_actions_len

        total_visits = 0
        best_score = 0
        cand = candidates[lead_role]
        for idx, score, visits in cand:
            actions[index_start + idx] = visits
            total_visits += visits

            if score > best_score:
                best_score = score

        cand_probs = []
        for idx, score, visits in cand:
            actions[index_start + idx] /= float(total_visits)
            cand_probs.append((idx, actions[index_start + idx]))

        return actions, best_score, lead_role_index, cand_probs

###############################################################################

def shuffle(*arrays):
    arrays = [np.array(a) for a in arrays]

    # shuffle data
    shuffle = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle)

    for i in range(len(arrays)):
        a = arrays[i]
        a.astype('float32')
        arrays[i] = arrays[i][shuffle]

    return arrays

def training_data(data_path, base_infos, sm_model, config):
    games_processed = 0
    log.debug("Processing games: %s" % config.game)

    # just gather all the data from the games for now
    samples_X0, samples_X1 = [], []
    samples_y0, samples_y1, samples_y2 = [], [], []
    the_samples = [samples_X0, samples_X1, samples_y0, samples_y1, samples_y2]

    from collections import Counter
    early_states = Counter()
    later_states = Counter()
    early_states_dropped = 0
    later_states_dropped = 0

    # incase we store the data (otherwise just warming up the cpu)
    persist_samples = []
    for data in get_from_json(data_path, config.game):
        game = Game(data, sm_model)

        for depth, play in enumerate(game.plays):
            if depth < 5:
                # drop 75% moves from start of game
                # lots of dupes here
                if random.random() < TrainConfig.DROP_MOVES_FROM_START_PCT:
                    continue

                early_states[play.state] += 1
                if early_states[play.state] > 5:
                    early_states_dropped += 1
                    continue

            else:
                # drop 30% moves from this point on
                if random.random() < TrainConfig.DROP_MOVES_AFTER_START_PCT:
                    continue

                later_states[play.state] += 1
                if later_states[play.state] > 1:
                    later_states_dropped += 1
                    continue

            X_0 = net_config.state_to_channels(play.state,
                                               play.lead_role_index,
                                               config, base_infos)
            samples_X0.append(X_0)

            X_1 = [v for v, base_info in zip(play.state, base_infos) if base_info.channel is None]
            samples_X1.append(X_1)

            samples_y0.append(play.actions)

            samples_y1.append(play.scores[:2])
            samples_y2.append(play.scores[2:])

            if TrainConfig.SAVE_DATA_BEFORE_TRAIN:
                d = OrderedDict()
                d["state"] = play.state
                d["lead_role_index"] = play.lead_role_index
                d["final_scores"] = play.scores[:2]
                d["best_scores"] = play.scores[2:]
                d["policy_dist"] = play.dist_desc
                persist_samples.append(d)

        games_processed += 1
        if games_processed % 100 == 0:
            log.debug("GAMES PROCESSED %s" % games_processed)
            log.info("early_states_dropped %s" % early_states_dropped)
            log.info("later_states_dropped %s" % later_states_dropped)

        if TrainConfig.MAX_GAMES is not None and games_processed > TrainConfig.MAX_GAMES:
            break

    log.info("Total games processed %s" % games_processed)
    log.warning("early_states_dropped %s" % early_states_dropped)
    log.warning("later_states_dropped %s" % later_states_dropped)

    for samp in the_samples[1:]:
        assert len(the_samples[0]) == len(samp)

    num_samples = len(samples_X0)
    log.info("gathered %s samples" % num_samples)

    # shuffle
    log.debug("Shuffling data")

    # keep indices around
    the_samples.append(range(len(samples_X0)))
    the_samples = shuffle(*the_samples)
    indices = the_samples.pop(-1)

    # split data into training/validation
    num_training_samples = int(num_samples * (1 - TrainConfig.VALIDATION_SPLIT))

    if TrainConfig.SAVE_DATA_BEFORE_TRAIN:
        persist_data = OrderedDict()
        persist_data["game"] = config.game
        persist_data["generation"] = "gen0"
        persist_data["num_samples"] = num_samples
        persist_data["num_training_samples"] = num_training_samples
        persist_data["num_validation_samples"] = num_samples - num_training_samples
        persist_data["samples"] = [persist_samples[i] for i in indices]

        fd, path = tempfile.mkstemp(suffix='.json',
                                    prefix="initial_data_%s_" % (config.game),
                                    dir=".")
        with os.fdopen(fd, 'w') as open_file:
            open_file.write(json.dumps(persist_data, indent=4))

    training_data = [a[:num_training_samples] for a in the_samples]
    validation_data = [a[num_training_samples:] for a in the_samples]

    for i, s in enumerate(training_data):
        log.debug("Shape of training data %d: %s" % (i, s.shape))

    for i, s in enumerate(validation_data):
        log.debug("Shape of validation data %d: %s" % (i, s.shape))

    return training_data[0:2], training_data[2:], validation_data[0:2], validation_data[2:]


def build_and_train_nn(data_path, game_name, postfix):
    for attr in vars(TrainConfig):
        if "__" in attr:
            continue
        log.info("TrainConfig.%s = %s" % (attr, getattr(TrainConfig, attr)))

    # lookup via game_name (this gets statemachine & statemachine model)
    info = lookup.by_name(game_name)

    # the bases config (XXX idea is not to use hard coded stuff)
    config = net_config.get_bases_config(game_name)

    # update the base_infos and config
    base_infos = net_config.create_base_infos(config, info.model)

    number_of_outputs = sum(len(l) for l in info.model.actions)
    nn_model = net.get_network_model(config, number_of_outputs)

    # one way to get print_summary to output string!
    lines = []
    nn_model.summary(print_fn=lines.append)
    for l in lines:
        log.verbose(l)

    (training_inputs, training_outputs,
     validation_inputs, validation_outputs) = training_data(data_path, base_infos, info.model, config)

    # train
    my_cb = net.MyCallback()
    progbar = net.MyProgbarLogger()


    nn_model.fit(training_inputs, training_outputs,
                 verbose=0,
                 batch_size=TrainConfig.BATCH_SIZE,
                 epochs=TrainConfig.EPOCHS,
                 validation_data=(validation_inputs, validation_outputs),
                 callbacks=[progbar, my_cb])

    # save model / weights
    with open("model_nn_%s_%s.json" % (game_name, postfix), "w") as f:
        f.write(nn_model.to_json())

    nn_model.save_weights("weights_nn_%s_%s.h5" % (game_name, postfix), overwrite=True)

###############################################################################

def main_wrap():
    import pdb
    import traceback

    from ggplib import interface
    try:
        path = sys.argv[1]
        game = sys.argv[2]
        postfix = sys.argv[3]

        log_name_base = "process_and_train__%s_%s_" % (game, postfix)
        interface.initialise_k273(1, log_name_base=log_name_base)
        log.initialise()

        build_and_train_nn(path, game, postfix)

    except Exception as exc:
        print exc
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

###############################################################################

if __name__ == "__main__":
    main_wrap()
