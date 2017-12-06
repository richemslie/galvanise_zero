import os
import sys
import json
import random

import numpy as np

import keras.callbacks

from ggplib.symbols import SymbolFactory
from ggplib.db import lookup

from ggplearn import net
from ggplib.util import log

# hyperparameters
class TrainConfig:
    BATCH_SIZE = 128
    EPOCHS = 24
    VALIDATION_SPLIT = 0.2
    MAX_GAMES = None
    DROP_MOVES_FROM_START_PCT = 0.9
    DROP_MOVES_AFTER_START_PCT = 0.6

###############################################################################

class MyCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        assert logs
        log.debug("Epoch %s/%s" % (epoch, self.epochs))

        def str_by_name(names, dp=3):
            fmt = "%%s = %%.%df" % dp
            strs = [fmt % (k, logs[k]) for k in names]
            return ", ".join(strs)

        loss_names = "loss policy_loss score0_loss score1_loss".split()
        val_loss_names = "val_loss val_policy_loss val_score0_loss val_score1_loss".split()

        log.info(str_by_name(loss_names, 4))
        log.info(str_by_name(val_loss_names, 4))

        # accuracy:
        for output in "policy score0 score1".split():
            acc = []
            val_acc = []
            for k in self.params['metrics']:
                if output not in k or "acc" not in k:
                    continue
                if "score" in output and "top" in k:
                    continue

                if 'val' in k:
                    val_acc.append(k)
                else:
                    acc.append(k)

            log.info("%s : %s" % (output, str_by_name(acc)))
            log.info("%s : %s" % (output, str_by_name(val_acc)))

class MyProgbarLogger(keras.callbacks.Callback):
    ' simple progress bar '
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch %d/%d' % (epoch + 1, self.epochs))

        self.target = self.params['samples']

        from keras.utils.generic_utils import Progbar
        self.progbar = Progbar(target=self.target)
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []
            self.progbar.update(self.seen, self.log_values)

    def on_batch_end(self, batch, logs=None):
        self.seen += logs.get('size')

        for k in logs:
            if "loss" in k and "val" not in k:
                self.log_values.append((k, logs[k]))

        self.progbar.update(self.seen, self.log_values)


###############################################################################

class BasesConfig(object):
    role_count = 2

    # will be updated later
    number_of_non_cord_states = 0

    @property
    def num_rows(self):
        return len(self.x_cords)

    @property
    def num_cols(self):
        return len(self.y_cords)

    @property
    def channel_size(self):
        return self.num_cols * self.num_rows

    @property
    def num_channels(self):
        # one for each role to indicate turn, one for each pieces
        return self.role_count + len(self.pieces)


class AtariGo_7x7(BasesConfig):
    game = "atariGo_7x7"
    x_cords = "1 2 3 4 5 6 7".split()
    y_cords = "1 2 3 4 5 6 7".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['black', 'white']


class Breakthrough(BasesConfig):
    game = "breakthrough"
    x_cords = "1 2 3 4 5 6 7 8".split()
    y_cords = "1 2 3 4 5 6 7 8".split()

    base_term = "cellHolds"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['white', 'black']


class Reversi(BasesConfig):
    game = "reversi"
    x_cords = "1 2 3 4 5 6 7 8".split()
    y_cords = "1 2 3 4 5 6 7 8".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['black', 'red']

class Connect4(BasesConfig):
    game = "connectFour"
    x_cords = "1 2 3 4 5 6 7 8".split()
    y_cords = "1 2 3 4 5 6".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['red', 'black']

class Hex(BasesConfig):
    game = "hex"
    x_cords = "a b c d e f g h i".split()
    y_cords = "1 2 3 4 5 6 7 8 9".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['red', 'blue']

def get_bases_config(game_name):
    for clz in AtariGo_7x7, Breakthrough, Reversi, Connect4, Hex:
        if clz.game == game_name:
            return clz()

######################################################################

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
    def __init__(self, state, actions, lead_role_index):
        self.state = state
        self.actions = actions

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
        # don't keep data

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

            actions, best_score, lead_role_index = self.process_actions(info['move'], info['candidates'])
            final_score = data["final_scores"][self.roles[lead_role_index]] / 100.0

            play = PlayOne(state, actions, lead_role_index)
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

        actions = [0 for i in range(self.expected_actions_len)]

        index_start = 0 if lead_role_index == 0 else self.player_0_actions_len

        total_visits = 0
        best_score = 0
        cand = candidates[lead_role]
        for idx, score, visits in cand:
            actions[index_start + idx] = visits
            total_visits += visits

            if score > best_score:
                best_score = score

        for idx, score, visits in cand:
            actions[index_start + idx] /= float(total_visits)

        return actions, best_score, lead_role_index

###############################################################################

class BaseInfo(object):
    def __init__(self, gdl_str, symbols):
        self.gdl_str = gdl_str
        self.symbols = symbols

        # drops true ie (true (control black)) -> (control black)
        self.terms = symbols[1]

        # populated in update_base_infos()
        self.channel = None
        self.cord_idx = None


def create_base_infos(config, sm_model):
    symbol_factory = SymbolFactory()
    base_infos = [BaseInfo(s, symbol_factory.symbolize(s)) for s in sm_model.bases]

    all_cords = []
    for x_cord in config.x_cords:
        for y_cord in config.y_cords:
            all_cords.append((x_cord, y_cord))

    # need to match up there terms.  There will be one channel each.  We don't care what
    # the order is, the NN doesn't care either.
    BASE_TERM = 0

    def match_terms(b_info, arg):
        return b_info.terms[config.piece_term] == arg

    for channel_count, arg in enumerate(config.pieces):
        count = 0
        for board_pos, (x_cord, y_cord) in enumerate(all_cords):
            # this is slow.  Will go through all the bases and match up terms.
            for b_info in base_infos:
                if b_info.terms[BASE_TERM] != config.base_term:
                    continue

                if b_info.terms[config.x_term] == x_cord and \
                   b_info.terms[config.y_term] == y_cord:

                    if match_terms(b_info, arg):
                        count += 1
                        b_info.channel = channel_count
                        b_info.cord_idx = board_pos
                        break

        log.info("init_state() found %s states for channel %s" % (count, channel_count))

    # update the config for non cord states
    config.number_of_non_cord_states = 0
    for b_info in base_infos:
        if b_info.channel is None:
            config.number_of_non_cord_states += 1

    log.info("Number of number_of_non_cord_states %d" % config.number_of_non_cord_states)

    return base_infos

def state_to_channels(basestate, lead_role_index, config, base_infos):
    # create a bunch of zero channels
    channels = [np.zeros(config.channel_size)
                for _ in range(len(config.pieces))]

    # simply add to channel
    for b_info, base_value in zip(base_infos, basestate):
        # XXX sanity
        assert isinstance(base_value, int) and abs(base_value) <= 1

        if base_value and b_info.channel is not None:
            channels[b_info.channel][b_info.cord_idx] = 1

    # here we add in who's turn it is, by adding a layer for each role and then setting
    # everything to 1.
    for ii in range(config.role_count):
        if lead_role_index == ii:
            channels.append(np.ones(config.channel_size))
        else:
            channels.append(np.zeros(config.channel_size))

    X_0 = np.array(channels)
    X_0 = np.rollaxis(X_0, -1)
    X_0 = np.reshape(X_0, (config.num_rows, config.num_cols, len(channels)))

    return X_0

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

            X_0 = state_to_channels(play.state, play.lead_role_index, config, base_infos)
            samples_X0.append(X_0)

            X_1 = [v for v, base_info in zip(play.state, base_infos) if base_info.channel is None]
            samples_X1.append(X_1)

            samples_y0.append(play.actions)

            samples_y1.append(play.scores[:2])
            samples_y2.append(play.scores[2:])

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

    log.info("gathered %s samples" % len(samples_X0))

    # shuffle
    log.debug("Shuffling data")
    the_samples = shuffle(*the_samples)

    for i, s in enumerate(the_samples):
        log.debug("Shape of sample %d: %s" % (i, s[0].shape))

    return the_samples[0:2], the_samples[2:]


def build_and_train_nn(data_path, game_name, postfix):

    for attr in vars(TrainConfig):
        if "__" in attr:
            continue
        log.info("TrainConfig.%s = %s" % (attr, getattr(TrainConfig, attr)))


    # lookup via game_name (this gets statemachine & statemachine model)
    info = lookup.by_name(game_name)

    # the bases config (XXX idea is not to use hard coded stuff)
    config = get_bases_config(game_name)

    # update the base_infos and config
    base_infos = create_base_infos(config, info.model)

    number_of_outputs = sum(len(l) for l in info.model.actions)
    nn_model = net.get_network_model(config, number_of_outputs)

    # one way to get print_summary to output string!
    lines = []
    nn_model.summary(print_fn=lines.append)
    for l in lines:
        log.verbose(l)

    inputs, outputs = training_data(data_path, base_infos, info.model, config)

    # train
    my_cb = MyCallback()
    progbar = MyProgbarLogger()

    nn_model.fit(inputs, outputs,
                 verbose=0,
                 batch_size=TrainConfig.BATCH_SIZE,
                 epochs=TrainConfig.EPOCHS,
                 validation_split=TrainConfig.VALIDATION_SPLIT,
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
