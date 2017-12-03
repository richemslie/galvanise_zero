import os
import sys
import json
import random

import numpy as np

from ggplib.symbols import SymbolFactory
from ggplib.db import lookup

from ggplearn import net


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


def get_bases_config(game_name):
    for clz in AtariGo_7x7, Breakthrough, Reversi:
        if clz.game == game_name:
            return clz()

######################################################################

def get_from_json(path, game_name):
    files = os.listdir(path)
    for f in files:
        if f.endswith(".json"):
            buf = open(os.path.join(path, f)).read()
            data = json.loads(buf)
            for g in data:
                if g["game"] == game_name:
                    yield g
                else:
                    print f, "NOT CORRECT GAME", g["game"]
                    break


class PlayOne(object):
    def __init__(self, state, actions, scores, lead_role_index):
        self.state = state
        self.actions = actions
        self.scores = scores

        # who's turn it is...
        self.lead_role_index = lead_role_index


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

            actions, scores, lead_role_index = self.process_actions(info['move'], info['candidates'])
            self.plays.append(PlayOne(state, actions, scores, lead_role_index))

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
        scores = [0 for i in range(self.expected_actions_len)]

        index_start = 0 if lead_role_index == 0 else self.player_0_actions_len

        total_visits = 0
        for idx, score, visits in candidates[lead_role]:
            actions[index_start + idx] = visits
            total_visits += visits

        for idx, score, visits in candidates[lead_role]:
            actions[index_start + idx] /= float(total_visits)


        for idx, score, visits in candidates[lead_role]:
            # scores has to be 0 and 1
            assert -0.1 < score < 1.1
            scores[index_start + idx] = score

        # XXX set noop to be the -best?  Or just leave it 0?
        return actions, scores, lead_role_index

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

        print "init_state() found %s states for channel %s" % (count, channel_count)

    # update the config for non cord states
    config.number_of_non_cord_states = 0
    for b_info in base_infos:
        if b_info.channel is None:
            config.number_of_non_cord_states += 1

    print "Number of number_of_non_cord_states", config.number_of_non_cord_states

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
    X_0 = np.reshape(X_0, (config.num_rows, config.num_rows, len(channels)))

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

def training_data(data_path, base_infos, sm_model, config, max_games):
    games_processed = 0
    print "Processing games:", config.game

    # just gather all the data from the games for now
    samples_X0, samples_X1, samples_y0, samples_y1 = [], [], [], []

    for data in get_from_json(data_path, config.game):
        game = Game(data, sm_model)

        for depth, play in enumerate(game.plays):
            if depth < 5:
                # drop 75% moves from start of game
                # lots of dupes here
                if random.random() < 0.75:
                    continue
            else:
                # drop 30% moves from this point on
                if random.random() < 0.3:
                    continue

            X_0 = state_to_channels(play.state, play.lead_role_index, config, base_infos)
            samples_X0.append(X_0)

            X_1 = [v for v, base_info in zip(play.state, base_infos) if base_info.channel is None]
            samples_X1.append(X_1)

            samples_y0.append(play.actions)
            samples_y1.append(play.scores)

        games_processed += 1
        if games_processed % 100 == 0:
            print "GAMES PROCESSED", games_processed

        if max_games is not None and games_processed > max_games:
            break

    print "Total games processed", games_processed

    assert len(samples_X0) == len(samples_X1) and len(samples_y0) == len(samples_y1) and len(samples_X0) == len(samples_y1)
    print "gathered %s samples" % len(samples_X0)

    # shuffle
    print "Shuffling data"
    X0, X1, y0, y1 = shuffle(samples_X0, samples_X1, samples_y0, samples_y1)

    print "Shapes of X0, X1", X0.shape, X1.shape
    print "Shape of y0, y1", y0.shape, y1.shape

    return X0, X1, y0, y1


def build_and_train_nn(data_path, game_name, max_games):
    BATCH_SIZE = 64
    EPOCHS = 12
    VALIDATION_SPLIT = 0.2

    # lookup via game_name (this gets statemachine & statemachine model)
    info = lookup.by_name(game_name)

    # the bases config (XXX idea is not to use hard coded stuff)
    config = get_bases_config(game_name)

    # update the base_infos and config
    base_infos = create_base_infos(config, info.model)

    number_of_outputs = sum(len(l) for l in info.model.actions)
    nn_model = net.get_network_model(config, number_of_outputs)
    print nn_model.summary()

    X0, X1, y0, y1 = training_data(data_path, base_infos, info.model, config, max_games)

    # train
    nn_model.fit([X0, X1], [y0, y1], verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)

    # save model / weights
    with open("model_nn_%s.json" % game_name, "w") as f:
        f.write(nn_model.to_json())

    nn_model.save_weights("weights_nn_%s.h5" % game_name, overwrite=True)

###############################################################################

def main_wrap():
    import pdb
    import traceback
    try:
        from ggplib import interface
        interface.initialise_k273(1, log_name_base="perf_test")

        import ggplib.util.log
        ggplib.util.log.initialise()

        path = sys.argv[1]
        game = sys.argv[2]

        max_games = None

        build_and_train_nn(path, game, max_games)

    except Exception as exc:
        print exc
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

###############################################################################

if __name__ == "__main__":
    main_wrap()
