"""

            #if depth > 2:
            #    if state in self.all_states:
            #        Game.dupes += 1
            #    else:
            #        Game.all_states.add(state)

    print "#games", len(games)
    print "#dupe states", Game.dupes
    print "#unique states", len(Game.all_states)
    print "% of dupe states", Game.dupes / float(Game.dupes + len(Game.all_states))



class TrainingData:
    def __init__(self, config, data, do_score_only=True):
        self.config = config
        self.model = config.get_model()
        self.data = data

    def gen(self, filter_by_role_index):
        for row in self.data:

            # # numerics please:
            # role_index = int(row[0])

            # # filtering?
            # if not self.do_scores:
            #     if role_index != filter_by_role_index:
            #         continue

            game_depth = int(row[1])
            if game_depth > 100:
                continue

            scores = [float(row[3]), float(row[4])]

            # XXX THESE HACKS WERE HERE FOR POLICY NETWORK...
            #if scores[0] > 0.499 and scores[1] < 0.499 and game_depth > 20:
            #    print "Playing randomly since game is already a draw"
            #    continue

            #if scores[role_index] < 0.01:
            #    print "Skip low score move - would be playing randomly"
            #    continue

            state = [1 if x == "1" else 0 for x in row[5:]]
            X_0 = self.config.map_state_with_roles(state,
                                                   self.analyser.base_state_info,
                                                   role_index)

            if self.do_scores:
                y_0 = scores
            else:
                action = int(row[2])
                y_0 = self.action_to_one_hot(role_index, action)

            yield role_index, X_0, y_0

###############################################################################
dupes = 0
all_states = set()

"""

import os
import sys
import json

class PlayOne(object):
    def __init__(self, state, inputs, best_score):
        self.state = state
        self.inputs = inputs
        self.best_score = best_score

class Game:
    def __init__(self, data, model):
        # don't keep data

        # should we do this everytime... thinking about speed
        self.roles = model.roles
        assert len(self.roles) == 2
        self.expected_state_len = len(model.bases)
        assert len(model.actions) == 2
        self.expected_inputs_len = sum(len(l) for l in model.actions)
        self.player_0_inputs_len = len(model.actions[0])

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

            best_score, inputs = self.process_inputs(info['move'], info['candidates'])
            self.plays.append(PlayOne(state, inputs, best_score))
            depth += 1

    def process_inputs(self, move, candidates):
        self.move = move
        assert len(move) == 2
        lead_role = self.roles[0 if move.index("noop") else 1]
        passive_role = self.roles[1 if move.index("noop") else 0]

        # so player0 goes towards 1, player1 goes towards -1
        m = 2 if move.index("noop") else -2
        k = -1 if move.index("noop") else 1

        # noop, throw away
        assert len(candidates[passive_role]) == 1

        inputs = [0 for i in range(self.expected_inputs_len)]

        # worst possible score
        best = k
        v = -k

        index_start = 0 if move.index("noop") else self.player_0_inputs_len
        for idx, score in candidates[lead_role]:
            normalise_score = score * m + k
            if (v* normalise_score > best *v):
                best = normalise_score

            inputs[index_start + idx] = normalise_score

        # XXX set noop to be the -best?  Or just leave it 0?
        return best, inputs


def get_all(path, game_name):
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









class AtariGo_7x7(object):
    game = "atariGo_7x7_"
    x_cords = "1 2 3 4 5 6 7".split()
    y_cords = "1 2 3 4 5 6 7".split()
    base_term = "cell"
    pieces = ['black', 'white']


def search_for_terms(self):
    return [(p,) for p in self.pieces]

def init_state(config, model):
    sf = SymbolFactory()
    sf.
    for b in model.bases:

    all_cords = []
    for x_cord in config.x_cords:
        for y_cord in config.y_cords:
            all_cords.append((x_cord, y_cord))

    for b_info in bs_info.bases:
        b_info.channel = None

    # need to match up there terms.  There will be one channel each.  We don't care what
    # the order is, the NN doesn't care either.
    for (channel_count, args) in enumerate(self.search_for_terms):
        count = 0
        for board_pos, (x_cord, y_cord) in enumerate(all_cords):
            # this is slow.  Will go through all the bases and match up terms.
            for b_info in bs_info.bases:
                if b_info.terms[BASE_TERM] != self.base_term:
                    continue

                if (b_info.terms[self.x_term] == x_cord and
                    b_info.terms[self.y_term] == y_cord):

                    if self.match_terms(b_info, args):
                        count += 1
                        b_info.channel = channel_count
                        b_info.cord_idx = board_pos
                        break

        print "init_state() found %s states for %s" % (count, args)

def training_data_rows(games, model):
    config = AtariGo_7x7()

    init_state(config, model.bases)


    for g in games:
        for p in g.plays:
            X_0 = map_state_with_roles(state, base_state_info, role_index)

            xxx

def go(path, game_name):
    from ggplib.db import lookup
    info = lookup.by_name(game)

    games = []
    for data in get_all(path, game_name):
        games.append(Game(data, info.model))
        if len(games) % 100 == 0:
            print len(games)

    assert games

###############################################################################

if __name__ == "__main__":
    import pdb
    import traceback
    try:
        from ggplib import interface
        interface.initialise_k273(1, log_name_base="perf_test")

        import ggplib.util.log
        ggplib.util.log.initialise()

        path = sys.argv[1]
        game = sys.argv[2]
        go(path, game)

    except Exception as exc:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
