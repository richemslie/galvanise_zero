'''
Doesn't actually self play.
Similar to fast n slow, approximates playing a full game and sampling one state/move.
This should give similar results and be a ton faster.

All stages use monte carlo player.

1.  Run a game using just the policy net and Dirichlet noise (with mc player and iterations == 1).
2.  Sample a single unique state from that game.
3.  Run monte carlo with 800 iterations on that state.  Record new policy.
4.  Play game to the end using a low number of iterations.  Record final score.
5.  Goto 1

'''


import os
import random
import tempfile

from collections import OrderedDict

import json

from ggplib.player.gamemaster import GameMaster
from ggplib.db import lookup
from ggplib.db.helper import get_gdl_for_game

from ggplearn.player import mc

# XXX monkey patch (cause I will likely forget to turn it off)
mc.VERBOSE = False

models_path = os.path.join(os.environ["GGPLEARN_PATH"], "src", "ggplearn", "models")


class Runner(object):
    def __init__(self, game_name, with_generation):
        self.game_name = game_name
        self.with_generation = with_generation

        self.gm = GameMaster(get_gdl_for_game(game_name))

        for role in self.gm.sm.get_roles():
            player = mc.NNMonteCarlo(with_generation)
            self.gm.add_player(player, role)

        self.basestate = self.gm.sm.new_base_state()

    def get_bases(self):
        self.gm.sm.get_current_state(self.basestate)
        return [self.basestate.get(i) for i in range(self.basestate.len())]

    def play_one_game(self):
        for player, _ in self.gm.players:
            player.NUM_OF_PLAYOUTS_PER_ITERATION = 1
            player.NUM_OF_PLAYOUTS_PER_ITERATION_NOOP = 1
            player.EXPERIMENTAL_MINMAX = False
            player.DIRICHLET_NOISE_ALPHA = 0.03
            player.CPUCT_CONSTANT = 4.0

        self.gm.reset()

        self.gm.start(meta_time=30, move_time=5)

        states = [self.get_bases()]

        last_move = None
        while not self.gm.finished():
            last_move = self.gm.play_single_move(last_move=last_move)
            states.append(self.get_bases())

        self.gm.play_to_end(last_move)
        return states

    def do_policy(self, state):
        for player, _ in self.gm.players:
            player.NUM_OF_PLAYOUTS_PER_ITERATION = 800
            player.NUM_OF_PLAYOUTS_PER_ITERATION_NOOP = 1
            player.DIRICHLET_NOISE_ALPHA = -1

        for i, v in enumerate(state):
            self.basestate.set(i, v)

        self.gm.reset()
        self.gm.start(meta_time=30, move_time=30, initial_basestate=self.basestate)

        # self.last_move used in playout_state
        self.last_move = self.gm.play_single_move(None)

        # fish for root
        if self.last_move[0] == "noop":
            root = self.gm.players[1][0].root
        else:
            root = self.gm.players[0][0].root

        children = root.sorted_children()

        def get_visits(c):
            if c.to_node is None:
                return 1.0
            else:
                return (c.to_node.mcts_visits + 1)

        total_visits = 1.0
        for child in children:
            total_visits += get_visits(child)

        dist = [(c.legal,
                 c.move,
                 get_visits(c) / total_visits,
                 c.p_visits_pct) for c in children]

        # XXX add temperature...

        return dist

    def playout_state(self):
        for player, _ in self.gm.players:
            player.NUM_OF_PLAYOUTS_PER_ITERATION = 42
            player.NUM_OF_PLAYOUTS_PER_ITERATION_NOOP = 1
            player.DIRICHLET_NOISE_ALPHA = -1

        # self.last_move was set in do_policy()
        self.gm.play_to_end(self.last_move)
        return self.gm.scores

    def generate_samples(self, number_samples):
        NUMBER_OF_SAMPLES_PER_GAME = 10
        self.unique_samples = {}
        while len(self.unique_samples) < number_samples:
            states = self.play_one_game()

            # we don't sample first or last state
            samples = random.sample(states[1:-1], NUMBER_OF_SAMPLES_PER_GAME)

            # one sample per game
            for s in samples:
                s = tuple(s)
                if tuple(s) in self.unique_samples:
                    continue

                policy_dist = self.do_policy(s)
                final_scores = self.playout_state()
                self.unique_samples[s] = (final_scores, policy_dist)
                break

    def write_to_file(self):
        json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

        data = OrderedDict()
        data["game"] = self.game_name
        data["with_generation"] = self.with_generation

        k = data["num_samples"] = len(self.unique_samples)

        data["samples"] = []
        for state, (final_scores, policy_dist) in self.unique_samples.items():

            sample = OrderedDict()
            sample["state"] = list(state)
            sample["final_scores"] = final_scores

            sample["policy_dists"] = []
            for legal, move_str, new_p, old_p in policy_dist:
                d = OrderedDict()
                d["legal"] = int(legal)
                d["new_prob"] = float(new_p)
                d["old_prob"] = float(old_p)
                sample["policy_dists"].append(d)

            # finally append to samples
            data["samples"].append(sample)

        fd, path = tempfile.mkstemp(suffix='.json',
                                    prefix="samples_%s_%d_" % (self.game_name, k),
                                    dir=".")
        with os.fdopen(fd, 'w') as open_file:
            open_file.write(json.dumps(data, indent=4))


def main(game_name):

    from ggplib import interface
    interface.initialise_k273(1)

    # turn off logging to reduce log
    # import ggplib.util.log
    # ggplib.util.log.initialise()

    # pre-initialise database - used in match for remapping
    lookup.get_database()

    from ggplearn.utils.keras import use_one_cpu_please
    use_one_cpu_please()

    game_name = sys.argv[1]
    with_generation = sys.argv[2]
    num_samples = int(sys.argv[3])

    # runs forever
    while True:
        runner = Runner(game_name, with_generation)
        runner.generate_samples(num_samples)
        runner.write_to_file()

if __name__ == "__main__":
    import pdb
    import sys
    import traceback

    try:
        main(sys.argv[1])

    except Exception as exc:
        print exc
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
