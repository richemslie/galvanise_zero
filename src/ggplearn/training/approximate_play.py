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
import time
import random
import tempfile

from collections import OrderedDict

import json

from ggplib.player.gamemaster import GameMaster
from ggplib.db import lookup
from ggplib.db.helper import get_gdl_for_game

from ggplearn.player import mc


class RunnerConf(Object):
    game_name = "breakthrough"
    policy_generation = "gen9_small"
    score_generation = "gen9_tiny"
    score_puct_player_conf = mc.PUCTPlayerConf()
    policy_puct_player_conf = mc.PUCTPlayerConf()


class Runner(object):
    def __init__(self, conf=None):
        if conf is None:
            conf = RunnerConf()
        self.conf = conf

        self.gm = GameMaster(get_gdl_for_game(self.confgame_name))

        for role in self.gm.sm.get_roles():
            player = mc.PUCTPlayer(self.conf.score_generation,
                                   self.conf.score_puct_player_conf)
            self.gm.add_player(player, role)

            c = self.conf.score_puct_player_conf
            c.NUM_OF_PLAYOUTS_PER_ITERATION = 100
            c.NUM_OF_PLAYOUTS_PER_ITERATION_NOOP = 1
            c.VERBOSE = False

        for role in self.gm_policy.sm.get_roles():
            player = mc.PUCTPlayer(self.conf.policy_generation,
                                   self.conf.policy_puct_player_conf)
            self.gm.add_player(player, role)

            c = self.conf.policy_puct_player_conf
            c.NUM_OF_PLAYOUTS_PER_ITERATION = 400
            c.NUM_OF_PLAYOUTS_PER_ITERATION_NOOP = 1
            c.VERBOSE = False

        self.basestate = self.gm.sm.new_base_state()

    def get_bases(self):
        self.gm.sm.get_current_state(self.basestate)
        return self.basestate.to_list()

    def play_one_game(self):
        self.gm.reset()

        self.gm.start(meta_time=30, move_time=5)

        states = [self.get_bases()]

        last_move = None
        while not self.gm.finished():
            last_move = self.gm.play_single_move(last_move=last_move)
            states.append(self.get_bases())

        # pop the final state, as we don't want terminal states
        states.pop()

        # cleanup
        self.gm.play_to_end(last_move)
        return states, self.gm.scores

    def do_policy(self, state):
        for i, v in enumerate(state):
            self.basestate.set(i, v)

        self.gm.reset()
        self.gm.start(meta_time=30, move_time=10, initial_basestate=self.basestate)

        # self.last_move used in playout_state
        self.last_move = self.gm.play_single_move(None)

        # fish for root
        if self.last_move[0] == "noop":
            root = self.gm.players[1][0].root
        else:
            root = self.gm.players[0][0].root

        children = root.sorted_children()
        total_visits = float(sum(child.visits() for child in children))
        return [(c.legal, c.visits() / total_visits)
                for c in children]

    def mk_sample_desc(self, states, policy, final_score):
        sample_desc = OrderedDict()
        sample_desc["state"] = list(state)
        sample_desc["final_score"] = final_score

        l = sample["policy_distribution"] = []
        for legal, move_str, prob in policy_dist:
            d = OrderedDict()
            d["legal"] = int(legal)
            d["probability"] = float(new_p)
            l.append(d)

        return sample_desc

    def generate_sample(self):
        states, final_score = self.play_one_game()

        # we don't sample last state as has no policy
        random.shuffle(states)

        while states:
            state = tuple(states.pop())

            if tuple(s) in self.unique_samples:
                continue

            policy_dist = self.do_policy(state)
            retun self.mk_sample_desc(states, policy, final_score)

def run(conf):
    runner = Runner(conf)
    while True:
        start = time.time()
        s = runner.generate_sample(num_samples)
        print "time taken to generate sample: %.2f" % (time.time() - start)
        yield s


