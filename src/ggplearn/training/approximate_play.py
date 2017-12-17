import time
import random

import numpy as np

from ggplib.util import log

from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game

from ggplearn.util import attrutil
from ggplearn.distributed import msgs

from ggplearn.player import mc


class Runner(object):
    def __init__(self, conf):
        assert isinstance(conf, msgs.ConfigureApproxTrainer)

        attrutil.pprint(conf)

        self.conf = conf

        # create two game masters, one for the score playout, and one for the policy evaluation
        self.gm_score = GameMaster(get_gdl_for_game(self.conf.game))
        self.gm_policy = GameMaster(get_gdl_for_game(self.conf.game))

        # add players to gamemasteres
        for role in self.gm_score.sm.get_roles():
            player = mc.PUCTPlayer(self.conf.score_generation,
                                   self.conf.score_puct_player_conf)
            self.gm_score.add_player(player, role)

        for role in self.gm_policy.sm.get_roles():
            player = mc.PUCTPlayer(self.conf.policy_generation,
                                   self.conf.policy_puct_player_conf)
            self.gm_policy.add_player(player, role)

        # cache a local statemachine basestate (doesn't matter which gm it comes from)
        self.basestate = self.gm_policy.sm.new_base_state()

        # we want unique samples per generation, so store a unique_set here
        self.unique_states = set()
        self.reset_debug()

    def reset_debug(self):
        # debug times
        self.acc_time_for_play_one_game = 0
        self.acc_time_for_do_policy = 0

    def add_to_unique_states(self, state):
        self.unique_states.add(state)

    def get_bases(self):
        self.gm_score.sm.get_current_state(self.basestate)
        return tuple(self.basestate.to_list())

    def play_one_game(self):
        self.gm_score.reset()

        self.gm_score.start(meta_time=20, move_time=10)

        states = [(0, self.get_bases())]

        last_move = None
        depth = 1
        while not self.gm_score.finished():
            last_move = self.gm_score.play_single_move(last_move=last_move)
            states.append((depth, self.get_bases()))
            depth += 1

        # cleanup
        self.gm_score.play_to_end(last_move)
        return states, self.gm_score.scores

    def do_policy(self, state):
        for i, v in enumerate(state):
            self.basestate.set(i, v)

        self.gm_policy.reset()
        self.gm_policy.start(meta_time=30, move_time=10, initial_basestate=self.basestate)

        # self.last_move used in playout_state
        self.last_move = self.gm_policy.play_single_move(None)

        # fish for root (XXX for game specific)
        lead_role_index = 1 if self.last_move[0] == "noop" else 0

        player = self.gm_policy.get_player(lead_role_index)

        # distx = [(c.move, p) for c, p in player.get_probabilities(self.conf.temperature)]
        # import pprint
        # print.pprint(distx)

        dist = [(c.legal, p) for c, p in player.get_probabilities(self.conf.temperature)]
        return dist, lead_role_index

    def generate_sample(self):
        # debug
        score_player = self.gm_score.get_player(0)
        policy_player = self.gm_policy.get_player(0)
        log.debug("generate_sample() gens score: %s, policy: %s" % (score_player.generation,
                                                                    policy_player.generation))
        log.debug("iterations score: %s, policy: %s" % (score_player.conf.num_of_playouts_per_iteration,
                                                        policy_player.conf.num_of_playouts_per_iteration))

        log.debug("unique_states: %s" % len(self.unique_states))

        start_time = time.time()
        states, final_score = self.play_one_game()
        game_length = len(states)

        log.debug("Done play_one_game(), game_length %d" % game_length)

        self.acc_time_for_play_one_game += time.time() - start_time

        shuffle_states = states[:]

        # pop the final state, as we don't want terminal states.  But keep in states intact
        shuffle_states.pop()
        np.random.shuffle(shuffle_states)

        duplicate_count = 0

        while shuffle_states:
            depth, state = shuffle_states.pop()

            if state in self.unique_states:
                duplicate_count += 1
                continue

            start_time = time.time()
            policy_dist, lead_role_index = self.do_policy(state)

            log.debug("Done do_policy()")
            self.acc_time_for_do_policy += time.time() - start_time

            prev2 = None  # states[depth - 3] if depth >= 3 else None
            prev1 = None  # states[depth - 2] if depth >= 2 else None
            prev0 = None  # states[depth - 1] if depth >= 1 else None

            sample = msgs.Sample(prev2, prev1, prev0,
                                 state, policy_dist, final_score,
                                 depth, game_length, lead_role_index)

            return sample, duplicate_count

        log.warning("Ran out of states, lots of duplicates.  Please do something about this, "
                    "shouldn't be playing with lots of duplicates.  Hack for now is to rerun.")
        return self.generate_sample()
