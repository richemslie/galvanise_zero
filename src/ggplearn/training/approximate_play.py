"""
back to square 1, more or less
  * first play is random-ish with policy player - gm_select
  * second uses puct player, just on selected state - gm_policy
  * third starting from the same state as policy was trained
    on (not the resultant state), policy player for score - gm_score

XXX still not sure whether this approach will lead to unstable network.

"""

import time
import random

import numpy as np

from ggplib.util import log

from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game

from ggplearn.util import attrutil

from ggplearn import msgdefs
from ggplearn.player.puctplayer import PUCTPlayer
from ggplearn.player.policyplayer import PolicyPlayer


class Session(object):
    def __init__(self):
        self.unique_states = set()
        self.samples = None
        self.duplicates_seen = None
        self.reset()

    def reset(self):
        self.samples = []
        self.duplicates_seen = 0

    def start_and_record(self, runner):
        sample, duplicates = runner.generate_sample(self)
        self.samples.append(sample)
        self.duplicates_seen += duplicates

    def add_to_unique_states(self, state):
        self.unique_states.add(state)


class Runner(object):
    def __init__(self, conf):
        assert isinstance(conf, msgdefs.ConfigureApproxTrainer)

        self.conf = conf

        # create two game masters, one for the score playout, and one for the policy evaluation
        self.gm_select = GameMaster(get_gdl_for_game(self.conf.game), fast_reset=True)
        self.gm_policy = GameMaster(get_gdl_for_game(self.conf.game), fast_reset=True)
        self.gm_score = GameMaster(get_gdl_for_game(self.conf.game), fast_reset=True)

        # add players to gamemasteres
        for role in self.gm_select.sm.get_roles():
            self.gm_select.add_player(PolicyPlayer(self.conf.player_select_conf), role)

        for role in self.gm_policy.sm.get_roles():
            self.gm_policy.add_player(PUCTPlayer(self.conf.player_policy_conf), role)

        for role in self.gm_score.sm.get_roles():
            self.gm_score.add_player(PolicyPlayer(self.conf.player_score_conf), role)

        # cache a local statemachine basestate (doesn't matter which gm it comes from)
        self.basestate = self.gm_select.sm.new_base_state()

        # and cache roles
        self.roles = self.gm_select.sm.get_roles()

        # we want unique samples per generation, so store a unique_set here
        self.reset_debug()

    def patch_players(self, scheduler):
        ''' patch the players to use a scheduler greenlet network, which provides concurrency -
            which turns to be much faster (even without a gpu) '''

        # patch each player, on each gamemaster
        for gm in self.gm_select, self.gm_policy, self.gm_score:
            for p, _ in gm.players:
                p.nn = scheduler

    def reset_debug(self):
        self.time_for_play_one_game = 0
        self.time_for_do_policy = 0
        self.time_for_do_score = 0

    def get_bases(self):
        self.gm_select.sm.get_current_state(self.basestate)
        return tuple(self.basestate.to_list())

    def play_one_game_for_selection(self):
        self.gm_select.reset()

        self.gm_select.start(meta_time=240, move_time=240)

        states = [(0, self.get_bases())]

        last_move = None
        depth = 1
        while not self.gm_select.finished():
            last_move = self.gm_select.play_single_move(last_move=last_move)
            states.append((depth, self.get_bases()))
            depth += 1

        # cleanup
        self.gm_select.play_to_end(last_move)
        return states

    def do_policy(self, depth, state):
        for i, v in enumerate(state):
            self.basestate.set(i, v)

        self.gm_policy.reset()
        self.gm_policy.start(meta_time=240, move_time=240, initial_basestate=self.basestate, game_depth=depth)

        # self.last_move used in playout_state
        self.last_move = self.gm_policy.play_single_move(None)

        assert self.gm_policy.get_game_depth() == depth

        # fish for root (XXX for game specific)
        lead_role_index = 1 if self.last_move[0] == "noop" else 0

        player = self.gm_policy.get_player(lead_role_index)

        dist = [(c.legal, p) for c, p in player.get_probabilities()]
        return dist, lead_role_index

    def do_score(self, depth, state):
        for i, v in enumerate(state):
            self.basestate.set(i, v)

        self.gm_score.reset()
        self.gm_score.start(meta_time=240, move_time=240, initial_basestate=self.basestate, game_depth=depth)
        self.gm_score.play_to_end()

        return [self.gm_score.get_score(r) / 100.0 for r in self.roles]

    def generate_sample(self, session):
        # debug
        # log.debug("Entering generate_sample(), unique_states: %s" % len(self.unique_states))

        start_time = time.time()
        states = self.play_one_game_for_selection()
        self.time_for_play_one_game = time.time() - start_time

        game_length = len(states)
        # log.debug("Done play_one_game_for_selection(), game_length %d" % game_length)

        shuffle_states = states[:]

        # pop the final state, as we don't want terminal states.  But keep in states intact
        shuffle_states.pop()
        random.shuffle(shuffle_states)
        np.random.shuffle(shuffle_states)
        random.shuffle(shuffle_states)

        duplicate_count = 0

        while shuffle_states:
            depth, state = shuffle_states.pop()

            if state in session.unique_states:
                duplicate_count += 1
                continue

            start_time = time.time()
            policy_dist, lead_role_index = self.do_policy(depth, state)
            self.time_for_do_policy = time.time() - start_time

            # start from state and not from what policy returns (which would add bias)
            start_time = time.time()
            final_score = self.do_score(depth, state)
            self.time_for_do_score = time.time() - start_time

            prev_state = states[depth - 1] if depth >= 1 else None
            sample = msgdefs.Sample(prev_state,
                                    state, policy_dist, final_score,
                                    depth, game_length, lead_role_index)


            session.add_to_unique_states(tuple(state))

            log.debug("select/policy/score %.2f/%.2f/%.2f depth %s/%s score %s" % (self.time_for_play_one_game,
                                                                                   self.time_for_do_policy,
                                                                                   self.time_for_do_score,
                                                                                   depth, game_length,
                                                                                   final_score))

            return sample, duplicate_count

        log.warning("Ran out of states, lots of duplicates.  Please do something about this, "
                    "shouldn't be playing with lots of duplicates.  Hack for now is to rerun.")
        return self.generate_sample()
