'''

specing things out mostly for proper implementation in c++
----------------------------------------------------------

 * use just a single PUCTEvaluator.
  * means there is just a single tree being built - rather than 6 we have in approximate_play!

 * lower PUCT iterations for non-policy moves (likely between 40-100)
   * allow for resign if > .1 % probabilty of winning.  Ensure 10% of resigns are ignored.

 * allow for selection from starting point (random number with expected number of moves per game)

 * specifiy maximum number of moves for policy
  * 800 iterations
  * these are the samples to learn policy.
  * typically > 1, as 'previous states' also taken into account when training

* turning off selection and allowing maximum number of policy moves - is the AGZ/A0 method.

* uses temperature/noise

* performance:  want to keep running concurrent games rather than waiting for them all to finish.
  * waiting for them all to finish ends up with staggered batches to the gpu
    * which slows things down significantly
  * Updating the network in the middle of a game is fine.
  * This is the plan for c++.  Whether we should do the same thing here in python?

XXX code does not reflect the above yet.

Currently take 2500 predictions for one sample.  If we have lower iterations at 64 and higher at
800 and expected game depth is 70.  And we take 5 samples.  Then that is ~8500 predictions.  Which
turns out at 1700 predictions per sample.
With two GPUs that should give us approx 25 samples per second. And each generation (which was set
5000 samples for v5) - will take < 4 minutes.  (Currently take 2-3 hours).

If it is that fast, will spend most of the time being bottlenecked with training.  Will need to
that differently.

'''


import time
import random

import numpy as np

from ggplib.util import log

from ggplib.db.helper import get_gdl_for_game

from ggpzero.defs import msgs, confs
from ggpzero.player.puctplayer import PUCTEvaluator


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
        for sample, duplicates in runner.generate_sample(self):
            # XXX should we add to unique_states ?
            self.unique_states.add(tuple(sample))
            self.samples.append(sample)
            self.duplicates_seen += duplicates

    def add_to_unique_states(self, state):
        self.unique_states.add(state)


class Runner(object):
    def __init__(self, conf):
        assert isinstance(conf, msgs.ConfigureApproxTrainer)

        self.conf = conf

        gdl = get_gdl_for_game(self.conf.game)
        gdl = gdl

        # XXX no gamemastes
        # create game masters, one for the score playout, and one for the policy evaluation
        # self.gm_select = GameMaster(gdl, fast_reset=True)
        # self.gm_policy = GameMaster(gdl, fast_reset=True)
        # self.gm_score = GameMaster(gdl, fast_reset=True)

        # add players to gamemasteres
        # for role in self.gm_select.sm.get_roles():
        #    self.gm_select.add_player(PUCTPlayer(self.conf.player_select_conf), role)

        # for role in self.gm_policy.sm.get_roles():
        #    self.gm_policy.add_player(PUCTPlayer(self.conf.player_policy_conf), role)

        # for role in self.gm_score.sm.get_roles():
        #    self.gm_score.add_player(PUCTPlayer(self.conf.player_score_conf), role)

        # just one puctevalutor
        self.evaluator = PUCTEvaluator(None)  # XXXX todo

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

        results = []
        for i in range(3):
            policy_dist, lead_role_index = self.do_policy(depth, state)
            results.append((depth, state, policy_dist, lead_role_index))

            # advance state/depth with best move
            depth += 1

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

    def generate_samples(self, session):
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

            self.time_for_do_policy = time.time() - start_time

            # start from state and not from what policy returns (which would add bias)
            start_time = time.time()
            final_score = self.do_score(depth, state)
            self.time_for_do_score = time.time() - start_time

            prev_state = states[depth - 1] if depth >= 1 else None
            policy_dist = None  # XXXX
            lead_role_index = None  # XXXX
            sample = confs.Sample(prev_state,
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
