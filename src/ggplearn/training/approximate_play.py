import time
import random

import attr

from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game

from ggplearn.player import mc


@attr.s
class Sample(object):
    state = attr.ib()
    policy = attr.ib()
    final_score = attr.ib()
    lead_role_index = attr.ib()


@attr.s
class RunnerConf(object):
    game_name = attr.ib("breakthrough")
    policy_generation = attr.ib("gen0_small")
    score_generation = attr.ib("gen0_smaller")
    score_puct_player_conf = attr.ib(None)
    policy_puct_player_conf = attr.ib(None)


class Runner(object):
    def __init__(self, conf):
        self.conf = conf

        # create two game masters, one for the score playout, and one for the policy evaluation
        self.gm_score = GameMaster(get_gdl_for_game(self.conf.game_name))
        self.gm_policy = GameMaster(get_gdl_for_game(self.conf.game_name))

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

        # we want unique samples per generation
        self.unique_samples = set()

        self.reset_debug()

    def reset_debug(self):
        # debug times
        self.acc_time_for_play_one_game = 0
        self.acc_time_for_do_policy = 0

    def get_bases(self):
        self.gm_score.sm.get_current_state(self.basestate)
        return self.basestate.to_list()

    def play_one_game(self):
        self.gm_score.reset()

        self.gm_score.start(meta_time=20, move_time=10)

        states = [self.get_bases()]

        last_move = None
        while not self.gm_score.finished():
            last_move = self.gm_score.play_single_move(last_move=last_move)
            states.append(self.get_bases())

        # pop the final state, as we don't want terminal states
        states.pop()

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

        # fish for root (XXX for game specific, XXX for horrible fishing)
        lead_role_index = 1 if self.last_move[0] == "noop" else 0
        player = self.gm_policy.get_player(lead_role_index)
        dist = [(c.legal, p) for c, p in player.get_probabilities(1.5)]
        return dist, lead_role_index

    def generate_sample(self):
        start_time = time.time()
        states, final_score = self.play_one_game()
        self.acc_time_for_play_one_game += time.time() - start_time
        # we don't sample last state as has no policy
        random.shuffle(states)

        while states:
            state = tuple(states.pop())

            if tuple(state) in self.unique_samples:
                continue

            start_time = time.time()
            policy_dist, lead_role_index = self.do_policy(state)
            self.acc_time_for_do_policy += time.time() - start_time
            return Sample(state, policy_dist, final_score, lead_role_index)
