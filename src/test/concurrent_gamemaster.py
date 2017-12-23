from builtins import super

import time
from collections import deque

import greenlet

from ggplib.db.helper import get_gdl_for_game
from ggplib.db import lookup
from ggplib.player.gamemaster import GameMaster

from ggplearn import msgdefs
from ggplearn.nn import bases

from ggplearn.player.puctplayer import PUCTPlayer
from ggplearn.player.policyplayer import PolicyPlayer


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    #from ggplearn.util.keras import constrain_resources
    #constrain_resources()

BATCH_SIZE = 256

class NetworkScheduler(object):
    def __init__(self, nn):
        self.nn = nn
        self.buf_states = []
        self.buf_lead_role_indexes = []

        self.main = greenlet.getcurrent()
        self.runnables = deque()
        self.requestors = []
        self.before_time = -1
        self.after_time = -1

        self.acc_python_time = 0
        self.acc_predict_time = 0

    def add_runnable(self, fn, arg):
        self.runnables.append((greenlet.greenlet(fn), arg))

    def predict_n(self, states, lead_role_indexes):
        # cur is running inside player
        cur = greenlet.getcurrent()

        self.buf_states += states
        self.buf_lead_role_indexes += lead_role_indexes
        self.requestors += [cur] * len(states)

        # print 'HERE predict_n', len(self.buf_states)
        # print 'HERE self.requestors', len(self.requestors)

        # switching to main
        return self.main.switch(cur)

    def predict_1(self, state, lead_role_index):
        return self.predict_n([state], [lead_role_index])[0]

    def set_result(self, g, result):
        self.runnables.append((g, result))

    def do_predictions(self):
        results = []
        while self.buf_states:
            next_states = self.buf_states[:BATCH_SIZE]
            next_lead_role_indexes = self.buf_lead_role_indexes[:BATCH_SIZE]

            self.before_time = time.time()
            if self.after_time > 0:
                self.acc_python_time += self.before_time - self.after_time

            results += self.nn.predict_n(next_states, next_lead_role_indexes)

            self.after_time = time.time()
            self.acc_predict_time += self.after_time - self.before_time

            self.buf_states = self.buf_states[BATCH_SIZE:]
            self.buf_lead_role_indexes = self.buf_lead_role_indexes[BATCH_SIZE:]

        # go through results, assign them to particular requestor and unblock requestor
        assert len(results) == len(self.requestors)

        cur_requestor = None
        cur_result = None
        for i in range(len(results)):
            if cur_requestor is None:
                cur_requestor = self.requestors[i]
                cur_result = [results[i]]

            elif cur_requestor != self.requestors[i]:
                self.set_result(cur_requestor, cur_result)

                cur_requestor = self.requestors[i]
                cur_result = [results[i]]
            else:
                cur_result.append(results[i])

        if cur_requestor is not None:
            self.set_result(cur_requestor, cur_result)

        self.requestors = []

    def run_all(self):
        # this is us
        self.main = greenlet.getcurrent()
        print 'run_all', self.main

        while True:
            # print 'runnables len', len(self.runnables)
            if not self.runnables:
                # done
                if not self.buf_states:
                    break
                self.do_predictions()

            else:
                g, arg = self.runnables.popleft()
                g.switch(arg)

                if len(self.buf_states) >= BATCH_SIZE:
                    self.do_predictions()


class GameMasterMagic(GameMaster):
    def set_network(self, network_proxy):
        self.network_proxy = network_proxy

    def patch_players(self):
        for p, _ in self.players:
            p.nn = self.network_proxy

def test_concurrrent_gamemaster():

    default_generation = "v5xx_gen_normal_9"

    # first get the game
    game_info = lookup.by_name("reversi")

    # create a network
    bases_config = bases.get_config(game_info.game, game_info.model, default_generation)
    nn = bases_config.create_network()
    nn.load()

    # create a NetworkScheduler
    network_proxy = NetworkScheduler(nn)

    # create some gamemaster/players
    policy_config = msgdefs.PolicyPlayerConf(generation="noone",
                                             choose_exponential_scale=0.25,
                                             verbose=False)

    puct_config = msgdefs.PUCTPlayerConf(name="rev2-test",
                                         generation="noone",
                                         verbose=False,
                                         playouts_per_iteration=800,
                                         playouts_per_iteration_noop=0,
                                         expand_root=100,
                                         dirichlet_noise_alpha=0.3,
                                         cpuct_constant_first_4=0.75,
                                         cpuct_constant_after_4=0.75,
                                         choose="choose_top_visits")

    gamemasters = []

    # create a bunch of gamemaster and players [slow first step]
    for i in range(BATCH_SIZE):
        gm = GameMasterMagic(get_gdl_for_game("reversi"), fast_reset=True)
        # gm.add_player(PolicyPlayer(policy_config), "black")
        # gm.add_player(PolicyPlayer(policy_config), "red")
        gm.add_player(PUCTPlayer(conf=puct_config), "black")
        gm.add_player(PUCTPlayer(conf=puct_config), "red")

        gm.set_network(network_proxy)
        gm.patch_players()
        gm.start(meta_time=300, move_time=300)

        gamemasters.append(gm)

    for gm in gamemasters:
        last_move = None
        network_proxy.add_runnable(gm.play_single_move, last_move)

    s = time.time()
    network_proxy.run_all()
    print network_proxy.acc_python_time
    print network_proxy.acc_predict_time
    print "TIME TAKEN", time.time() - s

    # for gm in gamemasters:
    #     print gm.depth,
    # print

    # # rerun
    # for gm in gamemasters:
    #     gm.reset()
    #     gm.start(meta_time=30, move_time=1000)

    #     last_move = None
    #     network_proxy.add_runnable(gm.play_to_end, last_move)

    # s = time.time()
    # network_proxy.run_all()
    # print "TIME TAKEN2", time.time() - s
    # for gm in gamemasters:
    #     print gm.depth,
    # print


if __name__ == "__main__":

    test_concurrrent_gamemaster()
