import time
from collections import deque, Counter

import greenlet

from ggplib.db import lookup

from ggplearn.nn import bases


class NetworkScheduler(object):
    ''' proxies a network to so that can use predict_n() which async (like really async - say goodbye to the stack '''

    def __init__(self, nn, batch_size=256):
        self.nn = nn
        self.batch_size = batch_size

        self.buf_states = []
        self.buf_lead_role_indexes = []

        self.main = greenlet.getcurrent()
        self.runnables = deque()
        self.requestors = []

        # states
        self.before_time = -1
        self.after_time = -1
        self.acc_python_time = 0
        self.acc_predict_time = 0
        self.count_prediction_size = Counter()

    def add_runnable(self, fn, arg=None):
        self.runnables.append((greenlet.greenlet(fn), arg))

    def predict_n(self, states, lead_role_indexes):
        # cur is running inside player
        cur = greenlet.getcurrent()

        self.buf_states += states
        self.buf_lead_role_indexes += lead_role_indexes
        self.requestors += [cur] * len(states)

        # switching to main
        return self.main.switch(cur)

    def predict_1(self, state, lead_role_index):
        return self.predict_n([state], [lead_role_index])[0]

    def do_predictions(self):
        results = []
        while self.buf_states:
            next_states = self.buf_states[:self.batch_size]
            next_lead_role_indexes = self.buf_lead_role_indexes[:self.batch_size]

            self.before_time = time.time()
            if self.after_time > 0:
                self.acc_python_time += self.before_time - self.after_time

            self.count_prediction_size[len(next_states)] += 1
            results += self.nn.predict_n(next_states, next_lead_role_indexes)

            self.after_time = time.time()
            self.acc_predict_time += self.after_time - self.before_time

            self.buf_states = self.buf_states[self.batch_size:]
            self.buf_lead_role_indexes = self.buf_lead_role_indexes[self.batch_size:]

        # go through results, assign them to particular requestor and unblock requestor
        assert len(results) == len(self.requestors)

        cur_requestor = None
        cur_result = None
        for i in range(len(results)):
            if cur_requestor is None:
                cur_requestor = self.requestors[i]
                cur_result = [results[i]]

            elif cur_requestor != self.requestors[i]:
                self.runnables.append((cur_requestor, cur_result))

                cur_requestor = self.requestors[i]
                cur_result = [results[i]]
            else:
                cur_result.append(results[i])

        if cur_requestor is not None:
            self.runnables.append((cur_requestor, cur_result))

        self.requestors = []

    def run(self):
        # this is us
        self.before_time = self.after_time -1
        self.acc_predict_time = self.acc_python_time = 0
        self.count_prediction_size = Counter()

        self.main = greenlet.getcurrent()

        # magic below:
        while True:
            if not self.runnables:
                # done
                if not self.buf_states:
                    break
                self.do_predictions()

            else:
                g, arg = self.runnables.popleft()
                g.switch(arg)

                if len(self.buf_states) >= self.batch_size:
                    self.do_predictions()


def create_scheduler(game, generation, batch_size=256):
    game_info = lookup.by_name(game)
    bases_config = bases.get_config(game_info.game, game_info.model, generation)
    nn = bases_config.create_network()
    nn.load()
    return NetworkScheduler(nn, batch_size=batch_size)
