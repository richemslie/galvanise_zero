from builtins import super

import time

import attr
import numpy as np

import ggpzero_interface
from ggpzero.defs import confs, datadesc


def sm_to_ptr(sm):
    from ggplib.interface import ffi
    cffi_ptr = sm.c_statemachine
    ptr_as_long = int(ffi.cast("intptr_t", cffi_ptr))
    return ptr_as_long


def joint_move_to_ptr(joint_move):
    from ggplib.interface import ffi
    cffi_ptr = joint_move.c_joint_move
    ptr_as_long = int(ffi.cast("intptr_t", cffi_ptr))
    return ptr_as_long


def basestate_to_ptr(basestate):
    from ggplib.interface import ffi
    cffi_ptr = basestate.c_base_state
    ptr_as_long = int(ffi.cast("intptr_t", cffi_ptr))
    return ptr_as_long


def create_c_transformer(transformer):
    TransformerClz = ggpzero_interface.GdlBasesTransformer
    c_transformer = TransformerClz(transformer.channel_size,
                                   transformer.raw_channels_per_state,
                                   transformer.num_of_controls_channels,
                                   transformer.num_previous_states,
                                   transformer.policy_dist_count)

    # build it up
    for b in transformer.board_space:
        index = transformer.channel_size * b.channel_id + b.y_idx * transformer.num_rows + b.x_idx
        c_transformer.add_board_base(b.base_indx, index)

    for c in transformer.control_space:
        c_transformer.add_control_base(c.base_indx, c.channel_id, c.value)

    return c_transformer


class PollerBase(object):
    POLL_AGAIN = "poll_again"

    def __init__(self, sm, nn, batch_size=1024, sleep_between_poll=-1):
        self.sm = sm
        self.nn = nn

        # maximum number of batches we can do
        self.batch_size = batch_size

        # if defined, will sleep between polls
        self.sleep_between_poll = sleep_between_poll

        self.poll_last = None
        self.reset_stats()

    def _get_poller(self):
        raise NotImplemented

    def reset_stats(self):
        self.num_predictions_calls = 0
        self.total_predictions = 0
        self.acc_time_polling = 0
        self.acc_time_prediction = 0

    def poll(self, do_stats=False):
        ''' POLL_AGAIN is returned, to indicate we need to call poll() again. '''

        transformer = self.nn.gdl_bases_transformer
        expect_num_arrays = len(transformer.policy_dist_count) + 1

        if self.poll_last is None:
            dummy = np.zeros(0)
            arrays = [dummy for _ in range(expect_num_arrays)]
        else:
            arrays = list(self.poll_last)

        assert len(arrays) == expect_num_arrays

        if do_stats:
            s0 = time.time()

        pred_array = self._get_poller().poll(len(arrays[0]), arrays)
        if pred_array is None:
            self.poll_last = None
            return

        if do_stats:
            s1 = time.time()

        t = self.nn.gdl_bases_transformer
        num_predictions = len(pred_array) / (t.num_channels * t.channel_size)
        assert num_predictions <= self.batch_size

        # make sure array is correct shape for keras/tensorflow (no memory is allocated)
        pred_array = pred_array.reshape(num_predictions, t.num_channels, t.num_cols, t.num_rows)
        self.poll_last = self.nn.get_model().predict_on_batch(pred_array)

        if do_stats:
            s2 = time.time()

            self.num_predictions_calls += 1
            self.total_predictions += num_predictions
            self.acc_time_polling += s1 - s0
            self.acc_time_prediction += s2 - s1

        return self.POLL_AGAIN

    def poll_loop(self, cb=None, do_stats=False):
        ''' will poll until we are done '''

        # calls back every n times
        cb_every_n = 100
        count = 1
        while self.poll(do_stats=do_stats) == self.POLL_AGAIN:
            if count % cb_every_n == 0:
                if cb is not None:
                    if cb():
                        break
            count += 1
            if self.sleep_between_poll > 0:
                time.sleep(self.sleep_between_poll)

    def update_nn(self, nn):
        self.nn = nn

    def dump_stats(self):
        print "num of prediction calls", self.num_predictions_calls
        print "predictions", self.total_predictions
        print "acc_time_polling", self.acc_time_polling
        print "acc_time_prediction", self.acc_time_prediction


class PlayPoller(PollerBase):
    def __init__(self, sm, nn, conf):
        super().__init__(sm, nn, batch_size=1)
        transformer = nn.gdl_bases_transformer
        self.c_transformer = create_c_transformer(transformer)
        self.c_player = ggpzero_interface.Player(sm_to_ptr(sm),
                                                 self.c_transformer,
                                                 conf)

        for name in "reset apply_move move get_move".split():
            name = "player_" + name
            setattr(self, name, getattr(self.c_player, name))

    def _get_poller(self):
        return self.c_player


class Supervisor(PollerBase):
    def __init__(self, sm, nn, batch_size=1024, sleep_between_poll=-1, workers=None):

        transformer = nn.gdl_bases_transformer
        self.c_transformer = create_c_transformer(transformer)

        self.c_supervisor = ggpzero_interface.Supervisor(sm_to_ptr(sm),
                                                         self.c_transformer,
                                                         batch_size)
        if workers:
            self.c_supervisor.set_num_workers(workers)

        self.bs_for_unique_states = sm.new_base_state()

        super().__init__(sm, nn, batch_size=batch_size, sleep_between_poll=sleep_between_poll)

    def _get_poller(self):
        return self.c_supervisor

    def start_self_play(self, conf, num_workers):
        assert isinstance(conf, confs.SelfPlayConfig)
        return self.c_supervisor.start_self_play(num_workers, attr.asdict(conf))

    def fetch_samples(self):
        res = self.c_supervisor.fetch_samples()
        if res:
            return [datadesc.Sample(**d) for d in res]
        else:
            return []

    def add_unique_state(self, l):
        self.bs_for_unique_states.from_list(l)
        self.c_supervisor.add_unique_state(basestate_to_ptr(self.bs_for_unique_states))

    def clear_unique_states(self):
        self.c_supervisor.clear_unique_states()
