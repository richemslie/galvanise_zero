from builtins import super

import time

import numpy as np

import ggpzero_interface


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
                                   transformer.channel_size * transformer.raw_channels_per_state)

    # build it up
    for b in transformer.base_infos:
        if b.channel is not None:
            index = transformer.channel_size * b.channel + b.y_idx * transformer.num_rows + b.x_idx
        else:
            index = -1
        c_transformer.add_base_info(b.channel is not None, index)

    for indx in transformer.control_states:
        c_transformer.add_control_state(indx)

    return c_transformer


class SupervisorBase(object):
    POLL_AGAIN = "poll_again"

    def __init__(self, sm, nn, batch_size=1024, use_threaded=False):
        self.sm = sm
        self.nn = nn

        # maximum number of batches we can do
        self.batch_size = batch_size

        transformer = nn.gdl_bases_transformer
        self.c_transformer = create_c_transformer(transformer)

        # XXX option for inline / threaded
        if use_threaded:
            assert False, "TODO"
        else:
            self.c_supervisor = ggpzero_interface.Supervisor(sm_to_ptr(sm),
                                                             self.c_transformer,
                                                             batch_size,
                                                             transformer.policy_dist_count,
                                                             transformer.policy_1_index_start)

        self.poll_last = None

        # stats
        self.total_predictions = 0
        self.acc_time_c_supervisor = 0
        self.acc_time_prediction = 0

    def create_job(self, num_selfplays, base_iterations, sample_iterations):
        self.c_supervisor.start_self_play_test(num_selfplays, base_iterations, sample_iterations)

    def poll(self, do_stats=False):
        ''' POLL_AGAIN is returned, to indicate we need to call poll() again. '''

        if self.poll_last is None:
            dummy = np.zeros(0)
            policy_dists, final_scores = dummy, dummy
        else:
            policy_dists, final_scores = self.poll_last

        if do_stats:
            s0 = time.time()

        pred_array = self.c_supervisor.poll(policy_dists, final_scores)
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
        policy_dists, final_scores = self.nn.get_model().predict(pred_array,
                                                                 batch_size=num_predictions)

        self.poll_last = policy_dists, final_scores

        if do_stats:
            s2 = time.time()

            self.total_predictions += num_predictions
            self.acc_time_c_supervisor += s1 - s0
            self.acc_time_prediction += s2 - s1

        return self.POLL_AGAIN

    def poll_loop(self, cb=None, do_stats=False):
        ''' will poll until we are done '''
        while self.poll(do_stats=do_stats) == self.POLL_AGAIN:
            if cb is not None:
                cb()

    def dump_stats(self):
        print "predictions", self.total_predictions
        print "acc_time_c_supervisor", self.acc_time_c_supervisor
        print "acc_time_prediction", self.acc_time_prediction


class SupervisorSelfPlay(SupervisorBase):
    def __init__(self, sm, nn, batch_size=1024):
        super().__init__(sm, nn, batch_size=batch_size)

    # start_test = self.c_supervisor.start_self_play_test


class SupervisorPlay(SupervisorBase):
    def __init__(self, sm, nn):
        super().__init__(sm, nn, batch_size=1)

        for name in "start reset apply_move move get_move".split():
            name = "player_" + name
            setattr(self, name, getattr(self.c_supervisor, name))
