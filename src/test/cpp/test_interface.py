import os
import time

import numpy as np

from ggplib.db import lookup

from ggpzero.nn.manager import get_manager

# XXX shouldn't import this directly
import ggpzero_interface
from ggpzero_interface import SupervisorDummy

from ggpzero.util import cppinterface


def float_formatter0(x):
    return "%.0f" % x


def float_formatter1(x):
    return "%.2f" % x


def setup(self):
    from ggplib.util.init import setup_once
    setup_once()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_cffi_to_cmodule():
    # XXX this is not a valid test anymore???
    wrap = ggpzero_interface.start(42, 42)

    info = lookup.by_name("breakthrough")
    sm_ptr = cppinterface.sm_to_ptr(info.get_sm())
    print sm_ptr

    wrap.test_sm(sm_ptr)


def test_transformer():

    # game = "breakthrough"
    # gen = "v5_84"

    game = "reversi"
    gen = "v6_65"

    game_info = lookup.by_name(game)
    sm = game_info.get_sm()

    man = get_manager()
    t = man.get_transformer(game)

    # create transformer wrapper object
    c_transformer = cppinterface.create_c_transformer(t)

    nn = man.load_network(game, gen)
    verbose = False

    total_predictions = 0
    total_s0 = 0
    total_s1 = 0
    total_s2 = 0
    for ii in range(1000):
        start = time.time()
        array = c_transformer.test(cppinterface.sm_to_ptr(sm))
        total_s0 += time.time() - start

        sz = len(array) / (t.num_channels * t.channel_size)

        total_predictions += sz

        array = array.reshape(sz, t.num_channels, t.num_cols, t.num_rows)
        total_s1 += time.time() - start

        if verbose:
            np.set_printoptions(threshold=np.inf, formatter={'float_kind' : float_formatter0})
            print array

        # test we can actually predict
        res = nn.get_model().predict(array, batch_size=sz)
        # print res[0].shape
        # print res[1].shape

        total_s2 += time.time() - start

        if verbose:
            np.set_printoptions(threshold=np.inf, formatter={'float_kind' : float_formatter1})
            print res

    print total_predictions, "time taken", [s * 1000 for s in (total_s0, total_s1, total_s2)]


def test_cgreenlets():
    ggpzero_interface.cgreenlet_test()


def test_dummy_supervisor():

    # game = "breakthrough"
    # gen = "v5_84"
    game = "reversi"
    gen = "v6_65"
    # game = "breakthroughSmall"
    # gen = "v1_0"
    # game = "connectFour"
    # gen = "v6_27"

    game_info = lookup.by_name(game)
    sm = game_info.get_sm()

    man = get_manager()
    t = man.get_transformer(game)

    # create transformer wrapper object
    c_transformer = cppinterface.create_c_transformer(t)

    nn = man.load_network(game, gen)

    for batch_size in (1, 128, 1024):

        d = SupervisorDummy(cppinterface.sm_to_ptr(sm), c_transformer, batch_size,
                            t.policy_dist_count, t.policy_1_index_start)

        dummy = np.zeros(0)
        policy_dists, final_scores = dummy, dummy
        total_preds = 0
        acc_0 = 0
        acc_1 = 0
        while True:
            s = time.time()
            pred_array = d.test(policy_dists, final_scores)
            if pred_array is None:
                break

            sz = len(pred_array) / (t.num_channels * t.channel_size)
            total_preds += sz
            pred_array = pred_array.reshape(sz, t.num_channels, t.num_cols, t.num_rows)
            e = time.time()
            acc_0 += e - s
            policy_dists, final_scores = nn.get_model().predict(pred_array, batch_size=sz)
            acc_1 += time.time() - e

        print "total predictions", total_preds
        print "times %.2f %.2f" % (acc_0, acc_1)
        relative_time = (acc_0 + acc_1) / batch_size
        print "relative time per game %.1f" % relative_time
