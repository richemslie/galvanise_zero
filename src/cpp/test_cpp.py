import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np


def test_cffi_to_cmodule():
    from ggplib.db import lookup
    from ggplib.interface import ffi
    import ggpzero_interface
    wrap = ggpzero_interface.start(42, 42)

    info = lookup.by_name("breakthrough")

    cffi_ptr = info.get_sm().c_statemachine
    ptr_as_long = int(ffi.cast("intptr_t", cffi_ptr))
    print wrap.test_sm(ptr_as_long)


def sm_to_ptr(sm):
    from ggplib.interface import ffi
    cffi_ptr = sm.c_statemachine
    ptr_as_long = int(ffi.cast("intptr_t", cffi_ptr))
    return ptr_as_long


def create_c_transformer(transformer):
    from ggpzero_interface import GdlBasesTransformer
    c_transformer = GdlBasesTransformer(transformer.channel_size,
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


def test_transformer():
    from ggplib.util.init import setup_once
    setup_once()

    # game = "breakthrough"
    # gen = "v5_84"

    game = "reversi"
    gen = "v6_65"

    from ggplib.db import lookup
    game_info = lookup.by_name(game)
    sm = game_info.get_sm()

    from ggpzero.nn.manager import get_manager
    man = get_manager()
    t = man.get_transformer(game)

    # create transformer wrapper object
    c_transformer = create_c_transformer(t)

    float_formatter0 = lambda x: "%.0f" % x
    float_formatter1 = lambda x: "%.0f" % x

    nn = man.load_network(game, gen)
    verbose = False

    total_predictions = 0
    total_s0 = 0
    total_s1 = 0
    total_s2 = 0
    for ii in range(1000):
        start = time.time()
        array = c_transformer.test(sm_to_ptr(sm))
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
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero_interface import cgreenlet_test
    cgreenlet_test()


def test_dummy_supervisor():
    from ggplib.util.init import setup_once
    setup_once()

    # game = "breakthrough"
    # gen = "v5_84"
    # game = "reversi"
    # gen = "v6_65"
    game = "breakthroughSmall"
    gen = "v1_0"

    from ggplib.db import lookup
    game_info = lookup.by_name(game)
    sm = game_info.get_sm()

    from ggpzero.nn.manager import get_manager
    man = get_manager()
    t = man.get_transformer(game)

    # create transformer wrapper object
    c_transformer = create_c_transformer(t)

    from ggpzero_interface import SupervisorDummy
    d = SupervisorDummy(sm_to_ptr(sm), c_transformer, 1024,
                        t.policy_dist_count, t.policy_1_index_start)

    nn = man.load_network(game, gen)

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
