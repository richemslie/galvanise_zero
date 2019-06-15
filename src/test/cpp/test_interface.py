''' note these tests really need a GPU.  XXX add a skip, or CPU versions of test. '''

import os
import time

import numpy as np
import py.test

import tensorflow as tf

from ggplib.db import lookup

from ggpzero.nn.manager import get_manager
from ggpzero.util import cppinterface
from ggpzero.defs import confs, templates


def float_formatter0(x):
    return "%.0f" % x


def float_formatter1(x):
    return "%.2f" % x


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    np.set_printoptions(threshold=100000)


def do_transformer(num_previous_states):
    game = "breakthrough"

    game_info = lookup.by_name(game)
    sm = game_info.get_sm()

    man = get_manager()

    # only multiple_policy_heads supported in c++
    generation_descr = templates.default_generation_desc(game,
                                                         multiple_policy_heads=True,
                                                         num_previous_states=num_previous_states)

    t = man.get_transformer(game, generation_descr)

    # create transformer wrapper object
    c_transformer = cppinterface.create_c_transformer(t)

    nn = man.create_new_network(game, "small", generation_descr)
    verbose = True

    total_predictions = 0
    total_s0 = 0
    total_s1 = 0
    total_s2 = 0
    for ii in range(10):
        print ii
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


def test_transformer():
    do_transformer(0)


def test_transformer_with_prev_states():
    do_transformer(3)


def test_inline_supervisor_creation():
    games = "breakthrough reversi breakthroughSmall connectFour".split()

    man = get_manager()

    for game in games:
        game_info = lookup.by_name(game)

        # get statemachine
        sm = game_info.get_sm()

        # only multiple_policy_heads supported in c++
        generation_descr = templates.default_generation_desc(game,
                                                             multiple_policy_heads=True)

        nn = man.create_new_network(game, "small", generation_descr)

        for batch_size in (1, 128, 1024):
            supervisor = cppinterface.Supervisor(sm, nn, batch_size=batch_size)
            supervisor = supervisor
            continue


def setup_c4(batch_size=1024):
    game = "connectFour"
    man = get_manager()

    # only multiple_policy_heads supported in c++
    generation_descr = templates.default_generation_desc(game,
                                                         multiple_policy_heads=True)

    nn = man.create_new_network(game, "small", generation_descr)

    game_info = lookup.by_name(game)

    supervisor = cppinterface.Supervisor(game_info.get_sm(), nn, batch_size=batch_size)

    conf = templates.selfplay_config_template()
    return supervisor, conf


def get_ctx():
    class X:
        pass
    return X()


def do_test(batch_size, get_sample_count, num_workers=0):
    supervisor, conf = setup_c4(batch_size=batch_size)
    supervisor.start_self_play(conf, num_workers)

    nonlocal = get_ctx
    nonlocal.samples = []

    def cb():
        new_samples = supervisor.fetch_samples()
        if new_samples:
            nonlocal.samples += new_samples
            print "Total rxd", len(nonlocal.samples)

            if len(nonlocal.samples) > get_sample_count:
                return True

    supervisor.poll_loop(do_stats=True, cb=cb)
    supervisor.dump_stats()

    # should be resumable
    nonlocal.samples = []
    supervisor.reset_stats()
    supervisor.poll_loop(do_stats=True, cb=cb)
    supervisor.dump_stats()


def test_inline_one():
    do_test(batch_size=1, get_sample_count=42, num_workers=0)


def test_inline_batched():
    do_test(batch_size=1024, get_sample_count=5000, num_workers=1)


def test_workers_batched():
    do_test(batch_size=1024, get_sample_count=5000, num_workers=2)


def test_inline_unique_states():
    py.test.skip("too slow without GPU")
    get_sample_count = 250
    supervisor, conf = setup_c4(batch_size=1)
    supervisor.start_self_play(conf, num_workers=0)

    nonlocal = get_ctx
    nonlocal.samples = []
    nonlocal.added_unique_states = False

    def cb():
        new_samples = supervisor.fetch_samples()
        if new_samples:
            nonlocal.samples += new_samples
            print "Total rxd", len(nonlocal.samples)

            if len(nonlocal.samples) > get_sample_count:
                return True

        if not nonlocal.added_unique_states:
            if len(nonlocal.samples) > 100:
                nonlocal.added_unique_states = True
                for s in nonlocal.samples:
                    supervisor.add_unique_state(s.state)

            elif len(nonlocal.samples) > 20:
                print 'clearing unique states'
                supervisor.clear_unique_states()

    supervisor.poll_loop(do_stats=True, cb=cb)
    supervisor.dump_stats()

    supervisor.clear_unique_states()
    nonlocal.samples = []
    nonlocal.added_unique_states = False

    supervisor.poll_loop(do_stats=True, cb=cb)
    supervisor.dump_stats()
