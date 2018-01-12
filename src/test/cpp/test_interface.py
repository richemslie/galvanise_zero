
import os
import time

import numpy as np

from ggplib.db import lookup

from ggpzero.nn.manager import get_manager
from ggpzero.util import cppinterface


def float_formatter0(x):
    return "%.0f" % x


def float_formatter1(x):
    return "%.2f" % x


def setup():
    from ggplib.util.init import setup_once
    setup_once()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_call_fn():
    import ggpzero_interface
    ggpzero_interface.hello_test('bobthebuilder')


def test_transformer():
    game = "breakthrough"

    game_info = lookup.by_name(game)
    sm = game_info.get_sm()

    man = get_manager()
    t = man.get_transformer(game)

    # create transformer wrapper object
    c_transformer = cppinterface.create_c_transformer(t)

    nn = man.create_new_network(game)
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


def test_inline_supervisor_creation():
    games = "breakthrough reversi breakthroughSmall connectFour".split()

    man = get_manager()

    for game in games:
        game_info = lookup.by_name(game)

        # get statemachine
        sm = game_info.get_sm()
        nn = man.create_new_network(game)

        for batch_size in (1, 128, 1024):
            supervisor = cppinterface.SupervisorSelfPlay(sm, nn, batch_size=batch_size)
            continue


def test_inline_supervisor_run():
    game = "breakthroughSmall"
    man = get_manager()
    nn = man.create_new_network(game, "smaller")

    game_info = lookup.by_name(game)

    supervisor = cppinterface.SupervisorSelfPlay(game_info.get_sm(), nn, batch_size=4096)
    supervisor.create_job(4096, 42, 42)
    supervisor.poll_loop(do_stats=True)
    supervisor.dump_stats()

    time.sleep(0.5)

    supervisor.create_job(4096, 42, 42)
    supervisor.poll_loop(do_stats=True)
    supervisor.dump_stats()


def test_inline_supervisor_speed():
    game = "reversi"
    gen = "v6_65"

    man = get_manager()
    nn = man.load_network(game, gen)
    print nn.summary()
    game_info = lookup.by_name(game)

    supervisor = cppinterface.SupervisorSelfPlay(game_info.get_sm(), nn, batch_size=4096)
    supervisor.create_job(1024, 500, 500)
    supervisor.poll_loop(do_stats=True)
    supervisor.dump_stats()


def test_inline_more_games():
    ' tests where we have more games than batch_size '
    batch_size = 128
    games_size = 512

    game = "breakthroughSmall"
    man = get_manager()
    nn = man.create_new_network(game, "smaller")

    game_info = lookup.by_name(game)

    supervisor = cppinterface.SupervisorSelfPlay(game_info.get_sm(), nn, batch_size=batch_size)
    supervisor.create_job(games_size, 21, 21)
    supervisor.poll_loop(do_stats=True)
    supervisor.dump_stats()
