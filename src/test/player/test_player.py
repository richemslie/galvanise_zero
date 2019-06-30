import os

from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db import lookup

from ggpzero.util import attrutil
from ggpzero.defs import confs, templates
from ggpzero.nn.manager import get_manager

from ggpzero.player.puctplayer import PUCTPlayer

GAME = "breakthroughSmall"
RANDOM_GEN = "rand_0"

GOOD_GEN1 = "x1_132"


def setup():
    import tensorflow as tf

    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    import numpy as np
    np.set_printoptions(threshold=100000)

    man = get_manager()
    if not man.can_load(GAME, RANDOM_GEN):
        network = man.create_new_network(GAME)
        man.save_network(network, RANDOM_GEN)


def get_game_info(board_size):
    # add players
    if board_size == 8:
        game = "breakthrough"
    else:
        assert board_size == 6
        game = "breakthroughSmall"

    return lookup.by_name(game)


def pretty_board(board_size, sm):
    ' pretty print board current state of match '

    from ggplib.util.symbols import SymbolFactory
    as_str = get_game_info(board_size).model.basestate_to_str(sm.get_current_state())
    sf = SymbolFactory()
    states = sf.to_symbols(as_str)
    mapping = {}
    control = None
    for s in list(states):
        if s[1][0] == "control":
            control = s[1][1]
        else:
            if board_size == 8:
                assert s[1][0] == "cellHolds"
            else:
                assert s[1][0] == "cell"

            key = int(s[1][1]), int(s[1][2])
            mapping[key] = s[1][3]

    lines = []
    line_len = board_size * 4 + 1
    lines.append("    +" + "-" * (line_len - 2) + "+")
    for i in reversed(range(1, board_size + 1)):
        ll = [" %s  |" % i]
        for j in reversed(range(1, board_size + 1)):
            key = j, i
            if key in mapping:
                if mapping[key] == "black":
                    ll.append(" %s |" % u"\u2659")
                else:
                    assert mapping[key] == "white"
                    ll.append(" %s |" % u"\u265F")
            else:
                ll.append("   |")

        lines.append("".join(ll))
        if i > 1:
            lines.append("    " + "-" * line_len)

    lines.append("    +" + "-" * (line_len - 2) + "+")
    if board_size == 8:
        lines.append("     " + ' '.join(' %s ' % c for c in 'abcdefgh'))
    else:
        lines.append("     " + ' '.join(' %s ' % c for c in 'abcdef'))

    print
    print
    print "\n".join(lines)
    print "Control:", control


def play(player_white, player_black, move_time=0.5):
    gm = GameMaster(lookup.by_name(GAME), verbose=True)
    gm.add_player(player_white, "white")
    gm.add_player(player_black, "black")

    gm.start(meta_time=15, move_time=move_time)

    move = None
    while not gm.finished():

        # print out the board
        pretty_board(6, gm.sm)

        move = gm.play_single_move(last_move=move)

    gm.finalise_match(move)


def test_random():
    # add two players
    # simplemcts vs RANDOM_GEN
    pymcs = get.get_player("simplemcts")
    pymcs.max_run_time = 0.25

    eval_config = templates.base_puct_config(verbose=True,
                                             max_dump_depth=1)
    puct_config = confs.PUCTPlayerConfig("gzero",
                                         True,
                                         100,
                                         0,
                                         RANDOM_GEN,
                                         eval_config)

    attrutil.pprint(puct_config)

    puct_player = PUCTPlayer(puct_config)

    play(pymcs, puct_player)


def test_trained():
    # simplemcts vs GOOD_GEN
    simple = get.get_player("simplemcts")
    simple.max_run_time = 0.5

    eval_config = confs.PUCTEvaluatorConfig(verbose=True,
                                            puct_constant=0.85,
                                            puct_constant_root=3.0,

                                            dirichlet_noise_pct=-1,

                                            fpu_prior_discount=0.25,
                                            fpu_prior_discount_root=0.15,

                                            choose="choose_temperature",
                                            temperature=2.0,
                                            depth_temperature_max=10.0,
                                            depth_temperature_start=0,
                                            depth_temperature_increment=0.75,
                                            depth_temperature_stop=1,
                                            random_scale=1.0,
                                            batch_size=1,
                                            max_dump_depth=1)

    puct_config = confs.PUCTPlayerConfig("gzero",
                                         True,
                                         200,
                                         0,
                                         GOOD_GEN1,
                                         eval_config)
    attrutil.pprint(puct_config)

    puct_player = PUCTPlayer(puct_config)

    play(simple, puct_player)
    #play(puct_player, simple)
