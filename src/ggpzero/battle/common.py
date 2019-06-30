from builtins import super

import os
import re
import pdb
import sys
import time
import pprint
import hashlib
import traceback

import tensorflow as tf

from ggplib.util.init import setup_once
from ggplib.util import log

from ggplib.player import get
from ggplib.player.base import MatchPlayer
from ggplib.player.gamemaster import GameMaster

from ggpzero.util.keras import init
from ggpzero.util import attrutil as at

from ggpzero.defs import confs

from ggpzero.player.puctplayer import PUCTPlayer


def setup(log_name_base=None):
    if log_name_base is not None:
        setup_once(log_name_base=log_name_base)
    else:
        setup_once(log_name_base=log_name_base)
    init()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)


def run(main_fn, log_name_base=None):
    try:
        setup(log_name_base=log_name_base)
        main_fn()

    except Exception as exc:
        print exc
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


class PromptPlayer(MatchPlayer):
    def __init__(self):
        super().__init__("prompt player")

    def on_apply_move(self, moves):
        pass

    def on_next_move(self, finish_time):
        sm = self.match.sm
        ri = self.match.our_role_index
        ls = sm.get_legal_state(ri)

        def f(i):
            return sm.legal_to_move(ri, ls.get_legal(i))

        if ls.get_count() == 1 and f(0) == "noop":
            gdl_role_move = "noop"

        else:
            while True:
                try:
                    print "Make your move?",
                    move = raw_input()
                    move = move.strip().lower()
                    print "you played", move
                    low, high = move[0].lower(), int(move[1:])
                    assert 'a' <= low <= 'z'
                    break
                except Exception as exc:
                    print "what??", exc

            gdl_role_move = "(place %s %s)" % (low, high)

        print "the gdl move", gdl_role_move

        the_moves = [f(ii) for ii in range(ls.get_count())]
        choice = the_moves.index(gdl_role_move)
        return ls.get_legal(choice)


def get_puct_config(gen, **kwds):
    eval_config = confs.PUCTEvaluatorV2Config(verbose=True,
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

                                              max_dump_depth=2,

                                              minimax_backup_ratio=0.75,

                                              top_visits_best_guess_converge_ratio=0.8,

                                              think_time=2.0,
                                              converged_visits=2000,

                                              batch_size=32,
                                              extra_uct_exploration=-1.0)

    config = confs.PUCTPlayerConfig(name="puct",
                                    verbose=True,
                                    generation=gen,
                                    playouts_per_iteration=-1,
                                    playouts_per_iteration_noop=0,
                                    evaluator_config=eval_config)

    for k, v in kwds.items():
        updated = False
        if at.has(eval_config, k):
            updated = True
            setattr(eval_config, k, v)

        if at.has(config, k):
            updated = True
            setattr(config, k, v)

        if not updated:
            log.warning("Unused setting %s:%s" % (k, v))

    print "get_puct_config:"
    at.pprint(config)

    return config


class MatchTooLong(Exception):
    ""


def get_player(player_type, move_time, gen=None, **extra_opts):
    if gen:
        assert player_type == "puct", "unknown player type: %s" % player_type
        return PUCTPlayer(get_puct_config(gen, **extra_opts))

    add_extra_opts = False
    assert gen is None
    if player_type == "s":
        player = get.get_player("simplemcts")
        player.max_tree_search_time = move_time
        add_extra_opts = True
    elif player_type == "m":
        player = get.get_player("pymcs")
        player.max_run_time = move_time
        add_extra_opts = True

    elif player_type == "r":
        player = get.get_player("random")

    elif player_type == "g":
        from gurgeh.player import GurgehPlayer
        player = GurgehPlayer()
        add_extra_opts = True

        player.max_tree_search_time = move_time
    else:
        assert False, "unknown player type: %s" % player_type

    if add_extra_opts:
        for k, v in extra_opts.items():
            setattr(player, k, v)

    return player


def parse_sgf(text):
    tokens = re.split(r'([a-zA-Z]+\[[^\]]+\])', text)
    tokens = [t for t in tokens if ']' in t.strip()]

    def game_info():
        for t in tokens:
            key, value = re.search(r'([a-zA-Z]+)\[([^\]]+)\]', t).groups()
            yield key, value

    moves = []
    for k, v in game_info():
        if k.upper() in "WB":
            moves.append(v)

    return moves


class MatchGameInfo(object):
    def __init__(self, game_info):
        self.game_info = game_info
        self.max_match_length = 1000

    @property
    def name(self):
        return self.game_info.game

    def print_board(self, sm):
        print ""

    def convert_move_to_gdl(self, move):
        return move

    def make_moves(self, moves):
        sm = self.game_info.get_sm()
        sm.reset()

        # get some objects
        base_state = sm.get_initial_state()

        joint_moves = []

        def f(ri, ls, i):
            return sm.legal_to_move(ri, ls.get_legal(i))

        def next_sm(role_index, gdl_str_move):
            joint_move = sm.get_joint_move()
            for ri in range(2):
                ls = sm.get_legal_state(ri)
                if ls.get_count() == 1 and f(ri, ls, 0) == "noop":
                    choice = 0
                else:
                    the_moves = [f(ri, ls, ii) for ii in range(ls.get_count())]
                    choice = the_moves.index(gdl_str_move)

                joint_move.set(ri, ls.get_legal(choice))

            joint_moves.append(joint_move)

            # update state machine
            sm.next_state(joint_move, base_state)
            sm.update_bases(base_state)

        match_depth = 0
        role_index = 0

        for move in moves:
            # convert fn can yield more than one move (useful for games like amazons etc)
            for gdl_move in self.convert_move_to_gdl(move):
                next_sm(role_index, gdl_move)
                match_depth += 1

            role_index = 1 if role_index == 0 else 0

        return role_index, joint_moves, base_state, match_depth

    def get_gm(self, players, move_time, moves):
        gm = GameMaster(self.game_info)
        sm = self.game_info.get_sm()

        for player, role in zip(players, sm.get_roles()):
            gm.add_player(player, role)

        if moves:
            _, _, initial_state, match_depth = self.make_moves(moves)
        else:
            match_depth = 0
            initial_state = None

        gm.reset()
        if initial_state:
            gm.start(meta_time=15,
                     initial_basestate=initial_state,
                     game_depth=match_depth,  # XXX rename to match_depth on gm
                     move_time=move_time)
        else:
            gm.start(meta_time=15,
                     move_time=move_time)

        return gm, match_depth

    def play_cb(self, players, match_depth):
        ''' called before playing move '''
        pass

    def play(self, players, move_time, max_moves=None, moves=None, resign_score=None, verbose=True):
        if verbose:
            print "max_moves", max_moves
        start_time = time.time()

        # add players
        gm, match_depth = self.get_gm(players, move_time, moves)
        sm = self.game_info.get_sm()

        save_moves = []

        move = None
        resigned = False
        resigned_ri = -1
        while not gm.finished():

            # here we print out the board, if one is defined
            if verbose:
                self.print_board(gm.sm)

            # allow any update to players config
            self.play_cb(players, match_depth)

            move = gm.play_single_move(last_move=move)

            # store moves/probabilities
            move_info = []
            for ri, p in enumerate(players):
                prob = -1
                node_count = -1
                if hasattr(p, "last_probability"):
                    prob = p.last_probability
                    node_count = p.last_node_count

                elif hasattr(p, "before_apply_info"):
                    s = p.before_apply_info()
                    # this is rough...
                    try:
                        d = eval("dict(%s)" % s)
                        prob = d["best_prob"]
                        node_count = d["nodes"]
                    except Exception, exc:
                        print exc

                # resign?  Only resign if move was not noop
                if resign_score is not None and move[ri] != 'noop':
                    if float(prob) > 0 and float(prob) < resign_score:
                        resigned = True
                        resigned_ri = ri

                move_info.append("nodes %d / prob %.3f" % (node_count, float(prob)))

            match_depth += 1
            save_moves.append((match_depth, move, move_info))

            if resigned:
                resign_move = ['', '']
                resign_move[resigned_ri] = "resign"
                save_moves.append((match_depth, resign_move, ""))
                break

            if max_moves is not None:
                if len(save_moves) == max_moves:
                    break

            if match_depth > self.max_match_length:
                raise MatchTooLong()

        if gm.finished():
            gm.finalise_match(move)

        end_time = time.time()

        if verbose:
            self.print_board(gm.sm)

        # return the result
        scores = []
        if gm.finished():
            for player, role in zip(players, sm.get_roles()):
                scores.append((player.get_name(), gm.get_score(role)))

        if resigned:
            assert not scores
            scores.append((players[0].get_name(), 100 if resigned_ri == 1 else 0))
            scores.append((players[1].get_name(), 0 if resigned_ri == 1 else 100))

        # hang on to the sm... just in case we want to do something with it
        self.gm = gm

        time_taken = end_time - start_time
        return time_taken, scores, save_moves

    def parse_sgf(self, sgf):
        return parse_sgf(sgf)

    def export(self, players, result):
        for match_depth, move, move_info in result:
            print move



def dump_results(game, players, results):
    # for now XXX, should allow any number of players
    assert len(players) == 2
    player_0, player_1 = players

    run_id = hashlib.md5(hashlib.sha1("%.5f" % time.time()).hexdigest()).hexdigest()[:6]
    filename = "%s_games_%s__vs__%s_%s.log" % (game,
                                               player_0.get_name(),
                                               player_1.get_name(), run_id)

    with open(filename, "w") as f:
        for p in players:
            print p.get_name()
            print >>f, p.get_name()

            if isinstance(p, PUCTPlayer):
                print at.pformat(p.conf)
                print >>f, at.pformat(p.conf)

            print "------"
            print >>f, "------"

        for time_taken, score, moves in results:
            print "------"
            print >>f, "------"

            time_taken = "time taken for match: %s seconds" % int(time_taken)
            print time_taken
            print >>f, time_taken

            print "scores:", score
            print >>f, "scores:", score

            s = pprint.pformat(moves)
            print s
            print >>f, s

            print "------"
            print >>f, "------"
