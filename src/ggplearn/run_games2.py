import os
import random
import tempfile

from collections import OrderedDict

import json

import numpy as np

from keras.models import model_from_json

from ggplib.player.gamemaster import GameMaster
from ggplib.db import lookup
from ggplib.db.helper import get_gdl_for_game

from ggplib.player.base import MatchPlayer

from ggplearn import net_config
from ggplearn.player.mc import NNMonteCarlo
from ggplib.util import log

models_path = os.path.join(os.environ["GGPLEARN_PATH"], "src", "ggplearn", "models")


class Child(object):
    def __init__(self, parent, move, legal):
        self.parent = parent
        self.move = move
        self.legal = legal

        # from NN
        self.p_visits_pct = 0
        self.visits_pct_noisy = 0

        self.to_node = None


class Node(object):
    def __init__(self, state, lead_role_index, is_terminal):
        self.state = state
        self.lead_role_index = lead_role_index
        self.is_terminal = is_terminal
        self.children = []

        self.predicted = False

        # from NN
        self.prior_scores = None
        self.final_scores = None

        # from sm.get_goal_value() (0 - 100)
        self.terminal_scores = None

    def add_child(self, move, legal):
        self.children.append(Child(self, move, legal))

    @property
    def expanded(self):
        for c in self.children:
            if c.to_node is None:
                return False
        return True

    def get_top_k(self, k):
        assert self.predicted

        children = self.children[:]
        children.sort(key=lambda c: c.p_visits_pct, reverse=True)
        return children[:k]


def get_legals(ls):
    return [ls.get_legal(i) for i in range(ls.get_count())]


def bs_to_state(bs):
    return tuple(bs.get(i) for i in range(bs.len()))


def state_to_bs(state, bs):
    for i, v in enumerate(state):
        bs.set(i, v)


class NNExpander(MatchPlayer):
    top_k = 3
    do_noise = True
    noise_pct = 0.75
    dirichlet_alpha = (0.03, 1)

    def __init__(self, game_name, postfix):
        MatchPlayer.__init__(self, "NNExpander-" + postfix)

        # do everything upfront for speed
        self.postfix = postfix

        # lookup via game_name (this gets statemachine & statemachine model)
        game_info = lookup.by_name(game_name)

        sm = game_info.get_sm()

        self.nn_config = net_config.get_bases_config(game_info.game)
        self.base_infos = net_config.create_base_infos(self.nn_config, game_info.model)

        # load neural network model and weights
        model_filename = os.path.join(models_path, "model_nn_%s_%s.json" % (game_info.game, self.postfix))
        weights_filename = os.path.join(models_path, "weights_nn_%s_%s.h5" % (game_info.game, self.postfix))

        with open(model_filename, "r") as f:
            self.nn_model = model_from_json(f.read())

        self.nn_model.load_weights(weights_filename)

        # cache joint move, and basestate
        self.joint_move = sm.get_joint_move()
        self.basestate_expand_node = sm.new_base_state()
        self.basestate_expanded_node = sm.new_base_state()

        self.sm = sm

        def get_noop_idx(actions):
            for idx, a in enumerate(actions):
                if "noop" in a:
                    return idx
            assert False, "did not find noop"

        self.role0_noop_legal, self.role1_noop_legal = map(get_noop_idx, game_info.model.actions)

    def on_meta_gaming(self, finish_time):
        self.nodes_to_predict = []
        self.root = None
        log.info("NNExpander, match id: %s" % self.match.match_id)

    def predict_n(self, nodes):
        num_states = len(nodes)

        X_0 = [net_config.state_to_channels(n.state,
                                            n.lead_role_index,
                                            self.nn_config,
                                            self.base_infos) for n in nodes]

        X_0 = np.array(X_0).reshape(num_states, self.nn_config.num_rows,
                                    self.nn_config.num_cols, self.nn_config.num_channels)

        X_1 = [[v for v, base_info in zip(n.state, self.base_infos)
                if base_info.channel is None] for n in nodes]
        X_1 = np.array(X_1).reshape(num_states, len(X_1[0]))

        Y = self.nn_model.predict([X_0, X_1], batch_size=num_states)

        assert len(Y) == 3

        result = []
        for i in range(num_states):
            policy, scores0, scores1 = Y[0][i], Y[1][i], Y[2][i]
            result.append((policy, scores0, scores1))

        return result

    def predict_1(self, node):
        return self.predict_n([node])[0]

    def do_predictions(self):
        result = self.predict_n(self.nodes_to_predict)
        for node, (pred_policy, priors, finals) in zip(self.nodes_to_predict, result):
            self.update_node(node, pred_policy, priors, finals)
        self.nodes_to_predict = []

    def create_node(self, basestate):
        sm = self.match.sm
        sm.update_bases(basestate)

        if sm.get_legal_state(0).get_count() == 1 and sm.get_legal_state(0).get_legal(0) == self.role0_noop_legal:
            lead_role_index = 1
        else:
            assert (sm.get_legal_state(1).get_count() == 1 and
                    sm.get_legal_state(1).get_legal(0) == self.role1_noop_legal)
            lead_role_index = 0

        node = Node(bs_to_state(basestate),
                    lead_role_index,
                    sm.is_terminal())

        ls = sm.get_legal_state(0) if lead_role_index == 0 else sm.get_legal_state(1)

        for l in get_legals(ls):
            node.add_child(sm.legal_to_move(lead_role_index, l), l)

        if node.is_terminal:
            node.terminal_scores = [sm.get_goal_value(i) for i in range(2)]

        return node

    def expand_children(self, children):
        # note children don't necessarilt need to have same parent

        sm = self.match.sm

        for c in children:
            assert c.to_node is None
            node = c.parent
            state_to_bs(node.state, self.basestate_expand_node)
            sm.update_bases(self.basestate_expand_node)

            if node.lead_role_index == 0:
                self.joint_move.set(1, self.role1_noop_legal)
            else:
                self.joint_move.set(0, self.role0_noop_legal)

            self.joint_move.set(node.lead_role_index, c.legal)
            sm.next_state(self.joint_move, self.basestate_expanded_node)

            c.to_node = self.create_node(self.basestate_expanded_node)

    def update_node(self, node, pred_policy, priors, finals):
        assert not node.predicted
        if node.lead_role_index == 0:
            start_pos = 0
        else:
            start_pos = len(self.match.game_info.model.actions[0])

        for c in node.children:
            ridx = start_pos + c.legal
            c.p_visits_pct = pred_policy[ridx]

        node.prior_scores = priors
        node.final_scores = finals
        node.predicted = True

    def on_next_move(self, finish_time):
        sm = self.sm
        sm.update_bases(self.match.get_current_state())

        root = self.root = self.create_node(self.match.get_current_state())
        assert not root.is_terminal

        # not much to do if not our "move"
        if root.lead_role_index != self.match.our_role_index:
            self.root = None
            if self.match.our_role_index:
                return self.role1_noop_legal
            else:
                return self.role0_noop_legal

        pred_policy, priors, finals = self.predict_1(root)
        self.update_node(root, pred_policy, priors, finals)

        # expand and predict
        self.expand_children(root.children)

        # check top k for terminal states
        for child in root.get_top_k(self.top_k):
            if child.to_node is not None and child.to_node.is_terminal:
                if child.to_node.terminal_scores[root.lead_role_index] == 100:
                    return child.legal

        dirichlet_noise = np.random.dirichlet(self.dirichlet_alpha, len(root.children))

        choice_value = choice_noisy = None
        best_value = best_noisy_value = -1
        for child, d in zip(root.children, dirichlet_noise):
            child.visits_pct_noisy = ((1 - self.noise_pct) * child.p_visits_pct +
                                      self.noise_pct * d[0])

            if child.p_visits_pct > best_value:
                best_value = child.p_visits_pct
                choice_value = child

            if child.visits_pct_noisy > best_noisy_value:
                best_noisy_value = child.visits_pct_noisy
                choice_noisy = child

        if self.do_noise:
            if choice_noisy != choice_value:
                print "Choice %s -> %s (%s / %s)" % (choice_value.move,
                                                     choice_noisy.move,
                                                     choice_value.visits_pct_noisy,
                                                     choice_noisy.visits_pct_noisy)
            choice = choice_noisy
        else:
            choice = choice_value

        return choice.legal


class Runner(object):
    def __init__(self, game_name, postfix):
        self.game_name = game_name
        self.postfix = postfix

        self.gm = GameMaster(get_gdl_for_game(game_name))

        for role in self.gm.sm.get_roles():
            player = NNExpander(game_name, postfix)
            self.gm.add_player(player, role)

        # for acquring info on game
        self.gm_playout = GameMaster(get_gdl_for_game(game_name))
        for role in self.gm.sm.get_roles():
            player = NNMonteCarlo(postfix)
            self.gm_playout.add_player(player, role)

        self.basestate = self.gm.sm.new_base_state()

    def get_bases(self):
        self.gm.sm.get_current_state(self.basestate)
        return [self.basestate.get(i) for i in range(self.basestate.len())]

    def play_one_game(self):
        self.gm.reset()

        self.gm.start(meta_time=30, move_time=5)

        states = [self.get_bases()]

        last_move = None
        while not self.gm.finished():
            last_move = self.gm.play_single_move(last_move=last_move)
            states.append(self.get_bases())

        self.gm.play_to_end(last_move)
        return states

    def do_policy(self, state):
        for player, _ in self.gm_playout.players:
            player.NUM_OF_PLAYOUTS_PER_ITERATION = 420
            print 'here', player, player.NUM_OF_PLAYOUTS_PER_ITERATION

        for i, v in enumerate(state):
            self.basestate.set(i, v)

        self.gm_playout.reset()
        self.gm_playout.start(meta_time=30, move_time=30, initial_basestate=self.basestate)
        self.last_move = self.gm_playout.play_single_move(None)

        # fish for root
        if self.last_move[0] == "noop":
            root = self.gm_playout.players[1][0].root
        else:
            root = self.gm_playout.players[0][0].root

        def f(x):
            return -1 if x.to_node is None else x.to_node.mcts_visits
        children = root.children[:]
        children.sort(key=f, reverse=True)

        def get_visits(c):
            if c.to_node is None:
                return 1.0
            else:
                return (c.to_node.mcts_visits + 1)

        total_visits = 1.0
        for child in children:
            total_visits += get_visits(child)

        dist = [(c.legal,
                 c.move,
                 get_visits(c) / total_visits,
                 c.p_visits_pct) for c in children]

        best_scores = children[0].to_node.mcts_score
        return dist, best_scores

    def playout_state(self):
        for player, _ in self.gm_playout.players:
            player.NUM_OF_PLAYOUTS_PER_ITERATION = 42
            print 'here', player, player.NUM_OF_PLAYOUTS_PER_ITERATION

        # self.last_move was set in do_policy()
        self.gm_playout.play_to_end(self.last_move)
        return self.gm.scores

    def generate_samples(self, number_samples):
        NUMBER_OF_SAMPLES_PER_GAME = 3
        self.unique_samples = {}
        while len(self.unique_samples) < number_samples:
            states = self.play_one_game()

            # we don't sample first or last state
            samples = random.sample(states[1:-1], NUMBER_OF_SAMPLES_PER_GAME)

            # one sample per game
            for s in samples:
                s = tuple(s)
                if tuple(s) in self.unique_samples:
                    continue

                policy_dist, best_scores = self.do_policy(s)
                final_scores = self.playout_state()
                self.unique_samples[s] = (final_scores, policy_dist, best_scores)
                break

    def write_to_file(self):
        json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

        data = OrderedDict()
        data["game"] = self.game_name
        data["postfix"] = self.postfix

        k = data["num_samples"] = len(self.unique_samples)

        data["samples"] = []
        for state, (final_scores, policy_dist, best_scores) in self.unique_samples.items():

            sample = OrderedDict()
            sample["state"] = list(state)
            sample["final_scores"] = final_scores
            sample["best_scores"] = [float(s) for s in best_scores]

            sample["policy_dists"] = []
            for legal, move_str, new_p, old_p in policy_dist:
                d = OrderedDict()
                d["legal"] = int(legal)
                d["new_prob"] = float(new_p)
                d["old_prob"] = float(old_p)
                sample["policy_dists"].append(d)

            # finally append to samples
            data["samples"].append(sample)

        fd, path = tempfile.mkstemp(suffix='.json',
                                    prefix="samples_%s_%d_" % (self.game_name, k),
                                    dir=".")
        with os.fdopen(fd, 'w') as open_file:
            open_file.write(json.dumps(data, indent=4))


def main(game_name):

    from ggplib import interface
    interface.initialise_k273(1)

    # XXX reduce log
    # import ggplib.util.log
    # ggplib.util.log.initialise()

    # pre-initialise database - used in match for remapping
    lookup.get_database()

    from ggplearn.utils.keras import use_one_cpu_please
    use_one_cpu_please()

    game_name = sys.argv[1]
    postfix = sys.argv[2]
    num_samples = int(sys.argv[3])
    runner = Runner(game_name, postfix)
    runner.generate_samples(num_samples)
    runner.write_to_file()

if __name__ == "__main__":
    import pdb
    import sys
    import traceback

    try:
        main(sys.argv[1])

    except Exception as exc:
        print exc
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
