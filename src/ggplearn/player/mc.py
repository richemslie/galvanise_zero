import math
import time
import numpy as np

from ggplib.util import log
from ggplib.player.base import MatchPlayer

from ggplearn.utils.bt import pretty_print_board

from ggplearn import net_config


###############################################################################

VERBOSE = True


def opp(role_index):
    ' helper '
    return 0 if role_index else 1


###############################################################################

def sort_by_policy_key(c):
    return c.policy_dist_pct


class Child(object):
    def __init__(self, parent, move, legal):
        self.parent = parent
        self.move = move
        self.legal = legal
        self.traversals = 0

        # from NN
        self.policy_dist_pct = None

        # to the next node
        # XXX this deviates from AlphaGoZero paper, where the keep statistics on child.  But I am
        # following how I did things in galvanise, as it is simpler to keep it my head.
        self.to_node = None

        # debug
        self.debug_node_score = -1
        self.debug_puct_score = -1

    def __repr__(self):
        n = self.to_node
        if n:
            ri = self.parent.lead_role_index
            if n.is_terminal:
                score = n.terminal_scores[ri] / 95.0
            else:
                score = n.final_score[ri] or 0.0

            return "%s %.2f%%   %.2f %s" % (self.move,
                                            self.policy_dist_pct * 100,
                                            score,
                                            "T " if n.is_terminal else "* ")
        else:
            return "%s %.2f%%   ---- ? " % (self.move,
                                            self.policy_dist_pct * 100)
    __str__ = __repr__


class Node(object):
    def __init__(self, state, lead_role_index, is_terminal):
        self.state = state
        self.lead_role_index = lead_role_index
        self.is_terminal = is_terminal
        self.children = []

        self.predicted = False

        # from NN
        self.final_score = None

        # from sm.get_goal_value() (0 - 100)
        self.terminal_scores = None

        self.mcts_visits = 0
        self.mcts_score = [0.0, 0.0]

    def add_child(self, move, legal):
        self.children.append(Child(self, move, legal))

    def sorted_children(self, by_score=False):
        ' sorts by mcts visits OR score '

        if not self.children:
            return self.children

        if by_score:
            def f(x):
                return -1 if x.to_node is None else x.to_node.mcts_score[self.lead_role_index]
        else:
            def f(x):
                return -1 if x.to_node is None else x.to_node.mcts_visits

        children = self.children[:]
        children.sort(key=f, reverse=True)
        return children


class NNMonteCarlo(MatchPlayer):
    player_name = "MC"

    NUM_OF_PLAYOUTS_PER_ITERATION = 800
    NUM_OF_PLAYOUTS_PER_ITERATION_NOOP = 800

    # 2 / 2 v gurgeh 10 seconds
    # CPUCT_CONSTANT = 0.75
    # DEPTH_0_CPUCT_CONSTANT = 0.75

    # 3 / 1 v gurgeh 10 seconds
    # CPUCT_CONSTANT = 0.85
    # DEPTH_0_CPUCT_CONSTANT = 0.75

    CPUCT_CONSTANT = 0.95
    DEPTH_0_CPUCT_CONSTANT = 0.95

    # only added to child policy pct (less than 0 is off)
    DIRICHLET_NOISE_ALPHA = 0.15
    DIRICHLET_NOISE_PCT = 0.25

    # XXX need to really nip this one on the bud
    DIRICHLET_EXPANDED_ONLY = True

    # XXX Not sure I need this
    PUCT_VISIT_INITIAL_BOOST = 7

    EXPAND_ROOT = 5

    def __init__(self, generation="latest"):
        identifier = "%s_%s_%s" % (self.player_name, self.NUM_OF_PLAYOUTS_PER_ITERATION, generation)
        MatchPlayer.__init__(self, identifier)
        self.nn = None
        self.root = None
        self.generation = generation

    def on_meta_gaming(self, finish_time):
        self.root = None
        log.info("NNExpander, match id: %s" % self.match.match_id)

        sm = self.match.sm
        game_info = self.match.game_info

        # this is a performance hack, where once we get the nn/config we don't reget it.
        # if latest is set will always get the latest

        if self.generation == 'latest' or self.nn is None:
            self.base_config = net_config.get_bases_config(game_info.game,
                                                           game_info.model,
                                                           self.generation)

            self.nn = self.base_config.create_network()
            self.nn.load()

            # cache joint move, and basestate
            self.joint_move = sm.get_joint_move()
            self.basestate_expand_node = sm.new_base_state()
            self.basestate_expanded_node = sm.new_base_state()

            def get_noop_idx(actions):
                for idx, a in enumerate(actions):
                    if "noop" in a:
                        return idx
                assert False, "did not find noop"

            self.role0_noop_legal, self.role1_noop_legal = map(get_noop_idx, game_info.model.actions)

        self.root = None
        self.nodes_to_predict = []

    def update_node_policy(self, node, pred_policy):
        if node.lead_role_index == 0:
            start_pos = 0
        else:
            start_pos = len(self.match.game_info.model.actions[0])

        for c in node.children:
            ridx = start_pos + c.legal
            c.policy_dist_pct = pred_policy[ridx]

    def do_predictions(self):
        actual_nodes_to_predict = []
        for node in self.nodes_to_predict:
            if node.is_terminal:
                node.mcts_score = [s / 95.0 for s in node.terminal_scores]
            else:
                assert not node.predicted
                actual_nodes_to_predict.append(node)

        self.nodes_to_predict = []

        # nothing to do
        if not actual_nodes_to_predict:
            return

        states = [n.state for n in actual_nodes_to_predict]
        lead_role_indexs = [n.lead_role_index for n in actual_nodes_to_predict]

        result = self.nn.predict_n(states, lead_role_indexs)

        for node, (pred_policy, pred_final_score) in zip(actual_nodes_to_predict,
                                                         result):
            node.predicted = True
            node.final_score = pred_final_score
            node.mcts_score = pred_final_score[:]
            self.update_node_policy(node, pred_policy)

    def create_node(self, basestate):
        sm = self.match.sm
        sm.update_bases(basestate)

        if sm.get_legal_state(0).get_count() == 1 and sm.get_legal_state(0).get_legal(0) == self.role0_noop_legal:
            lead_role_index = 1
        else:
            assert (sm.get_legal_state(1).get_count() == 1 and
                    sm.get_legal_state(1).get_legal(0) == self.role1_noop_legal)
            lead_role_index = 0

        node = Node(basestate.to_list(),
                    lead_role_index,
                    sm.is_terminal())

        if node.is_terminal:
            node.terminal_scores = [sm.get_goal_value(i) for i in range(2)]
        else:
            legal_state = sm.get_legal_state(0) if lead_role_index == 0 else sm.get_legal_state(1)
            for l in legal_state.to_list():
                node.add_child(sm.legal_to_move(lead_role_index, l), l)

        return node

    def expand_child(self, child):
        sm = self.match.sm
        assert child.to_node is None
        node = child.parent

        self.basestate_expand_node.from_list(node.state)
        sm.update_bases(self.basestate_expand_node)

        if node.lead_role_index == 0:
            self.joint_move.set(1, self.role1_noop_legal)
        else:
            self.joint_move.set(0, self.role0_noop_legal)

        self.joint_move.set(node.lead_role_index, child.legal)
        sm.next_state(self.joint_move, self.basestate_expanded_node)

        child.to_node = self.create_node(self.basestate_expanded_node)

    def back_propagate(self, path, scores):
        self.count_bp += 1

        for _, node, child in reversed(path):
            node.mcts_visits += 1

            for i, s in enumerate(scores):
                node.mcts_score[i] = (node.mcts_visits * node.mcts_score[i] + s) / float(node.mcts_visits + 1)

            if child is not None:
                child.traversals += 1

    def get_dirichlet_noise(self, node, depth):
        if self.DIRICHLET_NOISE_ALPHA < 0:
            return None

        return np.random.dirichlet((self.DIRICHLET_NOISE_ALPHA, 1.0), len(node.children))

    def select_child(self, node, depth):
        # get best
        best_child = None
        best_score = -1

        dirichlet_noise = self.get_dirichlet_noise(node, depth)
        do_dirichlet_noise = dirichlet_noise is not None
        cpuct_constant = self.DEPTH_0_CPUCT_CONSTANT if depth == 0 else self.CPUCT_CONSTANT

        for idx, child in enumerate(node.children):
            cn = child.to_node

            child_visits = 0.0
            if cn is not None:
                # break early?
                child_visits = float(cn.mcts_visits)
                node_score = cn.mcts_score[node.lead_role_index]

            else:
                # as per alphago paper
                node_score = 0.0

            child_pct = child.policy_dist_pct

            if do_dirichlet_noise:
                if self.DIRICHLET_EXPANDED_ONLY and child.to_node is None:
                    do_dirichlet_noise = False

            if do_dirichlet_noise:
                noise_pct = self.DIRICHLET_NOISE_PCT
                child_pct = (1 - noise_pct) * child_pct - noise_pct * dirichlet_noise[idx][0]

            v = node.mcts_visits - child_visits + self.PUCT_VISIT_INITIAL_BOOST

            puct_score = cpuct_constant * child_pct * math.sqrt(v) / (child_visits + 1)

            score = node_score + puct_score

            # use for debug/display
            child.debug_node_score = node_score
            child.debug_puct_score = puct_score

            if score > best_score:
                best_child = child
                best_score = score

        assert best_child is not None
        return best_child

    def playout(self, current):
        assert current is not None and not current.is_terminal

        path = []
        depth = 0
        scores = None
        while True:
            child = self.select_child(current, depth)
            path.append((depth, current, child))

            if child.to_node is None:
                self.expand_child(child)
                self.nodes_to_predict.append(child.to_node)
                self.do_predictions()
                scores = child.to_node.mcts_score

                depth += 1
                path.append((depth, child.to_node, None))
                break

            current = child.to_node

            # already expanded terminal
            if current.is_terminal:
                depth += 1
                path.append((depth, child.to_node, None))
                scores = [s / 95.0 for s in child.to_node.terminal_scores]
                break

            depth += 1

        assert scores is not None
        self.back_propagate(path, scores)
        return depth

    def on_apply_move(self, joint_move):
        # need to fish for it in children?
        if self.root is not None:
            lead, other = self.root.lead_role_index, opp(self.root.lead_role_index)
            if other == 0:
                assert joint_move.get(other) == self.role0_noop_legal
            else:
                assert joint_move.get(other) == self.role0_noop_legal

            played = joint_move.get(lead)

            for c in self.root.children:
                c.parent = None
                if c.legal == played:
                    found = True
                    # might be none, that is fine
                    new_root = c.to_node

            assert found
            self.root = new_root

            def visit_count(node):
                if node is None:
                    return 0
                total = 1
                for c in node.children:
                    total += visit_count(c.to_node)
                return total

            if VERBOSE:
                print "ROOT FOUND:", new_root, visit_count(new_root)

    def dump_node(self, node, indent=0):
        indent_str = " " * indent
        for child in node.sorted_children():
            print indent_str,
            print child, "\t->  ",
            if child.to_node is not None:
                n = child.to_node
                print "%d @ %.3f / %.3f" % (n.mcts_visits, n.mcts_score[0], n.mcts_score[1]),
            else:
                print "--- @ ---- / ----",

            print "\t  %.2f + %.2f = %.3f" % (child.debug_node_score,
                                              child.debug_puct_score,
                                              child.debug_node_score + child.debug_puct_score)

    def on_next_move(self, finish_time):
        self.count_bp = 0
        sm = self.match.sm
        sm.update_bases(self.match.get_current_state())

        start_time = time.time()
        if self.root is not None:
            assert self.root.state == self.match.get_current_state().to_list()
        else:
            if VERBOSE:
                print 'creating root'

            self.root = self.create_node(self.match.get_current_state())
            assert not self.root.is_terminal

            # predict root
            self.nodes_to_predict.append(self.root)

        # expand and predict all children
        if self.EXPAND_ROOT:
            for c in self.root.children[:self.EXPAND_ROOT]:
                if c.to_node is None:
                    self.expand_child(c)
                    self.nodes_to_predict.append(c.to_node)

        self.do_predictions()

        print "time taken for root", time.time() - start_time

        max_depth = -1
        total_depth = 0
        iterations = 0

        start_time = time.time()
        while True:
            if time.time() > finish_time:
                print "RAN OUT OF TIME"
                break

            depth = self.playout(self.root)
            max_depth = max(depth, max_depth)
            total_depth += depth

            iterations += 1

            if self.root.lead_role_index != self.match.our_role_index:
                if (self.NUM_OF_PLAYOUTS_PER_ITERATION_NOOP > 0 and
                    iterations == self.NUM_OF_PLAYOUTS_PER_ITERATION_NOOP):
                    # exit early
                    break

            if (self.NUM_OF_PLAYOUTS_PER_ITERATION > 0 and
                iterations == self.NUM_OF_PLAYOUTS_PER_ITERATION):
                # exit early
                break

        print "Time taken for %s iteratons %.1f" % (iterations,
                                                    time.time() - start_time)

        print "The average depth explored: %.2f, max depth: %d" % (total_depth / float(iterations),
                                                                   max_depth)

        if VERBOSE:
            current = self.root
            dump_depth = 0
            while current is not None:
                self.dump_node(current, indent=dump_depth * 4)
                if current.is_terminal:
                    break
                current = current.sorted_children()[0].to_node
                dump_depth += 1

                if dump_depth == 4:
                    break

            if self.match.game_info.game == "breakthrough":
                pretty_print_board(sm, self.root.state)
                print

        if self.root.lead_role_index != self.match.our_role_index:
            if self.match.our_role_index:
                return self.role1_noop_legal
            else:
                return self.role0_noop_legal

        return self.choose().legal

    def choose(self):
        # XXX should call choice_move()... or something.
        best_visits = self.root.sorted_children()[0]
        best_score = self.root.sorted_children(by_score=True)[0]
        best = best_visits
        if best_visits != best_score:
            log.warning("Conflicting between score and visits...")
            x = self.root.sorted_children()[1]
            if x == best_score:
                log.warning("Switching to score")
                best = x
        print "BEST", best
        print
        return best


##############################################################################

class NNMonteCarloTest(NNMonteCarlo):
    player_name = "test"
    NUM_OF_PLAYOUTS_PER_ITERATION = 120
    NUM_OF_PLAYOUTS_PER_ITERATION_NOOP = 1


def main():
    import sys
    from twisted.internet import reactor
    from twisted.web import server

    from ggplib.util import log
    from ggplib.server import GGPServer
    from ggplib import interface

    port = int(sys.argv[1])
    generation = sys.argv[2]

    interface.initialise_k273(1, log_name_base="mcplayer")
    log.initialise()

    if len(sys.argv) > 3 and sys.argv[3] == "-t":
        player = NNMonteCarloTest(generation)
    else:
        player = NNMonteCarlo(generation)

    # this still uses more than once cpu :(
    import tensorflow as tf
    config = tf.ConfigProto(device_count=dict(CPU=1),
                            allow_soft_placement=False,
                            log_device_placement=False,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    sess = tf.Session(config=config)

    from keras import backend
    backend.set_session(sess)

    ggp = GGPServer()
    ggp.set_player(player)
    site = server.Site(ggp)

    reactor.listenTCP(port, site)
    reactor.run()


if __name__ == "__main__":
    main()
