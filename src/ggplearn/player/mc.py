
import math
import time

import numpy as np

from ggplib.util import log
from ggplib.player.base import MatchPlayer
from ggplib.play import play_runner

from ggplearn.util.bt import pretty_print_board

from ggplearn import net_config


###############################################################################

def opp(role_index):
    ' helper '
    return 0 if role_index else 1


###############################################################################

class PUCTPlayerConf(object):
    NAME = "PUCTPlayer"
    VERBOSE = True

    NUM_OF_PLAYOUTS_PER_ITERATION = 800
    NUM_OF_PLAYOUTS_PER_ITERATION_NOOP = 1

    CPUCT_CONSTANT = 0.75
    DEPTH_0_CPUCT_CONSTANT = 0.75

    # added to root child policy pct (less than 0 is off)
    DIRICHLET_NOISE_ALPHA = 0.01
    DIRICHLET_NOISE_PCT = 0.25

    # MAYBE useful for when small number of iterations.  otherwise pretty much the same
    EXPAND_ROOT = -1
    EXPAND_EVERY_X = -1
    MM = False


def sort_by_policy_key(c):
    return c.policy_dist_pct


class Child(object):
    def __init__(self, parent, move, legal):
        self.parent = parent
        self.move = move
        self.legal = legal

        # from NN
        self.policy_dist_pct = None

        # to the next node
        # XXX this deviates from AlphaGoZero paper, where the keep statistics on child.  But I am
        # following how I did things in galvanise, as it is simpler to keep it my head.
        self.to_node = None

        # debug
        self.debug_node_score = -1
        self.debug_puct_score = -1

    def visits(self):
        if self.to_node is None:
            return 0
        return self.to_node.mc_visits

    def __repr__(self):
        n = self.to_node
        if n:
            ri = self.parent.lead_role_index
            if n.is_terminal:
                score = n.terminal_scores[ri] / 100.0
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

        self.mc_visits = 0
        self.mc_score = [0.0, 0.0]

        self.rollout_score = [0.0, 0.0]

    def add_child(self, move, legal):
        self.children.append(Child(self, move, legal))

    def sorted_children(self, by_score=False):
        ' sorts by mcts visits OR score '

        if not self.children:
            return self.children

        if by_score:
            def f(x):
                return -1 if x.to_node is None else x.to_node.mc_score[self.lead_role_index]
        else:
            def f(x):
                return -1 if x.to_node is None else x.to_node.mc_visits

        children = self.children[:]
        children.sort(key=f, reverse=True)
        return children


class PUCTPlayer(MatchPlayer):
    def __init__(self, generation="latest", conf=None):
        self.nn = None
        self.root = None
        self.generation = generation
        if conf is None:
            conf = PUCTPlayerConf()
        self.conf = conf

        identifier = "%s_%s_%s" % (self.conf.NAME, self.conf.NUM_OF_PLAYOUTS_PER_ITERATION, generation)
        MatchPlayer.__init__(self, identifier)

    def on_meta_gaming(self, finish_time):
        self.root = None
        log.info("PUCTPlayer, match id: %s" % self.match.match_id)

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

        total = 0
        for c in node.children:
            ridx = start_pos + c.legal
            c.policy_dist_pct = pred_policy[ridx]

            total += c.policy_dist_pct

        # normalise
        for c in node.children:
            c.policy_dist_pct /= total

    def do_predictions(self):
        actual_nodes_to_predict = []
        for node in self.nodes_to_predict:
            if node.is_terminal:
                node.mc_score = [s / 100.0 for s in node.terminal_scores]
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
            node.mc_score = pred_final_score[:]
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
            for i, s in enumerate(scores):
                node.mc_score[i] = (node.mc_visits *
                                    node.mc_score[i] + s) / float(node.mc_visits + 1)
            node.mc_visits += 1

    def dirichlet_noise(self, node, depth):
        if depth != 0:
            return None

        if self.conf.DIRICHLET_NOISE_ALPHA < 0:
            return None

        return np.random.dirichlet([self.conf.DIRICHLET_NOISE_ALPHA] * len(node.children))

    def minimax_before(self, node):
        # get all the children expanded
        top_children = node.sorted_children()
        if not top_children:
            return

        top_children = [c for c in top_children
                        if top_children[0].visits() == c.visits()]

        best = None
        best_score = -2
        for c in top_children:
            if not c.to_node:
                continue

            cn = c.to_node

            if cn.mc_score[node.lead_role_index] > best_score:
                best = cn
                best_score = cn.mc_score[node.lead_role_index]

        if best:
            node.mc_score = best.mc_score[:]

    def select_child(self, node, depth):
        if self.conf.MM and node.mc_visits > 20 and node.mc_visits % 4 == 0:
            for child in node.children:
                if child.visits() and child.to_node:
                    if not child.to_node.is_terminal:
                        self.minimax_before(child.to_node)

        # get best
        best_child = None
        best_score = -1

        dirichlet_noise = self.dirichlet_noise(node, depth)
        cpuct_constant = self.conf.DEPTH_0_CPUCT_CONSTANT if depth == 0 else self.conf.CPUCT_CONSTANT

        for idx, child in enumerate(sorted(node.children,
                                           key=sort_by_policy_key, reverse=True)):
            cn = child.to_node

            child_visits = 0.0

            # prior... (alpha go zero said 0 but there score ranges from [-1,1]
            node_score = 0.0

            # force expansion every 20...
            if self.conf.EXPAND_EVERY_X > 0 and node.mc_visits % self.conf.EXPAND_EVERY_X == 0:
                if cn is None:
                    node_score = 1.0

            if cn is not None:
                child_visits = float(cn.mc_visits)
                node_score = cn.mc_score[node.lead_role_index]

                # ensure terminals are enforced more than other nodes
                if node.is_terminal:
                    node_score * 1.02

            child_pct = child.policy_dist_pct

            if dirichlet_noise is not None:
                noise_pct = self.conf.DIRICHLET_NOISE_PCT
                child_pct = (1 - noise_pct) * child_pct + noise_pct * dirichlet_noise[idx]

            k = cpuct_constant
            v = float(node.mc_visits + 1)
            cv = float(child_visits + 1)
            puct_score = k * child_pct * (v ** 0.5) / cv

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
                scores = child.to_node.mc_score

                depth += 1
                path.append((depth, child.to_node, None))
                break

            current = child.to_node

            # already expanded terminal
            if current.is_terminal:
                depth += 1
                path.append((depth, child.to_node, None))
                scores = [s / 100.0 for s in child.to_node.terminal_scores]
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

            if self.conf.VERBOSE:
                print "ROOT FOUND:", new_root, visit_count(new_root)

    def dump_node(self, node, indent=0):
        indent_str = " " * indent
        print indent_str, "node %s %s" % (node.mc_visits, node.final_score)
        for child in node.sorted_children():
            print indent_str,
            print child, "\t->  ",
            if child.to_node is not None:
                n = child.to_node
                print "%d @ %.3f / %.3f" % (n.mc_visits, n.mc_score[0], n.mc_score[1]),
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
            if self.conf.VERBOSE:
                print 'creating root'

            self.root = self.create_node(self.match.get_current_state())
            assert not self.root.is_terminal

            # predict root
            self.nodes_to_predict.append(self.root)

        # expand and predict all children
        if self.conf.EXPAND_ROOT > 0:
            children = self.root.children[:]
            children.sort(key=sort_by_policy_key, reverse=True)
            for c in children[:self.conf.EXPAND_ROOT]:
                if c.to_node is None:
                    self.expand_child(c)
                    self.nodes_to_predict.append(c.to_node)

        self.do_predictions()

        if self.conf.VERBOSE:
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
                if (self.conf.NUM_OF_PLAYOUTS_PER_ITERATION_NOOP > 0 and
                    iterations == self.conf.NUM_OF_PLAYOUTS_PER_ITERATION_NOOP):
                    # exit early
                    break

            if (self.conf.NUM_OF_PLAYOUTS_PER_ITERATION > 0 and
                iterations == self.conf.NUM_OF_PLAYOUTS_PER_ITERATION):
                # exit early
                break

        print "Time taken for %s iteratons %.1f" % (iterations,
                                                    time.time() - start_time)
        if self.conf.VERBOSE:
            print "The average depth explored: %.2f, max depth: %d" % (total_depth / float(iterations),
                                                                       max_depth)

        choice = self.choose(finish_time)

        if self.conf.VERBOSE:
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

            self.dump_temperature(self.root)

            if self.match.game_info.game == "breakthrough":
                pretty_print_board(sm, self.root.state)
                print

        if self.root.lead_role_index != self.match.our_role_index:
            if self.match.our_role_index:
                return self.role1_noop_legal
            else:
                return self.role0_noop_legal

        return choice.legal

    def dump_temperature(self, node):
        children = node.children[:]
        children.sort(key=sort_by_policy_key, reverse=True)

        total_visits = float(sum(c.visits() for c in children))

        TEMP = 2
        probs = [(c.visits() / total_visits) ** TEMP for c in children]
        probs_tot = sum(probs)
        probs = [p / probs_tot for p in probs]
        for p, c in zip(probs, children):
            print c, "prob", p * 100

    def choose_converge(self, finish_time):
        best_visit = self.root.sorted_children()[0]
        best_score = self.root.sorted_children(by_score=True)[0]
        best = best_visit
        if best_visit != best_score:
            if self.conf.VERBOSE:
                log.info("Conflicting between score and visits... visits : %s score : %s" % (best_visit, best_score))

            # we should run a few more iterations (say 5) and if the majority as best score, we
            # should return score (assuming it will converge/catchup/overtake)
            # alternatively we could just look at debug scores from a select

            for i in range(int(self.conf.NUM_OF_PLAYOUTS_PER_ITERATION / 2)):
                if time.time() > finish_time:
                    break

                self.playout(self.root)

                best_visit = self.root.sorted_children()[0]
                best_score = self.root.sorted_children(by_score=True)[0]

                if best_visit == best_score:
                    log.warning("Converged")
                    break

            best_visit = self.root.sorted_children()[0]
            if self.conf.VERBOSE:
                best_score = self.root.sorted_children(by_score=True)[0]
                if best_visit != best_score:
                    log.info("Failed to converge - switching to best score")

            if best != best_visit:
                log.warning("best visits now: %s -> %s" % (best, best_visit))
                best = best_visit

        if self.conf.VERBOSE:
            print "BEST", best
            print
        return best

    def choose_minmax(self, finish_time):
        # perform a 2-ply minmax over top 8s

        ri = self.root.lead_role_index
        best_score = -2
        best_child = None
        for c in self.root.sorted_children()[:5]:
            if best_child is None:
                best_child = c

            if not c.to_node:
                continue

            children = c.to_node.sorted_children()
            if children:
                c2 = children[0]
                print "minmaxed for %s : %s" % (c, c2)
                if c2.to_node and c2.to_node.mc_score[ri] > best_score:
                    print 'WTF', c2.to_node.mc_score[ri], best_score
                    best_score = c2.to_node.mc_score[ri]
                    best_child = c

        print "best score minmax over top x", best_child
        return best_child

    def choose_top_visits(self, finish_time):
        return self.root.sorted_children()[0]

    choose = choose_top_visits


##############################################################################

class TestConfig(PUCTPlayerConf):
    NAME = "test"
    EXPAND_EVERY_X = 16
    DIRICHLET_NOISE_ALPHA = 0.01


def main():
    import sys
    from ggplib.play import play_runner
    from ggplearn.util.keras import constrain_resources

    constrain_resources()

    port = int(sys.argv[1])
    generation = sys.argv[2]

    conf = None
    if len(sys.argv) > 3 and sys.argv[3] == "-t":
        conf = TestConfig()

    player = PUCTPlayer(generation, conf=conf)
    play_runner(player, port)

if __name__ == "__main__":
    main()
