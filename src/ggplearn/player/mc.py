import math
import time
import random
from operator import itemgetter, attrgetter

import attr

import numpy as np

from ggplib.util import log
from ggplib.player.base import MatchPlayer

from ggplearn.util.bt import pretty_print_board

from ggplearn.nn import bases


###############################################################################

@attr.s
class PUCTPlayerConf(object):
    name = attr.ib("PUCTPlayer")
    verbose = attr.ib(True)

    num_of_playouts_per_iteration = attr.ib(800)
    num_of_playouts_per_iteration_noop = attr.ib(1)
    cpuct_constant_first_4 = attr.ib(0.75)
    cpuct_constant_after_4 = attr.ib(0.75)

    # added to root child policy pct (less than 0 is off)
    dirichlet_noise_alpha = attr.ib(0.1)
    dirichlet_noise_pct = attr.ib(0.25)

    # MAYBE useful for when small number of iterations.  otherwise pretty much the same
    expand_root = attr.ib(-1)

    choose = attr.ib("choose_top_visits")

    max_dump_depth = attr.ib(2)


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

            return "%s %d %.2f%%   %.2f %s" % (self.move,
                                               self.visits(),
                                               self.policy_dist_pct * 100,
                                               score,
                                               "T " if n.is_terminal else "* ")
        else:
            return "%s %d %.2f%%   ---- ? " % (self.move,
                                               self.visits(),
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

        if self.conf.verbose:
            assert self.conf.num_of_playouts_per_iteration_noop > 0, "DONT KNOW WHY THIS DOESNT WORK, BUT XXX"

        self.choose = getattr(self, self.conf.choose)

        identifier = "%s_%s_%s" % (self.conf.name, self.conf.num_of_playouts_per_iteration, generation)
        MatchPlayer.__init__(self, identifier)

    def on_meta_gaming(self, finish_time):
        self.root = None
        log.info("PUCTPlayer, match id: %s" % self.match.match_id)

        self.game_depth = 0
        sm = self.match.sm
        game_info = self.match.game_info

        # this is a performance hack, where once we get the nn/config we don't reget it.
        # if latest is set will always get the latest

        if self.generation == 'latest' or self.nn is None:
            self.base_config = bases.get_config(game_info.game, game_info.model, self.generation)
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

        self.our_noop_legal = self.role1_noop_legal if self.match.our_role_index == 1 else self.role0_noop_legal

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

        # sort the children now rather than every iteration
        node.children.sort(key=attrgetter("policy_dist_pct"), reverse=True)

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

        if self.conf.dirichlet_noise_alpha < 0:
            return None

        return np.random.dirichlet([self.conf.dirichlet_noise_alpha] * len(node.children))

    def cpuct_constant(self, node):
        cpuct = self.conf.cpuct_constant_after_4
        expanded = sum(1 for c in node.children if c.to_node is not None)
        if expanded < 3:
            cpuct = self.conf.cpuct_constant_first_4

        return cpuct

    def select_child(self, node, depth):

        dirichlet_noise = self.dirichlet_noise(node, depth)
        cpuct_constant = self.cpuct_constant(node)

        # get best
        best_child = None
        best_score = -1

        for idx, child in enumerate(node.children):
            cn = child.to_node

            child_visits = 0.0

            # prior... (alpha go zero said 0 but there score ranges from [-1,1]
            node_score = 0.0

            if cn is not None:
                child_visits = float(cn.mc_visits)
                node_score = cn.mc_score[node.lead_role_index]

                # ensure terminals are enforced more than other nodes
                if cn.is_terminal:
                    node_score * 1.02

            child_pct = child.policy_dist_pct

            if dirichlet_noise is not None:
                noise_pct = self.conf.dirichlet_noise_pct
                child_pct = (1 - noise_pct) * child_pct + noise_pct * dirichlet_noise[idx]

            v = float(node.mc_visits + 1)
            cv = float(child_visits + 1)
            puct_score = cpuct_constant * child_pct * (v ** 0.5) / cv

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

    def playout_loop(self, node, finish_time, cb=None):
        max_depth = -1
        total_depth = 0
        iterations = 0

        start_time = time.time()

        if self.root.lead_role_index != self.match.our_role_index:
            max_iterations = self.conf.num_of_playouts_per_iteration_noop
        else:
            max_iterations = self.conf.num_of_playouts_per_iteration

        while True:
            if time.time() > finish_time:
                log.info("RAN OUT OF TIME")
                break

            depth = self.playout(node)
            max_depth = max(depth, max_depth)
            total_depth += depth

            iterations += 1

            if max_iterations > 0 and iterations > max_iterations:
                break

            if cb and cb():
                break

        if self.conf.verbose:
            log.info("Time taken for %s iteratons %.3f" % (iterations,
                                                           time.time() - start_time))

            log.debug("The average depth explored: %.2f, max depth: %d" % (total_depth / float(iterations),
                                                                           max_depth))

    def on_apply_move(self, joint_move):
        self.game_depth += 1

        # need to fish for it in children?
        if self.root is not None:
            lead = self.root.lead_role_index
            other = 0 if lead else 1
            if other == 0:
                assert joint_move.get(other) == self.role0_noop_legal
            else:
                assert joint_move.get(other) == self.role1_noop_legal

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

            if self.conf.verbose:
                log.verbose("ROOT FOUND: %s / %d" % (new_root, visit_count(new_root)))

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

    def noop(self):
        if self.root.lead_role_index != self.match.our_role_index:
            if self.match.our_role_index:
                return self.role1_noop_legal
            else:
                return self.role0_noop_legal
        return None

    def on_next_move(self, finish_time):
        self.count_bp = 0
        sm = self.match.sm
        sm.update_bases(self.match.get_current_state())

        # break early as possible
        if not self.conf.verbose and self.conf.num_of_playouts_per_iteration_noop == 0:
            ri = self.match.our_role_index
            ls = sm.get_legal_state(ri)
            if ls.get_count() == 1 and ls.get_legal(0) == self.our_noop_legal:
                return self.our_noop_legal

        start_time = time.time()
        if self.root is not None:
            assert self.root.state == self.match.get_current_state().to_list()
        else:
            if self.conf.verbose:
                log.info('creating root')

            self.root = self.create_node(self.match.get_current_state())
            assert not self.root.is_terminal

            # predict root
            self.nodes_to_predict.append(self.root)

        self.do_predictions()

        # expand and predict some of root children
        if self.conf.expand_root > 0:
            for c in self.root.children[:self.conf.expand_root]:
                if c.to_node is None:
                    self.expand_child(c)
                    self.nodes_to_predict.append(c.to_node)

            self.do_predictions()

        if self.conf.verbose:
            log.debug("time taken for root %.3f" % (time.time() - start_time))

        self.playout_loop(self.root, finish_time)

        # this saves time and time is of the essence for training
        # becuase choice here is based on node.lead_role_index, not our_role_index
        choice = self.choose(finish_time)

        if self.conf.verbose:
            current = self.root

            dump_depth = 0
            while current is not None:
                if dump_depth == self.conf.max_dump_depth:
                    break

                self.dump_node(current, indent=dump_depth * 4)
                if current.is_terminal:
                    break

                current = current.sorted_children()[0].to_node
                dump_depth += 1

            if self.match.game_info.game == "breakthrough":
                pretty_print_board(sm, self.root.state)
                print

        noop_res = self.noop()
        if noop_res is not None:
            return noop_res
        else:
            return choice.legal

    def get_probabilities(self, temperature=1):
        total_visits = float(sum(c.visits() for c in self.root.children))

        temps = [((c.visits() + 1) / total_visits) ** temperature for c in self.root.children]
        temps_tot = sum(temps)

        probs = [(c, t / temps_tot) for c, t in zip(self.root.children, temps)]
        probs.sort(key=itemgetter(1), reverse=True)

        return probs

    def choose_converge_check(self):
        best_visit = self.root.sorted_children()[0]
        best_score = self.root.sorted_children(by_score=True)[0]
        if best_visit == best_score:
            if self.conf.verbose:
                log.info("Converged - breaking")
            return True
        return False

    def choose_converge(self, finish_time):
        if self.root.lead_role_index != self.match.our_role_index:
            return self.choose_top_visits(finish_time)

        best_visit = self.root.sorted_children()[0]

        score = best_visit.to_node.mc_score[self.root.lead_role_index]
        if score > 0.9 or score < 0.1:
            return best_visit

        best = best_visit
        best_score = self.root.sorted_children(by_score=True)[0]
        if best_visit != best_score:
            if self.conf.verbose:
                log.info("Conflicting between score and visits... visits : %s score : %s" % (best_visit,
                                                                                             best_score))

            store_current_alpha = self.conf.dirichlet_noise_alpha
            self.conf.dirichlet_noise_alpha = -1
            self.playout_loop(self.root, finish_time, self.choose_converge_check)
            self.conf.dirichlet_noise_alpha = store_current_alpha

            best_visit = self.root.sorted_children()[0]

            if self.conf.verbose:
                best_score = self.root.sorted_children(by_score=True)[0]
                if best_visit != best_score:
                    log.info("Failed to converge")

            if best != best_visit:
                if self.conf.verbose:
                    log.info("best visits now: %s -> %s" % (best, best_visit))
                best = best_visit

        if self.conf.verbose:
            log.info("BEST %s" % best)

        return best

    def choose_top_visits(self, finish_time):
        return self.root.sorted_children()[0]

    def choose_temperature(self, finish_time):
        depth = max(1, self.game_depth)
        if depth > 16:
            return self.choose_top_visits(finish_time)

        temp = 1.0
        random_range = 0.5 / math.sqrt(depth + 2) + 0.1

        temperature_depth_end = 16
        if self.game_depth < temperature_depth_end:
            temp = 0.5 + 0.5 * (0.85 ** (temperature_depth_end - depth))

        if self.conf.verbose:
            log.info("depth %d, temperature is %.3f, random range %.3f" % (depth, temp, random_range))

        random_expected = random.random() * random_range

        total_seen = 0
        res = None
        for c, p in self.get_probabilities(temp):
            if self.conf.verbose:
                print c, 100 * p
            total_seen += p
            if res is None and total_seen > random_expected:
                res = c, p
                if self.conf.verbose:
                    print ":::::::::: GOT"

        assert res
        return res[0]


##############################################################################

def get_test_config():
    return PUCTPlayerConf(name="xx",
                          verbose=True,
                          num_of_playouts_per_iteration=32,
                          num_of_playouts_per_iteration_noop=1,
                          expand_root=100,
                          dirichlet_noise_alpha=0.5,
                          cpuct_constant_first_4=0.75,
                          cpuct_constant_after_4=0.75,
                          choose="choose_temperature",
                          max_dump_depth=2)

def get_test_config():
    return PUCTPlayerConf(name="xx",
                          verbose=True,
                          num_of_playouts_per_iteration=200,
                          num_of_playouts_per_iteration_noop=1,
                          expand_root=100,
                          dirichlet_noise_alpha=0.1,
                          cpuct_constant_first_4=0.75,
                          cpuct_constant_after_4=0.75,
                          choose="choose_converge",
                          max_dump_depth=2)


def main():
    import sys
    from ggplib.play import play_runner
    from ggplearn.util.keras import constrain_resources

    constrain_resources()

    port = int(sys.argv[1])
    generation = sys.argv[2]

    conf = None
    if len(sys.argv) > 3 and sys.argv[3] == "-t":
        conf = get_test_config()

    player = PUCTPlayer(generation, conf=conf)
    play_runner(player, port)


if __name__ == "__main__":
    main()
