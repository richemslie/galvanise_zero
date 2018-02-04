from builtins import super

import sys
import time
import random
from operator import itemgetter, attrgetter

import numpy as np
from tabulate import tabulate

from ggplib.util import log
from ggplib.player.base import MatchPlayer

from ggpzero.defs import confs, templates

from ggpzero.nn.manager import get_manager


###############################################################################

class Child(object):
    def __init__(self, parent, move, legal):
        # for repr:
        self.parent = parent

        self.move = move
        self.legal = legal

        # from NN
        self.init_policy_prob = None
        self.policy_prob = None

        # to the next node
        # this deviates from AlphaGoZero paper, where the keep statistics on child.  But I am
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
                                               self.policy_prob * 100,
                                               score,
                                               "T " if n.is_terminal else "* ")
        else:
            return "%s %d %.2f%%   ---- ? " % (self.move,
                                               self.visits(),
                                               self.policy_prob * 100)
    __str__ = __repr__


class Node(object):
    def __init__(self, state, lead_role_index, is_terminal, parent):
        self.state = state
        self.lead_role_index = lead_role_index
        self.is_terminal = is_terminal
        self.parent = parent

        self.children = []

        # from NN
        self.final_score = None

        # from sm.get_goal_value() (0 - 100)
        self.terminal_scores = None

        self.mc_visits = 0
        self.mc_score = None

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


class PUCTEvaluator(object):
    def __init__(self, conf=None):
        if conf is None:
            conf = confs.PUCTPlayerConfig()
        self.conf = conf

        self.nn = None
        self.initial_root = self.root = None

        self.choose = getattr(self, self.conf.choose)

        self.identifier = "%s_%s_%s" % (self.conf.name, self.conf.playouts_per_iteration, conf.generation)

        self.sm = None

    def init(self, game_info, current_state):
        self.game_info = game_info

        # HACK XXXX hack memory leak fix...
        if self.sm is None:
            self.sm = game_info.get_sm()

            # cache joint move, and basestate
            self.joint_move = self.sm.get_joint_move()
            self.basestate_expand_node = self.sm.new_base_state()

        # This is a performance hack, where once we get the nn/config we don't re-get it.
        # If latest is set will always get the latest
        if self.conf.generation == 'latest' or self.nn is None:
            self.nn = get_manager().load_network(game_info.game, self.conf.generation)

        def get_noop_idx(actions):
            for idx, a in enumerate(actions):
                if "noop" in a:
                    return idx
            assert False, "did not find noop"

        self.role0_noop_legal, self.role1_noop_legal = map(get_noop_idx, game_info.model.actions)

        self.initial_root = self.root = None
        self.game_depth = 0

        start_time = time.time()

        if self.conf.verbose:
            log.info('creating root')

        self.root = self.create_node(current_state, None)
        assert not self.root.is_terminal

        if self.conf.verbose:
            log.debug("time taken for root %.3f" % (time.time() - start_time))

        return self.root

    def update_node_policy(self, node, policies):
        policy = policies[node.lead_role_index]

        total = 0
        assert len(node.children)
        disaster = False
        for c in node.children:
            c.init_policy_prob = policy[c.legal]
            c.policy_prob = policy[c.legal]
            if c.policy_prob < 0.001:
                c.policy_prob = 0.001
                disaster = True
            total += c.policy_prob

        # was it really a disaster?

        if False and len(node.children) > 1 and disaster and max(node.final_score) < 0.97:
            log.warning("disasterdisasterdisaster!!!!")
            print total
            for p in policies:
                print "policy:"
                print p

            for c in node.children:
                print c
                if c.policy_prob < 0.0001:
                    print c.legal, policy[c.legal], c.policy_prob

        if total < 0.0001:
            # no real predictions, set it uniform
            for c in node.children:
                c.policy_prob = 1.0 / len(node.children)
        else:
            # normalise
            for c in node.children:
                c.policy_prob /= total

        # sort the children now rather than every iteration
        node.children.sort(key=attrgetter("policy_prob"), reverse=True)

    def create_node(self, basestate, parent):
        self.sm.update_bases(basestate)

        if (self.sm.get_legal_state(0).get_count() == 1 and
            self.sm.get_legal_state(0).get_legal(0) == self.role0_noop_legal):
            lead_role_index = 1

        else:
            assert (self.sm.get_legal_state(1).get_count() == 1 and
                    self.sm.get_legal_state(1).get_legal(0) == self.role1_noop_legal)
            lead_role_index = 0

        node = Node(basestate.to_list(), lead_role_index, self.sm.is_terminal(), parent)

        if node.is_terminal:
            node.terminal_scores = [self.sm.get_goal_value(i) for i in range(2)]
            node.mc_score = [s / 100.0 for s in node.terminal_scores]
        else:
            legal_state = self.sm.get_legal_state(0) if lead_role_index == 0 else self.sm.get_legal_state(1)

            for l in legal_state.to_list():
                node.add_child(self.sm.legal_to_move(lead_role_index, l), l)

            # Fish for max number of previous states.
            max_prev_states = self.nn.gdl_bases_transformer.num_previous_states

            prev_states = []
            cur = node.parent
            while len(prev_states) < max_prev_states:
                if cur is None:
                    break
                prev_states.append(cur.state)
                cur = cur.parent

            predictions = self.nn.predict_1(node.state)

            node.final_score = predictions.scores[:]
            node.mc_score = predictions.scores[:]
            self.update_node_policy(node, predictions.policies)
        return node

    def expand_child(self, node, child):
        assert child.to_node is None

        self.basestate_expand_node.from_list(node.state)
        self.sm.update_bases(self.basestate_expand_node)

        if node.lead_role_index == 0:
            self.joint_move.set(1, self.role1_noop_legal)
        else:
            self.joint_move.set(0, self.role0_noop_legal)

        self.joint_move.set(node.lead_role_index, child.legal)
        self.sm.next_state(self.joint_move, self.basestate_expand_node)

        new_node = self.create_node(self.basestate_expand_node, node)
        child.to_node = new_node

    def back_propagate(self, path, scores):
        for node in reversed(path):
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

    def puct_constant(self, node):
        constant = self.conf.puct_constant_after

        expansions = self.conf.puct_before_root_expansions if node is self.root else self.conf.puct_before_expansions

        expanded = sum(1 for c in node.children if c.to_node is not None)
        if expanded == len(node.children) or expanded < expansions:
            constant = self.conf.puct_constant_before

        return constant

    def select_child(self, node, depth):
        dirichlet_noise = self.dirichlet_noise(node, depth)
        puct_constant = self.puct_constant(node)

        # get best
        best_child = None
        best_score = -1

        v = float(node.mc_visits + 1)
        pre_mult = (v ** 0.5) * puct_constant
        noise_pct = self.conf.dirichlet_noise_pct

        for idx, child in enumerate(node.children):
            cn = child.to_node

            child_visits = 0.0

            # prior... (alpha go zero said 0 but there score ranges from [-1,1]
            node_score = 0.0

            if cn is not None:
                child_visits = float(cn.mc_visits)
                node_score = cn.mc_score[node.lead_role_index]

                # ensure terminals are enforced more than other nodes (network can return 1.0 for
                # basically dumb moves, if it thinks it will win regardless)
                if cn.is_terminal:
                    node_score *= 1.02

            child_pct = child.policy_prob

            if dirichlet_noise is not None:
                child_pct = (1 - noise_pct) * child_pct + noise_pct * dirichlet_noise[idx]

            puct_score = (pre_mult * child_pct) / (child_visits + 1.0)

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
        scores = None

        while True:
            path.append(current)

            # already expanded terminal
            if current.is_terminal:
                scores = [s / 100.0 for s in current.terminal_scores]
                break

            assert len(current.children) != 0, "MESSED UP - state %s" % str(current.state)

            child = self.select_child(current, len(path) - 1)

            if child.to_node is None:
                self.expand_child(current, child)
                scores = child.to_node.mc_score

                path.append(child.to_node)
                break

            current = child.to_node

        assert scores is not None
        self.back_propagate(path, scores)
        return len(path)

    def playout_loop(self, node, max_iterations, finish_time, cb=None):
        max_depth = -1
        total_depth = 0
        iterations = 0

        start_time = time.time()

        if max_iterations < 0:
            max_iterations = sys.maxint

        while iterations < max_iterations:
            if time.time() > finish_time:
                log.info("RAN OUT OF TIME")
                break

            depth = self.playout(node)
            max_depth = max(depth, max_depth)
            total_depth += depth

            iterations += 1

            if cb and cb():
                break

        if self.conf.verbose:
            if iterations:
                log.info("Iterations: %d" % iterations)
                log.info("Time taken  %.3f" % (time.time() - start_time))

                log.debug("The average depth explored: %.2f, max depth: %d" % (total_depth / float(iterations),
                                                                               max_depth))
            else:
                log.debug("Did no iterations.")

    def on_apply_move(self, joint_move):
        self.game_depth += 1
        if self.conf.verbose:
            log.verbose("on_apply_move @ depth %s" % self.game_depth)

        assert self.root is not None

        lead = self.root.lead_role_index
        other = 0 if lead else 1
        if other == 0:
            assert joint_move.get(other) == self.role0_noop_legal
        else:
            assert joint_move.get(other) == self.role1_noop_legal

        played = joint_move.get(lead)

        found = None
        for c in self.root.children:
            if c.legal == played:
                found = c
            else:
                # allow it to be garbage collected
                c.parent = None
                if c.to_node is not None:
                    c.to_node.parent = None
                    c.to_node = None

        assert found
        new_root = found.to_node
        if new_root is None:
            # need to create it
            self.expand_child(self.root, found)
            new_root = found.to_node
            if self.conf.verbose:
                log.verbose("new root created in apply_move")

        assert new_root is not None
        self.root = new_root

    def on_next_move(self, max_iterations, finish_time):
        if self.conf.root_expansions_preset_visits > 0:
            for c in self.root.children:
                if c.to_node is None:
                    self.expand_child(self.root, c)
                    c.to_node.mc_visits = self.conf.root_expansions_preset_visits

        self.playout_loop(self.root, max_iterations, finish_time)
        return self.choose(finish_time)

    def dump_node(self, node, choice, indent=0):
        class Color:
            PURPLE = '\033[95m'
            CYAN = '\033[96m'
            DARKCYAN = '\033[36m'
            BLUE = '\033[94m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            RED = '\033[91m'
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'
            END = '\033[0m'

            @classmethod
            def pr(self, color, s):
                print "%s%s%s" % (color, s, self.END)

        indent_str = " " * indent
        role = self.sm.get_roles()[node.lead_role_index]

        Color.pr(Color.YELLOW, "%s>>> lead: %s, visits: %s, reward: %s" % (indent_str,
                                                                           role,
                                                                           node.mc_visits,
                                                                           node.final_score[node.lead_role_index]))

        colors = [Color.BOLD]
        rows = []
        temp = 1.0
        if choice is not None:
            temp = max(temp, self.get_temperature())

        for child, prob in self.get_probabilities(node, temp):

            if child == choice:
                colors.append(Color.GREEN)
            else:
                colors.append(Color.CYAN)

            cols = []

            cols.append(child.move)
            cols.append(child.visits())
            cols.append(child.init_policy_prob * 100)
            cols.append(prob * 100)

            node_type = '?'
            if child.to_node is not None:
                node_type = "T" if child.to_node.is_terminal else "*"
            cols.append(node_type)

            if child.to_node is not None:
                n = child.to_node
                cols.append(n.mc_score[node.lead_role_index])
            else:
                cols.append(None)

            cols.append(child.debug_puct_score)
            cols.append(child.debug_node_score + child.debug_puct_score)

            rows.append(cols)

        headers = "move visits policy prob type score ~puct ~select".split()
        for c, line in zip(colors, tabulate(rows, headers, floatfmt=".2f", tablefmt="plain").splitlines()):
            Color.pr(c, indent_str + line)

    def debug_output(self, choice):
        current = self.root

        dump_depth = 0
        dump_node_choice = choice
        while dump_depth < self.conf.max_dump_depth:
            assert not current.is_terminal

            self.dump_node(current, dump_node_choice, indent=dump_depth * 4)
            dump_node_choice = None

            current = current.sorted_children()[0].to_node

            if current is None or current.is_terminal:
                break

            dump_depth += 1

        print "Choice", choice

    def get_probabilities(self, node=None, temperature=1):
        if node is None:
            node = self.root

        total_visits = float(sum(c.visits() for c in node.children))
        if total_visits < 1.0:
            total_visits = 1.0

        temps = [((c.visits() + 1) / total_visits) ** temperature for c in node.children]
        temps_tot = sum(temps)

        probs = [(c, t / temps_tot) for c, t in zip(node.children, temps)]
        probs.sort(key=itemgetter(1), reverse=True)

        return probs

    def noop(self, role_index):
        if self.root.lead_role_index != role_index:
            if role_index == 1:
                return self.role1_noop_legal
            else:
                return self.role0_noop_legal
        return None

    def choose_converge_check(self):
        best_visit = self.root.sorted_children()[0]
        best_score = self.root.sorted_children(by_score=True)[0]
        if best_visit == best_score:
            if self.conf.verbose:
                log.info("Converged - breaking")
            return True
        return False

    def choose_converge(self, finish_time):
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

    def get_temperature(self):
        c = self.conf
        if self.game_depth > c.depth_temperature_stop:
            return -1

        assert c.temperature > 0

        multiplier = 1.0 + (self.game_depth - c.depth_temperature_start) * c.depth_temperature_increment
        multiplier = max(1.0, multiplier)
        return min(c.temperature * float(multiplier), c.depth_temperature_max)

    def choose_temperature(self, finish_time):
        # apply temperature

        temp = self.get_temperature()
        if temp < 0:
            return self.choose_top_visits(finish_time)

        dist = self.get_probabilities(self.root, temp)
        expected_prob = random.random() * self.conf.random_scale

        if self.conf.verbose:
            log.info("* temperature: %s, expected_prob:%s" % (temp, expected_prob))

        seen_prob = 0
        for child, prob in dist:
            seen_prob += prob
            if seen_prob > expected_prob:
                break

        return child


class PUCTPlayer(MatchPlayer):
    ''' puct_evaluator is match agnostic.  '''

    def __init__(self, conf):
        self.puct_evaluator = PUCTEvaluator(conf)
        super().__init__(self.puct_evaluator.identifier)

    def on_meta_gaming(self, finish_time):
        if self.puct_evaluator.conf.verbose:
            log.info("PUCTPlayer, match id: %s" % self.match.match_id)

        self.puct_evaluator.init(self.match.game_info,
                                 self.match.get_current_state())

    def on_apply_move(self, joint_move):
        self.puct_evaluator.on_apply_move(joint_move)

    def on_next_move(self, finish_time):
        pe = self.puct_evaluator
        conf = self.puct_evaluator.conf

        assert pe.root is not None

        resign = False
        if conf.resign_score_value > 0 and pe.root.mc_score is not None:
            for i in range(2):
                s = pe.root.mc_score[i]
                if s < conf.resign_score_value:
                    resign = True
                    break

        if pe.root.lead_role_index == self.match.our_role_index:
            max_iterations = conf.playouts_per_iteration
        else:
            max_iterations = conf.playouts_per_iteration_noop

        if resign:
            max_iterations = min(max_iterations, conf.playouts_per_iteration_resign)
            print "RESIGN", max_iterations

        # choice here is always based on lead_role_index, and not our_role_index
        choice = pe.on_next_move(max_iterations, finish_time)

        noop_res = pe.noop(self.match.our_role_index)
        if noop_res is not None:
            return noop_res
        else:
            if conf.verbose:
                pe.debug_output(choice)

            return choice.legal

    def get_probabilities(self, node=None, temperature=1):
        return self.puct_evaluator.get_probabilities(node=node,
                                                     temperature=temperature)


##############################################################################

def main():
    from ggpzero.util.keras import init

    init()

    port = int(sys.argv[1])
    generation = sys.argv[2]

    config_name = "default"

    if len(sys.argv) > 3:
        config_name = sys.argv[3]

    conf = templates.puct_config_template(generation, config_name)
    player = PUCTPlayer(conf=conf)

    from ggplib.play import play_runner
    play_runner(player, port)


if __name__ == "__main__":
    main()
