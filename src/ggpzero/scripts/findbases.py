import random

from ggplib.db import lookup

from ggpzero.defs import confs


MAX_STATES_FOR_ROLLOUT = 500


class Rollout(object):
    def __init__(self, game_info):
        self.game_info = game_info
        self.sm = game_info.get_sm()

        self.states = [self.sm.new_base_state() for _ in range(MAX_STATES_FOR_ROLLOUT)]

        # get and cache fast move
        self.static_joint_move = self.sm.get_joint_move()
        self.lookahead_joint_move = self.sm.get_joint_move()

        self.depth = None
        self.legals = None
        self.scores = None

        def get_noop_idx(actions):
            for idx, a in enumerate(actions):
                if "noop" in a:
                    return idx
            assert False, "did not find noop"

        self.role0_noop_legal, self.role1_noop_legal = map(get_noop_idx, game_info.model.actions)

        # this is really approximate, works for some games
        assert len(self.game_info.model.roles) == 2
        role0, role1 = self.game_info.model.roles
        self.piece_counts = []
        for b in self.game_info.model.bases:
            if 'control' in b:
                self.piece_counts.append(None)
            elif role0 in b and role1 not in b:
                self.piece_counts.append(role0)
            elif role1 in b and role0 not in b:
                self.piece_counts.append(role1)
            else:
                self.piece_counts.append(None)

    def count_states(self, basestate, ri):
        role = self.game_info.model.roles[ri]
        total = 0
        for i in range(basestate.len()):
            if basestate.get(i) == 0:
                continue
            if self.piece_counts[i] == role:
                total += 1
        return total

    def reset(self):
        self.sm.reset()

        # (lead_role_index, legal)
        self.legals = []
        self.sm.get_current_state(self.states[0])

    def make_data(self, unique_states):
        state = None
        for _ in range(self.depth):
            d = random.randrange(self.depth)
            a_state = tuple(self.states[d].to_list())

            final_score = [s / 100.0 for s in self.scores]
            lead_role_index, main_legal = self.legals[d]

            if a_state not in unique_states:
                state = a_state
                unique_states.add(state)
                break

        if state is None:
            return None

        self.sm.update_bases(self.states[d])

        ls = self.sm.get_legal_state(lead_role_index)
        ADD_K = 0.3
        total = float(ls.get_count() * (1.0 + ADD_K))

        legals = ls.to_list()
        policy_dist = [(l, (1 / total + random.random() / (10 * ls.get_count()))) for l in ls.to_list()]
        policy_dist[legals.index(main_legal)] = (main_legal, (ls.get_count() * ADD_K + 1) / total)

        total = sum(p for _, p in policy_dist)
        policy_dist = [(l, p / total) for l, p in policy_dist]

        # now we can create a sample :)
        return confs.Sample(None, state, policy_dist, final_score, d, self.depth, lead_role_index)

    def get_current_state(self):
        return self.states[self.depth]

    def choose_move(self, lead_role_index):
        other_role_index = 0 if lead_role_index else 1

        # set other move
        ls_other = self.sm.get_legal_state(other_role_index)
        assert ls_other.get_count() == 1
        self.lookahead_joint_move.set(other_role_index, ls_other.get_legal(0))

        # steal new state for now...
        next_state = self.states[self.depth + 1]

        ls = self.sm.get_legal_state(lead_role_index)
        best_moves = []
        best_count = -1

        # want to reduce this
        for ii in range(ls.get_count()):
            legal = ls.get_legal(ii)
            self.lookahead_joint_move.set(lead_role_index, legal)
            self.sm.next_state(self.lookahead_joint_move, next_state)

            # move forward and see if we won the game?
            self.sm.update_bases(next_state)
            if self.sm.is_terminal():
                if self.sm.get_goal_value(lead_role_index) == 100:
                    # return this move (but fix the state of statemachine first)
                    self.sm.update_bases(self.get_current_state())
                    return legal

            count = self.count_states(next_state, other_role_index)
            if count > best_count:
                best_moves = [legal]
                best_count = count
            elif count == best_count:
                best_moves.append(legal)

            # revert statemachine
            self.sm.update_bases(self.get_current_state())

        return random.choice(best_moves)

    def run(self):
        self.reset()

        self.depth = 0
        self.legals = []
        while True:
            if self.sm.is_terminal():
                break

            # play move
            ls = self.sm.get_legal_state(0)
            if ls.get_count() == 1 and ls.get_legal(0) == self.role0_noop_legal:
                lead_role_index = 1
                self.static_joint_move.set(0, self.role0_noop_legal)
            else:
                lead_role_index = 0
                self.static_joint_move.set(1, self.role1_noop_legal)

            choice = self.choose_move(lead_role_index)

            self.static_joint_move.set(lead_role_index, choice)
            self.legals.append((lead_role_index, choice))

            # borrow the next state (side affect of assigning it)
            next_state = self.states[self.depth + 1]
            self.sm.next_state(self.static_joint_move, next_state)
            self.sm.update_bases(next_state)

            self.depth += 1

        self.scores = []
        for ii, _ in enumerate(self.sm.get_roles()):
            self.scores.append(self.sm.get_goal_value(ii))


def do_data_samples(game):
    game_info = lookup.by_name(game)
    r = Rollout(game_info)

    # perform a bunch of rollouts
    unique_states = set()

    samples = []
    try:
        for i in range(1000):
            r.run()
            sample = None
            for _ in range(10):
                sample = r.make_data(unique_states)
                if sample is not None:
                    break

            if sample is None:
                print "DUPE NATION", i
                continue

            samples.append(sample)

            if i % 50 == 0:
                print i

    except KeyboardInterrupt:
        pass

    for s in samples:
        print s.final_score


def determine_cords(game):
    # go through each base by "base term" and determine if is
    # (a) control
    # (b) a useful control
    # (c) a cord state
    # (d) something else

    game_info = lookup.by_name(game)

    for b in game_info.model.bases:
        print b

    # control :1
    # captureCount - scaled [0-16] :1

    # canEnPassantCapture :1 (full column)

    # aRookHasMoved :2
    # hRookHasMoved :2
    # kingHasMoved :2

    # step 1-101 :1 scaled
    # pieces (king queen rook bishop knight pawn) * (white black) :12
    # total channels 12 + 10 == 22


if __name__ == "__main__":
    import sys
    try:
        game = sys.argv[1]
    except IndexError:
        game = "skirmishNew"

    # determine_cords(game)
    do_data_samples(game)
