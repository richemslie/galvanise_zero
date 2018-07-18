import random

from ggplib.player.proxy import ProxyPlayer

from interface import create_gurgeh_cpp_player


class MatchInfo(object):
    def __init__(self, sm):
        self.sm = sm
        self.two_player_fixed_sum = True
        self.simultaneous_game_detected = False

        self.static_joint_move = self.sm.get_joint_move()
        self.static_basestate = self.sm.new_base_state()

    def do_basic_depth_charge(self):
        ''' identifies types of moves '''
        self.sm.reset()
        role_count = len(self.sm.get_roles())

        while True:
            if self.sm.is_terminal():
                break

            choice_counts_more_than_1 = 0
            for idx, r in enumerate(range(role_count)):
                ls = self.sm.get_legal_state(idx)
                choice = ls.get_legal(random.randrange(0, ls.get_count()))
                self.static_joint_move.set(idx, choice)

                assert ls.get_count()

                if ls.get_count() > 1:
                    choice_counts_more_than_1 += 1

            if not self.simultaneous_game_detected and choice_counts_more_than_1 > 1:
                self.simultaneous_game_detected = True

            self.sm.next_state(self.static_joint_move, self.static_basestate)
            self.sm.update_bases(self.static_basestate)

        if role_count > 1:
            total_score = 0
            for idx, _ in enumerate(self.sm.get_roles()):
                total_score += self.sm.get_goal_value(idx)

            if total_score != 100:
                self.two_player_fixed_sum = False


class GurgehPlayer(ProxyPlayer):
    thread_workers = 3
    skip_single_moves = True

    max_tree_search_time = 30
    max_number_of_nodes = 100000000
    max_memory = 1024 * 1024 * 1024 * 20
    max_tree_playout_iterations = 100000000

    initial_ucb_constant = 0.5
    upper_adjust_ucb_constant = 1.15
    lower_adjust_ucb_constant = 0.25

    lead_first_node_ucb_constant = 1.15
    lead_first_node_time_pct = 0.75

    select_random_move_count = 16
    selection_use_scores = True

    dump_depth = 2
    next_time = 2.5

    def meta_create_player(self):
        role_count = len(self.sm.get_roles())
        info = MatchInfo(self.sm)
        if role_count > 1:
            for _ in range(5):
                info.do_basic_depth_charge()

        return create_gurgeh_cpp_player(self.sm,
                                        self.match.our_role_index,

                                        info.two_player_fixed_sum,
                                        self.thread_workers,
                                        self.skip_single_moves,

                                        self.max_tree_search_time,
                                        self.max_number_of_nodes,
                                        self.max_memory,
                                        self.max_tree_playout_iterations,

                                        self.initial_ucb_constant,
                                        self.upper_adjust_ucb_constant,
                                        self.lower_adjust_ucb_constant,

                                        self.lead_first_node_ucb_constant,
                                        self.lead_first_node_time_pct,

                                        self.select_random_move_count,
                                        self.selection_use_scores,

                                        self.dump_depth,
                                        self.next_time)


def main():
    import sys
    from ggplib.play import play_runner
    port = int(sys.argv[1])

    # if second argument, set to player name
    try:
        player_name = sys.argv[2]
    except IndexError:
        player_name = "Gurgeh"

    player = GurgehPlayer(player_name)
    play_runner(player, port)


if __name__ == "__main__":
    main()
