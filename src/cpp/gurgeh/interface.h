#pragma once

#define StateMachine void
#define PlayerBase void
#define boolean int

#ifdef __cplusplus
extern "C" {
#endif

    // CFFI START INCLUDE

PlayerBase*  Player__createGurgehPlayer(StateMachine* _sm,
                                        int our_role_index,

                                        int fixed_sum_game,
                                        int thread_workers,
                                        int skip_single_moves,

                                        double max_tree_search_time,
                                        int max_number_of_nodes,
                                        long max_memory,
                                        long max_tree_playout_iterations,

                                        double initial_ucb_constant,
                                        double upper_adjust_ucb_constant,
                                        double lower_adjust_ucb_constant,

                                        double lead_first_node_ucb_constant,
                                        double lead_first_node_time_pct,

                                        int select_random_move_count,
                                        int selection_use_scores,

                                        int dump_depth,
                                        double next_time);

    // CFFI END INCLUDE

#ifdef __cplusplus
}
#endif

#undef StateMachine
#undef boolean
#undef PlayerBase

