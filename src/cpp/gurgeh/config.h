#pragma once

namespace PlayerMcts {

    struct Config {
        bool fixed_sum_game;
        unsigned int thread_workers;
        bool skip_single_moves;

        double max_tree_search_time;
        int max_number_of_nodes;
        long max_memory;
        long max_tree_playout_iterations;

        double initial_ucb_constant;
        double upper_adjust_ucb_constant;
        double lower_adjust_ucb_constant;

        double lead_first_node_ucb_constant;
        double lead_first_node_time_pct;

        int select_random_move_count;

        bool selection_use_scores = true;

        int dump_depth;
        double next_time;
    };
}

