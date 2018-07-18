#include "interface.h"

#include "config.h"
#include "mcts.h"

#include <player/player.h>
#include <statemachine/statemachine.h>
#include "statemachine/goalless_sm.h"
#include "statemachine/combined.h"

void* Player__createGurgehPlayer(void* _sm, int our_role_index,

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
                                 double next_time) {

    PlayerMcts::Config* config = new PlayerMcts::Config;
    config->fixed_sum_game = (bool) fixed_sum_game;
    config->thread_workers = thread_workers;
    config->skip_single_moves = (bool) skip_single_moves;

    config->max_tree_search_time = max_tree_search_time;
    config->max_number_of_nodes = max_number_of_nodes;
    config->max_memory = max_memory;
    config->max_tree_playout_iterations = max_tree_playout_iterations;

    config->initial_ucb_constant = initial_ucb_constant;
    config->upper_adjust_ucb_constant = upper_adjust_ucb_constant;
    config->lower_adjust_ucb_constant = lower_adjust_ucb_constant;

    config->lead_first_node_ucb_constant = lead_first_node_ucb_constant;
    config->lead_first_node_time_pct = lead_first_node_time_pct;

    config->select_random_move_count = select_random_move_count;
    config->selection_use_scores = (bool) selection_use_scores;

    config->dump_depth = dump_depth;
    config->next_time = next_time;

    GGPLib::StateMachineInterface* sm = dynamic_cast<GGPLib::StateMachineInterface*> (static_cast<GGPLib::StateMachine*> (_sm));
    GGPLib::PlayerBase* player = new PlayerMcts::Player(sm, our_role_index, config);
    return (void *) player;
}
