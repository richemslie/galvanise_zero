#pragma once

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <vector>
#include <utility>
#include <string>

namespace GGPZero {

    struct Sample {
        GGPLib::BaseState* state;
        std::vector <GGPLib::BaseState*> prev_states;
        std::vector <std::pair <int, float>> policy;
        int depth;
        int lead_role_index;

        // update at end of game
        int game_length;
        std::vector <float> final_score;

        std::string match_identifier;
        bool has_resigned;
        bool resign_false_positive;
        int starting_sample_depth;
    };

}
