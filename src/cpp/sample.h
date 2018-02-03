#pragma once

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <vector>
#include <utility>
#include <string>

namespace GGPZero {

    struct Sample {
        typedef std::vector <std::pair <int, float>> Policy;

        GGPLib::BaseState* state;
        std::vector <GGPLib::BaseState*> prev_states;
        std::vector <Policy>  policies;
        std::vector <float> final_score;
        int depth;
        int game_length;
        std::string match_identifier;
        bool has_resigned;
        bool resign_false_positive;
        int starting_sample_depth;
        std::vector <float> resultant_puct_score;
        int resultant_puct_visits;
    };

}
