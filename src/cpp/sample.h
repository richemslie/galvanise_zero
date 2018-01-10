#pragma once

#include <vector>

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

namespace GGPZero {

    struct Sample {
        GGPLib::BaseState* state;
        std::vector <GGPLib::BaseState*> prev_states;
        std::vector <float*> policy;
        std::vector <int> final_score;
        int depth;
        int game_length;
        int lead_role_index;
    };

}
