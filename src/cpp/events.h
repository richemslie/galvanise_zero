#pragma once

#include <vector>

namespace GGPZero {

    struct PredictDoneEvent {
        int pred_count;
        std::vector <float*> policies;
        float* final_scores;
    };

    struct ReadyEvent {
        int pred_count;
        float* channel_buf;
    };
}
