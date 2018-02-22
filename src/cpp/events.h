#pragma once

#include <vector>

namespace GGPZero {

    struct PredictDoneEvent {
        int pred_count;
        std::vector <float*> policies;
        float* final_scores;
    };

    struct ReadyEvent {
        // how much of the buffer is used (must be an exact multiple of channels*channel_size)
        int buf_count;
        float* channel_buf;
    };
}
