#pragma once

namespace GGPZero {

    struct PredictDoneEvent {
        int pred_count;
        float* policies;
        float* final_scores;
    };

    struct ReadyEvent {
        int pred_count;
        float* channel_buf;
    };
}
