#pragma once

#include "scheduler.h"
#include "puct/evaluator.h"

#include <statemachine/statemachine.h>

namespace GGPZero {
    // forwards
    class PuctEvaluator;

    class TestSelfPlay {
    public:
        TestSelfPlay(NetworkScheduler* scheduler, const GGPLib::BaseState* initial_state,
                     int base_iterations, int sample_iterations);
        ~TestSelfPlay();

    public:
        void playOnce();

    private:
        NetworkScheduler* scheduler;
        PuctEvaluator pe;

        // config... XXX add config object
        const GGPLib::BaseState* initial_state;
        int base_iterations;
        int sample_iterations;
    };

}
