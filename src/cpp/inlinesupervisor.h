#pragma once

#include "bases.h"
#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"
#include "greenlet/greenlet.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <deque>
#include <vector>

namespace GGPZero {

    // forwards
    class PuctEvaluator;

    class Supervisor {
    public:
        Supervisor(GGPLib::StateMachineInterface* sm);
        ~Supervisor();

    public:
        // create a scheduler
        NetworkScheduler* createScheduler(const GdlBasesTransformer* transformer,
                                          int batch_size,
                                          int expected_policy_size,
                                          int role_1_index);

        // python side (add level of indirection)...
        void puctPlayerStart();
        void puctApplyMove(const GGPLib::JointMove* move);
        void puctPlayerMove(const GGPLib::BaseState* state, int iterations, double end_time);
        int puctPlayerGetMove(int lead_role_index);

        void selfPlayTest(int num_selfplays, int base_iterations, int sample_iterations);

        int poll(float* policies, float* final_scores, int pred_count);

        float* getBuf() const;

    private:
        GGPLib::StateMachineInterface* sm;
        PuctEvaluator* player_pe;
        void* self_play_manager;

        // workers/pools of schedulers XXX

        // inline scheduler
        NetworkScheduler* inline_scheduler;
    };
}
