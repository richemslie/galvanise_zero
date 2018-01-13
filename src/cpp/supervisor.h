#pragma once

#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"
#include "gdltransformer.h"

#include "greenlet/greenlet.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <deque>
#include <vector>

namespace GGPZero {

    // forwards
    class PuctEvaluator;
    class SelfPlay;
    class SelfPlayConfig;

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
        void puctPlayerStart(PuctConfig* conf);
        void puctPlayerReset();
        void puctApplyMove(const GGPLib::JointMove* move);
        void puctPlayerMove(const GGPLib::BaseState* state, int iterations, double end_time);
        int puctPlayerGetMove(int lead_role_index);

        void startSelfPlay(const SelfPlayConfig* config);
        std::vector <Sample*> getSamples();

        int poll(float* policies, float* final_scores, int pred_count);

        float* getBuf() const;

    private:
        GGPLib::StateMachineInterface* sm;
        PuctEvaluator* player_pe;
        bool first_play;

        // inline only XXX
        std::vector <SelfPlay*> self_plays;

        // workers/pools of schedulers XXX

        // inline scheduler
        NetworkScheduler* inline_scheduler;
    };
}
