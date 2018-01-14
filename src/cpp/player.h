#pragma once

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <deque>
#include <vector>

namespace GGPZero {

    // forwards
    class PuctConfig;
    class PuctEvaluator;
    class NetworkScheduler;
    class GdlBasesTransformer;

    // this is a bit of hack, wasnt really designed to actually play from c++
    class Player {
    public:
        Player(GGPLib::StateMachineInterface* sm,
               const GdlBasesTransformer* transformer,
               PuctConfig* conf);
        ~Player();

    public:
        // python side
        void puctPlayerReset();
        void puctApplyMove(const GGPLib::JointMove* move);
        void puctPlayerMove(const GGPLib::BaseState* state, int iterations, double end_time);
        int puctPlayerGetMove(int lead_role_index);

        NetworkScheduler* getScheduler() {
            return this->scheduler;
        }

    private:
        PuctEvaluator* evaluator;
        NetworkScheduler* scheduler;

        bool first_play;
    };

}
