#pragma once

#include "events.h"
#include "scheduler.h"
#include "gdltransformer.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <vector>

namespace GGPZero::PuctV2 {

    // forwards
    class PuctEvaluator;
    struct PuctConfig;
    struct PuctNodeChild;

    // this is a bit of hack, wasnt really designed to actually play from c++
    class Player {

    public:
        Player(GGPLib::StateMachineInterface* sm,
               const GGPZero::GdlBasesTransformer* transformer,
               PuctConfig* conf);
        ~Player();

    public:
        // python side
        void updateConfig(float think_time, int converge_relaxed,
                          int converge_non_relaxed, bool verbose);

        void puctPlayerReset(int game_depth);
        void puctApplyMove(const GGPLib::JointMove* move);
        void puctPlayerMove(const GGPLib::BaseState* state, int iterations, double end_time);
        std::tuple <int, float, int> puctPlayerGetMove(int lead_role_index);

        const ReadyEvent* poll(int predict_count, std::vector <float*>& data);

    private:
        const GdlBasesTransformer* transformer;
        PuctConfig* config;

        PuctEvaluator* evaluator;
        NetworkScheduler* scheduler;

        bool first_play;

        // store the choice of onNextMove()...
        const PuctNodeChild* on_next_move_choice;

        // Events
        ReadyEvent ready_event;
        PredictDoneEvent predict_done_event;
    };

}
