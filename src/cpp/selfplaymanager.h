#pragma once

#include "gdltransformer.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <vector>

namespace GGPZero {

    // forwards
    class SelfPlay;
    class PuctEvaluator;
    class SelfPlayConfig;

    class SelfPlayManager {
    public:
        SelfPlayManager(GGPLib::StateMachineInterface* sm,
                   const GdlBasesTransformer* transformer,
                   int batch_size);
        ~SelfPlayManager();

    public:
        // only called from self player
        const GGPLib::BaseState::HashSet* getUniqueStates() const {
            return &this->unique_states;
        };

        void addSample(Sample* sample);

    public:
        void startSelfPlayers(const SelfPlayConfig* config);
        std::vector <Sample*> getSamples();

        int poll(float* policies, float* final_scores, int pred_count);
        float* getBuf() const;


        void addUniqueState(const GGPLib::BaseState* bs);
        void clearUniqueStates();

    private:
        GGPLib::StateMachineInterface* sm;
        const GdlBasesTransformer* transformer;
        int batch_size;

        std::vector <SelfPlay*> self_plays;

        // local scheduler
        NetworkScheduler* scheduler;

        std::vector <Sample*> samples;
        GGPLib::BaseState::HashSet unique_states;
        std::vector <GGPLib::BaseState*> states_allocated;
    };
}
