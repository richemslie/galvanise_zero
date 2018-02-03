#pragma once

#include "events.h"
#include "uniquestates.h"
#include "gdltransformer.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <string>
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
                        int batch_size,
                        int number_of_previous_states,
                        UniqueStates* unique_states,
                        std::string identifier);
        ~SelfPlayManager();

    public:
        // the following are only called from self player
        Sample* createSample(const PuctEvaluator* pe, const PuctNode* node);
        void addSample(Sample* sample);

        UniqueStates* getUniqueStates() const {
            return this->unique_states;
        };

        void incrDupes() {
            this->saw_dupes++;
        }

        void incrNoSamples() {
            this->no_samples_taken++;
        }

        void incrResign0FalsePositives() {
            this->false_positive_resigns0++;
        }

        void incrResign1FalsePositives() {
            this->false_positive_resigns1++;
        }

    public:
        void startSelfPlayers(const SelfPlayConfig* config);

        void poll();

        void reportAndResetStats();

        std::vector <Sample*>& getSamples() {
            return this->samples;
        }

        ReadyEvent* getReadyEvent() {
            return &this->ready_event;
        }

        PredictDoneEvent* getPredictDoneEvent() {
            return &this->predict_done_event;
        }

    private:
        GGPLib::StateMachineInterface* sm;
        const GdlBasesTransformer* transformer;
        int batch_size;

        // number of previous states to store in sample
        int number_of_previous_states;

        std::vector <SelfPlay*> self_plays;

        // local scheduler
        NetworkScheduler* scheduler;

        std::vector <Sample*> samples;
        UniqueStates* unique_states;
        std::string identifier;

        std::vector <GGPLib::BaseState*> states_allocated;

        // Events
        ReadyEvent ready_event;
        PredictDoneEvent predict_done_event;


        // stats
        int saw_dupes;
        int no_samples_taken;
        int false_positive_resigns0;
        int false_positive_resigns1;
    };
}
