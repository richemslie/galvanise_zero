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
    struct SelfPlayConfig;
    class PuctEvaluator;

    class SelfPlayManager {
    public:
        SelfPlayManager(GGPLib::StateMachineInterface* sm,
                        const GdlBasesTransformer* transformer,
                        int batch_size,
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

        void incrEarlyRunToEnds() {
            this->number_early_run_to_ends++;
        }

        void incrResigns() {
            this->number_resigns++;
        }

        void incrAbortsGameLength() {
            this->number_aborts_game_length++;
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
        int number_early_run_to_ends;
        int number_resigns;
        int number_aborts_game_length;
    };
}
