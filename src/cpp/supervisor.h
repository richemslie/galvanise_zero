#pragma once

#include "uniquestates.h"

#include "events.h"

#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"
#include "gdltransformer.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <string>
#include <vector>

#include <k273/util.h>

namespace GGPZero {

    // forwards
    class PuctEvaluator;
    class SelfPlay;
    class SelfPlayManager;
    struct SelfPlayConfig;

    typedef K273::LockedQueue <SelfPlayManager*> ReadyQueue;
    typedef K273::LockedQueue <SelfPlayManager* > PredictDoneQueue;

    class SelfPlayWorker : public K273::WorkerInterface {
    public:
        SelfPlayWorker(SelfPlayManager* man0, SelfPlayManager* man1,
                       const SelfPlayConfig* config);
        virtual ~SelfPlayWorker();

    private:
        void doWork();

    public:
        // supervisor side:
        SelfPlayManager* pull() {
            if (!this->outbound_queue.empty()) {
                return this->outbound_queue.pop();
            }

            return nullptr;
        }

        // supervisor side:
        void push(SelfPlayManager* manager) {
            this->inbound_queue.push(manager);
            this->getThread()->promptWorker();
        }

    private:
        // worker pulls from here when no more workers available
        PredictDoneQueue inbound_queue;

        // worker pushes on here when done
        ReadyQueue outbound_queue;

        bool enter_first_time;
        const SelfPlayConfig* config;

        // will be two
        SelfPlayManager* man0;
        SelfPlayManager* man1;
    };

    class Supervisor {
    public:
        Supervisor(GGPLib::StateMachineInterface* sm,
                   const GdlBasesTransformer* transformer,
                   int batch_size,
                   std::string identifier);
        ~Supervisor();

    private:
        void slowPoll(SelfPlayManager* manager);

    public:
        void createInline(const SelfPlayConfig* config);
        void createWorkers(const SelfPlayConfig* config);

        std::vector <Sample*> getSamples();

        const ReadyEvent* poll(int predict_count, std::vector <float*>& data);

        void addUniqueState(const GGPLib::BaseState* bs);
        void clearUniqueStates();

    private:
        GGPLib::StateMachineInterface* sm;
        const GdlBasesTransformer* transformer;
        const int batch_size;
        const std::string identifier;

        int slow_poll_counter;

        SelfPlayManager* inline_sp_manager;

        SelfPlayManager* in_progress_manager;
        SelfPlayWorker* in_progress_worker;
        std::vector <SelfPlayWorker*> self_play_workers;

        std::vector <Sample*> samples;
        UniqueStates unique_states;
    };
}
