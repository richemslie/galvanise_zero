#pragma once

#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"
#include "gdltransformer.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <vector>

#include <k273/util.h>

namespace GGPZero {

    // forwards
    class PuctEvaluator;
    class SelfPlay;
    class SelfPlayConfig;
    class SelfPlayManager;

    typedef K273::LockedQueue <SelfPlayManager*> ReadyQueue;

    class SelfPlayWorker : public K273::WorkerInterface {
    public:
        SelfPlayWorker(std::vector <SelfPlayManager*> self_players) :
            managers_available(self_players) {
        }

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
        std::vector <SelfPlayManager*> managers_available;

        // worker pulls from here when no more workers available
        ReadyQueue inbound_queue;

        // worker pushes on here when done
        ReadyQueue outbound_queue;
    };


    class Supervisor {
    public:
        Supervisor(GGPLib::StateMachineInterface* sm,
                   const GdlBasesTransformer* transformer,
                   int batch_size);
        ~Supervisor();

    public:
        void createInline(const SelfPlayConfig* config);

        // basically proxies to manager/workers
        std::vector <Sample*> getSamples();

        int poll(float* policies, float* final_scores, int pred_count);
        float* getBuf() const;

        void addUniqueState(const GGPLib::BaseState* bs);
        void clearUniqueStates();

    private:
        GGPLib::StateMachineInterface* sm;
        const GdlBasesTransformer* transformer;
        const int batch_size;

        SelfPlayManager* inline_sp_manager;

        std::vector <SelfPlayWorker*> self_play_workers;
        SelfPlayManager* current;
    };
}
