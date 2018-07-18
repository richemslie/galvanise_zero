#pragma once

#include "config.h"

#include <player/player.h>

#include <player/node.h>
#include <player/path.h>
#include <player/rollout.h>

#include <statemachine/statemachine.h>
#include <statemachine/basestate.h>
#include <statemachine/legalstate.h>
#include <statemachine/jointmove.h>

#include <k273/util.h>

namespace PlayerMcts {

    class Worker;

    class Event {
    public:
        enum EventType {
            EXPANSION = 0,
            ROLLOUT = 1
        };

    public:
        Event(EventType t, Worker* w) :
            event_type(t),
            worker(w) {
        }

    public:
        EventType event_type;
        Worker* worker;
    };

    typedef K273::LockedQueue <Event*> WorkerEventQueue;

    class Worker : public K273::WorkerInterface {
    public:
        Worker(WorkerEventQueue* event_queue, GGPLib::StateMachineInterface* sm,
               const Config*, int our_role_index);
        virtual ~Worker();

    public:
        void reset();

    private:
        double normaliseScore(int score0, int score1, int role_index);
        GGPLib::Node* createNode(const GGPLib::BaseState* bs);

        bool expandNode();
        void doRollout(const GGPLib::Node* node);

        void doWork();

    public:
        bool is_reset;
        GGPLib::Node* new_node;

        std::vector<double> score;
        GGPLib::Path::Selected path;

        bool did_rollout;

        // stats
        double time_for_expansion;
        double time_for_rollout;

    private:
        const Config* config;
        const int our_role_index;

        GGPLib::DepthChargeRollout* rollout;

        WorkerEventQueue* event_queue;
        Event event_expansion;
        Event event_rollout;

        GGPLib::StateMachineInterface* sm;

        // malloc this
        GGPLib::BaseState* base_state;
        K273::Random random;
    };
}
