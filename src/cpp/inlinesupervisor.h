#pragma once

#include "supervisorbase.h"
#include "bases.h"
#include "puct/node.h"
#include "greenlet/greenlet.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <deque>
#include <vector>

namespace GGPZero {

    struct Runnable {
        Runnable(greenlet_t* g, void* a) :
            greenlet(g),
            arg(a) {
        }
        greenlet_t* greenlet;
        void* arg;
    };

    // forwards
    class PuctEvaluator;

    class InlineSupervisor : public SupervisorBase {
    public:
        InlineSupervisor(GGPLib::StateMachineInterface* sm,
                        GdlBasesTransformer* transformer,
                        int batch_size,
                        int expected_policy_size,
                        int role_1_index);

        virtual ~InlineSupervisor() {
        }

    public:
        PuctNode* expandChild(PuctEvaluator* pe, const PuctNode* parent, const PuctNodeChild* child);
        PuctNode* createNode(PuctEvaluator* pe, const GGPLib::BaseState* bs);
        void finish();

        void mainLoop();

        void start_greenlet(void* arg) {
            this->mainLoop();
        }

        int test(float* policies, float* final_scores, int pred_count);
        float* getBuf() const {
            return this->channel_buf;
        }

    private:
        GdlBasesTransformer* transformer;
        const unsigned int batch_size;

        GGPLib::BaseState* basestate_expand_node;
        std::vector <greenlet_t*> requestors;
        std::deque <Runnable> runnables;

        int expected_policy_size;
        int role_1_index;

        float* policies;
        float* final_scores;
        int pred_count;

        float* channel_buf;
        int channel_buf_indx;
        greenlet_t* master;
        greenlet_t* top;
    };
}
