#pragma once

#include "supervisorbase.h"
#include "bases.h"
#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"
#include "greenlet/greenlet.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
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
        // C++ side...
        PuctNode* expandChild(PuctEvaluator* pe, const PuctNode* parent, const PuctNodeChild* child);
        PuctNode* createNode(PuctEvaluator* pe, const GGPLib::BaseState* bs);
        void runScheduler();

        // python side (add level of indirection)...

        void puctPlayerStart();
        void puctApplyMove(const GGPLib::JointMove* move);
        void puctPlayerMove(const GGPLib::BaseState* state, int iterations, double end_time);
        int puctPlayerGetMove(int lead_role_index);

        void selfPlayTest(int num_selfplays, int base_iterations, int sample_iterations);
        int poll(float* policies, float* final_scores, int pred_count);
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

        PuctEvaluator* player_pe;

        greenlet_t* top;
        greenlet_t* scheduler;
    };
}
