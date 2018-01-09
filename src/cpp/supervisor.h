#pragma once

#include "puctnode.h"
#include "greenlet/greenlet.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <vector>

namespace GGPZero {

    class MasterGreenlet: greenlet {
        void *run(void *arg) {
            printf("HERE1\n");

            greenlet* other = (greenlet*) arg;
            other->switch_to();
            return nullptr;
        }
    };

    // forwards
    class PuctEvaluator;
    class GdlBasesTransformer;

    struct Sample {
        GGPLib::BaseState* state;
        std::vector <GGPLib::BaseState*> prev_states;
        std::vector <float*> policy;
        std::vector <int> final_score;
        int depth;
        int game_length;
        int lead_role_index;
    };

    struct JumpArgs {
        JumpArgs(PuctNode* parent, PuctNodeChild* child, GGPLib::BaseState* state) :
            parent(parent),
            child(child),
            state(state) {
        }

        PuctNode* parent;
        PuctNodeChild* child;
        GGPLib::BaseState* state;
    };


    class Supervisor {
    public:
        Supervisor(GGPLib::StateMachineInterface* sm, GdlBasesTransformer* transformer);
        ~Supervisor();

    public:
        int getRoleCount() const {
            return this->sm->getRoleCount();
        }

        PuctNode* expandChild(PuctEvaluator* pe, PuctNode* parent, PuctNodeChild* child);
        PuctNode* createNode(PuctEvaluator* pe, const GGPLib::BaseState* bs);


        int sampleCount() const {
            return this->num_samples;
        }

        int predictionCount() const {
            return 42;
        }

        float* getChannelsToPrediction() {
            return nullptr;
        }

         void predictionsMade(float* predictions) {
            return;
        }

    private:
        GGPLib::StateMachineInterface* sm;
        GGPLib::BaseState* basestate_expand_node;
        GdlBasesTransformer* transformer;

        int num_samples;

        // two buffers, as keeps adding predictions while tensortflow predicts
        int use_buf_id;

        float* channel_buffer0;
        int buffer_next_index0;
        std::vector <PuctEvaluator*> requestors0;

        float* channel_buffer1;
        int buffer_next_index1;
        std::vector <PuctEvaluator*> requestors1;


        float* channel_buffer;
        int buffer_next_index;
        std::vector <PuctEvaluator*> requestors;

        MasterGreenlet master;
    };

}
