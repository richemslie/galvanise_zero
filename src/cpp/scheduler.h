#pragma once

#include "greenlet/greenlet.h"
#include "puct/node.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <k273/exception.h>

#include <deque>
#include <vector>
#include <string>


namespace GGPZero {

    // forwards
    class Sample;
    class PuctEvaluator;
    class GdlBasesTransformer;

    class NetworkScheduler {
    public:
        NetworkScheduler(GGPLib::StateMachineInterface* sm,
                         const GdlBasesTransformer* transformer,
                         int batch_size);
        ~NetworkScheduler();

    public:
        // interface called back from puct evaluator:
        std::string moveString(const GGPLib::JointMove& move);
        void dumpNode(const PuctNode* node, const PuctNodeChild* highlight,
                      const std::string& indent, bool sort_by_next_probability);

        Sample* createSample(const PuctNode* node);

        int getRoleCount() const {
            return this->sm->getRoleCount();
        }

        PuctNode* expandChild(PuctEvaluator* pe, const PuctNode* parent,
                              const PuctNodeChild* child);

        PuctNode* createNode(PuctEvaluator* pe, const GGPLib::BaseState* bs);


    public:
        // called from client:
        template <typename Callable>
        void addRunnable(Callable& f) {
            ASSERT(this->main_loop != nullptr);
            greenlet_t* g = createGreenlet<Callable> (f, this->main_loop);
            this->runnables.push_back(g);
        }

        void createMainLoop() {
            ASSERT(this->main_loop == nullptr);
            this->main_loop = createGreenlet([this]() {
                    return this->mainLoop();
                });
        }

        int poll(float* policies, float* final_scores, int pred_count);
        float* getBuf() const {
            return this->channel_buf;
        }

    private:
        void mainLoop();

    private:
        GGPLib::StateMachineInterface* sm;
        const GdlBasesTransformer* transformer;
        const unsigned int batch_size;

        GGPLib::BaseState* basestate_expand_node;
        std::vector <greenlet_t*> requestors;
        std::deque <greenlet_t*> runnables;

        // the main looper
        greenlet_t* main_loop;

        // exit in and of the main_loop (and is parent of main_loop)
        greenlet_t* top;

        // inbound predictions
        float* policies;
        float* final_scores;
        int pred_count;

        // outbound predictions
        float* channel_buf;
        int channel_buf_indx;
   };
}
