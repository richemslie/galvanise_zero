#pragma once

#include "events.h"

#include "puct/node.h"

#include "greenlet/greenlet.h"

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
        NetworkScheduler(const GdlBasesTransformer* transformer,
                         int role_count, int batch_size);
        ~NetworkScheduler();

    private:
        void updateFromPolicyHead(const int idx, PuctNode* node);
        void updateFromValueHead(const int idx, PuctNode* node);

    public:
        void evaluateNode(PuctEvaluator* pe, PuctNode* node);

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

        void poll(const PredictDoneEvent* predict_done_event, ReadyEvent* ready_event);

    private:
        void mainLoop();

    private:
        const GdlBasesTransformer* transformer;
        const int role_count;
        const unsigned int batch_size;

        std::vector <greenlet_t*> requestors;
        std::deque <greenlet_t*> runnables;

        // the main looper
        greenlet_t* main_loop;

        // exit in and of the main_loop (and is parent of main_loop)
        greenlet_t* top;

        // outbound predictions (we own this memory - although it will end up in python/tensorflow
        // for predictions, but that point we will be in a preserved state.)
        float* channel_buf;
        int channel_buf_indx;

        const PredictDoneEvent* predict_done_event;
   };
}
