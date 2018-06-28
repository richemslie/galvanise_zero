#pragma once

#include "sample.h"
#include "events.h"

#include "greenlet/greenlet.h"

#include <k273/inplist.h>
#include <k273/exception.h>

#include <deque>
#include <vector>
#include <string>

namespace GGPZero {

    // forwards
    class GdlBasesTransformer;

    ///////////////////////////////////////////////////////////////////////////////

    class ModelResult {
    public:
        ModelResult() :
            basestate(nullptr) {
        }

        void set(const GGPLib::BaseState* basestate,
                 int idx, const PredictDoneEvent* evt,
                 const GdlBasesTransformer* transformer);

        const float* getPolicy(int index) const {
            return this->policies[index];
        }

        float getReward(int index) const {
            return this->rewards[index];
        }

    private:
        // XXX better if all of this was one chunk of memory... but maybe later

        std::vector <float*> policies;
        std::vector <float> rewards;

        // could follow leela here and just store a hash
        const GGPLib::BaseState* basestate;
    };

    using ModelResultList = K273::InplaceList <ModelResult>;

    ///////////////////////////////////////////////////////////////////////////////
    // pure abstract interface

    class ModelRequestInterface {
    public:
        ModelRequestInterface() {
        }

        virtual ~ModelRequestInterface() {
        }

    public:
        // called to check if in NN cache
        virtual const GGPLib::BaseState* getBaseState() const = 0;

        // low level adds info to buffer
        virtual void add(float* buf, const GdlBasesTransformer* transformer) = 0;

        // given a result, populated
        virtual void reply(const ModelResult& result,
                           const GdlBasesTransformer* transformer) = 0;
    };

    ///////////////////////////////////////////////////////////////////////////////
    // XXX finish LRU NN cache

    class NetworkScheduler {
    public:
        NetworkScheduler(const GdlBasesTransformer* transformer,
                         int role_count, int batch_size, int lru_cache_size=1000);
        ~NetworkScheduler();

    public:
        // called an evaluator engine
        void evaluate(ModelRequestInterface* request);
        void yield();

    public:
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

        // called directly/indirectly from python, sending events to/fro:
        void poll(const PredictDoneEvent* predict_done_event,
                  ReadyEvent* ready_event);

    private:
        void mainLoop();

    private:
        const GdlBasesTransformer* transformer;
        const int role_count;
        const unsigned int batch_size;

        std::vector <greenlet_t*> requestors;
        std::vector <greenlet_t*> yielders;
        std::deque <greenlet_t*> runnables;

        // the main looper
        greenlet_t* main_loop;

        // exit in and of the main_loop (and is parent of main_loop)
        greenlet_t* top;

        // outbound predictions (we malloc/own this memory - although it will end up in
        // python/tensorflow for predictions, but that point we will be in a preserved state.)
        float* channel_buf;
        int channel_buf_indx;

        // size of the lru cachce
        int lru_cache_size;

        ModelResultList free_list;
        ModelResultList lru_list;
        GGPLib::BaseState::HashMap < ModelResultList::Node*> lru_lookup;

        // set via poll().  Don't own this memory.  However, it won't change under feet.
        const PredictDoneEvent* predict_done_event;
   };
}
