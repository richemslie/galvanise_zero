#pragma once

#include "sample.h"
#include "events.h"

#include "greenlet/greenlet.h"

#include <k273/inplist.h>
#include <k273/exception.h>

#include <deque>
#include <vector>
#include <string>

// forwards fun:
namespace GGPZero {
    class GdlBasesTransformer;
}

namespace SchedulerV2 {

    ///////////////////////////////////////////////////////////////////////////////

    class ModelResult {
    public:
        ModelResult() :
            basestate(nullptr) {
        }

        void set(const GGPLib::BaseState* basestate,
                 int idx, const GGPZero::PredictDoneEvent* evt,
                 const GGPZero::GdlBasesTransformer* transformer);

        const float* getPolicy(int index) const {
            return this->policies[index];
        }

        float getReward(int index) const {
            return this->rewards[index];
        }

    private:
        // better to be a chunk of memory...
        std::vector <float*> policies;
        std::vector <float> rewards;

        // could follow leela here and just store a hash
        const GGPLib::BaseState* basestate;
    };

    using ModelResultList = K273::InplaceList <ModelResult>;

    ///////////////////////////////////////////////////////////////////////////////
    // pure interface

    class NodeRequestInterface {
    public:
        NodeRequestInterface() {
        }

        virtual ~NodeRequestInterface() {
        }

    public:
        // low level adds info to buffer
        virtual void add(float* buf, const GGPZero::GdlBasesTransformer* transformer) = 0;

        // low level fetches from evt, and sets stuff
        virtual void reply(const ModelResult& result,
                           const GGPZero::GdlBasesTransformer* transformer) = 0;
    };

    ///////////////////////////////////////////////////////////////////////////////

    class NetworkScheduler {
    public:
        NetworkScheduler(const GGPZero::GdlBasesTransformer* transformer,
                         int role_count, int batch_size, int lru_cache_size=1000);
        ~NetworkScheduler();

    public:
        // called from a puct evaluator
        void evaluateNode(const GGPLib::BaseState* state,
                          std::vector <const GGPLib::BaseState*>& prev_states,
                          NodeRequestInterface* request);

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

        void poll(const GGPZero::PredictDoneEvent* predict_done_event,
                  GGPZero::ReadyEvent* ready_event);

    private:
        void mainLoop();

    private:
        const GGPZero::GdlBasesTransformer* transformer;
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

        // all the linked lists
        int lru_cache_size;

        ModelResultList free_list;
        ModelResultList lru_list;
        GGPLib::BaseState::HashMap < ModelResultList::Node*> lru_lookup;

        const GGPZero::PredictDoneEvent* predict_done_event;
   };
}
