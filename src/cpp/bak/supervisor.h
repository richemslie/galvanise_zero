#pragma once
XXX

#include "supervisorbase.h"
#include "bases.h"
#include "sample.h"
#include "puct/node.h"
#include "greenlet/greenlet.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/exception.h>

#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace GGPZero {

    // forwards
    class PuctEvaluator;

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


    struct BufCtx {
        // the size when full
        int channel_size;

        float* channel_buffer;
        int buffer_next_index;
        std::vector <PuctEvaluator*> requestors;
    };

    class Supervisor : public SupervisorBase {
    public:
        Supervisor(GGPLib::StateMachineInterface* sm,
                   GdlBasesTransformer* transformer, int batch_size);
        virtual ~Supervisor();

    public:
        int getRoleCount() const {
            return this->sm->getRoleCount();
        }

        PuctNode* expandChild(PuctEvaluator* pe, const PuctNode* parent, const PuctNodeChild* child);
        PuctNode* createNode(PuctEvaluator* pe, const GGPLib::BaseState* bs);

    public:
        // outside interface...
        int sampleCount() {
            std::lock_guard <std::mutex> lk(this->mutex);
            return this->num_samples;
        }

        bool predictionsReady() {
            // XXX use atomics?
            std::lock_guard <std::mutex> lk(this->mutex);
            return this->predictions_ready;
        }

        int getChannelsSize() const {
            return this->batch_size * this->transformer->totalSize();
        }

        float* getChannelsToPredict() {
            std::lock_guard <std::mutex> lk(this->mutex);
            ASSERT (this->predictions_ready);
            ASSERT (this->prediction_ctx != nullptr);
            this->predictions_ready = false;
            return this->prediction_ctx->channel_buffer;
        }

         void predictionsMade(float* predictions) {
            std::lock_guard <std::mutex> lk(this->mutex);
            ASSERT (this->prediction_ctx != nullptr);
            ASSERT (this->predictions_made != nullptr);

            // consumes memory
            this->predictions_made = predictions;

            return;
        }

        void mainLoop() {
            this->running = true;
            while (this->running) {

                if (this->predictions_in_progress) {
                    std::lock_guard <std::mutex> lk(this->mutex);
                    if (this->predictions_made != nullptr) {
                        ASSERT (this->predictions_in_progress);
                        this->predictions_in_progress = false;
                        this->prediction_ctx = nullptr;
                    }
                }

                if (this->current_ctx->requestors.size() == this->batch_size) {
                    std::lock_guard <std::mutex> lk(this->mutex);
                    if (!this->predictions_in_progress) {
                        ASSERT (this->prediction_ctx = nullptr);
                        this->prediction_ctx = this->current_ctx;
                        if (this->current_ctx == &buf_ctx[0]) {
                            this->current_ctx = &buf_ctx[1];
                        } else {
                            this->current_ctx = &buf_ctx[0];
                        }
                    } else {
                        while (true) {
                            // buffers are full...

                        }
                    }
                }

                for (int ii=0; ii<10; ii++) {
                    // don't run unless we have free requestors
                    if (this->current_ctx->requestors.size() == this->batch_size) {
                        break;
                    }

                    // pop the first runnable...
                    if (this->runnables.empty()) {
                        break;
                    }

                    greenlet* self_play = this->runnables.front();
                    this->runnables.pop_front();
                    self_play->switch_to();
                }
            }
        }

        void start() {
            std::lock_guard <std::mutex> lk(this->mutex);
            new std::thread(&Supervisor::mainLoop, this);
        }

        void stop() {
            std::lock_guard <std::mutex> lk(this->mutex);
            this->running = false;
        }

    private:
        GdlBasesTransformer* transformer;
        const unsigned int batch_size;

        GGPLib::BaseState* basestate_expand_node;

        bool running;
        int num_samples;

        BufCtx buf_ctx[2];
        BufCtx* current_ctx;
        BufCtx* prediction_ctx;

        bool predictions_ready;
        bool predictions_in_progress;

        // this is the array of predictions made by nn
        float* predictions_made;

        std::deque <greenlet*> runnables;
        greenlet_t* master;
        std::mutex mutex;
        std::thread* the_thread;
    };

}
