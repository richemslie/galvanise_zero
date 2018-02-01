#include "scheduler.h"

#include "sample.h"
#include "puct/node.h"
#include "gdltransformer.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <k273/logging.h>
#include <k273/exception.h>

#include <string>

using namespace GGPZero;

NetworkScheduler::NetworkScheduler(const GdlBasesTransformer* transformer,
                                   int role_count, int batch_size) :
    transformer(transformer),
    role_count(role_count),
    batch_size(batch_size),
    main_loop(nullptr),
    top(nullptr),
    channel_buf(nullptr),
    channel_buf_indx(0) {

    const int num_floats = this->transformer->totalSize() * this->batch_size;
    K273::l_debug("Creating channel_buf of size %d", num_floats);

    this->channel_buf = (float*) new float[num_floats];
}

NetworkScheduler::~NetworkScheduler() {
    delete[] this->channel_buf;
    delete this->transformer;
}

void NetworkScheduler::updateFromPolicyHead(const int idx, PuctNode* node) {
    // XXX for now we are only interested in new nodes lead_role_index.  For PUCTPlus will want to
    // populate all policies.

    float* policies_start = this->predict_done_event->policies[node->lead_role_index];
    policies_start += idx * this->transformer->getPolicySize(node->lead_role_index);

    // Update children in new_node with prediction
    float total_prediction = 0.0f;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->role_count, ii);
        c->policy_prob = *(policies_start + c->move.get(node->lead_role_index));
        if (c->policy_prob < 0.001) {
            // XXX stats?
            c->policy_prob = 0.001;
        }

        total_prediction += c->policy_prob;
    }

    if (total_prediction > std::numeric_limits<float>::min()) {
        // normalise:
        for (int ii=0; ii<node->num_children; ii++) {
            PuctNodeChild* c = node->getNodeChild(this->role_count, ii);
            c->policy_prob /= total_prediction;
        }

    } else {
        // well that sucks - absolutely no predictions, just make it uniform then...
        for (int ii=0; ii<node->num_children; ii++) {
            PuctNodeChild* c = node->getNodeChild(this->role_count, ii);
            c->policy_prob = 1.0 / node->num_children;
        }
    }
}

void NetworkScheduler::updateFromValueHead(const int idx, PuctNode* node) {
    const int final_score_incr = idx * role_count;
    float* final_scores_start = this->predict_done_event->final_scores + final_score_incr;

    for (int ii=0; ii<role_count; ii++) {
        float s = *(final_scores_start + ii);
        node->setFinalScore(ii, s);
        node->setCurrentScore(ii, s);
    }
}

void NetworkScheduler::evaluateNode(PuctEvaluator* pe, PuctNode* node) {

    // XXX be more optimal to have this vector the scheduler
    std::vector <const GGPLib::BaseState*> prev_states;
    const PuctNode* cur = node->parent;
    for (int ii=0; ii<this->transformer->getNumberPrevStates(); ii++) {
        if (cur != nullptr) {
            prev_states.push_back(cur->getBaseState());
            cur = cur->parent;
        }
    }

    // index into the current position of the channel buffer
    float* buf = this->channel_buf + this->channel_buf_indx;
    this->transformer->toChannels(node->getBaseState(), prev_states, buf);

    // increment index for the next evaluateNode()
    this->channel_buf_indx += this->transformer->totalSize();

    // hang onto the position of where inserted into requestors
    const int idx = this->requestors.size();

    this->requestors.emplace_back(greenlet_current());

    // see you later...
    greenlet_switch_to(this->main_loop);

    // back now! at this point we have predicted
    ASSERT(this->predict_done_event->pred_count >= idx);

    this->updateFromPolicyHead(idx, node);
    this->updateFromValueHead(idx, node);

    // we return before continuing, we will be pushed back onto runnables queue
    greenlet_switch_to(this->main_loop);
}

void NetworkScheduler::mainLoop() {
    K273::l_verbose("entering Scheduler::mainLoop()");

    // this is the main_loop greenlet - just to prove this point:
    ASSERT(greenlet_current() == this->main_loop);

    while (true) {
        bool jump_to_top = false;

        if (this->runnables.empty()) {
            // done - nothing else to do
            if (this->requestors.empty()) {
                break;
            }

            jump_to_top = true;
        }

        if (!jump_to_top) {
            // full?
            if (this->requestors.size() == this->batch_size) {
                jump_to_top = true;
            }
        }

        if (jump_to_top) {
            greenlet_switch_to(this->top);

            // once we return, we must handle the results before doing anything else
            ASSERT(this->predict_done_event->pred_count == (int) this->requestors.size());

            for (greenlet_t* req : this->requestors) {
                // handle them straight away (will jump back to "see you later" in
                // NetworkScheduler::createNode()
                greenlet_switch_to(req);

                // and then add them to runnables
                this->runnables.emplace_back(req);
            }

            // clean up
            this->requestors.clear();
        }

        greenlet_t *g = this->runnables.front();
        this->runnables.pop_front();
        greenlet_switch_to(g);
    }

    K273::l_verbose("requestors.size on exiting runScheduler():  %zu", this->requestors.size());
}

void NetworkScheduler::poll(const PredictDoneEvent* predict_done_event, ReadyEvent* ready_event) {
    //poll() must be called with an event.  The even resides in the parent process (which is the
    //self play manager / player).  This is passed to the main_loop()..

    // this is top
    if (this->top == nullptr) {
        this->top = greenlet_current();
    } else {
        // very important that this relationship is maintainede
        ASSERT(greenlet_current() == this->top);
        ASSERT(!greenlet_isdead(this->top));
    }

    // This should be an assert also...
    ASSERT_MSG(this->main_loop != nullptr, "NetworkScheduler::poll without mainLoop() set");
    ASSERT(!greenlet_isdead(this->main_loop));

    this->predict_done_event = predict_done_event;

    // we may / or may not set the pred_count
    this->channel_buf_indx = 0;
    greenlet_switch_to(this->main_loop);
    this->predict_done_event = nullptr;

    // the main_loop will die if it runs out of things to do
    if (this->channel_buf_indx == 0) {
        ASSERT(greenlet_isdead(this->main_loop));
        this->main_loop = nullptr;
    }

    // populate ready_event
    ready_event->channel_buf = this->channel_buf;
    ready_event->pred_count = this->channel_buf_indx;
}
