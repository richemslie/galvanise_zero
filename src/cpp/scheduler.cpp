#include "scheduler.h"

#include "gdltransformer.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <k273/logging.h>
#include <k273/exception.h>

#include <string>

using namespace GGPZero;

///////////////////////////////////////////////////////////////////////////////

void ModelResult::set(const GGPLib::BaseState* bs,
                      int idx,
                      const PredictDoneEvent* evt,
                      const GdlBasesTransformer* transformer) {

    // XXX how can we ensure this lives on??? with lru
    this->basestate = bs;

    // XXX todo go back and rename final_scores -> rewards...
    this->policies.resize(transformer->getNumberPolicies());
    this->rewards.resize(transformer->getNumberRewards());

    for (int ii=0; ii<transformer->getNumberPolicies(); ii++) {
        float* pt_policy = evt->policies[ii];
        pt_policy += idx * transformer->getPolicySize(ii);
        this->policies[ii] = pt_policy;
    }

    for (int ii=0; ii<transformer->getNumberRewards(); ii++) {
        float* pt_reward = evt->final_scores + idx * transformer->getNumberRewards() + ii;
        this->rewards[ii] = *pt_reward;
    }
}

///////////////////////////////////////////////////////////////////////////////

NetworkScheduler::NetworkScheduler(const GdlBasesTransformer* transformer,
                                   int batch_size, int lru_cache_size) :
    transformer(transformer),
    batch_size(batch_size),
    main_loop(nullptr),
    top(nullptr),
    channel_buf(nullptr),
    channel_buf_indx(0),
    lru_cache_size(lru_cache_size) {

    const int num_floats = this->transformer->totalSize() * this->batch_size;
    K273::l_debug("Creating channel_buf with batch size %d", this->batch_size);

    this->channel_buf = (float*) new float[num_floats];

    // create a free list
    for (int ii=0; ii<this->lru_cache_size; ii++) {
        this->free_list.emplaceBack();
    }
}

NetworkScheduler::~NetworkScheduler() {
    delete[] this->channel_buf;
    delete this->transformer;
}

void NetworkScheduler::evaluate(ModelRequestInterface* request) {

    // check if in LRU, if so return that, update position
    auto const found = this->lru_lookup.find(request->getBaseState());
    if (found != this->lru_lookup.end()) {
        ModelResultList::Node* item = found->second;

        // call the requestor
        request->reply(item->get(), transformer);

        // move the item to front
        this->lru_list.remove(item);
        this->lru_list.pushFront(item);

        return;
    }

    // index into the current position of the channel buffer
    float* buf = this->channel_buf + this->channel_buf_indx;
    request->add(buf, this->transformer);

    // increment index for the next evaluateNode()
    this->channel_buf_indx += this->transformer->totalSize();

    // hang onto the position of where inserted into requestors
    const int idx = this->requestors.size();

    this->requestors.emplace_back(greenlet_current());

    // see you later...
    greenlet_switch_to(this->main_loop);

    // back now! at this point we have predicted
    ASSERT(this->predict_done_event->pred_count >= idx);

    // * get next from free list
    // * populate
    // * call requester
    // * add to lru
    //if (this->free_list->empty()) {
    //    // remove last 5% from lru_list/
    //}

    //ModelResult* this->free_list->remove(this->free_list->head();

    ModelResult tmp_xxx_res;

    // careful with requester->getBaseState() here XXX if added to LRU, needs to create a new
    // basestate
    tmp_xxx_res.set(request->getBaseState(), idx, this->predict_done_event, this->transformer);
    request->reply(tmp_xxx_res, this->transformer);

    // we return before continuing, we will be pushed back onto runnables queue
    greenlet_switch_to(this->main_loop);
}

void NetworkScheduler::yield() {
    // yields until next request returns
    this->yielders.emplace_back(greenlet_current());
    greenlet_switch_to(this->main_loop);
}

void NetworkScheduler::mainLoop() {
    //K273::l_verbose("entering Scheduler::mainLoop()");

    // this is the main_loop greenlet - just to prove this point:
    ASSERT(greenlet_current() == this->main_loop);

    while (true) {
        bool jump_to_top = false;

        if (this->runnables.empty()) {

            // done - nothing else to do
            if (this->requestors.empty()) {

                // push all yielders onto to runnables queue
                if (!this->yielders.empty()) {
                    for (greenlet_t* y : this->yielders) {
                        this->runnables.emplace_back(y);
                    }

                    this->yielders.clear();
                    continue;
                }

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

            if (!this->requestors.empty()) {
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

            // push all yielders onto to runnables queue
            if (!this->yielders.empty()) {
                for (greenlet_t* y : this->yielders) {
                    this->runnables.emplace_back(y);
                }

                this->yielders.clear();
            }
        }

        greenlet_t *g = this->runnables.front();
        this->runnables.pop_front();
        greenlet_switch_to(g);
    }

    //K273::l_verbose("requestors.size on exiting runScheduler():  %zu", this->requestors.size());
}

void NetworkScheduler::poll(const PredictDoneEvent* predict_done_event,
                            ReadyEvent* ready_event) {
    // poll() must be called with an event.  The even resides in the parent process (which is the
    // self play manager / player).  This is passed to the main_loop()..

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
    ready_event->buf_count = this->channel_buf_indx;
}
