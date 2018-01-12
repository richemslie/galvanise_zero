#include "scheduler.h"

#include "bases.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <k273/logging.h>
#include <k273/exception.h>

#include <string>

using namespace GGPZero;

NetworkScheduler::NetworkScheduler(GGPLib::StateMachineInterface* sm,
                                   const GdlBasesTransformer* transformer,
                                   int batch_size,
                                   int expected_policy_size,
                                   int role_1_index) :
    sm(sm),
    transformer(transformer),
    batch_size(batch_size),
    expected_policy_size(expected_policy_size),
    role_1_index(role_1_index),
    basestate_expand_node(nullptr),
    main_loop(nullptr),
    top(nullptr),
    policies(nullptr),
    final_scores(nullptr),
    pred_count(0),
    channel_buf_indx(0) {

    this->basestate_expand_node = this->sm->newBaseState();

    const int num_floats = this->transformer->totalSize() * this->batch_size;
    K273::l_debug("Creating channel_buf of size %d", num_floats);

    this->channel_buf = (float*) malloc(sizeof(float) * num_floats);
}

NetworkScheduler::~NetworkScheduler() {
    free(this->basestate_expand_node);
    free(this->channel_buf);

    delete this->sm;
    delete this->transformer;
}

std::string NetworkScheduler::moveString(const GGPLib::JointMove& move) {
    return PuctNode::moveString(move, this->sm);
}

void NetworkScheduler::dumpNode(const PuctNode* node,
                                const PuctNodeChild* highlight,
                                const std::string& indent,
                                bool sort_by_next_probability) {
    PuctNode::dumpNode(node, highlight, indent, sort_by_next_probability, this->sm);
}

PuctNode* NetworkScheduler::expandChild(PuctEvaluator* pe,
                                        const PuctNode* parent, const PuctNodeChild* child) {
    // update the statemachine
    this->sm->updateBases(parent->getBaseState());
    this->sm->nextState(&child->move, this->basestate_expand_node);

    // create node
    return this->createNode(pe, this->basestate_expand_node);
}

PuctNode* NetworkScheduler::createNode(PuctEvaluator* pe, const GGPLib::BaseState* bs) {
   // update the statemachine
    this->sm->updateBases(bs);

    const int role_count = this->sm->getRoleCount();
    PuctNode* new_node = PuctNode::create(role_count, bs, this->sm);

    if (new_node->is_finalised) {
        for (int ii=0; ii<role_count; ii++) {
            int score = this->sm->getGoalValue(ii);
            new_node->setFinalScore(ii, score / 100.0);
            new_node->setCurrentScore(ii, score / 100.0);
        }

        return new_node;
    }

    static std::vector <GGPLib::BaseState*> dummy;
    float* buf = this->channel_buf + this->channel_buf_indx;
    this->transformer->toChannels(bs, dummy, buf);
    this->channel_buf_indx += this->transformer->totalSize();

    // hang onto the position of where we in requestors
    int idx = this->requestors.size();
    this->requestors.emplace_back(greenlet_current());

    greenlet_switch_to(this->main_loop);

    ASSERT (this->pred_count > 0 && idx < this->pred_count);

    int policy_incr = new_node->lead_role_index ? this->role_1_index : 0;
    policy_incr += idx * this->expected_policy_size;

    int final_score_incr = idx * role_count;

    // Update children in new_node with prediction
    float total_prediction = 0.0f;

    for (int ii=0; ii<new_node->num_children; ii++) {
        PuctNodeChild* c = new_node->getNodeChild(this->sm->getRoleCount(), ii);
        c->policy_prob = *(this->policies + policy_incr + c->move.get(new_node->lead_role_index));
        total_prediction += c->policy_prob;
    }

    if (total_prediction > std::numeric_limits<float>::min()) {
        // normalise:
        for (int ii=0; ii<new_node->num_children; ii++) {
            PuctNodeChild* c = new_node->getNodeChild(this->sm->getRoleCount(), ii);
            c->policy_prob /= total_prediction;
        }

    } else {
        for (int ii=0; ii<new_node->num_children; ii++) {
            PuctNodeChild* c = new_node->getNodeChild(this->sm->getRoleCount(), ii);
            c->policy_prob = 1.0 / new_node->num_children;
        }
    }

    for (int ii=0; ii<role_count; ii++) {
        float s = (float) *(this->final_scores + final_score_incr + ii);
        new_node->setFinalScore(ii, s);
        new_node->setCurrentScore(ii, s);
    }

    // we return before continueing, we will be pushed back onto runnables queue
    greenlet_switch_to(this->main_loop);

    return new_node;
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
            ASSERT (this->policies != nullptr && this->final_scores != nullptr);
            ASSERT (this->pred_count == (int) this->requestors.size());

            for (greenlet_t* req : this->requestors) {
                // handle them straight away
                greenlet_switch_to(req);

                // and then add them to runnables
                this->runnables.emplace_back(req);
            }

            // clean up
            this->requestors.clear();
            this->policies = nullptr;
            this->final_scores = nullptr;
            this->pred_count = 0;
        }

        greenlet_t *g = this->runnables.front();
        this->runnables.pop_front();
        greenlet_switch_to(g);
    }

    K273::l_verbose("requestors.size on exiting runScheduler():  %zu", this->requestors.size());
}

int NetworkScheduler::poll(float* policies, float* final_scores, int pred_count) {
    // this is top
    this->top = greenlet_current();
    ASSERT(!greenlet_isdead(this->top));

    if (this->main_loop == nullptr) {
        K273::l_warning("poll called with no job");
        return 0;
    }

    ASSERT(!greenlet_isdead(this->main_loop));

    ASSERT (this->policies == nullptr && this->final_scores == nullptr && this->pred_count == 0);
    this->policies = policies;
    this->final_scores = final_scores;
    this->pred_count = pred_count;

    this->channel_buf_indx = 0;

    greenlet_switch_to(this->main_loop);

    if (this->channel_buf_indx == 0) {
        ASSERT(greenlet_isdead(this->main_loop));
        this->main_loop = nullptr;
    }

    return this->channel_buf_indx;
}
