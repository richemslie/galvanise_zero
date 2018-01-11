#include "inlinesupervisor.h"

#include "selfplay.h"
#include "puct/evaluator.h"
#include "greenlet/greenlet.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/logging.h>

using namespace GGPZero;


void* g_playOnce(void* arg) {
    auto f = std::mem_fn(&TestSelfPlay::playOnce);
    f((TestSelfPlay*) arg);
    return nullptr;
}

void* g_mainLoop(void* arg) {
    auto f = std::mem_fn(&InlineSupervisor::mainLoop);
    f((InlineSupervisor*) arg);
    return nullptr;
}


InlineSupervisor::InlineSupervisor(GGPLib::StateMachineInterface* sm,
                                   GdlBasesTransformer* transformer,
                                   int batch_size,
                                   int expected_policy_size,
                                   int role_1_index) :
    SupervisorBase(sm),
    transformer(transformer),
    batch_size(batch_size),
    basestate_expand_node(nullptr),
    expected_policy_size(expected_policy_size),
    role_1_index(role_1_index),
    policies(nullptr),
    final_scores(nullptr),
    pred_count(0),
    channel_buf_indx(0) {

    this->basestate_expand_node = this->sm->newBaseState();
    this->master = nullptr;

    const int num_floats = this->transformer->totalSize() * this->batch_size;
    K273::l_debug("Creating channel_buf of size %d", num_floats);

    this->channel_buf = (float*) malloc(sizeof(float) * num_floats);
}


PuctNode* InlineSupervisor::expandChild(PuctEvaluator* pe,
                                        const PuctNode* parent, const PuctNodeChild* child) {
    // update the statemachine
    this->sm->updateBases(parent->getBaseState());
    this->sm->nextState(&child->move, this->basestate_expand_node);

    // create node
    return this->createNode(pe, this->basestate_expand_node);
}


PuctNode* InlineSupervisor::createNode(PuctEvaluator* pe, const GGPLib::BaseState* bs) {
    static xoroshiro32plus16 random;

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

    greenlet_switch_to(this->master, nullptr);

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
        double s = (double) *(this->final_scores + final_score_incr + ii);
        new_node->setFinalScore(ii, s);
        new_node->setCurrentScore(ii, s);
    }

    return new_node;
}

void InlineSupervisor::finish() {
    greenlet_switch_to(this->master, nullptr);
}

void InlineSupervisor::mainLoop() {
    GGPLib::BaseState* bs = this->sm->newBaseState();
    bs->assign(this->sm->getInitialState());

    // create a bunch of self plays (for now only 1)
    for (unsigned int ii=0; ii<this->batch_size; ii++) {
        TestSelfPlay* sp = new TestSelfPlay(this, bs);
        greenlet_t *g = greenlet_new(g_playOnce, nullptr, 0);
        this->runnables.emplace_back(g, sp);
    }

    while (true) {
        if (this->runnables.empty()) {
            if (this->requestors.empty()) {
                break;
            }

            greenlet_switch_to(this->top, nullptr);

            ASSERT (this->policies != nullptr && this->final_scores != nullptr);
            ASSERT (this->pred_count == (int) this->requestors.size());

            for (auto req : this->requestors) {
                this->runnables.emplace_back(req, nullptr);
            }

            this->requestors.clear();

        } else {
            Runnable r = this->runnables.front();
            this->runnables.pop_front();
            greenlet_switch_to(r.greenlet, r.arg);
        }
    }

    K273::l_verbose("requestors.size at end of mainLoop() x:  %zu", this->requestors.size());
}

int InlineSupervisor::test(float* policies, float* final_scores, int pred_count) {
    //K273::l_debug("In pred_count %d", pred_count);

    ASSERT (this->policies == nullptr && this->final_scores == nullptr && this->pred_count == 0);
    this->policies = policies;
    this->final_scores = final_scores;
    this->pred_count = pred_count;

    this->channel_buf_indx = 0;
    if (this->master == nullptr) {
        this->top = greenlet_current();
        this->master = greenlet_new(g_mainLoop, nullptr, 0);
        greenlet_switch_to(this->master, this);
    } else {
        greenlet_switch_to(this->master, nullptr);
    }

    if (this->channel_buf_indx == 0) {
        ASSERT(greenlet_isdead(this->master));
    }

    this->policies = nullptr;
    this->final_scores = nullptr;
    this->pred_count = 0;

    return this->channel_buf_indx;
}
