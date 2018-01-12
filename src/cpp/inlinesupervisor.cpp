#include "inlinesupervisor.h"

#include "selfplay.h"
#include "puct/evaluator.h"
#include "greenlet/greenlet.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <k273/logging.h>

#include <functional>

using namespace GGPZero;

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
    channel_buf_indx(0),
    top(nullptr),
    scheduler(nullptr) {

    this->basestate_expand_node = this->sm->newBaseState();

    const int num_floats = this->transformer->totalSize() * this->batch_size;
    K273::l_debug("Creating channel_buf of size %d", num_floats);

    this->channel_buf = (float*) malloc(sizeof(float) * num_floats);

    this->top = greenlet_current();
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

    greenlet_switch_to(this->scheduler);

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

void InlineSupervisor::puctPlayerStart() {
    K273::l_verbose("InlineSupervisor::puctPlayerStart()");
    this->player_pe = new PuctEvaluator(PuctConfig::defaultConfig(), this);
}

void InlineSupervisor::puctApplyMove(const GGPLib::JointMove* move) {
    ASSERT (this->player_pe != nullptr);
    if (this->player_pe->hasRoot()) {
        this->player_pe->applyMove(move);
    }
}

void InlineSupervisor::puctPlayerMove(const GGPLib::BaseState* state, int iterations, double end_time) {

    // returns the lead_role_index's legal of the root.

    K273::l_verbose("InlineSupervisor::puctPlayerIterations() - %d", iterations);

    ASSERT (this->scheduler == nullptr);

    this->scheduler = createGreenlet([this]() {
            return this->runScheduler();
        });

    ASSERT (this->player_pe != nullptr);

    if (!this->player_pe->hasRoot()) {
        auto f = [this, state, iterations, end_time]() {
            this->player_pe->establishRoot(state, 0);
            this->player_pe->onNextMove(iterations, end_time);
        };

        this->runnables.emplace_back(createGreenlet(f, this->scheduler),
                                     nullptr);

    } else {
        auto f = [this, iterations, end_time]() {
            this->player_pe->onNextMove(iterations, end_time);
        };

        this->runnables.emplace_back(createGreenlet(f, this->scheduler),
                                     nullptr);
    }
}

int InlineSupervisor::puctPlayerGetMove(int lead_role_index) {
    ASSERT (this->player_pe != nullptr);
    PuctNodeChild* child = this->player_pe->chooseTopVisits();
    if (child == nullptr) {
        return -1;
    }

    return child->move.get(lead_role_index);
}

void InlineSupervisor::selfPlayTest(int num_selfplays, int base_iterations, int sample_iterations) {
    K273::l_verbose("InlineSupervisor::selfPlayTest(%d, %d, %d)",
                    num_selfplays, base_iterations, sample_iterations);

    ASSERT (this->scheduler == nullptr);

    this->scheduler = createGreenlet([this]() {
            return this->runScheduler();
        });

    GGPLib::BaseState* bs = this->sm->newBaseState();
    bs->assign(this->sm->getInitialState());

    // create a bunch of self plays
    for (int ii=0; ii<num_selfplays; ii++) {
        TestSelfPlay* sp = new TestSelfPlay(this, bs, base_iterations, sample_iterations);

        auto f = [sp]() {
            return sp->playOnce();
        };

        this->runnables.emplace_back(createGreenlet(f, this->scheduler),
                                     nullptr);
    }
}

void InlineSupervisor::runScheduler() {
    K273::l_verbose("entering InlineSupervisor::runScheduler()");

    while (true) {
        if (this->runnables.empty()) {
            if (this->requestors.empty()) {
                break;
            }

            greenlet_switch_to(this->top);

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

    K273::l_verbose("requestors.size on exiting runScheduler():  %zu", this->requestors.size());
}

int InlineSupervisor::poll(float* policies, float* final_scores, int pred_count) {
    ASSERT(!greenlet_isdead(this->top));

    if (this->scheduler == nullptr) {
        K273::l_warning("poll called with no job");
        return 0;
    }

    ASSERT(!greenlet_isdead(this->scheduler));

    ASSERT (this->policies == nullptr && this->final_scores == nullptr && this->pred_count == 0);
    this->policies = policies;
    this->final_scores = final_scores;
    this->pred_count = pred_count;

    this->channel_buf_indx = 0;

    greenlet_switch_to(this->scheduler);

    if (this->channel_buf_indx == 0) {
        ASSERT(greenlet_isdead(this->scheduler));
        this->scheduler = nullptr;
    }

    this->policies = nullptr;
    this->final_scores = nullptr;
    this->pred_count = 0;

    return this->channel_buf_indx;
}
