#include "supervisor.h"

#include "selfplay.h"
#include "puct/evaluator.h"
#include "greenlet/greenlet.h"

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <k273/logging.h>
#include <k273/exception.h>

using namespace GGPZero;

Supervisor::Supervisor(GGPLib::StateMachineInterface* sm) :
    sm(sm->dupe()),
    player_pe(nullptr),
    self_play_manager(nullptr),
    inline_scheduler(nullptr) {
}

Supervisor::~Supervisor() {
    delete this->sm;
}

NetworkScheduler* Supervisor::createScheduler(const GdlBasesTransformer* transformer,
                                              int batch_size,
                                              int expected_policy_size,
                                              int role_1_index) {

    this->inline_scheduler = new NetworkScheduler(this->sm->dupe(),
                                                  transformer,
                                                  batch_size,
                                                  expected_policy_size,
                                                  role_1_index);
    return this->inline_scheduler;
}

void Supervisor::puctPlayerStart() {
    K273::l_verbose("InlineSupervisor::puctPlayerStart()");
    this->player_pe = new PuctEvaluator(PuctConfig::defaultConfig(), this->inline_scheduler);
}

void Supervisor::puctApplyMove(const GGPLib::JointMove* move) {
    ASSERT (this->player_pe != nullptr);
    if (this->player_pe->hasRoot()) {
        this->player_pe->applyMove(move);
    }
}

void Supervisor::puctPlayerMove(const GGPLib::BaseState* state, int iterations, double end_time) {

    // returns the lead_role_index's legal of the root.

    K273::l_verbose("InlineSupervisor::puctPlayerIterations() - %d", iterations);

    ASSERT (this->player_pe != nullptr);

    if (!this->player_pe->hasRoot()) {
        auto f = [this, state, iterations, end_time]() {
            this->player_pe->establishRoot(state, 0);
            this->player_pe->onNextMove(iterations, end_time);
        };

        this->inline_scheduler->addRunnable(f);

    } else {
        auto f = [this, iterations, end_time]() {
            this->player_pe->onNextMove(iterations, end_time);
        };

        this->inline_scheduler->addRunnable(f);
    }
}

int Supervisor::puctPlayerGetMove(int lead_role_index) {
    ASSERT (this->player_pe != nullptr);
    PuctNodeChild* child = this->player_pe->chooseTopVisits();
    if (child == nullptr) {
        return -1;
    }

    return child->move.get(lead_role_index);
}

void Supervisor::selfPlayTest(int num_selfplays, int base_iterations, int sample_iterations) {
    K273::l_verbose("InlineSupervisor::selfPlayTest(%d, %d, %d)",
                    num_selfplays, base_iterations, sample_iterations);

    GGPLib::BaseState* bs = this->sm->newBaseState();
    bs->assign(this->sm->getInitialState());

    // create a bunch of self plays
    for (int ii=0; ii<num_selfplays; ii++) {
        TestSelfPlay* sp = new TestSelfPlay(this->inline_scheduler,
                                            bs,
                                            base_iterations,
                                            sample_iterations);

        auto f = [sp]() {
            return sp->playOnce();
        };

        this->inline_scheduler->addRunnable(f);
    }
}

int Supervisor::poll(float* policies, float* final_scores, int pred_count) {
    return this->inline_scheduler->poll(policies, final_scores, pred_count);
}

float* Supervisor::getBuf() const {
    return this->inline_scheduler->getBuf();
}
