#include "supervisor.h"

#include "selfplay.h"
#include "scheduler.h"
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
    first_play(false),
    inline_scheduler(nullptr) {
}

Supervisor::~Supervisor() {
    delete this->sm;
}

NetworkScheduler* Supervisor::createScheduler(const GdlBasesTransformer* transformer,
                                              int batch_size,
                                              int expected_policy_size,
                                              int role_1_index) {

    ASSERT(this->inline_scheduler == nullptr);
    this->inline_scheduler = new NetworkScheduler(this->sm->dupe(),
                                                  transformer,
                                                  batch_size,
                                                  expected_policy_size,
                                                  role_1_index);
    return this->inline_scheduler;
}


int Supervisor::poll(float* policies, float* final_scores, int pred_count) {
    return this->inline_scheduler->poll(policies, final_scores, pred_count);
}

float* Supervisor::getBuf() const {
    return this->inline_scheduler->getBuf();
}







void Supervisor::puctPlayerStart(PuctConfig* conf) {
    K273::l_verbose("Supervisor::puctPlayerStart()");
    this->player_pe = new PuctEvaluator(conf, this->inline_scheduler);
}

void Supervisor::puctPlayerReset() {
    K273::l_verbose("Supervisor::reset()");
    this->player_pe->reset();
    this->first_play = true;
}

void Supervisor::puctApplyMove(const GGPLib::JointMove* move) {
    ASSERT (this->player_pe != nullptr);

    this->inline_scheduler->createMainLoop();

    auto f = [this, move]() {
        this->player_pe->applyMove(move);
    };

    this->inline_scheduler->addRunnable(f);
}

void Supervisor::puctPlayerMove(const GGPLib::BaseState* state, int iterations, double end_time) {
    this->inline_scheduler->createMainLoop();

    K273::l_verbose("Supervisor::puctPlayerIterations() - %d", iterations);

    ASSERT (this->player_pe != nullptr);

    // this should only happen as first move in the game
    if (this->first_play) {
        this->first_play = false;
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
    const PuctNodeChild* child = this->player_pe->choose();
    if (child == nullptr) {
        return -1;
    }

    return child->move.get(lead_role_index);
}









void Supervisor::startSelfPlay(const SelfPlayConfig* config) {
    this->inline_scheduler->createMainLoop();

    // create a bunch of self plays
    for (int ii=0; ii<1; ii++) {
        SelfPlay* sp = new SelfPlay(this->inline_scheduler, config, this->sm);
        this->self_plays.push_back(sp);

        auto f = [sp]() {
            sp->playGamesForever();
        };

        this->inline_scheduler->addRunnable(f);
    }
}

std::vector <Sample*> Supervisor::getSamples() {
    std::vector <Sample*> result;

    for (auto sp : this->self_plays) {
        std::vector <Sample*>& sp_samples = sp->getSamples();
        if (!sp_samples.empty()) {
            result.insert(result.end(),
                          sp_samples.begin(),
                          sp_samples.end());
            sp_samples.clear();
        }
    }

    return result;
}

