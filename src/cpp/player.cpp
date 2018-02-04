#include "player.h"

#include "events.h"
#include "scheduler.h"
#include "gdltransformer.h"

#include "puct/config.h"
#include "puct/evaluator.h"

#include <k273/logging.h>
#include <k273/exception.h>

using namespace GGPZero;

Player::Player(GGPLib::StateMachineInterface* sm,
               const GdlBasesTransformer* transformer,
               PuctConfig* conf) :
    transformer(transformer),
    evaluator(nullptr),
    scheduler(nullptr),
    first_play(false) {

    // first create a scheduler
    const int batch_size = 1;
    this->scheduler = new NetworkScheduler(transformer, sm->getRoleCount(), batch_size);

    // ... and then the evaluator...
    // dupe statemachine here, as the PuctEvaluator thinks it is sharing a statemachine (ie it
    // doesn't dupe the statemachine itself)
    this->evaluator = new PuctEvaluator(sm->dupe(), conf, this->scheduler);
}

Player::~Player() {
    delete this->evaluator;
    delete this->scheduler;
}

void Player::puctPlayerReset() {
    K273::l_verbose("Player::puctPlayerReset()");
    this->evaluator->reset();
    this->first_play = true;
}

void Player::puctApplyMove(const GGPLib::JointMove* move) {
    this->scheduler->createMainLoop();

    if (this->first_play) {
        this->first_play = false;
        auto f = [this, move]() {
            this->evaluator->establishRoot(nullptr, 0);
            this->evaluator->applyMove(move);
        };

        this->scheduler->addRunnable(f);

    } else {
        auto f = [this, move]() {
            this->evaluator->applyMove(move);
        };

        this->scheduler->addRunnable(f);
    }
}

void Player::puctPlayerMove(const GGPLib::BaseState* state, int iterations, double end_time) {
    this->scheduler->createMainLoop();

    K273::l_verbose("Player::puctPlayerMove() - %d", iterations);

    // this should only happen as first move in the game
    if (this->first_play) {
        this->first_play = false;
        auto f = [this, state, iterations, end_time]() {
            this->evaluator->establishRoot(state, 0);
            this->evaluator->onNextMove(iterations, end_time);
        };

        this->scheduler->addRunnable(f);

    } else {
        auto f = [this, iterations, end_time]() {
            this->evaluator->onNextMove(iterations, end_time);
        };

        this->scheduler->addRunnable(f);
    }
}

int Player::puctPlayerGetMove(int lead_role_index) {
    const PuctNodeChild* child = this->evaluator->choose();
    if (child == nullptr) {
        return -1;
    }

    return child->move.get(lead_role_index);
}

const ReadyEvent* Player::poll(int predict_count, std::vector <float*>& data) {
    // when pred_count == 0, it is used to bootstrap the main loop in scheduler
    this->predict_done_event.pred_count = predict_count;

    // XXX holds pointers to data - maybe we should just copy it like in supervisor case.  It isn't
    // like this is an optimisation, I am just being lazy.

    int index = 0;
    this->predict_done_event.policies.resize(this->transformer->getNumberPolicies());
    for (int ii=0; ii<this->transformer->getNumberPolicies(); ii++) {
        this->predict_done_event.policies[ii] = data[index++];
    }

    this->predict_done_event.final_scores = data[index++];

    this->scheduler->poll(&this->predict_done_event, &this->ready_event);

    return &this->ready_event;
}

