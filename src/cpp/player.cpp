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
    evaluator(nullptr),
    scheduler(nullptr),
    first_play(false) {

    // first create a scheduler
    const int batch_size = 1;
    this->scheduler = new NetworkScheduler(sm->dupe(), transformer, batch_size);

    // and then the evaluator
    this->evaluator = new PuctEvaluator(conf, this->scheduler);
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

    auto f = [this, move]() {
        this->evaluator->applyMove(move);
    };

    this->scheduler->addRunnable(f);
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

const ReadyEvent* Player::poll(float* policies, float* final_scores, int pred_count) {
    // when pred_count == 0, it is used to bootstrap the main loop in scheduler
    this->predict_done_event.pred_count = pred_count;
    this->predict_done_event.policies = policies;
    this->predict_done_event.final_scores = final_scores;

    this->scheduler->poll(&this->predict_done_event, &this->ready_event);

    return &this->ready_event;
}

