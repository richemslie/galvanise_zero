#include "player.h"

#include "puct/config.h"
#include "puct/evaluator.h"

#include "scheduler.h"
#include "gdltransformer.h"

#include <k273/logging.h>
#include <k273/exception.h>

#include <vector>

using namespace GGPZero;

Player::Player(GGPLib::StateMachineInterface* sm,
               const GdlBasesTransformer* transformer,
               PuctConfig* conf) :
    transformer(transformer),
    config(conf),
    evaluator(nullptr),
    scheduler(nullptr),
    first_play(false),
    on_next_move_choice(nullptr) {

    ASSERT(conf->batch_size >= 1);

    // first create a scheduler
    this->scheduler = new GGPZero::NetworkScheduler(transformer, conf->batch_size);

    // ... and then the evaluator...
    // dupe statemachine here, as the PuctEvaluator thinks it is sharing a statemachine (ie it
    // doesn't dupe the statemachine itself)
    this->evaluator = new PuctEvaluator(sm->dupe(), this->scheduler, transformer);
    this->evaluator->updateConf(conf);
}


Player::~Player() {
    delete this->evaluator;
    delete this->scheduler;
}

void Player::updateConfig(float think_time, int converged_visits, bool verbose) {
    this->config->think_time = think_time;
    this->config->converged_visits = converged_visits;
    this->config->verbose = verbose;

    this->evaluator->updateConf(this->config);
}

void Player::puctPlayerReset(int game_depth) {
    K273::l_verbose("V2 Player::puctPlayerReset()");
    this->evaluator->reset(game_depth);
    this->first_play = true;
}


void Player::puctApplyMove(const GGPLib::JointMove* move) {
    this->scheduler->createMainLoop();

    if (this->first_play) {
        this->first_play = false;
        auto f = [this, move]() {
            this->evaluator->establishRoot(nullptr);
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

void Player::puctPlayerMove(const GGPLib::BaseState* state, int evaluations, double end_time) {
    this->on_next_move_choice = nullptr;
    this->scheduler->createMainLoop();

    K273::l_verbose("V2 Player::puctPlayerMove() - %d", evaluations);

    // this should only happen as first move in the game
    if (this->first_play) {
        this->first_play = false;
        auto f = [this, state, evaluations, end_time]() {
            this->evaluator->establishRoot(state);
            this->on_next_move_choice = this->evaluator->onNextMove(evaluations, end_time);
        };

        this->scheduler->addRunnable(f);

    } else {
        auto f = [this, evaluations, end_time]() {
            this->on_next_move_choice = this->evaluator->onNextMove(evaluations, end_time);
        };

        this->scheduler->addRunnable(f);
    }
}

std::tuple <int, float, int> Player::puctPlayerGetMove(int lead_role_index) {
    if (this->on_next_move_choice == nullptr) {
        return std::make_tuple(-1, -1.0f, -1);
    }

    float probability = -1;
    const PuctNode* node = this->on_next_move_choice->to_node;
    if (node != nullptr) {
        probability = node->getCurrentScore(lead_role_index);
    }

    return std::make_tuple(this->on_next_move_choice->move.get(lead_role_index),
                           probability,
                           this->evaluator->nodeCount());
}

std::vector <PuctNodeDebug> Player::treeDebugInfo() {
    std::vector <PuctNodeDebug> res;
    const PuctNode* root = this->evaluator->getRootNode();

    if (root == nullptr) {
        return res;
    }

    // XXX learn c++
    struct X {
        X(int x, int y) :
            x(x),
            y(y) {
        }

        int x, y;
    };

    std::vector <X> spec;
    spec.emplace_back(0, -1);
    spec.emplace_back(0, 0);
    spec.emplace_back(0, 1);
    spec.emplace_back(0, 2);
    spec.emplace_back(0, 3);

    spec.emplace_back(1, -1);
    spec.emplace_back(1, 0);

    spec.emplace_back(2, -1);
    spec.emplace_back(2, 0);

    for (X& x : spec) {

        PuctNodeDebug info;
        PuctNode::debug(root, x.x, x.y, 10, info);
        if (info.index_0 < 0) {
            return res;
        }

        res.push_back(info);
    }

    return res;
}

const GGPZero::ReadyEvent* Player::poll(int predict_count, std::vector <float*>& data) {
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
