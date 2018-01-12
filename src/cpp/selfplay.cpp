#include "selfplay.h"

#include "scheduler.h"
#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"

#include <statemachine/statemachine.h>

#include <k273/logging.h>

using namespace GGPZero;


SelfPlay::SelfPlay(NetworkScheduler* scheduler, const SelfPlayConfig* conf) :
    scheduler(scheduler),
    conf(conf) {
    // everything is initialised in configure()
}


SelfPlay::~SelfPlay() {
}

void SelfPlay::configure(GGPLib::StateMachineInterface* sm) {
    // sm is available for initialising self play, but it can't hang on to it
}

void SelfPlay::playOnce() {
    this->pe->reset();

    const PuctNode* root = this->pe->establishRoot(this->initial_state, 0);
    int game_depth = 0;
    int total_iterations = 0;
    while (true) {
        if (root->isTerminal()) {
            break;
        }

        int iterations = this->conf->select_iterations;
        if (game_depth >= 14 && game_depth <= 17) {
            iterations = this->conf->sample_iterations;
        }

        const PuctNodeChild* choice = this->pe->onNextMove(iterations);
        root = this->pe->fastApplyMove(choice);
        total_iterations += iterations;
        game_depth++;
    }

    K273::l_verbose("done TestSelfPlay::playOnce() depth/iterations :: %d / %d", game_depth,
                    total_iterations);
}

void SelfPlay::playGamesForever() {
    while (true) {
        this->playOnce();
    }
}
