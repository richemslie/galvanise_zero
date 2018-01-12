#include "selfplay.h"

#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"
#include "supervisorbase.h"

#include <statemachine/statemachine.h>

#include <k273/logging.h>

using namespace GGPZero;


TestSelfPlay::TestSelfPlay(SupervisorBase* supervisor, const GGPLib::BaseState* state,
                           int base_iterations, int sample_iterations) :
    supervisor(supervisor),
    pe(PuctConfig::defaultConfig(), supervisor),
    initial_state(state),
    base_iterations(base_iterations),
    sample_iterations(sample_iterations) {
}


TestSelfPlay::~TestSelfPlay() {
}


void TestSelfPlay::playOnce() {
    this->pe.reset();

    PuctNode* root = this->pe.establishRoot(this->initial_state, 0);
    int game_depth = 0;
    int total_iterations = 0;
    while (true) {
        if (root->isTerminal()) {
            break;
        }

        int iterations = this->base_iterations;
        if (game_depth >= 14 && game_depth <= 17) {
            iterations = this->sample_iterations;
        }

        PuctNodeChild* choice = pe.onNextMove(iterations);
        root = pe.fastApplyMove(choice);
        total_iterations += iterations;
        game_depth++;
    }

    K273::l_verbose("done TestSelfPlay::playOnce() depth/iterations :: %d / %d", game_depth,
                    total_iterations);
}
