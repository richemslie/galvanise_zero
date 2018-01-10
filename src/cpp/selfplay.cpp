#include "selfplay.h"
#include "puctnode.h"
#include "pucteval.h"
#include "supervisorbase.h"

#include <statemachine/statemachine.h>

#include <k273/logging.h>

using namespace GGPZero;


PUCTEvalConfig* defaultConfig() {
    PUCTEvalConfig* conf = new PUCTEvalConfig;
    conf->name = "test_evaluator";
    conf->verbose = false;
    conf->generation = "noone";

    conf->puct_before_expansions = 3;
    conf->puct_before_root_expansions = 3;

    conf->puct_constant_before = 5.0;
    conf->puct_constant_after = 0.75;

    conf->dirichlet_noise_pct = 0.25;
    conf->dirichlet_noise_alpha = 0.5;

    conf->choose = ChooseFn::choose_top_visits;

    conf->max_dump_depth = 2;

    conf->random_scale = 0.75;
    conf->temperature = 1.0;
    conf->depth_temperature_start = 8;
    conf->depth_temperature_stop = 16;
    conf->depth_temperature_increment = 0.5;
    return conf;
}


TestSelfPlay::TestSelfPlay(SupervisorBase* supervisor, const GGPLib::BaseState* state) :
    supervisor(supervisor),
    pe(defaultConfig(), supervisor),
    initial_state(state) {
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

        int iterations = 100;
        if (game_depth >= 14 && game_depth <= 17) {
            iterations = 800;
        }

        PuctNodeChild* choice = pe.onNextNove(iterations);
        root = pe.fastApplyMove(choice);
        total_iterations += iterations;
        game_depth++;
    }

    K273::l_warning("done TestSelfPlay::playOnce() depth/iterations :: %d / %d", game_depth, total_iterations);
    this->supervisor->finish();
}

