#include "selfplay.h"

#include "sample.h"
#include "scheduler.h"
#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"

#include <statemachine/statemachine.h>

#include <k273/logging.h>

using namespace GGPZero;


SelfPlay::SelfPlay(NetworkScheduler* scheduler, const SelfPlayConfig* conf,
                   GGPLib::StateMachineInterface* sm) :
    scheduler(scheduler),
    conf(conf),
    initial_state(sm->getInitialState()) {
    this->pe = new PuctEvaluator(conf->select_puct_config, this->scheduler);
}


SelfPlay::~SelfPlay() {
}

void SelfPlay::playOnce() {
    /*
      XXX
      * revert to playing whole game, then randomly sampling start point
      * add exand root back as an option
      * resignation false positives must be recorded, we'll need to increase the pct
      * 3 functions, rather than one big function
      * stats
      * duplicates
      */


    // reset the puct evaluator
    this->pe->reset();

    // this is what we are trying to get, samples
    std::vector <Sample*> this_game_samples;

    // choose a starting point for when to start sampling
    int sample_start_depth = this->rng.getWithMax(this->conf->expected_game_length -
                                                  this->conf->max_number_of_samples);

    PuctNode* root = this->pe->establishRoot(this->initial_state, 0);
    int game_depth = 0;
    int total_iterations = 0;

    // ok simple loop until we start taking samples
    this->pe->updateConf(this->conf->select_puct_config);
    int iterations = this->conf->select_iterations;

    // selecting
    while (true) {
        // we haven't started taking samples and we hit the end of the road... woops
        if (root->isTerminal()) {
            root = this->pe->backupRoot(this->conf->max_number_of_samples);
            break;
        }

        if (game_depth == sample_start_depth) {
            break;
        }

        const PuctNodeChild* choice = this->pe->onNextMove(iterations);
        root = this->pe->fastApplyMove(choice);
        total_iterations += iterations;
        game_depth++;
    }

    //K273::l_verbose("XXX1 :: %d / %d", game_depth, sample_start_depth);

    // sampling:
    this->pe->updateConf(this->conf->sample_puct_config);
    iterations = this->conf->sample_iterations;
    while (true) {
        const int sample_count = this_game_samples.size();
        if (sample_count == this->conf->max_number_of_samples) {
            //K273::l_verbose("XXX2 :: %d / %d", game_depth, sample_count);
            break;
        }

        if (root->isTerminal()) {
            K273::l_info("terminal during samples :: %d %d", game_depth, iterations);
            //K273::l_verbose("XXX3 :: %d / %d", game_depth, sample_count);
            break;
        }

        //K273::l_verbose("XXX4 :: %d / %d", game_depth, sample_count);

        // create a sample (call getProbabilities() to ensure probabilities are right for policy)
        this->pe->getProbabilities(root, 1.0f);
        Sample* s = this->scheduler->createSample(root);
        s->depth = game_depth;

        // store it for later
        this_game_samples.push_back(s);

        const PuctNodeChild* choice = this->pe->onNextMove(iterations);
        root = this->pe->fastApplyMove(choice);
        total_iterations += iterations;
        game_depth++;
    }

    // scoring:
    this->pe->updateConf(this->conf->score_puct_config);
    iterations = this->conf->score_iterations;

    const bool can_resign = this->rng.get() > this->conf->resign_false_positive_retry_percentage;
    while (true) {
        if (root->isTerminal()) {
            break;
        }

        if (can_resign) {
            if (root->getCurrentScore(root->lead_role_index) < this->conf->resign_score_probability) {
                //K273::l_verbose("early resign :: %d %d", game_depth, iterations);
                break;
            }
        }

        const PuctNodeChild* choice = this->pe->onNextMove(iterations);
        root = this->pe->fastApplyMove(choice);
        total_iterations += iterations;
        game_depth++;
    }

    // update samples
    for (auto sample : this_game_samples) {
        sample->game_length = game_depth;
        for (int ii=0; ii<this->scheduler->getRoleCount(); ii++) {
            // need to clamp scores over > 0.9
            double score = root->getCurrentScore(ii);
            if (score < this->conf->resign_score_probability) {
                score = 0.0;
            } else if (score > (1.0 - this->conf->resign_score_probability)) {
                score = 1.0;
            }

            sample->final_score.push_back(score);
        }
    }

    int sample_count = this_game_samples.size();
    if (sample_count != this->conf->max_number_of_samples) {
        K273::l_verbose("done TestSelfPlay::playOnce() depth/iterations/samples :: %d / %d / %d", game_depth,
                        total_iterations, (int) this_game_samples.size());
    }

    this->samples.insert(this->samples.end(),
                         this_game_samples.begin(),
                         this_game_samples.end());
}

void SelfPlay::playGamesForever() {
    while (true) {
        this->playOnce();
    }
}
