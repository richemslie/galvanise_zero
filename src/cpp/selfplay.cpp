#include "selfplay.h"

#include "sample.h"
#include "scheduler.h"
#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"
#include "selfplaymanager.h"

#include <statemachine/statemachine.h>

#include <k273/logging.h>

using namespace GGPZero;


SelfPlay::SelfPlay(NetworkScheduler* scheduler, const SelfPlayConfig* conf,
                   SelfPlayManager* manager, const GGPLib::BaseState* initial_state) :
    scheduler(scheduler),
    conf(conf),
    manager(manager),
    initial_state(initial_state) {
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


    int saw_dupes = 0;
    auto dupe = [this, &saw_dupes] (const GGPLib::BaseState* bs) {
        auto unique_states = this->manager->getUniqueStates();
        auto it = unique_states->find(bs);
        saw_dupes++;

        return it != this->manager->getUniqueStates()->end();
    };

    // reset the puct evaluator
    this->pe->reset();

    // this is what we are trying to get, samples
    std::vector <Sample*> this_game_samples;

    PuctNode* root = this->pe->establishRoot(this->initial_state, 0);
    int game_depth = 0;
    int total_iterations = 0;

    // ok simple loop until we start taking samples
    this->pe->updateConf(this->conf->select_puct_config);
    int iterations = this->conf->select_iterations;

    // selecting - we playout whole game and then choose a random move
    while (true) {
        // we haven't started taking samples and we hit the end of the road... woops
        if (root->isTerminal()) {
            break;
        }

        const PuctNodeChild* choice = this->pe->onNextMove(iterations);
        root = this->pe->fastApplyMove(choice);
        total_iterations += iterations;
        game_depth++;
    }

    // ok, we have a complete game.  Choose a starting point and start running samples from there.

    // choose a starting point for when to start sampling
    int sample_start_depth = 0;
    if (game_depth - this->conf->max_number_of_samples > 0) {
        sample_start_depth = this->rng.getWithMax(game_depth -
                                                  this->conf->max_number_of_samples);
    }

    root = this->pe->jumpRoot(sample_start_depth);
    game_depth = sample_start_depth;

    const bool can_resign = this->rng.get() > this->conf->resign_false_positive_retry_percentage;

    // don't start as if the game is done
    if (can_resign) {
        while (root->getCurrentScore(root->lead_role_index) < this->conf->resign_score_probability) {
            sample_start_depth--;
            // XXX add to configuration (although it is obscure)
            if (sample_start_depth < 10) {
                break;
            }

            root = this->pe->jumpRoot(sample_start_depth);
        }
    }

    // sampling:
    this->pe->updateConf(this->conf->sample_puct_config);
    iterations = this->conf->sample_iterations;
    while (true) {
        const int sample_count = this_game_samples.size();
        if (sample_count == this->conf->max_number_of_samples) {
            break;
        }

        if (root->isTerminal()) {
            break;
        }

        if (dupe(root->getBaseState())) {
            // need random choice
            int choice = rng.getWithMax(root->num_children);
            const PuctNodeChild* child = root->getNodeChild(this->scheduler->getRoleCount(), choice);

            root = this->pe->fastApplyMove(child);
            total_iterations += iterations;
            game_depth++;
            continue;
        }

        // we will create a sample, add to unique states here before jumping out of continuation
        this->manager->addUniqueState(root->getBaseState());

        const PuctNodeChild* choice = this->pe->onNextMove(iterations);

        // create a sample (call getProbabilities() to ensure probabilities are right for policy)
        this->pe->getProbabilities(root, 1.0f);
        Sample* s = this->scheduler->createSample(root);
        s->depth = game_depth;

        // keep a local ref to it for when we score it
        this_game_samples.push_back(s);

        root = this->pe->fastApplyMove(choice);
        total_iterations += iterations;
        game_depth++;
    }

    // scoring:
    this->pe->updateConf(this->conf->score_puct_config);
    iterations = this->conf->score_iterations;

    if (!this_game_samples.empty()) {
        while (true) {
            if (root->isTerminal()) {
                break;
            }

            if (can_resign) {
                if (root->getCurrentScore(root->lead_role_index) < this->conf->resign_score_probability) {
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

            this->manager->addSample(sample);
        }
    }

    // this was defined right at the top...
    // XXXXX add to some stats and report at end of cycle... way too much noise
    if (saw_dupes) {
        //K273::l_verbose("saw dupe states %d", saw_dupes);
    }

}

void SelfPlay::playGamesForever() {
    while (true) {
        this->playOnce();
    }
}

