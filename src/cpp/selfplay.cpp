#include "selfplay.h"

#include "sample.h"
#include "scheduler.h"
#include "uniquestates.h"
#include "selfplaymanager.h"

#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"

#include <statemachine/statemachine.h>

#include <k273/logging.h>
#include <k273/strutils.h>

#include <cmath>
#include <vector>


using namespace GGPZero;


SelfPlay::SelfPlay(SelfPlayManager* manager, const SelfPlayConfig* conf, PuctEvaluator* pe,
                   const GGPLib::BaseState* initial_state, int role_count, std::string identifier) :
    manager(manager),
    conf(conf),
    pe(pe),
    initial_state(initial_state),
    role_count(role_count),
    identifier(identifier),
    match_count(0) {
}


SelfPlay::~SelfPlay() {
}

float clamp(float value, float amount) {
    // 1.05 -> fixes rounding issues
    if (value < amount * 1.05) {
        return 0.0;

    } else if (value > (1.0f - amount * 1.05)) {
        return 1.0;

    } else {
        return value;
    }
}

PuctNode* SelfPlay::selectNode() {
    PuctNode* node = this->pe->establishRoot(this->initial_state, 0);

    // ok simple loop until we start taking samples
    this->pe->updateConf(this->conf->select_puct_config);
    const int iterations = this->conf->select_iterations;

    // start from initial_state if select is turned off
    if (iterations < 0) {
        return node;
    }

    // selecting - we playout whole game and then choose a random move
    int game_depth = 0;
    while (true) {
        if (node->isTerminal()) {
            break;
        }

        const PuctNodeChild* choice = this->pe->onNextMove(iterations);
        node = this->pe->fastApplyMove(choice);
        game_depth++;
    }

    ASSERT(node->game_depth == game_depth);

    for (int ii=0; ii<10; ii++) {
        // ok, we have a complete game.  Choose a starting point and start running samples from there.

        // choose a starting point for when to start sampling
        int sample_start_depth = 0;
        if (game_depth - this->conf->max_number_of_samples > 0) {
            sample_start_depth = this->rng.getWithMax(game_depth -
                                                      this->conf->max_number_of_samples);
        }

        ASSERT(sample_start_depth >= 0);

        node = this->pe->jumpRoot(sample_start_depth);
        ASSERT(node->game_depth == sample_start_depth);

        // don't start as if the game is done
        if (this->can_resign) {
            // note this score isn't that reliable...  since likely we didn't do any iterations
            while (node->getCurrentScore(node->lead_role_index) < this->conf->resign_score_probability) {
                sample_start_depth--;
                node = this->pe->jumpRoot(sample_start_depth);
            }
        }

        if (this->manager->getUniqueStates()->isUnique(node->getBaseState())) {
            return node;
        }
    }

    return nullptr;
}

bool SelfPlay::resign(PuctNode* node) {
    ASSERT(!node->isTerminal());

    // should never get here if already resigned
    ASSERT(!this->has_resigned);

    float score = node->getCurrentScore(node->lead_role_index);
    bool should_resign = score < this->conf->resign_score_probability;

    if (this->can_resign) {
        this->has_resigned = should_resign;
        return should_resign;

    } else if (should_resign && this->false_positive_resign_check) {
        // check for false positive on resigns...

        this->false_positive_resign_check = false;

        // add the scores to check later
        for (int ii=0; ii<this->role_count; ii++) {
            this->resign_false_positive_check_scores.push_back(node->getCurrentScore(ii));
        }
    }

    return false;
}

PuctNode* SelfPlay::collectSamples(PuctNode* node) {

    // sampling:
    this->pe->updateConf(this->conf->sample_puct_config);
    const int iterations = this->conf->sample_iterations;

    while (true) {
        // sample_count < 0, run to the end with samples
        const int sample_count = this->game_samples.size();
        if (sample_count > 0 && sample_count == this->conf->max_number_of_samples) {
            break;
        }

        if (node->isTerminal() || this->resign(node)) {
            break;
        }

        if (!this->manager->getUniqueStates()->isUnique(node->getBaseState())) {
            this->manager->incrDupes();

            // break out here, no point playing randomly when samples have been taken
            if (sample_count > 0) {
                break;
            }

            // move to next state via selector
            {
                this->pe->updateConf(this->conf->select_puct_config);
                const PuctNodeChild* choice = this->pe->onNextMove(this->conf->select_iterations);
                node = this->pe->fastApplyMove(choice);

                this->pe->updateConf(this->conf->sample_puct_config);
            }

            continue;
        }

        // we will create a sample, add to unique states here
        this->manager->getUniqueStates()->add(node->getBaseState());

        // run the simulations
        const PuctNodeChild* choice = this->pe->onNextMove(iterations);

        // create a sample (call getProbabilities() to ensure probabilities are right for policy)
        this->pe->getProbabilities(node, 1.0f, true);
        Sample* s = this->manager->createSample(this->pe, node);

        // keep a local ref to it for when we score it
        this->game_samples.push_back(s);

        node = this->pe->fastApplyMove(choice);
    }

    return node;
}

PuctNode* SelfPlay::runToEnd(PuctNode* node) {
    this->pe->updateConf(this->conf->score_puct_config);
    const int iterations = this->conf->score_iterations;

    // simply run the game to end
    while (true) {
        if (node->isTerminal() || this->resign(node)) {
            break;
        }

        const PuctNodeChild* choice = this->pe->onNextMove(iterations);
        node = this->pe->fastApplyMove(choice);
    }

    return node;
}

void SelfPlay::playOnce() {
    this->match_count++;

    // reset everything
    this->game_samples.clear();

    // reset resignation status
    this->has_resigned = false;

    // randomly choose if we *can* resign or not
    this->can_resign = this->rng.get() > this->conf->resign_false_positive_retry_percentage;

    // if we *can not* resign, check for false positives
    this->false_positive_resign_check = !this->can_resign;
    this->resign_false_positive_check_scores.clear();

    // reset the puct evaluator
    this->pe->reset();

    // first select a starting point
    PuctNode* node = this->selectNode();

    if (node == nullptr) {
        K273::l_verbose("Failed to select a node - restarting");
        this->manager->incrNoSamples();
        return;
    }

    ASSERT(!node->isTerminal());

    int starting_sample_depth = node->game_depth;

    node = this->collectSamples(node);

    // no samples :(
    if (this->game_samples.empty()) {
        K273::l_verbose("Failed to produce no samples - restarting");
        this->manager->incrNoSamples();
        return;
    }

    // final stage, scoring:
    if (!this->has_resigned && !node->isTerminal()) {
        node = this->runToEnd(node);
    }

    // determine final score / and if a false positive if resigned
    bool is_resign_false_positive = false;
    std::vector <float> final_score;

    for (int ii=0; ii<this->role_count; ii++) {
        float score = node->getCurrentScore(ii);

        if (this->has_resigned) {
            score = clamp(score, this->conf->resign_score_probability);

        } else {
            if (!is_resign_false_positive && !this->resign_false_positive_check_scores.empty()) {
                float was_resign_score = clamp(this->resign_false_positive_check_scores[ii],
                                               this->conf->resign_score_probability);

                if (std::fabs(score - was_resign_score) > 0.01f) {
                    is_resign_false_positive = true;

                    K273::l_verbose("Was a false positive resign %.2f, clamp %.2f, final %.2f",
                                    this->resign_false_positive_check_scores[ii],
                                    was_resign_score,
                                    score);

                    this->manager->incrResignFalsePositives();
                }
            }
        }

        final_score.push_back(score);
    }

    // update current samples (remember we can have multiple per game)
    for (auto sample : this->game_samples) {
        sample->final_score = final_score;
        sample->game_length = node->game_depth;
        sample->match_identifier = this->identifier + K273::fmtString("_%d", this->match_count);
        sample->has_resigned = this->has_resigned;
        sample->resign_false_positive = is_resign_false_positive;
        sample->starting_sample_depth = starting_sample_depth;
        this->manager->addSample(sample);
    }
}

void SelfPlay::playGamesForever() {
    while (true) {
        this->playOnce();
    }
}
