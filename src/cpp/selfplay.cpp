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
    if (value < amount) {
        return 0.0;

    } else if (value > (1.0f - amount)) {
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
        // we haven't started taking samples and we hit the end of the road... woops
        if (node->isTerminal()) {
            break;
        }

        const PuctNodeChild* choice = this->pe->onNextMove(iterations);
        node = this->pe->fastApplyMove(choice);
        game_depth++;
    }

    // ok, we have a complete game.  Choose a starting point and start running samples from there.

    ASSERT(node->game_depth == game_depth);

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

            // XXX add to configuration (although it is obscure)
            if (sample_start_depth < 10) {
                break;
            }

            node = this->pe->jumpRoot(sample_start_depth);
        }
    }

    return node;
}

bool SelfPlay::resign(PuctNode* node) {
    float score = node->getCurrentScore(node->lead_role_index);
    bool should_resign = score < this->conf->resign_score_probability;

    // already resigned
    ASSERT (!this->has_resigned);

    if (this->can_resign) {
        this->has_resigned = should_resign;
        return should_resign;

    } else if (should_resign && this->false_postitive_resign_check) {
        // check for false postive on resigns...

        this->false_postitive_resign_check = false;

        // add the scores to check later
        for (int ii=0; ii<this->role_count; ii++) {
            float clamped_score = clamp(node->getCurrentScore(ii),
                                        this->conf->resign_score_probability);

            this->resign_false_positive_check_scores.push_back(clamped_score);
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

            // need random choice (XXX use the probabilty distribution)
            int choice = rng.getWithMax(node->num_children);
            const PuctNodeChild* child = node->getNodeChild(this->role_count, choice);

            node = this->pe->fastApplyMove(child);
            continue;
        }

        // we will create a sample, add to unique states here
        this->manager->getUniqueStates()->add(node->getBaseState());

        const PuctNodeChild* choice = this->pe->onNextMove(iterations);

        // create a sample (call getProbabilities() to ensure probabilities are right for policy)
        this->pe->getProbabilities(node, 1.0f);
        Sample* s = this->manager->createSample(node);
        s->depth = node->game_depth;

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

    this->has_resigned = false;
    this->false_postitive_resign_check = !can_resign;
    this->resign_false_positive_check_scores.clear();

    // reset the puct evaluator
    this->pe->reset();

    // randomly choose if we can resign or not
    this->can_resign = this->rng.get() > this->conf->resign_false_positive_retry_percentage;

    // first select a starting point
    PuctNode* node = this->selectNode();
    ASSERT(!node->isTerminal());

    int starting_sample_depth = node->game_depth;

    node = this->collectSamples(node);

    // no samples :(
    if (this->game_samples.empty()) {
        this->manager->incrNoSamples();
        return;
    }

    // final stage, scoring:
    if (!this->has_resigned && !node->isTerminal()) {
        node = this->runToEnd(node);
    }

    // update current samples (remember we can have multiple per game)
    for (auto sample : this->game_samples) {

        bool is_resign_false_positive = false;
        for (int ii=0; ii<this->role_count; ii++) {
            float score = clamp(node->getCurrentScore(ii),
                                this->conf->resign_score_probability);

            sample->final_score.push_back(score);

            if (!is_resign_false_positive && !resign_false_positive_check_scores.empty()) {
                if (std::fabs(score - resign_false_positive_check_scores[ii]) > 0.0001f) {

                    is_resign_false_positive = true;
                    this->manager->incrResignFalsePositives();
                }
            }
        }

        // game id?
        sample->match_identifier = this->identifier + K273::fmtString("_%d",
                                                                     this->match_count);
        sample->has_resigned = this->has_resigned;
        sample->game_length = node->game_depth;
        sample->starting_sample_depth = starting_sample_depth;
        sample->resign_false_positive = is_resign_false_positive;

        this->manager->addSample(sample);
    }
}

void SelfPlay::playGamesForever() {
    while (true) {
        this->playOnce();
    }
}
