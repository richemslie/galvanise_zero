#include "selfplay.h"

#include "sample.h"
#include "scheduler.h"
#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"
#include "selfplaymanager.h"

#include <statemachine/statemachine.h>

#include <k273/logging.h>

#include <cmath>
#include <vector>


using namespace GGPZero;


SelfPlay::SelfPlay(SelfPlayManager* manager, const SelfPlayConfig* conf, PuctEvaluator* pe,
                   const GGPLib::BaseState* initial_state, int role_count) :
    manager(manager),
    conf(conf),
    pe(pe),
    initial_state(initial_state),
    role_count(role_count) {
}


SelfPlay::~SelfPlay() {
}

int clamp(float value, float amount) {
    if (value < amount) {
        return 0.0;

    } else if (value > (1.0f - amount)) {
        return 1.0;
    } else {
        return value;
    }
}

PuctNode* SelfPlay::selectNode(const bool can_resign) {

    PuctNode* node = this->pe->establishRoot(this->initial_state, 0);

    // ok simple loop until we start taking samples
    this->pe->updateConf(this->conf->select_puct_config);
    const int iterations = this->conf->select_iterations;

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
    if (can_resign) {
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

PuctNode* SelfPlay::collectSamples(PuctNode* node) {

    // sampling:
    this->pe->updateConf(this->conf->sample_puct_config);
    const int iterations = this->conf->sample_iterations;

    while (true) {
        const int sample_count = this->game_samples.size();
        if (sample_count == this->conf->max_number_of_samples) {
            break;
        }

        if (node->isTerminal()) {
            break;
        }

        if (!this->isUnique(node->getBaseState())) {
            this->manager->incrDupes();

            // need random choice
            int choice = rng.getWithMax(node->num_children);
            const PuctNodeChild* child = node->getNodeChild(this->role_count, choice);

            node = this->pe->fastApplyMove(child);
            continue;
        }

        // we will create a sample, add to unique states here before jumping out of continuation
        this->manager->addUniqueState(node->getBaseState());

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

void SelfPlay::playOnce() {
    ASSERT(this->game_samples.empty());
    const bool can_resign = this->rng.get() > this->conf->resign_false_positive_retry_percentage;

    // reset the puct evaluator
    this->pe->reset();

    PuctNode* node = this->selectNode(can_resign);

    ASSERT(!node->isTerminal());

    // ok, we have a complete game.  Choose a starting point and start running samples from there.
    node = this->collectSamples(node);

    // no samples :(  XXX Worthy of a reporting statistic (add it to manager)
    if (this->game_samples.empty()) {
        this->manager->incrNoSamples();
        return;
    }

    // final stage, scoring:
    this->pe->updateConf(this->conf->score_puct_config);
    const int iterations = this->conf->score_iterations;

    // resignation and check for false positives
    bool resigned = false;
    bool false_postitive_resign_check = !can_resign;
    std::vector <float> resign_false_positive_check_scores;

    // run the game to end
    while (true) {
        if (node->isTerminal()) {
            break;
        }

        if (can_resign) {
            if (node->getCurrentScore(node->lead_role_index) < this->conf->resign_score_probability) {
                resigned = true;
                break;
            }
        } else {

            // check for false postive on resigns...
            if (false_postitive_resign_check) {
                for (int ii=0; ii<this->role_count; ii++) {
                    float score = node->getCurrentScore(ii);

                    if (score < this->conf->resign_score_probability) {
                        false_postitive_resign_check = false;

                        // add the scores to check later
                        for (int jj=0; jj<this->role_count; jj++) {
                            float clamped_score = clamp(node->getCurrentScore(jj),
                                                        this->conf->resign_score_probability);
                            resign_false_positive_check_scores.push_back(clamped_score);
                        }

                        // break out of the outer for loop
                        break;
                    }
                }
            }
        }

        const PuctNodeChild* choice = this->pe->onNextMove(iterations);
        node = this->pe->fastApplyMove(choice);
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

        sample->resigned = resigned;
        sample->game_length = node->game_depth;
        sample->resign_false_positive = is_resign_false_positive;

        this->manager->addSample(sample);
    }
}


void SelfPlay::playGamesForever() {
    while (true) {
        this->playOnce();
    }
}


bool SelfPlay::isUnique(const GGPLib::BaseState* bs) {
    auto unique_states = this->manager->getUniqueStates();

    auto it = unique_states->find(bs);
    return it == this->manager->getUniqueStates()->end();
}
