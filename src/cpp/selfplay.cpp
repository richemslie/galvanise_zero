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

// ZZZ XXX revamp the resign logic...

using namespace GGPZero;

class NetworkReloaded {
};

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
        // note this score isn't that reliable...  since likely we didn't do any iterations yet
        if (this->can_resign0) {
            float lead_score = node->getCurrentScore(node->lead_role_index);
            while (lead_score < this->conf->resign0_score_probability ||
                   lead_score > (1 - this->conf->resign0_score_probability)) {
                sample_start_depth--;

                // XXX some configurable amount?
                if (sample_start_depth < 4) {
                    break;
                }

                node = this->pe->jumpRoot(sample_start_depth);
            }
        }

        if (this->manager->getUniqueStates()->isUnique(node->getBaseState())) {
            return node;
        }
    }

    return nullptr;
}

bool SelfPlay::resign(const PuctNode* node) {
    ASSERT(!node->isTerminal());

    // should never get here if already resigned
    ASSERT(!this->has_resigned);

    float score = node->getCurrentScore(node->lead_role_index);

    if (this->can_resign0 && score < this->conf->resign0_score_probability) {
        this->has_resigned = true;

        // add the scores to check later
        for (int ii=0; ii<this->role_count; ii++) {
            this->resign0_false_positive_check_scores.push_back(node->getCurrentScore(ii));
        }

    } else if (this->can_resign1 && score < this->conf->resign1_score_probability) {
        this->has_resigned = true;

        // add the scores to check later
        for (int ii=0; ii<this->role_count; ii++) {
            this->resign1_false_positive_check_scores.push_back(node->getCurrentScore(ii));
        }
    }

    return this->has_resigned;
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

        if (node->isTerminal()) {
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

        const PuctNodeChild* choice = nullptr;
        try {
            // run the simulations
            choice = this->pe->onNextMove(iterations);

            // we will create a sample, add to unique states here
            this->manager->getUniqueStates()->add(node->getBaseState());

            // create a sample (call getProbabilities() to ensure probabilities are right for policy)
            // ZZZ should try changing this %
            this->pe->getProbabilities(node, 1.0f, true);
        } catch (NetworkReloaded&) {
            //this->pe->resetRoot();
            continue;
            break;
        }

        ASSERT (choice != nullptr);

        // XXX why we get the manager to do this????  Doesn't make sense(we can grab the statemachine from this->pe)...
        Sample* s = this->manager->createSample(this->pe, node);

        // tmp XXX ZZZ
        s->lead_role_index = node->lead_role_index;

        // keep a local ref to it for when we score it
        this->game_samples.push_back(s);

        // This doesnt stop from running the samples.  When adding the samples (to be trained on -
        // see "*** AFTER SAMPLING"), we will drop the moves that lead to the loss.  This is
        // because MCTS starts to play randomly or stupidly as it knows it has no chance to win,
        // and the network doesnt know any differently on the policy side whether it was in this
        // mode.  On the other hand the winning moves are added.
        // Note: If the score ends up being a false positive, then all moves are added.
        if (!this->has_resigned) {
            this->resign(node);
        }

        node = this->pe->fastApplyMove(choice);
    }

    return node;
}

PuctNode* SelfPlay::runToEnd(PuctNode* node) {
    this->pe->updateConf(this->conf->score_puct_config);
    const int iterations = this->conf->score_iterations;

    // simply run the game to end
    while (true) {
        if (node->isTerminal()) {
            break;
        }

        try {
            const PuctNodeChild* choice = this->pe->onNextMove(iterations);
            node = this->pe->fastApplyMove(choice);
        } catch (NetworkReloaded&) {
            //this->pe->resetRoot();
        }
    }

    return node;
}

bool SelfPlay::checkFalsePositive(const std::vector <float>& false_positive_check_scores,
                                  float resign_probability, float final_score,
                                  int role_index) {

    if (!false_positive_check_scores.empty()) {
        float was_resign_score = clamp(false_positive_check_scores[role_index],
                                       resign_probability);

        if (std::fabs(final_score - was_resign_score) > 0.01f) {
            K273::l_verbose("False positive resign %.2f, clamp %.2f, final %.2f",
                            false_positive_check_scores[role_index],
                            was_resign_score,
                            final_score);

            return true;
        }
    }

    return false;
}

void SelfPlay::playOnce() {
    this->match_count++;

    // reset everything
    this->game_samples.clear();

    // reset resignation status
    this->has_resigned = false;

    // randomly choose if we *can* resign or not
    double r = this->rng.get();
    this->can_resign0 = r > this->conf->resign0_false_positive_retry_percentage;
    this->can_resign1 = r > this->conf->resign1_false_positive_retry_percentage;

    this->resign0_false_positive_check_scores.clear();
    this->resign1_false_positive_check_scores.clear();

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
        //K273::l_verbose("Failed to produce any samples - restarting");
        this->manager->incrNoSamples();
        return;
    }

    // final stage, scoring:
    if (!node->isTerminal()) {
        node = this->runToEnd(node);
    }

    // *** AFTER SAMPLING

    ASSERT(node->isTerminal());

    // determine final score / and if a false positive if resigned
    bool is_resign0_false_positive = false;
    bool is_resign1_false_positive = false;
    std::vector <float> final_score;

    for (int ri=0; ri<this->role_count; ri++) {
        float score = node->getCurrentScore(ri);

        if (this->has_resigned) {

            if (!is_resign0_false_positive) {
                if (this->checkFalsePositive(this->resign0_false_positive_check_scores,
                                             this->conf->resign0_score_probability, score, ri)) {
                    is_resign0_false_positive = true;
                    this->manager->incrResign0FalsePositives();
                }
            }

            if (!is_resign1_false_positive) {
                if (this->checkFalsePositive(this->resign1_false_positive_check_scores,
                                             this->conf->resign1_score_probability, score, ri)) {
                    is_resign1_false_positive = true;
                    this->manager->incrResign1FalsePositives();
                }
            }
        }

        final_score.push_back(score);
    }

    // update current samples (remember we can have multiple per game)
    for (auto sample : this->game_samples) {

        if (this->has_resigned && !is_resign0_false_positive && !is_resign1_false_positive) {
            if (sample->resultant_puct_score[sample->lead_role_index] < this->conf->resign0_score_probability) {
                // dont add sample, wins go through though
                continue;
            }
        }

        sample->final_score = final_score;
        sample->game_length = node->game_depth;
        sample->match_identifier = this->identifier + K273::fmtString("_%d", this->match_count);
        sample->has_resigned = this->has_resigned;
        sample->resign_false_positive = is_resign0_false_positive || is_resign1_false_positive;
        sample->starting_sample_depth = starting_sample_depth;
        this->manager->addSample(sample);
    }
}

void SelfPlay::playGamesForever() {
    while (true) {
        this->playOnce();
    }
}
