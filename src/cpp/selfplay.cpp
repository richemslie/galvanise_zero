/*
General note of is_finalised versus isTerminal().  When collecting, we want to add everything
(especially if oscillating) even if game has been finalised.  Let resign threshold handle those
cases when game is finalised long before termination.
Therefore it makes sense to only use isTerminal() in collecting samples, but use is_finalised in
runToEnd(). */

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


bool SelfPlay::resign(const PuctNode* node) {
    ASSERT(!node->isTerminal());

    // should never get here if already resigned
    ASSERT(!this->has_resigned);

    // constraint, assumes we are in a choice node
    const float score = node->getCurrentScore(node->lead_role_index);

    // how are these even correct?  we don't know which one actually resigned to check these
    // values.  ANSWER: by abusing whether resignX_false_positive_check_scores is empty, we know.
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
    this->pe->updateConf(this->conf->puct_config);

    const bool do_oscillate_sampling = this->conf->oscillate_sampling_pct > 0;

    const int evals = this->conf->evals_per_move;

    auto& man = *this->manager->getUniqueStates();

    while (true) {
        if (this->conf->abort_max_length > 0 &&
            node->game_depth > this->conf->abort_max_length) {

            K273::l_verbose("Exiting collectSamples - abort_max_length exceeded %d/%d",
                            node->game_depth, this->conf->abort_max_length);

            break;
        }

        if (node->isTerminal()) {
            break;
        }

        const PuctNodeChild* choice = nullptr;

        bool do_skip = false;
        if (!man.isUnique(node->getBaseState(), node->game_depth)) {
            this->manager->incrDupes();
            do_skip = true;

        } else {
            // test whether to honour oscillate_sampling_pct
            if (do_oscillate_sampling &&
                this->rng.get() > this->conf->oscillate_sampling_pct) {
                do_skip = true;
            }
        }

        if (!do_skip) {
            // not atomic, but good enough
            man.add(node->getBaseState());

            // reset the node beforehand
            this->pe->resetRootNode();

            // run the simulations
            choice = this->pe->onNextMove(evals);

            // create a sample (call getProbabilities() to ensure probabilities are right for policy)
            this->pe->getProbabilities(node, this->conf->temperature_for_policy, false);

            // comment out this for debugging
            // this->pe->dumpNode(node, choice);

            Sample* s = this->manager->createSample(this->pe, node);

            // keep a local ref to it for when we score it
            this->game_samples.push_back(s);

        } else {
            const int skip_evals = std::max(16, (int) this->rng.getWithMax(evals / 3 + 1));
            this->pe->updateConf(this->conf->run_to_end_puct_config);
            choice = this->pe->onNextMove(skip_evals);

            // comment out this for debugging
            // this->pe->dumpNode(node, choice);

            // create a sample (call getProbabilities() to ensure probabilities are right for policy)
            this->pe->getProbabilities(node, this->conf->temperature_for_policy, false);

            // comment out this for debugging
            // this->pe->dumpNode(node, choice);

            Sample* s = this->manager->createSample(this->pe, node);

            // keep a local ref to it for when we score it
            this->game_samples.push_back(s);

            this->pe->updateConf(this->conf->puct_config);
        }

        // apply the move
        ASSERT(choice != nullptr);
        node = this->pe->fastApplyMove(choice);

        if (node->isTerminal()) {
            break;
        }

        // the node score is most accurate at this point, so can check resign here
        if (!this->has_resigned) {
            this->resign(node);
        }

        // some of the time actually resign (if we have at least one sample)
        if (this->has_resigned && this->game_samples.size() > 1) {
            this->manager->incrResigns();
            break;
        }
    }

    return node;
}

int SelfPlay::runToEnd(PuctNode* node, std::vector <float>& final_scores) {
    this->pe->updateConf(this->conf->run_to_end_puct_config);

    const int evals = this->conf->run_to_end_evals;
    const bool run_to_end_can_resign = (this->has_resigned &&
                                        this->rng.get() > this->conf->run_to_end_pct);

    auto done = [this] (const PuctNode* node) {
        if (this->conf->abort_max_length > 0 && node->game_depth > this->conf->abort_max_length) {
            return true;
        }

        if (node->is_finalised) {
            return true;
        }

        return false;
    };

    // simply run the game to end
    while (!done(node)) {
        const PuctNodeChild* choice = this->pe->onNextMove(evals);
        node = this->pe->fastApplyMove(choice);

        if (node->is_finalised) {
            break;
        }

        if (run_to_end_can_resign &&
            node->game_depth > this->conf->run_to_end_minimum_game_depth) {

            const float lead_score = node->getCurrentScore(node->lead_role_index);
            if (lead_score < this->conf->run_to_end_early_score) {
                this->manager->incrEarlyRunToEnds();

                for (int ri=0; ri<this->role_count; ri++) {
                    if (ri == node->lead_role_index) {
                        final_scores.push_back(0.0);
                    } else {
                        final_scores.push_back(1.0);
                    }
                }

                return node->game_depth;
            }
        }
    }

    if (this->conf->abort_max_length > 0 && node->game_depth > this->conf->abort_max_length) {
        return -1;
    }

    for (int ri=0; ri<this->role_count; ri++) {
        final_scores.push_back(node->getCurrentScore(ri));
    }

    return node->game_depth;
}

bool SelfPlay::checkFalsePositive(const std::vector <float>& false_positive_check_scores,
                                  float resign_probability, float final_score,
                                  int role_index) {

    if (!false_positive_check_scores.empty()) {
        const float score = false_positive_check_scores[role_index];

        // 1.05 -> fixes rounding issues
        if ((score < resign_probability * 1.05) && final_score > 0.49) {
            K273::l_verbose("False positive resign %.2f, final %.2f",
                            score, final_score);
            return true;
        }
    }


    return false;
}

void SelfPlay::addSamples(const std::vector <float>& final_scores,
                          int starting_sample_depth, int game_depth) {

    // determine final score / and if a false positive if resigned
    bool is_resign0_false_positive = false;
    bool is_resign1_false_positive = false;

    for (int ri=0; ri<this->role_count; ri++) {
        const float final_score = final_scores[ri];

        if (this->has_resigned) {
            if (!is_resign0_false_positive) {
                if (this->checkFalsePositive(this->resign0_false_positive_check_scores,
                                             this->conf->resign0_score_probability, final_score, ri)) {
                    is_resign0_false_positive = true;
                    this->manager->incrResign0FalsePositives();
                }
            }

            if (!is_resign1_false_positive) {
                if (this->checkFalsePositive(this->resign1_false_positive_check_scores,
                                             this->conf->resign1_score_probability, final_score, ri)) {
                    is_resign1_false_positive = true;
                    this->manager->incrResign1FalsePositives();
                }
            }
        }
    }

    // update current samples (remember we can have multiple per game)
    for (auto sample : this->game_samples) {
        sample->final_score = final_scores;
        sample->game_length = game_depth;
        sample->match_identifier = this->identifier + K273::fmtString("_%d", this->match_count);
        sample->has_resigned = this->has_resigned;
        sample->resign_false_positive = is_resign0_false_positive || is_resign1_false_positive;
        sample->starting_sample_depth = starting_sample_depth;
        this->manager->addSample(sample);
    }
}

///////////////////////////////////////////////////////////////////////////////

void SelfPlay::playOnce() {
    this->match_count++;

    // reset everything
    this->game_samples.clear();

    // reset resignation status
    this->has_resigned = false;

    // randomly choose if we *can* resign or not
    double r = this->rng.get();
    this->can_resign0 = r > this->conf->resign0_pct;
    this->can_resign1 = r > this->conf->resign1_pct;

    this->resign0_false_positive_check_scores.clear();
    this->resign1_false_positive_check_scores.clear();

    // reset the puct evaluator and establish root
    this->pe->reset(0);

    // Initial node
    PuctNode* node = this->pe->establishRoot(this->initial_state);
    ASSERT(!node->isTerminal());

    const int starting_sample_depth = node->game_depth;

    node = this->collectSamples(node);

    // no samples :(
    if (this->game_samples.empty()) {
        //K273::l_verbose("Failed to produce any samples - restarting");
        this->manager->incrNoSamples();
        return;
    }

    // final stage, scoring:
    std::vector <float> final_scores;
    const int game_depth = this->runToEnd(node, final_scores);

    if (game_depth == -1) {
        this->manager->incrAbortsGameLength();
        return;
    }

    this->addSamples(final_scores, starting_sample_depth, game_depth);
}

void SelfPlay::playGamesForever() {
    while (true) {
        this->playOnce();
    }
}
