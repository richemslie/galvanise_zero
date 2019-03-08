
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


PuctNode* SelfPlay::selectNode() {
    // reset the puct evaluator and establish root
    this->pe->reset(0);

    if (this->conf->number_repeat_states_draw != -1) {
        pe->setRepeatStateDraw(-1, -1);
    }

    PuctNode* node = this->pe->establishRoot(this->initial_state);

    this->pe->updateConf(this->conf->select_puct_config);
    const int iterations = this->conf->select_iterations;

    // start from initial_state if select is turned off
    if (iterations < 0 || this->play_full_game) {
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

        if (this->conf->abort_max_length > 0 && game_depth > this->conf->abort_max_length) {
            break;
        }
    }

    ASSERT(node->game_depth == game_depth);

    auto not_in_resign_state = [] (PuctNode* the_node, float prob) {
        const float lead_score = the_node->getCurrentScore(the_node->lead_role_index);

        // allow for a bit more than resign score
        prob += 0.025;

        return ! (lead_score < prob || lead_score > (1 - prob));
    };

    // set the new starting state before resign
    int modified_game_depth = game_depth - this->conf->max_number_of_samples;
    modified_game_depth = std::max(0, modified_game_depth);
    node = this->pe->jumpRoot(modified_game_depth);

    // don't start as if the game is done
    // note this score isn't that reliable...  since did *not* do any iterations yet
    if (this->can_resign0) {
        while (modified_game_depth > 0) {
            if (not_in_resign_state(node, this->conf->resign0_score_probability)) {
                break;
            }

            modified_game_depth -=1;
            modified_game_depth = std::max(0, modified_game_depth);
            node = this->pe->jumpRoot(modified_game_depth);
        }
    }

    if (this->can_resign1) {
        while (modified_game_depth > 0) {
            if (not_in_resign_state(node, this->conf->resign1_score_probability)) {
                break;
            }

            modified_game_depth -=1;
            modified_game_depth = std::max(0, modified_game_depth);
            node = this->pe->jumpRoot(modified_game_depth);
        }
    }

    if (modified_game_depth < 3) {
        return node;
    }

    int sample_start_depth = this->rng.getWithMax(modified_game_depth);
    node = this->pe->jumpRoot(sample_start_depth);

    return node;
}

bool SelfPlay::resign(const PuctNode* node) {
    ASSERT(!node->isTerminal());

    // should never get here if already resigned
    ASSERT(!this->has_resigned);

    // constraint, assumes we are in a choice node
    float score = node->getCurrentScore(node->lead_role_index);

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
    ASSERT(this->conf->max_number_of_samples > 0);

    // sampling:
    this->pe->updateConf(this->conf->sample_puct_config);

    if (this->conf->number_repeat_states_draw != -1) {
        pe->setRepeatStateDraw(this->conf->number_repeat_states_draw,
                               this->conf->repeat_states_score);
    }

    const int iterations = this->conf->sample_iterations;

    auto& man = *this->manager->getUniqueStates();

    while (true) {

        if (this->conf->abort_max_length > 0 &&
            node->game_depth > this->conf->abort_max_length) {

            K273::l_verbose("Exiting collectSamples - abort_max_length exceeded %d/%d",
                            node->game_depth, this->conf->abort_max_length);

            break;
        }

        const int sample_count = this->game_samples.size();
        if (sample_count >= this->conf->max_number_of_samples) {
            if (this->play_full_game) {
                if (node->is_finalised) {
                    break;
                }

            } else {
                break;
            }
        }

        const PuctNodeChild* choice = nullptr;


        bool do_skip = false;
        if (man.isUnique(node->getBaseState(), node->game_depth)) {
            // not atomic, but good enough
            man.add(node->getBaseState());

            // test whether to honour oscillate_sampling_pct
            if (this->play_full_game &&
                this->conf->oscillate_sampling_pct > 0 &&
                this->rng.get() > this->conf->oscillate_sampling_pct) {
                do_skip = true;
                //K273::l_verbose("Skip oscillate");
            }

        } else {
            this->manager->incrDupes();
            //K273::l_verbose("Skip dupe");
            do_skip = true;
        }

        if (!do_skip) {

            // run the simulations
            choice = this->pe->onNextMove(iterations);

            // create a sample (call getProbabilities() to ensure probabilities are right for policy)
            this->pe->getProbabilities(node, this->conf->temperature_for_policy, false);

            // XXX why we get the manager to do this????  Doesn't make sense(we can grab the
            // statemachine from this->pe)...
            Sample* s = this->manager->createSample(this->pe, node);

            // tmp XXX, we should also move the createSample() here
            s->lead_role_index = node->lead_role_index;

            // keep a local ref to it for when we score it
            this->game_samples.push_back(s);

        } else {
            const int rng_amount = std::min(10, iterations / 2 - 42);
            const int skip_iterations = 42 + this->rng.getWithMax(rng_amount);

            //K273::l_verbose("Skipping with %d iterations", skip_iterations);

            choice = this->pe->onNextMove(skip_iterations);
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
        // note, that this is done every turn, so over n time steps, more likely to actually resign
        if (this->has_resigned && this->game_samples.size() > 1) {
            if (this->rng.get() < this->conf->pct_actually_resign) {
                this->manager->incrActualResigns();
                break;
            }
        }
    }

    return node;
}

int SelfPlay::runToEnd(PuctNode* node, std::vector <float>& final_scores) {
    this->pe->updateConf(this->conf->score_puct_config);

    const int iterations = this->conf->score_iterations;
    const bool run_to_end_can_resign = (this->has_resigned &&
                                        this->rng.get() < this->conf->run_to_end_early_pct);

    auto done = [this] (const PuctNode* node) {
        if (node->isTerminal()) {
            return true;
        }

        if (this->conf->abort_max_length > 0) {
            if (node->game_depth > this->conf->abort_max_length) {
                return true;
            }

        } else {
            if (node->is_finalised) {
                return true;
            }
        }

        return false;
    };

    // simply run the game to end
    while (!done(node)) {
        const PuctNodeChild* choice = this->pe->onNextMove(iterations);
        node = this->pe->fastApplyMove(choice);

        // XXX do we need this?  node should still be ok.   XXX check & remove.
        if (node->isTerminal()) {
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

    if (this->conf->abort_max_length > 0 &&
        node->game_depth > this->conf->abort_max_length) {
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
        if ((score < resign_probability * 1.05) && final_score > 0.99) {
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

    this->play_full_game = false;
    if (this->conf->play_full_game_pct > 0) {
        double r = this->rng.get();
        if (r < this->conf->play_full_game_pct) {
            this->play_full_game = true;
        }
    }

    // randomly choose if we *can* resign or not
    double r = this->rng.get();
    this->can_resign0 = r > this->conf->resign0_false_positive_retry_percentage;
    this->can_resign1 = r > this->conf->resign1_false_positive_retry_percentage;

    this->resign0_false_positive_check_scores.clear();
    this->resign1_false_positive_check_scores.clear();

    // first select a starting point
    PuctNode* node = this->selectNode();

    if (node == nullptr) {
        K273::l_verbose("Failed to select a node - restarting");
        this->manager->incrNoSamples();
        return;
    }

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
