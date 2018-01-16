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


SelfPlay::SelfPlay(SelfPlayManager* manager, const SelfPlayConfig* conf, PuctEvaluator* pe,
                   const GGPLib::BaseState* initial_state, int role_count) :
    manager(manager),
    conf(conf),
    pe(pe),
    initial_state(initial_state),
    role_count(role_count),
    saw_dupes(0) {
}


SelfPlay::~SelfPlay() {
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
            this->saw_dupes++;

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
    /*
      * todo: see XXX
      * resignation false positives must be recorded, we'll need to increase the pct
      * stats / report duplicates, etc
      */

    ASSERT(this->game_samples.empty());
    const bool can_resign = this->rng.get() > this->conf->resign_false_positive_retry_percentage;

    // reset the puct evaluator
    this->pe->reset();

    PuctNode* node = this->selectNode(can_resign);

    ASSERT(!node->isTerminal());

    // ok, we have a complete game.  Choose a starting point and start running samples from there.
    node = this->collectSamples(node);

    // final stage, scoring:
    this->pe->updateConf(this->conf->score_puct_config);
    const int iterations = this->conf->score_iterations;

    bool resigned = false;
    if (!this->game_samples.empty()) {

        while (true) {
            if (node->isTerminal()) {
                break;
            }

            if (can_resign) {
                if (node->getCurrentScore(node->lead_role_index) < this->conf->resign_score_probability) {
                    resigned = true;
                    break;
                }
            }

            const PuctNodeChild* choice = this->pe->onNextMove(iterations);
            node = this->pe->fastApplyMove(choice);
        }

        // update samples
        for (auto sample : this->game_samples) {
            sample->game_length = node->game_depth;
            sample->resigned = resigned;
            for (int ii=0; ii<this->role_count; ii++) {
                // need to clamp scores over > 0.9
                double score = node->getCurrentScore(ii);
                if (score < this->conf->resign_score_probability) {
                    score = 0.0;
                } else if (score > (1.0 - this->conf->resign_score_probability)) {
                    score = 1.0;
                }

                sample->final_score.push_back(score);
            }

            this->manager->addSample(sample);
        }

        this->game_samples.clear();
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
