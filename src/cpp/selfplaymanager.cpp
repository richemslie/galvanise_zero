#include "supervisor.h"

#include "sample.h"
#include "selfplay.h"
#include "scheduler.h"
#include "selfplaymanager.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/logging.h>
#include <k273/strutils.h>
#include <k273/exception.h>

#include <string>

using namespace GGPZero;

SelfPlayManager::SelfPlayManager(GGPLib::StateMachineInterface* sm,
                                 const GdlBasesTransformer* transformer,
                                 int batch_size,
                                 int number_of_previous_states,
                                 UniqueStates* unique_states,
                                 std::string identifier) :
    sm(sm->dupe()),
    transformer(transformer),
    batch_size(batch_size),
    number_of_previous_states(number_of_previous_states),
    unique_states(unique_states),
    identifier(identifier),
    saw_dupes(0),
    no_samples_taken(0),
    false_positive_resigns(0) {

    this->scheduler = new NetworkScheduler(this->transformer,
                                           this->sm->getRoleCount(),
                                           this->batch_size);

    // allocate buffers for predict_done_event
    for (int ii=0; ii<this->transformer->getNumberPolicies(); ii++) {
        const int num_of_floats_policies = (this->transformer->getPolicySize(ii) *
                                            this->batch_size);
        float* mem = new float[num_of_floats_policies];
        this->predict_done_event.policies.push_back(mem);
    }

    const int num_of_floats_final_scores = (this->sm->getRoleCount() *
                                            this->batch_size);

    this->predict_done_event.final_scores = new float[num_of_floats_final_scores];
    this->predict_done_event.pred_count = 0;
}

SelfPlayManager::~SelfPlayManager() {
    delete this->sm;
    delete this->scheduler;

    for (float* mem : this->predict_done_event.policies) {
        delete[] mem;
    }

    delete[] this->predict_done_event.final_scores;
}


///////////////////////////////////////////////////////////////////////////////

// will create a new sample based on the root tree
Sample* SelfPlayManager::createSample(const PuctEvaluator* pe, const PuctNode* node) {
    Sample* sample = new Sample;
    sample->state = this->sm->newBaseState();
    sample->state->assign(node->getBaseState());

    // Add previous states
    const PuctNode* cur = node->parent;
    for (int ii=0; ii<this->number_of_previous_states; ii++) {
        if (cur == nullptr) {
            break;
        }

        GGPLib::BaseState* bs = this->sm->newBaseState();
        bs->assign(cur->getBaseState());
        sample->prev_states.push_back(bs);

        cur = cur->parent;
    }

    // create empty vectors
    sample->policies.resize(this->sm->getRoleCount());

    for (int ri=0; ri<this->sm->getRoleCount(); ri++) {
        Sample::Policy& policy = sample->policies[ri];
        for (int ii=0; ii<node->num_children; ii++) {
            const PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), ii);
            if (ri == node->lead_role_index) {
                policy.emplace_back(child->move.get(ri),
                                    child->next_prob);
            } else {
                // XXX huge hack to make it work (for now)
                policy.emplace_back(child->move.get(ri), 1.0);
                break;
            }
        }
    }

    sample->resultant_puct_visits = node->visits;
    for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
        sample->resultant_puct_score.push_back(node->getCurrentScore(ii));
    }

    return sample;
}

void SelfPlayManager::addSample(Sample* sample) {
    this->samples.push_back(sample);
}

///////////////////////////////////////////////////////////////////////////////

void SelfPlayManager::startSelfPlayers(const SelfPlayConfig* config) {
    K273::l_info("SelfPlayManager::startSelfPlayers - starting %d players", this->batch_size);

    this->scheduler->createMainLoop();

    // create a bunch of self plays
    for (int ii=0; ii<this->batch_size; ii++) {
        // the statemachine is shared between all puctevaluators of this mananger.  Just be careful.
        PuctEvaluator* pe = new PuctEvaluator(this->sm,
                                              config->select_puct_config,
                                              this->scheduler);
        std::string self_play_identifier = this->identifier + K273::fmtString("_%d", ii);
        SelfPlay* sp = new SelfPlay(this, config, pe, this->sm->getInitialState(),
                                    this->sm->getRoleCount(), self_play_identifier);
        this->self_plays.push_back(sp);

        auto f = [sp]() {
            sp->playGamesForever();
        };

        this->scheduler->addRunnable(f);
    }
}

void SelfPlayManager::poll() {
    // VERY IMPORTANT: This must be called in the thread that the scheduler resides (along with its
    // co-routines)
    // To make this super clear, the selfplaymanger should have a greenlet and we should assert it
    // is the correct one before continueing.

    this->scheduler->poll(&this->predict_done_event, &this->ready_event);
}

void SelfPlayManager::reportAndResetStats() {
    if (this->saw_dupes) {
        K273::l_info("Number of dupe states seen %d", this->saw_dupes);
        this->saw_dupes = 0;
    }

    if (this->no_samples_taken) {
        K273::l_info("Number of plays where no samples were taken %d", this->no_samples_taken);
        this->no_samples_taken = 0;
    }

    if (this->false_positive_resigns) {
        K273::l_info("Number of false positive resigns seen %d", this->false_positive_resigns);
        this->false_positive_resigns = 0;
    }
}
