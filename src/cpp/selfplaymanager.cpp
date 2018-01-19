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
                                 UniqueStates* unique_states,
                                 std::string identifier) :
    sm(sm->dupe()),
    transformer(transformer),
    batch_size(batch_size),
    unique_states(unique_states),
    identifier(identifier),
    saw_dupes(0),
    no_samples_taken(0),
    false_positive_resigns(0) {

    this->scheduler = new NetworkScheduler(this->sm->dupe(),
                                           this->transformer,
                                           this->batch_size);

    // allocate buffers for predict_done_event
    const int num_of_floats_policies = (this->transformer->getPolicySize() *
                                        this->batch_size);

    const int num_of_floats_final_scores = (this->sm->getRoleCount() *
                                            this->batch_size);

    this->predict_done_event.policies = new float[num_of_floats_policies];
    this->predict_done_event.final_scores = new float[num_of_floats_final_scores];
    this->predict_done_event.pred_count = 0;
}

SelfPlayManager::~SelfPlayManager() {
    delete this->sm;
    delete this->scheduler;

    delete[] this->predict_done_event.policies;
    delete[] this->predict_done_event.final_scores;
}


///////////////////////////////////////////////////////////////////////////////

// will create a new sample based on the root tree
Sample* SelfPlayManager::createSample(const PuctNode* node) {
    Sample* sample = new Sample;
    sample->state = this->sm->newBaseState();
    sample->state->assign(node->getBaseState());

    for (int ii=0; ii<node->num_children; ii++) {
        const PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), ii);
        sample->policy.emplace_back(child->move.get(node->lead_role_index),
                                    child->next_prob);
    }

    sample->lead_role_index = node->lead_role_index;
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
        PuctEvaluator* pe = new PuctEvaluator(config->select_puct_config, this->scheduler);
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
