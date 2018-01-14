#include "supervisor.h"

#include "sample.h"
#include "selfplay.h"
#include "scheduler.h"
#include "selfplaymanager.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/logging.h>
#include <k273/exception.h>

using namespace GGPZero;

SelfPlayManager::SelfPlayManager(GGPLib::StateMachineInterface* sm,
                                 const GdlBasesTransformer* transformer,
                                 int batch_size) :
    sm(sm->dupe()),
    transformer(transformer),
    batch_size(batch_size) {
    this->scheduler = new NetworkScheduler(this->sm->dupe(),
                                           this->transformer,
                                           this->batch_size);
}

SelfPlayManager::~SelfPlayManager() {
    delete this->sm;
    delete this->scheduler;
}

///////////////////////////////////////////////////////////////////////////////

void SelfPlayManager::addSample(Sample* sample) {
    this->samples.push_back(sample);
}

///////////////////////////////////////////////////////////////////////////////

void SelfPlayManager::startSelfPlayers(const SelfPlayConfig* config) {
    K273::l_info("SelfPlayManager::startSelfPlayers - starting %d players", this->batch_size);

    this->scheduler->createMainLoop();

    // create a bunch of self plays
    for (int ii=0; ii<this->batch_size; ii++) {
        SelfPlay* sp = new SelfPlay(this->scheduler, config,
                                    this, this->sm->getInitialState());
        this->self_plays.push_back(sp);

        auto f = [sp]() {
            sp->playGamesForever();
        };

        this->scheduler->addRunnable(f);
    }
}

std::vector <Sample*> SelfPlayManager::getSamples() {
    // move semantics? XXX
    std::vector <Sample*> result = this->samples;
    this->samples.clear();

    return result;
}

int SelfPlayManager::poll(float* policies, float* final_scores, int pred_count) {
    return this->scheduler->poll(policies, final_scores, pred_count);
}

float* SelfPlayManager::getBuf() const {
    return this->scheduler->getBuf();
}

void SelfPlayManager::addUniqueState(const GGPLib::BaseState* bs) {
    GGPLib::BaseState::HashSet::const_iterator it = this->unique_states.find(bs);
    if (it != this->unique_states.end()) {
        return;
    }

    // create a new basestate...
    GGPLib::BaseState* new_bs = this->sm->newBaseState();
    new_bs->assign(bs);
    this->states_allocated.push_back(new_bs);

    this->unique_states.insert(new_bs);
}


void SelfPlayManager::clearUniqueStates() {
    for (GGPLib::BaseState* bs : this->states_allocated) {
        ::free(bs);
    }

    this->unique_states.clear();
    this->states_allocated.clear();
}

