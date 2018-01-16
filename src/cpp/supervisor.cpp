#include "supervisor.h"

#include "events.h"
#include "selfplay.h"
#include "scheduler.h"
#include "selfplaymanager.h"
#include "puct/evaluator.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/logging.h>
#include <k273/exception.h>

using namespace GGPZero;

Supervisor::Supervisor(GGPLib::StateMachineInterface* sm,
                       const GdlBasesTransformer* transformer,
                       int batch_size) :
    sm(sm->dupe()),
    transformer(transformer),
    batch_size(batch_size),
    inline_sp_manager(nullptr),
    current(nullptr),
    unique_states(sm->dupe()) {
}

Supervisor::~Supervisor() {
    delete this->sm;
    // delete self_play_managers XXX
}

void Supervisor::createInline(const SelfPlayConfig* config) {
    // XXX going to have to change for threaded version
    K273::l_verbose("Supervisor::createInline()");
    this->inline_sp_manager = new SelfPlayManager(this->sm,
                                                  this->transformer,
                                                  this->batch_size,
                                                  &this->unique_states);

    this->inline_sp_manager->startSelfPlayers(config);
}


const ReadyEvent* Supervisor::poll(float* policies, float* final_scores, int pred_count) {
    ASSERT(0 <= pred_count && pred_count <= this->batch_size);

    if (this->inline_sp_manager != nullptr) {
        SelfPlayManager* manager = this->inline_sp_manager;

        PredictDoneEvent* event = manager->getPredictDoneEvent();

        // copy stuff to ready
        event->pred_count = pred_count;
        if (event->pred_count > 0) {
            memcpy(event->policies, policies,
                   sizeof(float) * pred_count * this->transformer->getPolicySize());

            memcpy(event->final_scores, final_scores,
                   sizeof(float) * pred_count * this->sm->getRoleCount());
        }

        this->inline_sp_manager->poll();
        return manager->getReadyEvent();
    }

    // XXX workers are very different
    ASSERT_MSG(false, "TODO XXX workers");
    return nullptr;
}

std::vector <Sample*> Supervisor::getSamples() {
    // XXX going to have to change for threaded version
    if (this->inline_sp_manager != nullptr) {
        this->inline_sp_manager->reportAndResetStats();
        return this->inline_sp_manager->getSamples();
    }

    ASSERT_MSG(false, "TODO XXX workers");
    return std::vector <Sample*>();
}

void Supervisor::addUniqueState(const GGPLib::BaseState* bs) {
    this->unique_states.add(bs);
}

void Supervisor::clearUniqueStates() {
    this->unique_states.clear();
}

