#include "supervisor.h"

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
    inline_sp_manager(nullptr) {
}

Supervisor::~Supervisor() {
    delete this->sm;
    // delete self_play_managers XXX
}

void Supervisor::createInline(const SelfPlayConfig* config) {
    K273::l_verbose("Supervisor::createInline()");
    this->inline_sp_manager = new SelfPlayManager(this->sm,
                                                  this->transformer,
                                                  this->batch_size);

    this->inline_sp_manager->startSelfPlayers(config);
}


int Supervisor::poll(float* policies, float* final_scores, int pred_count) {
    if (this->inline_sp_manager != nullptr) {
        return this->inline_sp_manager->poll(policies, final_scores, pred_count);
    }

    ASSERT_MSG(false, "TODO XXX workers");
    return -1;
}

float* Supervisor::getBuf() const {
    if (this->inline_sp_manager != nullptr) {
        return this->inline_sp_manager->getBuf();
    }

    ASSERT_MSG(false, "TODO XXX workers");
    return nullptr;
}

std::vector <Sample*> Supervisor::getSamples() {
    if (this->inline_sp_manager != nullptr) {
        return this->inline_sp_manager->getSamples();
    }

    ASSERT_MSG(false, "TODO XXX workers");
    return std::vector <Sample*>();
}

void Supervisor::addUniqueState(const GGPLib::BaseState* bs) {
    if (this->inline_sp_manager != nullptr) {
        this->inline_sp_manager->addUniqueState(bs);
        return;
    }

    ASSERT_MSG(false, "TODO XXX workers");
}

void Supervisor::clearUniqueStates() {
    if (this->inline_sp_manager != nullptr) {
        this->inline_sp_manager->clearUniqueStates();
        return;
    }

    ASSERT_MSG(false, "TODO XXX workers");
}

