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

#include <unistd.h>

using namespace GGPZero;

Supervisor::Supervisor(GGPLib::StateMachineInterface* sm,
                       const GdlBasesTransformer* transformer,
                       int batch_size) :
    sm(sm->dupe()),
    transformer(transformer),
    batch_size(batch_size),
    slow_poll_counter(0),
    inline_sp_manager(nullptr),
    in_progress_manager(nullptr),
    in_progress_worker(nullptr),
    unique_states(sm->dupe(), transformer, 200) {
}

Supervisor::~Supervisor() {
    delete this->sm;
    // XXX (for now kill -9)
    // * stop and join workers
    // * delete workers
    // * delete self_play_managers?
}


void Supervisor::slowPoll(SelfPlayManager* manager) {
    this->slow_poll_counter++;
    if (this->slow_poll_counter < 1024) {
        return;
    }

    this->slow_poll_counter = 0;

    manager->reportAndResetStats();

    std::vector<Sample* >& other = manager->getSamples();
    if (other.empty()) {
        return;
    }

    if (this->samples.empty()) {
        // fast move semantics
        this->samples = std::move(other);

    } else {
        this->samples.insert(this->samples.end(), other.begin(), other.end());
        other.clear();
    }
}

void Supervisor::createInline(const SelfPlayConfig* config) {
    K273::l_verbose("Supervisor::createInline()");
    this->inline_sp_manager = new SelfPlayManager(this->sm,
                                                  this->transformer,
                                                  this->batch_size,
                                                  &this->unique_states,
                                                  "inline");

    this->inline_sp_manager->startSelfPlayers(config);
}

void Supervisor::createWorkers(const SelfPlayConfig* config) {
    SelfPlayWorker* spw = new SelfPlayWorker(new SelfPlayManager(this->sm,
                                                                 this->transformer,
                                                                 this->batch_size,
                                                                 &this->unique_states,
                                                                 "sp0"),
                                             new SelfPlayManager(this->sm,
                                                                 this->transformer,
                                                                 this->batch_size,
                                                                 &this->unique_states,
                                                                 "sp1"),
                                             config);
    this->self_play_workers.push_back(spw);

    K273::WorkerThread* worker_thread = new K273::WorkerThread(spw);
    worker_thread->spawn();
    worker_thread->startPolling();

    worker_thread->promptWorker();
}

const ReadyEvent* Supervisor::poll(int predict_count, std::vector <float*>& data) {
    ASSERT(0 <= predict_count && predict_count <= this->batch_size);

    auto populateEvent = [&] (SelfPlayManager* manager) {
        PredictDoneEvent* event = manager->getPredictDoneEvent();

        // copy stuff to ready
        event->pred_count = predict_count;
        if (event->pred_count > 0) {
            int index = 0;
            event->policies.resize(this->transformer->getNumberPolicies());
            for (int ii=0; ii<this->transformer->getNumberPolicies(); ii++) {
                memcpy(event->policies[ii], data[index++],
                       sizeof(float) * predict_count * this->transformer->getPolicySize(ii));
            }

            memcpy(event->final_scores, data[index++],
                   sizeof(float) * predict_count * this->transformer->getNumberRewards());
        }
    };

    if (this->inline_sp_manager != nullptr) {
        populateEvent(this->inline_sp_manager);
        this->inline_sp_manager->poll();
        this->slowPoll(this->inline_sp_manager);
        return this->inline_sp_manager->getReadyEvent();
    }

    if (predict_count) {
        ASSERT(this->in_progress_worker != nullptr && this->in_progress_manager != nullptr);
        populateEvent(this->in_progress_manager);

        this->slowPoll(this->in_progress_manager);
        this->in_progress_worker->push(this->in_progress_manager);

        // wake up if need to...
        this->in_progress_worker->getThread()->promptWorker();

        this->in_progress_worker = nullptr;
        this->in_progress_manager = nullptr;
    }

    ASSERT(this->in_progress_worker == nullptr && this->in_progress_manager == nullptr);

    // workers run forever... kind of the whole point
    while (true) {

        for (auto worker : this->self_play_workers) {
            this->in_progress_manager = worker->pull();
            if (this->in_progress_manager != nullptr) {
                this->in_progress_worker = worker;
                break;
            }
        }

        if (this->in_progress_worker != nullptr) {
            break;
        }

        //::usleep(100);

        //for (auto worker : this->self_play_workers) {
        //    worker->getThread()->promptWorker();
        //}
    }

    return this->in_progress_manager->getReadyEvent();
}

std::vector <Sample*> Supervisor::getSamples() {
    std::vector <Sample*> result = std::move(this->samples);
    return result;
}

void Supervisor::addUniqueState(const GGPLib::BaseState* bs) {
    this->unique_states.add(bs);
}

void Supervisor::clearUniqueStates() {
    this->unique_states.clear();
}

SelfPlayWorker::SelfPlayWorker(SelfPlayManager* man0, SelfPlayManager* man1,
                               const SelfPlayConfig* config) :
    enter_first_time(true),
    config(config),
    man0(man0),
    man1(man1) {
}

SelfPlayWorker::~SelfPlayWorker() {
    delete this->man0;
    delete this->man1;
}

void SelfPlayWorker::doWork() {
    try {
        // prime the managers... after which they will run forever
        if (this->enter_first_time) {
            K273::l_warning("priming managers");
            this->enter_first_time = false;

            this->man0->startSelfPlayers(this->config);
            this->man0->getReadyEvent()->buf_count = 0;
            this->man0->poll();
            this->outbound_queue.push(this->man0);

            this->man1->startSelfPlayers(this->config);
            this->man1->getReadyEvent()->buf_count = 0;
            this->man1->poll();
            this->outbound_queue.push(this->man1);
        }

        int did_nothing = 0;
        while (true) {
            did_nothing++;
            while (!this->inbound_queue.empty()) {
                did_nothing = 0;

                SelfPlayManager* manager = this->inbound_queue.pop();
                manager->poll();
                this->thread_self->done();
                this->outbound_queue.push(manager);
            }

            if (did_nothing > 1000) {
                did_nothing = 0;
                // sleep for 1 millisecond
                usleep(5000);
            }
        }

        this->thread_self->done();
        return;

    } catch (const K273::Exception &exc) {
        K273::l_critical("In worker %p, Exception: %s", this, exc.getMessage().c_str());

    } catch (...) {
        K273::l_critical("In worker %p, Unknown exception caught.", this);
    }

    // XXX stop running thread here..
    ASSERT_MSG(false, "Worker needs to elegantly stop here");
}


