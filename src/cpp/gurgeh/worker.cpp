#include "worker.h"

#include "config.h"

#include "player/node.h"
#include "player/rollout.h"

#include <k273/util.h>
#include <k273/logging.h>
#include <k273/exception.h>

using namespace K273;
using namespace GGPLib;
using namespace PlayerMcts;

///////////////////////////////////////////////////////////////////////////////

Worker::Worker(WorkerEventQueue* event_queue, StateMachineInterface* sm,
               const Config* config, int our_role_index) :
    is_reset(false),
    new_node(nullptr),
    config(config),
    our_role_index(our_role_index),
    event_queue(event_queue),
    event_expansion(Event::EXPANSION, this),
    event_rollout(Event::ROLLOUT, this),
    sm(sm) {

    this->rollout = new DepthChargeRollout(this->sm);

    // Create basic stuff
    this->score.reserve(sm->getRoleCount());

    this->base_state = sm->newBaseState();
}

Worker::~Worker() {
    free(this->base_state);
    delete this->sm;
}

///////////////////////////////////////////////////////////////////////////////

void Worker::reset() {
    this->path.clear();

    // reset stats
    this->time_for_expansion = 0.0;
    this->time_for_rollout = 0.0;

    this->did_rollout = false;
    this->is_reset = true;
}

///////////////////////////////////////////////////////////////////////////////

double Worker::normaliseScore(int score0, int score1, int lead_role_index) {
    if (!this->config->fixed_sum_game) {
        double win_score = 0.0;
        if (lead_role_index == 0) {
            if (score0 > score1) {
                win_score += 0.5;
            } else if (score0 == score1) {
                win_score += 0.25;
            }
        } else {
            if (score1 > score0) {
                win_score += 0.5;
            } else if (score0 == score1) {
                win_score += 0.25;
            }
        }

        double s = (lead_role_index == 0 ? score0 : score1) / 200.0;
        return win_score + s;
    }

    if (lead_role_index == 0) {
        return score0 / 100.0;
    } else {
        return score1 / 100.0;
    }
}

Node* Worker::createNode(const BaseState* bs) {
    const int role_count = this->sm->getRoleCount();

    // simply create
    Node* node = Node::create(role_count,
                              this->our_role_index,
                              this->config->initial_ucb_constant,
                              bs,
                              this->sm);

    if (node->is_finalised) {
        for (int ii=0; ii<role_count; ii++) {
            const int score = this->sm->getGoalValue(ii);
            if (role_count == 2) {
                const double s = this->normaliseScore(this->sm->getGoalValue(0),
                                                      this->sm->getGoalValue(1),
                                                      ii);
                node->setScore(ii, s);

            } else {
                node->setScore(ii, score / 100.0);
            }
        }
    }

    return node;
}

bool Worker::expandNode() {
    bool do_rollout = true;

    const Path::Element* last = this->path.getLast();

    this->sm->updateBases(last->node->getBaseState());
    this->sm->nextState(&last->selection->move, this->base_state);

    // the basic create node
    this->new_node = this->createNode(this->base_state);

    if (this->new_node->is_finalised) {
        this->score.clear();
        for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
            this->score[ii] = this->new_node->getScore(ii);
        }

        do_rollout = false;
    }

    return do_rollout;
}

void Worker::doRollout(const Node* node) {
    const int role_count = this->sm->getRoleCount();

    this->rollout->doRollout(node->getBaseState(), 0);

    this->score.clear();
    for (int ii=0; ii<role_count; ii++) {
        if (role_count == 2) {
            this->score[ii] = this->normaliseScore(this->rollout->getScore(0),
                                                   this->rollout->getScore(1),
                                                   ii);
        } else {
            this->score[ii] = this->rollout->getScore(ii) / 100.0;
        }
    }
}

void Worker::doWork() {

    try {
        double start_time = get_rtc_relative_time();

        ASSERT (this->is_reset);
        this->is_reset = false;

        bool do_rollout = this->expandNode();

        this->time_for_expansion = get_rtc_relative_time() - start_time;

        // let main thread know expansion done, and acts as a memory barrier
        this->event_queue->push(&this->event_expansion);

        // do rollout?
        double rollout_start_time = get_rtc_relative_time();
        if (do_rollout) {
            this->doRollout(this->new_node);
            this->did_rollout = true;
        }

        this->time_for_rollout = get_rtc_relative_time() - rollout_start_time;

        // ok done.  Must be called before done_queue, or these is potential race condition
        // (although not likely to ever happen).
        this->thread_self->done();

        // let main thread know we are done, and acts as a memory barrier
        this->event_queue->push(&this->event_rollout);
        return;

    } catch (const K273::Exception &exc) {
        K273::l_critical("In worker %p, Exception: %s", this, exc.getMessage().c_str());

    } catch (...) {
        K273::l_critical("In worker %p, Unknown exception caught.", this);
    }

    // XXX stop running thread here..
    ASSERT_MSG (false, "Worker needs to elegantly stop here");
}
