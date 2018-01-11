#include "supervisor.h"

#include "bases.h"
#include "puct/node.h"
#include "puct/evaluator.h"

using namespace GGPZero;

struct Conf {
    // these are indexes into policy array
    int role0_start_index;
    int role1_start_index;
};

Supervisor::Supervisor(GGPLib::StateMachineInterface* sm,
                       GdlBasesTransformer* transformer,
                       int batch_size) :
    SupervisorBase(sm),
    transformer(transformer),
    batch_size(batch_size),
    basestate_expand_node(nullptr),
    running(false),
    num_samples(false),
    predictions_ready(false),
    predictions_in_progress(false) {
    this->basestate_expand_node = this->sm->newBaseState();
}

Supervisor::~Supervisor() {
    //XXX
}

PuctNode* Supervisor::createNode(PuctEvaluator* pe, const GGPLib::BaseState* bs) {
    // update the statemachine
    this->sm->updateBases(bs);

    const int role_count = this->sm->getRoleCount();
    PuctNode* new_node = PuctNode::create(role_count, bs, this->sm);

    if (new_node->is_finalised) {
        for (int ii=0; ii<role_count; ii++) {
            int score = this->sm->getGoalValue(ii);
            new_node->setFinalScore(ii, score / 100.0);
        }

        return new_node;
    }

    // Well, we need to:
    //  * convert it to channels
    //  * save the info
    //  * kansas is going bye bye
    this->current_ctx->requestors.push_back(pe);
    this->transformer->toChannels(bs,
                                  std::vector <GGPLib::BaseState*>(),
                                  this->current_ctx->channel_buffer + this->current_ctx->buffer_next_index);

    float* prediction_array = nullptr;  // ZZZ this->master->switch_to();

    // policy shared between roles... XXX
    //if (new_node->lead_role_index) {
    //    prediction_array += role1_start_index;
    //}


    // Update children in new_node with prediction
    float total_prediction = 0.0f;

    for (int ii=0; ii<new_node->num_children; ii++) {
        PuctNodeChild* c = new_node->getNodeChild(this->sm->getRoleCount(), ii);
        c->policy_prob = *(prediction_array + c->move.get(new_node->lead_role_index));
        total_prediction += c->policy_prob;
    }

    if (total_prediction > std::numeric_limits<float>::min()) {
        for (int ii=0; ii<new_node->num_children; ii++) {
            PuctNodeChild* c = new_node->getNodeChild(this->sm->getRoleCount(), ii);
            c->policy_prob /= total_prediction;
        }

    } else {
        // uniform - if we didn't have valid predictions... (unlikely)

        for (int ii=0; ii<new_node->num_children; ii++) {
            PuctNodeChild* c = new_node->getNodeChild(this->sm->getRoleCount(), ii);
            c->policy_prob = 1.0 / new_node->num_children;
        }
    }

    return new_node;
}



PuctNode* Supervisor::expandChild(PuctEvaluator* pe, const PuctNode* parent, const PuctNodeChild* child) {
    // update the statemachine
    this->sm->updateBases(parent->getBaseState());
    this->sm->nextState(&child->move, this->basestate_expand_node);

    // create node
    return this->createNode(pe, this->basestate_expand_node);
}
