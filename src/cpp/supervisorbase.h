#pragma once

#include "bases.h"
#include "puctnode.h"
#include "greenlet/greenlet.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/exception.h>

#include <deque>

namespace GGPZero {

    // forwards
    class PuctEvaluator;

    class SupervisorBase {
    public:
        SupervisorBase(GGPLib::StateMachineInterface* sm,
                       GdlBasesTransformer* transformer, int batch_size) :
            sm(sm->dupe()),
            transformer(transformer),
            batch_size(batch_size),
            master(nullptr) {
        }


        virtual ~SupervisorBase() {
            delete this->sm;
        }

    public:
        void dumpNode(const PuctNode* node, const PuctNodeChild* highlight, const std::string& indent) {
            PuctNode::dumpNode(node, highlight, indent, this->sm);
        }

        int getRoleCount() const {
            return this->sm->getRoleCount();
        }

        virtual PuctNode* expandChild(PuctEvaluator* pe, const PuctNode* parent, const PuctNodeChild* child) = 0;
        virtual PuctNode* createNode(PuctEvaluator* pe, const GGPLib::BaseState* bs) = 0;
        virtual void finish() = 0;

    protected:
        GGPLib::StateMachineInterface* sm;
        GdlBasesTransformer* transformer;
        const unsigned int batch_size;

        std::deque <greenlet*> runnables;
        greenlet_t* master;
    };
}

