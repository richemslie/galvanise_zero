#pragma once

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/exception.h>

#include <deque>
#include <string>


// XXX split up between python side and c++ interface.

namespace GGPZero {

    // forwards
    class PuctNode;
    class PuctEvaluator;
    class PuctNodeChild;
    class GdlBasesTransformer;


    class SupervisorBase {
    public:
        SupervisorBase(GGPLib::StateMachineInterface* sm) :
            sm(sm->dupe()) {
        }

        virtual ~SupervisorBase() {
            delete this->sm;
        }

    public:
        void dumpNode(const PuctNode* node, const PuctNodeChild* highlight, const std::string& indent);

        int getRoleCount() const {
            return this->sm->getRoleCount();
        }

        virtual PuctNode* expandChild(PuctEvaluator* pe, const PuctNode* parent,
                                      const PuctNodeChild* child) = 0;

        virtual PuctNode* createNode(PuctEvaluator* pe, const GGPLib::BaseState* bs) = 0;

    protected:
        GGPLib::StateMachineInterface* sm;
   };
}
