#pragma once

// local includes
#include "draughts_desc.h"
#include "draughts_board.h"

// ggplib includes
#include <statemachine/jointmove.h>
#include <statemachine/legalstate.h>
#include <statemachine/basestate.h>
#include <statemachine/roleinfo.h>
#include <statemachine/statemachine.h>


namespace InternationalDraughts {

    class SM : public GGPLib::StateMachineInterface {

    public:
        SM(Board* board, const Description* board_desc);
        virtual ~SM();

    public:
        // SM interface:

        GGPLib::StateMachineInterface* dupe() const;
        GGPLib::BaseState* newBaseState() const;
        const GGPLib::BaseState* getCurrentState() const;

        void setInitialState(const GGPLib::BaseState* bs);
        const GGPLib::BaseState* getInitialState() const;

        GGPLib::LegalState* getLegalState(int role_index);

        void updateBases(const GGPLib::BaseState* bs);
        const char* legalToMove(int role_index, int choice) const;
        GGPLib::JointMove* getJointMove();

        bool isTerminal() const;
        void nextState(const GGPLib::JointMove* move, GGPLib::BaseState* bs);

        int getGoalValue(int role_index);
        void reset();

        int getRoleCount() const;

    private:
        Board* board;
        const Description* board_desc;

        GGPLib::BaseState* current_state;
        GGPLib::BaseState* initial_state;
    };
}
