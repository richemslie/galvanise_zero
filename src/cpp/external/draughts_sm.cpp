// local includes
#include "draughts_sm.h"
#include "draughts_desc.h"
#include "draughts_board.h"


// k273 includes
#include <k273/logging.h>
#include <k273/exception.h>

// ggplib includes
#include <statemachine/jointmove.h>
#include <statemachine/legalstate.h>
#include <statemachine/basestate.h>


using namespace InternationalDraughts;


///////////////////////////////////////////////////////////////////////////////

SM::SM(Board* b, const Description* board_desc) :
    board(b),
    board_desc(board_desc) {

    this->current_state = this->newBaseState();
    this->board->setSquares(this->current_state);

    this->initial_state = this->newBaseState();
    this->board_desc->setInitialState(this->initial_state);

    // sm.reset() will be called in ggplib.interface.StateMachine()
}

SM::~SM() {
    K273::l_warning("In SM::~SM()");
    delete this->board;
    ::free(this->current_state);
    ::free(this->initial_state);
}

GGPLib::StateMachineInterface* SM::dupe() const {
    // create a new board / sm.  Note: it is imperative that the caller calls reset().
    return new SM(new Board(this->board_desc,
                            this->board->breakthrough_mode,
                            this->board->killer_mode),
                  this->board_desc);
}

void SM::reset() {
    this->updateBases(this->getInitialState());
}

void SM::updateBases(const GGPLib::BaseState* bs) {
    this->current_state->assign(bs);
    this->board->setSquares(this->current_state);
    this->board->updateLegalsChoices();
}

GGPLib::BaseState* SM::newBaseState() const {
    int num_bases = 8 * (this->board_desc->getNumPositions()) + 3;

    void* mem = ::malloc(GGPLib::BaseState::mallocSize(num_bases));
    GGPLib::BaseState* bs = static_cast <GGPLib::BaseState*>(mem);
    bs->init(num_bases);
    return bs;
}

const GGPLib::BaseState* SM::getCurrentState() const {
    return this->current_state;
}

void SM::setInitialState(const GGPLib::BaseState* bs) {
    ASSERT_MSG(false, "not supported");
}

const GGPLib::BaseState* SM::getInitialState() const {
    return this->initial_state;
}

GGPLib::LegalState* SM::getLegalState(int role_index) {
    // would be better to go via accessor

    if (role_index == 0) {
        return this->board->white_legalstate;
    } else {
        ASSERT(role_index == 1);
        return this->board->black_legalstate;
    }
}

GGPLib::JointMove* SM::getJointMove() {
    // zero array size malloc
    void* mem = malloc(GGPLib::JointMove::mallocSize(this->getRoleCount()));
    GGPLib::JointMove* move = static_cast <GGPLib::JointMove*>(mem);
    move->setSize(this->getRoleCount());
    return move;
}

void SM::nextState(const GGPLib::JointMove* move, GGPLib::BaseState* bs) {
    // we'll act on current state in playMove().
    // we'll assign it to bs when done, and then revert current_state.  Huge
    // hack, but will let things work for testing.

    bs->assign(this->current_state);
    this->board->setSquares(bs);

    // note: ensure this doesn't effect legal state
    // XXX to be sure should move legalstates to SM (makes senese logically)
    this->board->playMove(move);

    this->board->setSquares(this->current_state);
}

const char* SM::legalToMove(int role_index, int choice) const {
    // get from board_desc (same for both bt/non bt variants)
    return this->board_desc->legalToMove(role_index, choice);
}

bool SM::isTerminal() const {
    return this->board->done();
}

int SM::getGoalValue(int role_index) {
    // undefined, call getGoalValue() in a non terminal state at your peril
    if (!this->board->done()) {
        return -1;
    }

    if (role_index == 0) {
        return this->board->score(Role::White);
    } else {
        return this->board->score(Role::Black);
    }
}

int SM::getRoleCount() const {
    return 2;
}
