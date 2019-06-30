// local includes
#include "draughts_desc.h"

// k273 includes
#include <k273/logging.h>
#include <k273/strutils.h>
#include <k273/exception.h>


using namespace InternationalDraughts;


std::string Square::reprSquare() const {
    if (this->isEmpty()) {
        return "empty";
    }

    std::string res;
    if (this->data & WHITE_MAN) {
        res = "White man ";

    } else if (this->data & BLACK_MAN) {
        res = "Black man ";

    } else if (this->data & WHITE_KING) {
        res = "White king ";

    } else if (this->data & BLACK_KING) {
        res = "Black king ";

    } else {
        ASSERT_MSG(false, K273::fmtString("Impossible value: %d", this->data));
    }

    if (this->isCaptured()) {
        ASSERT(!this->isLast());

        res += "(captured) ";
    }

    if (this->isLast()) {
        ASSERT(!this->isCaptured());
        res += "(last) ";
    }

    return res;
}

std::string Square::reprMetaSquare() const {

    ASSERT(this->metaIntegrity());

    std::string res;
    if (this->whosTurn() == Role::White) {
        res = "White to play";

    } else {
        res = "Black to play";
    }

    if (this->interimStatus()) {
        res = " (interim move)";
    }

    return res;
}

///////////////////////////////////////////////////////////////////////////////

Description::Description(int board_size) :
    board_size(board_size) {

    // calls to generated code:

    if (board_size == 10) {
        this->initBoard_10x10();

    } else if (board_size == 8) {
        this->initBoard_8x8();

    } else {
        ASSERT_MSG(false, "board size not supported");
    }
}

Description::~Description() {
    K273::l_warning("In Description::~Description()");
}

///////////////////////////////////////////////////////////////////////////////

bool Description::isPromotionLine(Role role, Position pos) const {
    if (role == Role::White) {
        return this->white_promotion_line[pos - 1];
    } else {
        return this->black_promotion_line[pos - 1];
    }
}

const VectorDDI& Description::getDiagonalsForPosition(Role role,
                                                      Position pos,
                                                      Piece what) const {

    int index = role == Role::White ? 0 : 1;
    index = index * 2 + (what == Piece::Man ? 0 : 1);
    index = index * this->num_positions + (pos - 1);

    ASSERT(index >= 0 && index < (int) this->diagonal_data.size());
    return this->diagonal_data[index];
}

const ReverseLegalLookup* Description::getReverseLookup(const GGPLib::JointMove* move) const {
    // one side needs to be noop
    if (move->get(0) == this->white_noop) {
        ASSERT(move->get(1) >= 0 && move->get(1) < (int) this->reverse_legal_lookup_black.size());
        return &this->reverse_legal_lookup_black[move->get(1)];

    } else {
        ASSERT(move->get(1) == this->black_noop);
        ASSERT(move->get(0) >= 0 && move->get(0) < (int) this->reverse_legal_lookup_white.size());
        return &this->reverse_legal_lookup_white[move->get(0)];
    }
}

void Description::setInitialState(GGPLib::BaseState* bs) const {
    // only style loop, so like enumerate
    for (std::size_t ii=0; ii<this->initial_state.size(); ++ii) {
        ASSERT((int) ii < bs->size);

        bs->set(ii, this->initial_state[ii]);
    }
}

const char* Description::legalToMove(int role_index, Legal legal) const {
    ASSERT(legal >= 0);
    if (role_index == 0) {
        ASSERT(legal < (int) this->white_legal_moves.size());
        return this->white_legal_moves[legal];
    } else {
        ASSERT(legal < (int) this->black_legal_moves.size());
        return this->black_legal_moves[legal];
    }
}

int Description::legalsSize(Role role) const {
    if (role == Role::White) {
        return this->white_legal_moves.size();
    } else {
        return this->black_legal_moves.size();
    }
}
