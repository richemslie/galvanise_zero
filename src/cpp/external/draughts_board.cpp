// local includes
#include "draughts_desc.h"
#include "draughts_board.h"

// k273 includes
#include <k273/strutils.h>
#include <k273/logging.h>
#include <k273/exception.h>

// ggplib includes
#include <statemachine/jointmove.h>
#include <statemachine/legalstate.h>
#include <statemachine/basestate.h>

// std includes
#include <string>
#include <vector>

#include <ctype.h>
#include <string.h>
#include <stdio.h>

using namespace InternationalDraughts;

constexpr static bool DEBUG = false;

struct ImmediateCaptureResult  {
    ImmediateCaptureResult() :
        legal(-1),
        next_pos(-1),
        capture_pos(-1),
        first_after(false) {
    }

    Legal legal;
    Position next_pos;
    Position capture_pos;
    bool first_after;
};


class ImmediateCaptures {
    private:
    enum class Ctx {Enter, Next, KingReturn};

    public:
    ImmediateCaptures(const Board* board,
                      Role role, Piece what,
                      const VectorDDI& ddi) :
        board(board),
        role(role),
        what(what),
        ddi(ddi) {
    }

    class iterator {
    public:
        iterator(const ImmediateCaptures& ic, VectorDDI::const_iterator iter) :
            ic(ic),
            ctx(Ctx::Enter),
            king_dd_index(0),
            ddi_iter(iter) {
        }

        bool operator!= (const iterator& other) const {
            return this->ddi_iter != other.ddi_iter;
        }

        const ImmediateCaptureResult& operator*() const {
            return this->res;
        }

        void operator++();

    private:
        const ImmediateCaptures& ic;
        Ctx ctx;
        size_t king_dd_index;
        ImmediateCaptureResult res;
        VectorDDI::const_iterator ddi_iter;
    };

    iterator begin() const {
        auto ii = iterator(*this, this->ddi.begin());
        // required to set up coroutine.
        ++ii;
        return ii;
    }

    iterator end() const {
        return iterator(*this, this->ddi.end());
    }

private:
    const Board* board;
    Role role;
    Piece what;
    const VectorDDI& ddi;
};

///////////////////////////////////////////////////////////////////////////////

void ImmediateCaptures::iterator::operator++ () {

    if (this->ctx == Ctx::Next) {
        this->ddi_iter++;
    }

    while (this->ddi_iter != this->ic.ddi.end()) {
        const auto& dd = (*ddi_iter)->diagonals;

        if (this->ic.what == Piece::Man) {

            // diagonals here is specific to what, hence if not of size() == 2, not interested.
            if (dd.size() == 2) {
                // guaranteed that have two in advance
                const DiagonalInfo& captured = dd[0];
                const DiagonalInfo& next = dd[1];

                if (this->ic.board->get(next.pos)->isEmpty() &&
                    this->ic.board->get(captured.pos)->isOpponentAndNotCaptured(this->ic.role)) {

                    // populate result
                    this->res.legal = next.legal;
                    this->res.next_pos = next.pos;
                    this->res.capture_pos = captured.pos;

                    // used for killer, always first after for man capture
                    this->res.first_after = true;

                    this->ctx = Ctx::Next;
                    return;
                }
            }

        } else {
            ASSERT(this->ic.what == Piece::King);

            // coroutine part...
            if (this->ctx == Ctx::KingReturn) {
                // increment
                this->king_dd_index++;

                // for killer rule
                this->res.first_after = false;

                goto ctx_king_return;
            }

            // first find a capture, if any
            {
                bool found_capture = false;
                this->king_dd_index = 0;
                while (this->king_dd_index < dd.size()) {
                    const DiagonalInfo& next = dd[this->king_dd_index];

                    const Square* sq = this->ic.board->get(next.pos);

                    if (!sq->isEmpty()) {
                        //printf("here7\n");
                        if (sq->isOpponentAndNotCaptured(this->ic.role)) {

                            // populate res so far...
                            this->res.first_after = true;
                            this->res.capture_pos = next.pos;

                            this->king_dd_index++;

                            found_capture = true;
                        }

                        break;
                    }

                    this->king_dd_index++;
                }

                if (!found_capture) {
                    this->ddi_iter++;
                    continue;
                }
            }

        ctx_king_return:

            if (this->king_dd_index < dd.size()) {
                const DiagonalInfo& next = dd[this->king_dd_index];
                const Square* sq = this->ic.board->get(next.pos);

                if (sq->isEmpty()) {
                    // populate rest of result
                    this->res.legal = next.legal;
                    this->res.next_pos = next.pos;

                    this->ctx = Ctx::KingReturn;
                    return;
                }
            }
        }

        // not needed, but it is clearer if we set
        this->ctx = Ctx::Next;
        this->ddi_iter++;
    }
}

///////////////////////////////////////////////////////////////////////////////

Board::Board(const Description* board_desc,
             bool breakthrough_mode,
             bool killer_mode) :
    board_desc(board_desc),
    breakthrough_mode(breakthrough_mode),
    killer_mode(killer_mode),
    squares(nullptr) {

    // DONT MAKE IT hardcoded - FFS (it is a datastructure specific to propagation, not some
    // arbirtary list structure... and we have to abuse it for now... hence it *MUST* be the size
    // of the number of legals.)
    this->white_legalstate = new GGPLib::LegalState(board_desc->legalsSize(Role::White));
    this->black_legalstate = new GGPLib::LegalState(board_desc->legalsSize(Role::Black));
}

Board::~Board() {
    K273::l_warning("In Board::~Board()");
}

///////////////////////////////////////////////////////////////////////////////
// Board interface:

void Board::setSquares(GGPLib::BaseState* bs) {
    this->squares = (Square*) bs->data;
}

///////////////////////////////////////////////////////////////////////////////
// These are all private.  Maybe I should factor them out to wrap a basestate... or something.

void Board::clearCaptures() {
    Square* sq = this->squares;
    for (int ii=0; ii<this->board_desc->getNumPositions(); ii++, sq++) {
        if (sq->isCaptured()) {
            sq->clear();
        }
    }
}

int Board::score(Role role) const {
    bool king_white = false;
    bool king_black = false;
    if (this->breakthrough_mode) {
        Square* sq = this->squares;
        for (int ii=0; ii<this->board_desc->getNumPositions(); ii++, sq++) {
            if (sq->isWhiteKing()) {
                king_white = true;
                break;
            }

            if (sq->isBlackKing()) {
                king_black = true;
                break;
            }
        }
    }

    auto finalScore = [] (bool king_me, bool king_opp,
                          const GGPLib::LegalState* legals_me,
                          const GGPLib::LegalState* legals_opp) {
        // only for breakthrough_mode
        if (king_me) {
            return 100;
        }

        // only for breakthrough_mode
        if (king_opp) {
            return 0;
        }

        if (legals_opp->getCount() == 0) {
            return 100;
        }

        if (legals_me->getCount() == 0) {
            return 0;
        }

        // ZZZ don't need to do anything for 40_rule_count
        // must be a draw then
        return 50;
    };

    if (role == Role::White) {
        return finalScore(king_white, king_black, this->white_legalstate, this->black_legalstate);

    } else {
        return finalScore(king_black, king_white, this->black_legalstate, this->white_legalstate);
    }
}

bool Board::done() const {
    if (this->white_legalstate->getCount() == 0 || this->black_legalstate->getCount() == 0) {
        return true;
    }

    if (this->breakthrough_mode) {
        Square* sq = this->squares;
        for (int ii=0; ii<this->board_desc->getNumPositions(); ii++, sq++) {
            if (sq->isKing()) {
                return true;
            }
        }

    } else {
        // draw conditions XXX (done outside statemachine XXX)

        // ZZZ
        //if (this->getCounter() == this->board_desc->40_rule_count) {
        //    return true;
       // }
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////

void Board::pushLegal(Role role, Legal legal) {
    if (role == Role::White) {
        this->white_legalstate->insert(legal);
    } else {
        this->black_legalstate->insert(legal);
    }

    if (DEBUG) {
        void* mem = malloc(GGPLib::JointMove::mallocSize(2));
        GGPLib::JointMove* move = static_cast <GGPLib::JointMove*>(mem);
        move->setSize(2);

        if (role == Role::White) {
            move->set(0, legal);
            move->set(1, 0);
        } else {
            move->set(1, legal);
            move->set(0, 0);
        }

        const ReverseLegalLookup* rll = this->board_desc->getReverseLookup(move);
        K273::l_verbose("pushLegal (%s, %s) %d -> %d",
                        rll->role == Role::White ? "white" : "black",
                        rll->what == Piece::Man ? "man" : "king",
                        rll->from_pos, rll->to_pos);

        free(mem);
    }
}

void Board::pushLegalNoop(Role role) {
    Legal l = this->board_desc->getNoopLegal(role);
    if (role == Role::White) {
        this->white_legalstate->insert(l);

    } else {
        this->black_legalstate->insert(l);
    }
}

void Board::clearLegals(Role role) {
    if (role == Role::White) {
        this->white_legalstate->clear();

    } else {
        this->black_legalstate->clear();
    }
}

///////////////////////////////////////////////////////////////////////////////

void Board::nonCaptureLegals(Role role, Position pos, Piece what) {
    // constraint: must not be called if there are captures to be made

    auto& all_diagonals = this->board_desc->getDiagonalsForPosition(role, pos, what);

    if (what == Piece::Man) {
        for (DiagonalDirectionInfo* ddi : all_diagonals) {
            const auto& dd = ddi->diagonals;
            if (dd[0].legal > 0) {
                const Square* sq = this->get(dd[0].pos);
                if (sq->isEmpty()) {
                    this->pushLegal(role, dd[0].legal);
                }
            }
        }

    } else if (what == Piece::King) {

        for (DiagonalDirectionInfo* ddi : all_diagonals) {
            for (auto& d : ddi->diagonals) {

                const Square* sq = this->get(d.pos);
                if (!sq->isEmpty()) {
                    break;
                }

                this->pushLegal(role, d.legal);
            }
        }
    }
}

// dupe code ahead, just for efficiency sake: XXX actually lets not do that:)  Or create a lambda?

int Board::maximalLegals(Role role, Position pos, Piece what, int best_mc) {
    Square* pt_pos_sq = this->get(pos);
    Square restore_sq = *pt_pos_sq;
    pt_pos_sq->clear();

    ImmediateCaptures ici(this, role, what,
                          this->board_desc->getDiagonalsForPosition(role, pos, what));

    // find if there are immediate captures
    for (auto& cap : ici) {
        Square* pt_capture = this->get(cap.capture_pos);
        pt_capture->setCapture();

        int mc = this->maximalCaptures(role, cap.next_pos, what);

        if (DEBUG) {
            K273::l_verbose("Board::maximalLegals(), more %d/%d...",
                            cap.next_pos, mc);
        }

        pt_capture->unSetCapture();

        if (this->killer_mode && mc == 0) {
            // we are at last point, check capture_pos was king
            if (!cap.first_after) {
                if (pt_capture->isKing()) {
                    continue;
                }
            }
        }

        int count = mc + 1;
        if (count > best_mc) {
            best_mc = count;
            this->clearLegals(role);
        }

        if (count == best_mc) {
            this->pushLegal(role, cap.legal);
        }

    }

    // return to previous
    *pt_pos_sq = restore_sq;

    if (DEBUG && best_mc > 0) {
        K273::l_verbose("maximalLegals best_mc %d, %d %d %d", best_mc,
                        to_underlying(role), pos, to_underlying(what));
    }

    return best_mc;
}

int Board::maximalCaptures(Role role, Position pos, Piece what) {
    int best_mc = 0;

    ImmediateCaptures ici(this, role, what,
                          this->board_desc->getDiagonalsForPosition(role, pos, what));

    // find if there are immediate captures
    for (auto& cap : ici) {
        Square* pt_capture = this->get(cap.capture_pos);
        pt_capture->setCapture();
        int mc = this->maximalCaptures(role, cap.next_pos, what);
        pt_capture->unSetCapture();

        if (this->killer_mode && mc == 0) {
            // we are at last point, check capture_pos was king
            if (pt_capture->isKing()) {
                if (!cap.first_after) {
                    continue;
                }
            }
        }

        best_mc = std::max(best_mc, mc + 1);
    }

    return best_mc;
}

void Board::updateLegalsChoices() {
    // updates the legals,
    this->clearLegals(Role::White);
    this->clearLegals(Role::Black);

    // who's turn is it?
    const Square* meta = this->getMeta();
    Role role = meta->whosTurn();

    // add noop for opponent
    if (role == Role::White) {
        this->pushLegalNoop(Role::Black);
    } else {
        this->pushLegalNoop(Role::White);
    }

    // are we in interim position?
    if (meta->interimStatus()) {
        const Square* sq = this->squares;
        for (int ii=0; ii<this->board_desc->getNumPositions(); ii++, sq++) {
            if (sq->isLast()) {
                Position from_pos = ii + 1;
                this->maximalLegals(role, from_pos, sq->what(), 0);
                return;
            }
        }

        ASSERT_MSG(false, "Cant get here - board corrupt?");
    }

    // determine if any captures can be made
    {
        int best_mc = 0;
        const Square* sq = this->squares;
        for (int ii=0; ii<this->board_desc->getNumPositions(); ii++, sq++) {
            if (sq->isOccupied(role)) {
                Position from_pos = ii + 1;
                best_mc = this->maximalLegals(role, from_pos, sq->what(), best_mc);
            }
        }

        if (DEBUG) {
            K273::l_verbose("best_mc: %d", best_mc);
        }

        if (best_mc > 0) {
            return;
        }
    }

    // non-capture moves
    {
        const Square* sq = this->squares;
        for (int ii=0; ii<this->board_desc->getNumPositions(); ii++, sq++) {
            if (sq->isOccupied(role)) {
                Position from_pos = ii + 1;
                this->nonCaptureLegals(role, from_pos, sq->what());
            }
        }
    }
}

int Board::capturedFrom(Role role, Position from_pos, Position to_pos, Piece what, Direction direction) {
    for (auto& info : this->board_desc->getDiagonalsForPosition(role, from_pos, what)) {
        if (info->direction == direction) {
            for (auto& di : info->diagonals) {

                // all points must be empty, except we allow one and only one opponent piece
                // (the captured_pos)

                if (di.pos == to_pos) {
                    break;
                }

                Square* sq = this->get(di.pos);
                if (!sq->isEmpty()) {
                    // XXX assert that is opponent square?
                    // XXX or just pass all tests
                    return di.pos;
                }
            }
        }
    }

    return -1;
}

void Board::playMove(const GGPLib::JointMove* move) {
    const ReverseLegalLookup* rll = this->board_desc->getReverseLookup(move);

    if (DEBUG) {
        K273::l_verbose("Board::playMove (%s, %s) %d -> %d",
                        rll->role == Role::White ? "white" : "black",
                        rll->what == Piece::Man ? "man" : "king",
                        rll->from_pos, rll->to_pos);
    }

    // get the meta square, will always need to update
    Square* meta = this->getMeta();

    // unset interim status (doesnt matter if not set)
    meta->unSetInterimStatus();

    Square* from_sq = this->get(rll->from_pos);
    Square* to_sq = this->get(rll->to_pos);

    // move and add capture piece (if one)
    from_sq->clear();
    to_sq->set(rll->role, rll->what);

    int captured_pos = this->capturedFrom(rll->role,
                                          rll->from_pos,
                                          rll->to_pos,
                                          rll->what,
                                          rll->direction);

    if (DEBUG) {
        K273::l_verbose("playMove captured_pos %d", captured_pos);
    }

    if (captured_pos > 0) {
        Square* captured_sq = this->get(captured_pos);
        captured_sq->setCapture();

        // any more captures???
        ImmediateCaptures ici(this,
                              rll->role,
                              rll->what,
                              this->board_desc->getDiagonalsForPosition(rll->role, rll->to_pos, rll->what));

        for (auto x : ici) {
            x = x;
            to_sq->setLast();
            meta->setInterimStatus();
            return;
        }
    }

    // 40 rule - note in intermim status, we dont need to do anything
    // Square* counter_40r = this->getCounter();
    // if (rll->what == Piece::Man || captured_pos) {
    //     counter_40r->reset();
    // } else {
    //     counter_40r->incr;
    // }

    // ensure all captures are removed from board
    this->clearCaptures();

    // promote to king?
    if (rll->what == Piece::Man && this->board_desc->isPromotionLine(rll->role,
                                                                     rll->to_pos)) {
        to_sq->promote();
    }

    // and switch role...
    meta->switchTurn();
}

