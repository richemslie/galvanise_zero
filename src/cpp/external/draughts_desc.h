#pragma once

// local includes
#include "draughts_desc.h"
#include "common.h"

// std includes
#include <string>
#include <vector>
#include <cstdint>

// ggplib includes
#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>


///////////////////////////////////////////////////////////////////////////////

namespace InternationalDraughts {

    enum class Piece : int {Man = 0, King = 1};
    enum class Role : int {White = 0, Black = 1};

    // these are not strongly type; using for cosmetic reasons.
    using Legal = int;
    using Direction = int;

    const Direction NE = 0;
    const Direction NW = 1;
    const Direction SE = 2;
    const Direction SW = 3;

    // note: positions are counted from 1, it is *not* an index
    using Position = int;

    ///////////////////////////////////////////////////////////////////////////////

    class Square {

    public:
        Square(uint8_t value = 0) :
            data(value) {
        }

    private:

        constexpr static uint8_t WHITE_MAN = 1 << 0;
        constexpr static uint8_t WHITE_KING = 1 << 1;

        constexpr static uint8_t BLACK_MAN = 1 << 2;
        constexpr static uint8_t BLACK_KING = 1 << 3;

        constexpr static uint8_t LAST = 1 << 4;
        constexpr static uint8_t CAPTURED = 1 << 5;


        // lookup helpers

        constexpr static uint8_t WHITE_TURN = 1 << 0;
        constexpr static uint8_t BLACK_TURN = 1 << 1;
        constexpr static uint8_t INTERIM_STATUS = 1 << 2;


    public:
        void clear() {
            this->data = 0;
        }

        void set(Role role, Piece what) {
            constexpr static uint8_t lookup[2][2] = {{WHITE_MAN, WHITE_KING}, {BLACK_MAN, BLACK_KING}};
            this->data = lookup[to_underlying(role)][to_underlying(what)];
        }

        bool isEmpty() const {
            return this->data == 0;
        }

        bool isOccupied(Role role) const {
            constexpr static uint8_t lookup[2] = {WHITE_MAN + WHITE_KING, BLACK_MAN + BLACK_KING};
            uint8_t mask = lookup[to_underlying(role)];
            return mask & this->data;
        }

        Piece what() const {
            return to_enum <Piece> ((this->data & (WHITE_KING + BLACK_KING)) > 0);
        }

        bool isKing() const {
            return this->data & (WHITE_KING + BLACK_KING);
        }

        bool isWhiteKing() const {
            return this->data & WHITE_KING;
        }

        bool isBlackKing() const {
            return this->data & BLACK_KING;
        }

        bool isOpponentAndNotCaptured(Role role) const {
            constexpr static uint8_t opp_lookup[2] = {BLACK_MAN + BLACK_KING, WHITE_MAN + WHITE_KING};
            uint8_t mask = opp_lookup[to_underlying(role)];
            return (this->data & mask) && (this->data & CAPTURED) == 0;
        }

        bool isCaptured() const {
            return this->data & CAPTURED;
        }

        void setCapture() {
            this->data |= CAPTURED;
        }

        void unSetCapture() {
            this->data &= ~CAPTURED;
        }

        bool isLast() const {
            return this->data & LAST;
        }

        void setLast() {
            this->data |= LAST;
        }

        void unSetLast() {
            this->data &= ~LAST;
        }

        void promote() {
            if (this->data & WHITE_MAN) {
                this->data &= ~WHITE_MAN;
                this->data |= WHITE_KING;
            } else {
                this->data &= ~BLACK_MAN;
                this->data |= BLACK_KING;
            }
        }

        Role whosTurn() const {
            return this->data & WHITE_TURN ? Role::White : Role::Black;
        }

        void switchTurn() {
            if (this->data & WHITE_TURN) {
                this->data &= ~WHITE_TURN;
                this->data |= BLACK_TURN;

            } else {
                this->data &= ~BLACK_TURN;
                this->data |= WHITE_TURN;
            }
        }

        bool interimStatus() const {
            return this->data & INTERIM_STATUS;
        }

        void setInterimStatus() {
            this->data |= INTERIM_STATUS;
        }

        void unSetInterimStatus() {
            this->data &= ~INTERIM_STATUS;
        }

        bool metaIntegrity() const {
            return this->data & WHITE_TURN || this->data & BLACK_TURN;
        }

        std::string reprSquare() const;
        std::string reprMetaSquare() const;

    private:
        uint8_t data;
    };


    ///////////////////////////////////////////////////////////////////////////////

    struct ReverseLegalLookup {
        // just to resize vector - it is generated code anyways
        ReverseLegalLookup() {
        }

        ReverseLegalLookup(Role role,
                           Piece what,
                           Position from_pos,
                           Position to_pos,
                           Direction direction) :
            role(role),
            what(what),
            from_pos(from_pos),
            to_pos(to_pos),
            direction(direction) {
        }

        Role role;
        Piece what;
        Position from_pos;
        Position to_pos;
        Direction direction;
    };

    struct DiagonalInfo {
        DiagonalInfo(Position pos, Legal legal) :
            pos(pos),
            legal(legal) {
        }

        Position pos;
        Legal legal;
    };

    struct DiagonalDirectionInfo {
        Direction direction;
        std::vector <DiagonalInfo> diagonals;
    };

    using VectorDDI = std::vector <DiagonalDirectionInfo*>;
    using VectorDDI_iter = std::vector <DiagonalDirectionInfo*>::const_iterator;

    class Description {
    public:
        Description(int board_size);
        ~Description();

    private:
        // generated code
        void initBoard_8x8();
        void initBoard_10x10();

    public:
        bool getBoadSize() const {
            return this->board_size;
        }

        int getNumPositions() const {
            return this->num_positions;
        }

        bool isPromotionLine(Role role, Position pos) const;

        const VectorDDI& getDiagonalsForPosition(Role role, Position pos, Piece what) const;

        const ReverseLegalLookup* getReverseLookup(const GGPLib::JointMove* move) const;

        Legal getNoopLegal(Role role) const {
            if (role == Role::White) {
                return this->white_noop;
            } else {
                return this->black_noop;
            }
        }

        void setInitialState(GGPLib::BaseState* bs) const;
        const char* legalToMove(int role_index, Legal legal) const;

        int legalsSize(Role role) const;

    private:

        const bool board_size;

        // all below are populated by generated code:
        int num_positions;

        // for each role/pos/what combination
        // indexed by (role * 2 + what) * num_positions + pos
        std::vector <VectorDDI> diagonal_data;

        // XXX better to index these with Role::White == 0... if speed is an issue
        Legal white_noop;
        Legal black_noop;

        std::vector <ReverseLegalLookup> reverse_legal_lookup_white;
        std::vector <ReverseLegalLookup> reverse_legal_lookup_black;

        std::vector <bool> white_promotion_line;
        std::vector <bool> black_promotion_line;

        std::vector <const char*> white_legal_moves;
        std::vector <const char*> black_legal_moves;

        std::vector <bool> initial_state;
    };
}
