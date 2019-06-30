
// local includes
#include "catch2_runner.h"

#include "draughts_sm.h"
#include "draughts_desc.h"
#include "draughts_board.h"

// k273 includes
#include <k273/logging.h>
#include <k273/strutils.h>
#include <k273/exception.h>
#include <k273/rng.h>

// 3rd party
#include <catch.hpp>
#include <fmt/format.h>
#include <fmt/printf.h>

// std includes
#include <vector>
#include <string>
#include <iostream>

#include <sys/types.h>
#include <unistd.h>


using namespace InternationalDraughts;


TEST_CASE("SM access", "[sm]") {
    setLogging();

    try {
        Description desc(10);

        Board* board = new Board(&desc);

        // side effects
        SM sm(board, &desc);
        sm.reset();

        for (int ii=0; ii<desc.getNumPositions(); ii++) {
            fmt::printf("%s -> %s\n", ii + 1, board->get(ii + 1)->reprSquare());
        }

        fmt::printf("meta -> %s\n", board->getMeta()->reprMetaSquare());

        const GGPLib::BaseState* bs = sm.getCurrentState();
        const GGPLib::BaseState* bs2 = sm.getInitialState();

        for (int ii=0; ii<desc.getNumPositions() * 8 + 3; ++ii) {
            ASSERT(bs->get(ii) == bs2->get(ii));
        }

        for (int ii=0; ii<desc.getNumPositions(); ii++) {
            fmt::printf("%s -> %s\n", ii + 1, board->get(ii + 1)->reprSquare());
        }

        fmt::printf("meta -> %s\n", board->getMeta()->reprMetaSquare());

        ASSERT(bs->get(0) == 0);
        ASSERT(bs->get(1) == 0);
        ASSERT(bs->get(2) == 1);
        ASSERT(bs->get(3) == 0);
        ASSERT(bs->get(4) == 0);
        ASSERT(bs->get(5) == 0);
        ASSERT(bs->get(6) == 0);
        ASSERT(bs->get(7) == 0);

        // white first to play
        ASSERT(bs->get(50 * 8) == 1);
        ASSERT(bs->get(50 * 8 + 1) == 0);
        ASSERT(bs->get(50 * 8 + 2) == 0);

    } catch (const K273::Exception& exc) {
        FAIL(exc.getMessage());
    }
}

TEST_CASE("Squares 1", "[sm]") {
    setLogging();

    try {
        Description desc(10);
        Board* board = new Board(&desc);

        // side effects
        SM sm(board, &desc);
        sm.reset();

        // board should be set up.
        const Square* sq = board->get(1);
        Square orig = *sq;

        fmt::printf("%s\n", sq->reprSquare());
        REQUIRE(!sq->isEmpty());

        REQUIRE(sq->isOccupied(Role::Black));
        REQUIRE(!sq->isOccupied(Role::White));

        REQUIRE(sq->what() == Piece::Man);
        REQUIRE(!sq->isKing());
        REQUIRE(!sq->isWhiteKing());
        REQUIRE(!sq->isBlackKing());

        REQUIRE(!sq->isOpponentAndNotCaptured(Role::Black));
        REQUIRE(sq->isOpponentAndNotCaptured(Role::White));

        REQUIRE(!sq->isCaptured());
        REQUIRE(!sq->isLast());


        Square* sq2 = board->get(1);
        sq2->clear();

        fmt::printf("%s\n", sq->reprSquare());
        REQUIRE(sq->isEmpty());

        REQUIRE(!sq->isOccupied(Role::Black));
        REQUIRE(!sq->isOccupied(Role::White));

        REQUIRE(!sq->isKing());
        REQUIRE(!sq->isWhiteKing());
        REQUIRE(!sq->isBlackKing());

        REQUIRE(!sq->isOpponentAndNotCaptured(Role::Black));
        REQUIRE(!sq->isOpponentAndNotCaptured(Role::White));

        REQUIRE(!sq->isCaptured());
        REQUIRE(!sq->isLast());

        sq2->set(Role::White, Piece::King);

        fmt::printf("%s\n", sq->reprSquare());
        REQUIRE(!sq->isEmpty());

        REQUIRE(!sq->isOccupied(Role::Black));
        REQUIRE(sq->isOccupied(Role::White));

        REQUIRE(sq->isKing());
        REQUIRE(sq->isWhiteKing());
        REQUIRE(!sq->isBlackKing());

        REQUIRE(sq->isOpponentAndNotCaptured(Role::Black));
        REQUIRE(!sq->isOpponentAndNotCaptured(Role::White));

        REQUIRE(!sq->isCaptured());
        REQUIRE(!sq->isLast());

        sq2->setCapture();

        fmt::printf("%s\n", sq->reprSquare());
        REQUIRE(!sq->isEmpty());

        REQUIRE(!sq->isOccupied(Role::Black));
        REQUIRE(sq->isOccupied(Role::White));

        REQUIRE(sq->isKing());
        REQUIRE(sq->isWhiteKing());
        REQUIRE(!sq->isBlackKing());

        REQUIRE(!sq->isOpponentAndNotCaptured(Role::Black));
        REQUIRE(!sq->isOpponentAndNotCaptured(Role::White));

        REQUIRE(sq->isCaptured());
        REQUIRE(!sq->isLast());

        sq2->unSetCapture();
        sq2->setLast();

        fmt::printf("%s\n", sq->reprSquare());
        REQUIRE(!sq->isEmpty());

        REQUIRE(!sq->isOccupied(Role::Black));
        REQUIRE(sq->isOccupied(Role::White));

        REQUIRE(sq->isKing());
        REQUIRE(sq->isWhiteKing());
        REQUIRE(!sq->isBlackKing());

        REQUIRE(sq->isOpponentAndNotCaptured(Role::Black));
        REQUIRE(!sq->isOpponentAndNotCaptured(Role::White));

        REQUIRE(!sq->isCaptured());
        REQUIRE(sq->isLast());

        *sq2 = orig;

        fmt::printf("%s\n", sq->reprSquare());

        REQUIRE(!sq->isEmpty());

        REQUIRE(sq->isOccupied(Role::Black));
        REQUIRE(!sq->isOccupied(Role::White));

        REQUIRE(!sq->isKing());
        REQUIRE(!sq->isWhiteKing());
        REQUIRE(!sq->isBlackKing());

        REQUIRE(!sq->isOpponentAndNotCaptured(Role::Black));
        REQUIRE(sq->isOpponentAndNotCaptured(Role::White));

        REQUIRE(!sq->isCaptured());
        REQUIRE(!sq->isLast());

    } catch (const K273::Exception& exc) {
        FAIL(exc.getMessage());
    }
}


TEST_CASE("Create - interim status", "[simple]") {

    setLogging();

    try {
        Description desc(10);
        Board* board = new Board(&desc);

        // side effects
        SM sm(board, &desc);
        sm.reset();

        // clear
        for (int ii=0; ii<desc.getNumPositions(); ii++) {
            board->get(ii + 1)->clear();
        }

        board->get(20)->set(Role::White, Piece::Man);
        board->get(24)->set(Role::White, Piece::Man);

        for (int ii=0; ii<desc.getNumPositions(); ii++) {
            fmt::printf("%s -> %s\n", ii + 1, board->get(ii + 1)->reprSquare());
        }

        REQUIRE(!board->getMeta()->interimStatus());
        REQUIRE(!board->get(24)->isLast());

        board->get(24)->setLast();
        board->getMeta()->setInterimStatus();

        REQUIRE(board->getMeta()->interimStatus());
        REQUIRE(board->get(24)->isLast());

        board->get(24)->unSetLast();
        board->getMeta()->unSetInterimStatus();

        REQUIRE(!board->getMeta()->interimStatus());
        REQUIRE(!board->get(24)->isLast());

        // turn
        REQUIRE(board->getMeta()->whosTurn() == Role::White);
        board->getMeta()->switchTurn();
        REQUIRE(board->getMeta()->whosTurn() == Role::Black);

        for (int ii=0; ii<desc.getNumPositions(); ii++) {
            fmt::printf("%s -> %s\n", ii + 1, board->get(ii + 1)->reprSquare());
        }

    } catch (const K273::Exception& exc) {
        FAIL(exc.getMessage());
    }
}

struct MoveCheck {
    Piece what;
    int from_pos;
    int to_pos;
};

using MoveChecks = std::vector <MoveCheck>;

void checkLegals(const GGPLib::LegalState* ls, Role role,
                 Description* desc, MoveChecks moves) {

    void* mexm = malloc(10);

    // check the same size
    ASSERT(ls->getCount() == (int) moves.size());

    void* mem = malloc(GGPLib::JointMove::mallocSize(6));
    GGPLib::JointMove* joint_move = static_cast <GGPLib::JointMove*>(mem);
    joint_move->setSize(2);

    for (int ii=0; ii<ls->getCount(); ii++) {
        int legal = ls->getLegal(ii);

        if (role == Role::White) {
            joint_move->set(0, legal);
            joint_move->set(1, 0);

        } else {
            joint_move->set(1, legal);
            joint_move->set(0, 0);
        }

        const ReverseLegalLookup* rll = desc->getReverseLookup(joint_move);
        ASSERT(rll->role == role);

        bool found = false;
        for (auto& check : moves) {
            if (rll->what == check.what &&
                rll->from_pos == check.from_pos &&
                rll->to_pos == check.to_pos) {
                found = true;
                break;
            }
        }

        ASSERT_MSG(found, K273::fmtString("Not found (%s, %s) %d -> %d",
                                          rll->role == Role::White ? "white" : "black",
                                          rll->what == Piece::Man ? "man" : "king",
                                          rll->from_pos, rll->to_pos));
    }

    ::free(mexm);
    ::free(joint_move);
}

TEST_CASE("Legals 1 - non captures", "[legals]") {

    setLogging();

    try {
        Description desc(10);
        Board* board = new Board(&desc);

        // side effects
        SM sm(board, &desc);
        sm.reset();

        // clear board
        for (int ii=0; ii<desc.getNumPositions(); ii++) {
            board->get(ii + 1)->clear();
        }

        // two men just to spice things up
        board->get(22)->set(Role::White, Piece::Man);
        board->get(23)->set(Role::White, Piece::Man);

        auto subtest = [&] (Role role, Piece what, Position from_pos,
                            MoveChecks moves) {
            fmt::printf("clear legals\n");

            board->clearLegals(Role::White);
            board->clearLegals(Role::Black);
            board->nonCaptureLegals(role, from_pos, what);

            checkLegals(sm.getLegalState(to_underlying(role)), role, &desc,
                        moves);
        };

        subtest(Role::White, Piece::Man, 24,
                { { Piece::Man, 24, 20 }, { Piece::Man, 24, 19} });

        subtest(Role::Black, Piece::Man, 24,
                { { Piece::Man, 24, 29 }, { Piece::Man, 24, 30} });

        subtest(Role::Black, Piece::King, 24,
                { { Piece::King, 24, 20 }, { Piece::King, 24, 15},
                  { Piece::King, 24, 29 }, { Piece::King, 24, 33},
                  { Piece::King, 24, 38 }, { Piece::King, 24, 42},
                  { Piece::King, 24, 30 }, { Piece::King, 24, 35},
                  { Piece::King, 24, 19 }, { Piece::King, 24, 13},
                  { Piece::King, 24, 8 }, { Piece::King, 24, 2},
                  { Piece::King, 24, 47 }});

        subtest(Role::White, Piece::Man, 24,
                { { Piece::Man, 24, 20 }, { Piece::Man, 24, 19} });

        subtest(Role::Black, Piece::Man, 24,
                { { Piece::Man, 24, 29 }, { Piece::Man, 24, 30} });

        // corner cases - blocked
        subtest(Role::White, Piece::Man, 28,
                {  });

        subtest(Role::Black, Piece::Man, 18,
                {  });

        subtest(Role::White, Piece::Man, 6,
                { { Piece::Man, 6, 1 } });

        subtest(Role::White, Piece::Man, 31,
                { { Piece::Man, 31, 26 }, { Piece::Man, 31, 27 } });

        subtest(Role::White, Piece::King, 18,
                { { Piece::King, 18, 12 }, { Piece::King, 18, 13},
                  { Piece::King, 18, 7 }, { Piece::King, 18, 1},
                  { Piece::King, 18, 9 }, { Piece::King, 18, 4} });


    } catch (const K273::Exception& exc) {
        FAIL(exc.getMessage());
    }
}

TEST_CASE("Legals 2 - updateLegalsChoices()", "[legals]") {

    setLogging();

    try {
        Description desc(10);
        Board* board = new Board(&desc);

        // side effects
        SM sm(board, &desc);
        sm.reset();

        // noop
        REQUIRE(sm.getLegalState(to_underlying(Role::Black))->getCount() == 1);

        checkLegals(sm.getLegalState(to_underlying(Role::White)), Role::White, &desc,
                    { { Piece::Man, 31, 26 }, { Piece::Man, 31, 27 } ,
                      { Piece::Man, 32, 27 }, { Piece::Man, 32, 28 } ,
                      { Piece::Man, 33, 28 }, { Piece::Man, 33, 29 } ,
                      { Piece::Man, 34, 29 }, { Piece::Man, 34, 30 } ,
                      { Piece::Man, 35, 30 } });


        // set joint move


    } catch (const K273::Exception& exc) {
        FAIL(exc.getMessage());
    }
}



TEST_CASE("Legals 3 - capture kings()", "[legals]") {

    setLogging();

    try {
        Description desc(10);
        Board* board = new Board(&desc);

        // side effects
        SM sm(board, &desc);
        sm.reset();

        // clear board
        for (int ii=0; ii<desc.getNumPositions(); ii++) {
            board->get(ii + 1)->clear();
        }

        // two men just to spice things up
        board->get(22)->set(Role::White, Piece::Man);
        board->get(23)->set(Role::White, Piece::Man);
        board->get(18)->set(Role::Black, Piece::King);

        REQUIRE(board->getMeta()->whosTurn() == Role::White);
        board->getMeta()->switchTurn();

        auto subtest = [&] (Role role, Piece what, Position from_pos,
                            MoveChecks moves) {
            board->updateLegalsChoices();
            checkLegals(sm.getLegalState(to_underlying(role)), role, &desc, moves);
        };

        fmt::printf("START TEST...\n");
        subtest(Role::Black, Piece::King, 18,
                { { Piece::King, 18, 12 }, { Piece::King, 18, 13},
                  { Piece::King, 18, 7 }, { Piece::King, 18, 1},
                  { Piece::King, 18, 9 }, { Piece::King, 18, 4} });


    } catch (const K273::Exception& exc) {
        FAIL(exc.getMessage());
    }
}


