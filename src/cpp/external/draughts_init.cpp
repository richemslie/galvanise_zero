

// local includes
#include "draughts_desc.h"
#include "draughts_board.h"

// k273 includes
#include <k273/util.h>
#include <k273/logging.h>
#include <k273/exception.h>

using namespace InternationalDraughts;




void Description::initBoard_8x8() {

    this->num_positions = 32;
    this->white_noop = 0;
    this->black_noop = 0;
    this->diagonal_data.resize(128);

    // Reserve the map size upfront, hopefully memory will be contiguous (XXX check)
    this->reverse_legal_lookup_white.resize(402);
    this->reverse_legal_lookup_black.resize(402);

    // Initial state
    this->initial_state = {false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false};



    // generating promotion line for white
    this->white_promotion_line = {true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false};


    // generating moves for white
    this->white_legal_moves = {"noop", "(move man b 8 d 6)", "(move king b 8 c 7)", "(move king b 8 d 6)", "(move king b 8 e 5)", "(move king b 8 f 4)", "(move king b 8 g 3)", "(move king b 8 h 2)", "(move king b 8 a 7)", "(move man d 8 f 6)", "(move man d 8 b 6)", "(move king d 8 e 7)", "(move king d 8 f 6)", "(move king d 8 g 5)", "(move king d 8 h 4)", "(move king d 8 c 7)", "(move king d 8 b 6)", "(move king d 8 a 5)", "(move man f 8 h 6)", "(move man f 8 d 6)", "(move king f 8 g 7)", "(move king f 8 h 6)", "(move king f 8 e 7)", "(move king f 8 d 6)", "(move king f 8 c 5)", "(move king f 8 b 4)", "(move king f 8 a 3)", "(move man h 8 f 6)", "(move king h 8 g 7)", "(move king h 8 f 6)", "(move king h 8 e 5)", "(move king h 8 d 4)", "(move king h 8 c 3)", "(move king h 8 b 2)", "(move king h 8 a 1)", "(move man a 7 b 8)", "(move man a 7 c 5)", "(move king a 7 b 8)", "(move king a 7 b 6)", "(move king a 7 c 5)", "(move king a 7 d 4)", "(move king a 7 e 3)", "(move king a 7 f 2)", "(move king a 7 g 1)", "(move man c 7 d 8)", "(move man c 7 b 8)", "(move man c 7 e 5)", "(move man c 7 a 5)", "(move king c 7 d 8)", "(move king c 7 b 8)", "(move king c 7 d 6)", "(move king c 7 e 5)", "(move king c 7 f 4)", "(move king c 7 g 3)", "(move king c 7 h 2)", "(move king c 7 b 6)", "(move king c 7 a 5)", "(move man e 7 f 8)", "(move man e 7 d 8)", "(move man e 7 g 5)", "(move man e 7 c 5)", "(move king e 7 f 8)", "(move king e 7 d 8)", "(move king e 7 f 6)", "(move king e 7 g 5)", "(move king e 7 h 4)", "(move king e 7 d 6)", "(move king e 7 c 5)", "(move king e 7 b 4)", "(move king e 7 a 3)", "(move man g 7 h 8)", "(move man g 7 f 8)", "(move man g 7 e 5)", "(move king g 7 h 8)", "(move king g 7 f 8)", "(move king g 7 h 6)", "(move king g 7 f 6)", "(move king g 7 e 5)", "(move king g 7 d 4)", "(move king g 7 c 3)", "(move king g 7 b 2)", "(move king g 7 a 1)", "(move man b 6 c 7)", "(move man b 6 a 7)", "(move man b 6 d 8)", "(move man b 6 d 4)", "(move king b 6 c 7)", "(move king b 6 d 8)", "(move king b 6 a 7)", "(move king b 6 c 5)", "(move king b 6 d 4)", "(move king b 6 e 3)", "(move king b 6 f 2)", "(move king b 6 g 1)", "(move king b 6 a 5)", "(move man d 6 e 7)", "(move man d 6 c 7)", "(move man d 6 f 8)", "(move man d 6 b 8)", "(move man d 6 f 4)", "(move man d 6 b 4)", "(move king d 6 e 7)", "(move king d 6 f 8)", "(move king d 6 c 7)", "(move king d 6 b 8)", "(move king d 6 e 5)", "(move king d 6 f 4)", "(move king d 6 g 3)", "(move king d 6 h 2)", "(move king d 6 c 5)", "(move king d 6 b 4)", "(move king d 6 a 3)", "(move man f 6 g 7)", "(move man f 6 e 7)", "(move man f 6 h 8)", "(move man f 6 d 8)", "(move man f 6 h 4)", "(move man f 6 d 4)", "(move king f 6 g 7)", "(move king f 6 h 8)", "(move king f 6 e 7)", "(move king f 6 d 8)", "(move king f 6 g 5)", "(move king f 6 h 4)", "(move king f 6 e 5)", "(move king f 6 d 4)", "(move king f 6 c 3)", "(move king f 6 b 2)", "(move king f 6 a 1)", "(move man h 6 g 7)", "(move man h 6 f 8)", "(move man h 6 f 4)", "(move king h 6 g 7)", "(move king h 6 f 8)", "(move king h 6 g 5)", "(move king h 6 f 4)", "(move king h 6 e 3)", "(move king h 6 d 2)", "(move king h 6 c 1)", "(move man a 5 b 6)", "(move man a 5 c 7)", "(move man a 5 c 3)", "(move king a 5 b 6)", "(move king a 5 c 7)", "(move king a 5 d 8)", "(move king a 5 b 4)", "(move king a 5 c 3)", "(move king a 5 d 2)", "(move king a 5 e 1)", "(move man c 5 d 6)", "(move man c 5 b 6)", "(move man c 5 e 7)", "(move man c 5 a 7)", "(move man c 5 e 3)", "(move man c 5 a 3)", "(move king c 5 d 6)", "(move king c 5 e 7)", "(move king c 5 f 8)", "(move king c 5 b 6)", "(move king c 5 a 7)", "(move king c 5 d 4)", "(move king c 5 e 3)", "(move king c 5 f 2)", "(move king c 5 g 1)", "(move king c 5 b 4)", "(move king c 5 a 3)", "(move man e 5 f 6)", "(move man e 5 d 6)", "(move man e 5 g 7)", "(move man e 5 c 7)", "(move man e 5 g 3)", "(move man e 5 c 3)", "(move king e 5 f 6)", "(move king e 5 g 7)", "(move king e 5 h 8)", "(move king e 5 d 6)", "(move king e 5 c 7)", "(move king e 5 b 8)", "(move king e 5 f 4)", "(move king e 5 g 3)", "(move king e 5 h 2)", "(move king e 5 d 4)", "(move king e 5 c 3)", "(move king e 5 b 2)", "(move king e 5 a 1)", "(move man g 5 h 6)", "(move man g 5 f 6)", "(move man g 5 e 7)", "(move man g 5 e 3)", "(move king g 5 h 6)", "(move king g 5 f 6)", "(move king g 5 e 7)", "(move king g 5 d 8)", "(move king g 5 h 4)", "(move king g 5 f 4)", "(move king g 5 e 3)", "(move king g 5 d 2)", "(move king g 5 c 1)", "(move man b 4 c 5)", "(move man b 4 a 5)", "(move man b 4 d 6)", "(move man b 4 d 2)", "(move king b 4 c 5)", "(move king b 4 d 6)", "(move king b 4 e 7)", "(move king b 4 f 8)", "(move king b 4 a 5)", "(move king b 4 c 3)", "(move king b 4 d 2)", "(move king b 4 e 1)", "(move king b 4 a 3)", "(move man d 4 e 5)", "(move man d 4 c 5)", "(move man d 4 f 6)", "(move man d 4 b 6)", "(move man d 4 f 2)", "(move man d 4 b 2)", "(move king d 4 e 5)", "(move king d 4 f 6)", "(move king d 4 g 7)", "(move king d 4 h 8)", "(move king d 4 c 5)", "(move king d 4 b 6)", "(move king d 4 a 7)", "(move king d 4 e 3)", "(move king d 4 f 2)", "(move king d 4 g 1)", "(move king d 4 c 3)", "(move king d 4 b 2)", "(move king d 4 a 1)", "(move man f 4 g 5)", "(move man f 4 e 5)", "(move man f 4 h 6)", "(move man f 4 d 6)", "(move man f 4 h 2)", "(move man f 4 d 2)", "(move king f 4 g 5)", "(move king f 4 h 6)", "(move king f 4 e 5)", "(move king f 4 d 6)", "(move king f 4 c 7)", "(move king f 4 b 8)", "(move king f 4 g 3)", "(move king f 4 h 2)", "(move king f 4 e 3)", "(move king f 4 d 2)", "(move king f 4 c 1)", "(move man h 4 g 5)", "(move man h 4 f 6)", "(move man h 4 f 2)", "(move king h 4 g 5)", "(move king h 4 f 6)", "(move king h 4 e 7)", "(move king h 4 d 8)", "(move king h 4 g 3)", "(move king h 4 f 2)", "(move king h 4 e 1)", "(move man a 3 b 4)", "(move man a 3 c 5)", "(move man a 3 c 1)", "(move king a 3 b 4)", "(move king a 3 c 5)", "(move king a 3 d 6)", "(move king a 3 e 7)", "(move king a 3 f 8)", "(move king a 3 b 2)", "(move king a 3 c 1)", "(move man c 3 d 4)", "(move man c 3 b 4)", "(move man c 3 e 5)", "(move man c 3 a 5)", "(move man c 3 e 1)", "(move man c 3 a 1)", "(move king c 3 d 4)", "(move king c 3 e 5)", "(move king c 3 f 6)", "(move king c 3 g 7)", "(move king c 3 h 8)", "(move king c 3 b 4)", "(move king c 3 a 5)", "(move king c 3 d 2)", "(move king c 3 e 1)", "(move king c 3 b 2)", "(move king c 3 a 1)", "(move man e 3 f 4)", "(move man e 3 d 4)", "(move man e 3 g 5)", "(move man e 3 c 5)", "(move man e 3 g 1)", "(move man e 3 c 1)", "(move king e 3 f 4)", "(move king e 3 g 5)", "(move king e 3 h 6)", "(move king e 3 d 4)", "(move king e 3 c 5)", "(move king e 3 b 6)", "(move king e 3 a 7)", "(move king e 3 f 2)", "(move king e 3 g 1)", "(move king e 3 d 2)", "(move king e 3 c 1)", "(move man g 3 h 4)", "(move man g 3 f 4)", "(move man g 3 e 5)", "(move man g 3 e 1)", "(move king g 3 h 4)", "(move king g 3 f 4)", "(move king g 3 e 5)", "(move king g 3 d 6)", "(move king g 3 c 7)", "(move king g 3 b 8)", "(move king g 3 h 2)", "(move king g 3 f 2)", "(move king g 3 e 1)", "(move man b 2 c 3)", "(move man b 2 a 3)", "(move man b 2 d 4)", "(move king b 2 c 3)", "(move king b 2 d 4)", "(move king b 2 e 5)", "(move king b 2 f 6)", "(move king b 2 g 7)", "(move king b 2 h 8)", "(move king b 2 a 3)", "(move king b 2 c 1)", "(move king b 2 a 1)", "(move man d 2 e 3)", "(move man d 2 c 3)", "(move man d 2 f 4)", "(move man d 2 b 4)", "(move king d 2 e 3)", "(move king d 2 f 4)", "(move king d 2 g 5)", "(move king d 2 h 6)", "(move king d 2 c 3)", "(move king d 2 b 4)", "(move king d 2 a 5)", "(move king d 2 e 1)", "(move king d 2 c 1)", "(move man f 2 g 3)", "(move man f 2 e 3)", "(move man f 2 h 4)", "(move man f 2 d 4)", "(move king f 2 g 3)", "(move king f 2 h 4)", "(move king f 2 e 3)", "(move king f 2 d 4)", "(move king f 2 c 5)", "(move king f 2 b 6)", "(move king f 2 a 7)", "(move king f 2 g 1)", "(move king f 2 e 1)", "(move man h 2 g 3)", "(move man h 2 f 4)", "(move king h 2 g 3)", "(move king h 2 f 4)", "(move king h 2 e 5)", "(move king h 2 d 6)", "(move king h 2 c 7)", "(move king h 2 b 8)", "(move king h 2 g 1)", "(move man a 1 b 2)", "(move man a 1 c 3)", "(move king a 1 b 2)", "(move king a 1 c 3)", "(move king a 1 d 4)", "(move king a 1 e 5)", "(move king a 1 f 6)", "(move king a 1 g 7)", "(move king a 1 h 8)", "(move man c 1 d 2)", "(move man c 1 b 2)", "(move man c 1 e 3)", "(move man c 1 a 3)", "(move king c 1 d 2)", "(move king c 1 e 3)", "(move king c 1 f 4)", "(move king c 1 g 5)", "(move king c 1 h 6)", "(move king c 1 b 2)", "(move king c 1 a 3)", "(move man e 1 f 2)", "(move man e 1 d 2)", "(move man e 1 g 3)", "(move man e 1 c 3)", "(move king e 1 f 2)", "(move king e 1 g 3)", "(move king e 1 h 4)", "(move king e 1 d 2)", "(move king e 1 c 3)", "(move king e 1 b 4)", "(move king e 1 a 5)", "(move man g 1 h 2)", "(move man g 1 f 2)", "(move man g 1 e 3)", "(move king g 1 h 2)", "(move king g 1 f 2)", "(move king g 1 e 3)", "(move king g 1 d 4)", "(move king g 1 c 5)", "(move king g 1 b 6)", "(move king g 1 a 7)"};
    // generating for white man 1 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 6, legal: invalid
        ddi->diagonals.emplace_back(6, -1);

        // to position 10, legal: (legal white (move man b 8 d 6))
        ddi->diagonals.emplace_back(10, 1);

        this->diagonal_data[0].push_back(ddi);


        this->reverse_legal_lookup_white[1] = ReverseLegalLookup(Role::White, Piece::Man, 1, 10, SE);
    }
    // generating for white man 1 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 5, legal: invalid
        ddi->diagonals.emplace_back(5, -1);

        this->diagonal_data[0].push_back(ddi);


    }
    // generating for white man 2 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: invalid
        ddi->diagonals.emplace_back(7, -1);

        // to position 11, legal: (legal white (move man d 8 f 6))
        ddi->diagonals.emplace_back(11, 9);

        this->diagonal_data[1].push_back(ddi);


        this->reverse_legal_lookup_white[9] = ReverseLegalLookup(Role::White, Piece::Man, 2, 11, SE);
    }
    // generating for white man 2 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 6, legal: invalid
        ddi->diagonals.emplace_back(6, -1);

        // to position 9, legal: (legal white (move man d 8 b 6))
        ddi->diagonals.emplace_back(9, 10);

        this->diagonal_data[1].push_back(ddi);


        this->reverse_legal_lookup_white[10] = ReverseLegalLookup(Role::White, Piece::Man, 2, 9, SW);
    }
    // generating for white man 3 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: invalid
        ddi->diagonals.emplace_back(8, -1);

        // to position 12, legal: (legal white (move man f 8 h 6))
        ddi->diagonals.emplace_back(12, 18);

        this->diagonal_data[2].push_back(ddi);


        this->reverse_legal_lookup_white[18] = ReverseLegalLookup(Role::White, Piece::Man, 3, 12, SE);
    }
    // generating for white man 3 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: invalid
        ddi->diagonals.emplace_back(7, -1);

        // to position 10, legal: (legal white (move man f 8 d 6))
        ddi->diagonals.emplace_back(10, 19);

        this->diagonal_data[2].push_back(ddi);


        this->reverse_legal_lookup_white[19] = ReverseLegalLookup(Role::White, Piece::Man, 3, 10, SW);
    }
    // generating for white man 4 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: invalid
        ddi->diagonals.emplace_back(8, -1);

        // to position 11, legal: (legal white (move man h 8 f 6))
        ddi->diagonals.emplace_back(11, 27);

        this->diagonal_data[3].push_back(ddi);


        this->reverse_legal_lookup_white[27] = ReverseLegalLookup(Role::White, Piece::Man, 4, 11, SW);
    }
    // generating for white man 5 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal white (move man a 7 b 8))
        ddi->diagonals.emplace_back(1, 35);

        this->diagonal_data[4].push_back(ddi);


        this->reverse_legal_lookup_white[35] = ReverseLegalLookup(Role::White, Piece::Man, 5, 1, NE);
    }
    // generating for white man 5 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 9, legal: invalid
        ddi->diagonals.emplace_back(9, -1);

        // to position 14, legal: (legal white (move man a 7 c 5))
        ddi->diagonals.emplace_back(14, 36);

        this->diagonal_data[4].push_back(ddi);


        this->reverse_legal_lookup_white[36] = ReverseLegalLookup(Role::White, Piece::Man, 5, 14, SE);
    }
    // generating for white man 6 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal white (move man c 7 d 8))
        ddi->diagonals.emplace_back(2, 44);

        this->diagonal_data[5].push_back(ddi);


        this->reverse_legal_lookup_white[44] = ReverseLegalLookup(Role::White, Piece::Man, 6, 2, NE);
    }
    // generating for white man 6 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal white (move man c 7 b 8))
        ddi->diagonals.emplace_back(1, 45);

        this->diagonal_data[5].push_back(ddi);


        this->reverse_legal_lookup_white[45] = ReverseLegalLookup(Role::White, Piece::Man, 6, 1, NW);
    }
    // generating for white man 6 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: invalid
        ddi->diagonals.emplace_back(10, -1);

        // to position 15, legal: (legal white (move man c 7 e 5))
        ddi->diagonals.emplace_back(15, 46);

        this->diagonal_data[5].push_back(ddi);


        this->reverse_legal_lookup_white[46] = ReverseLegalLookup(Role::White, Piece::Man, 6, 15, SE);
    }
    // generating for white man 6 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: invalid
        ddi->diagonals.emplace_back(9, -1);

        // to position 13, legal: (legal white (move man c 7 a 5))
        ddi->diagonals.emplace_back(13, 47);

        this->diagonal_data[5].push_back(ddi);


        this->reverse_legal_lookup_white[47] = ReverseLegalLookup(Role::White, Piece::Man, 6, 13, SW);
    }
    // generating for white man 7 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal white (move man e 7 f 8))
        ddi->diagonals.emplace_back(3, 57);

        this->diagonal_data[6].push_back(ddi);


        this->reverse_legal_lookup_white[57] = ReverseLegalLookup(Role::White, Piece::Man, 7, 3, NE);
    }
    // generating for white man 7 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal white (move man e 7 d 8))
        ddi->diagonals.emplace_back(2, 58);

        this->diagonal_data[6].push_back(ddi);


        this->reverse_legal_lookup_white[58] = ReverseLegalLookup(Role::White, Piece::Man, 7, 2, NW);
    }
    // generating for white man 7 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 11, legal: invalid
        ddi->diagonals.emplace_back(11, -1);

        // to position 16, legal: (legal white (move man e 7 g 5))
        ddi->diagonals.emplace_back(16, 59);

        this->diagonal_data[6].push_back(ddi);


        this->reverse_legal_lookup_white[59] = ReverseLegalLookup(Role::White, Piece::Man, 7, 16, SE);
    }
    // generating for white man 7 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 10, legal: invalid
        ddi->diagonals.emplace_back(10, -1);

        // to position 14, legal: (legal white (move man e 7 c 5))
        ddi->diagonals.emplace_back(14, 60);

        this->diagonal_data[6].push_back(ddi);


        this->reverse_legal_lookup_white[60] = ReverseLegalLookup(Role::White, Piece::Man, 7, 14, SW);
    }
    // generating for white man 8 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 4, legal: (legal white (move man g 7 h 8))
        ddi->diagonals.emplace_back(4, 70);

        this->diagonal_data[7].push_back(ddi);


        this->reverse_legal_lookup_white[70] = ReverseLegalLookup(Role::White, Piece::Man, 8, 4, NE);
    }
    // generating for white man 8 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal white (move man g 7 f 8))
        ddi->diagonals.emplace_back(3, 71);

        this->diagonal_data[7].push_back(ddi);


        this->reverse_legal_lookup_white[71] = ReverseLegalLookup(Role::White, Piece::Man, 8, 3, NW);
    }
    // generating for white man 8 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 12, legal: invalid
        ddi->diagonals.emplace_back(12, -1);

        this->diagonal_data[7].push_back(ddi);


    }
    // generating for white man 8 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: invalid
        ddi->diagonals.emplace_back(11, -1);

        // to position 15, legal: (legal white (move man g 7 e 5))
        ddi->diagonals.emplace_back(15, 72);

        this->diagonal_data[7].push_back(ddi);


        this->reverse_legal_lookup_white[72] = ReverseLegalLookup(Role::White, Piece::Man, 8, 15, SW);
    }
    // generating for white man 9 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 6, legal: (legal white (move man b 6 c 7))
        ddi->diagonals.emplace_back(6, 82);

        // to position 2, legal: (legal white (move man b 6 d 8))
        ddi->diagonals.emplace_back(2, 84);

        this->diagonal_data[8].push_back(ddi);


        this->reverse_legal_lookup_white[82] = ReverseLegalLookup(Role::White, Piece::Man, 9, 6, NE);
        this->reverse_legal_lookup_white[84] = ReverseLegalLookup(Role::White, Piece::Man, 9, 2, NE);
    }
    // generating for white man 9 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 5, legal: (legal white (move man b 6 a 7))
        ddi->diagonals.emplace_back(5, 83);

        this->diagonal_data[8].push_back(ddi);


        this->reverse_legal_lookup_white[83] = ReverseLegalLookup(Role::White, Piece::Man, 9, 5, NW);
    }
    // generating for white man 9 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 14, legal: invalid
        ddi->diagonals.emplace_back(14, -1);

        // to position 18, legal: (legal white (move man b 6 d 4))
        ddi->diagonals.emplace_back(18, 85);

        this->diagonal_data[8].push_back(ddi);


        this->reverse_legal_lookup_white[85] = ReverseLegalLookup(Role::White, Piece::Man, 9, 18, SE);
    }
    // generating for white man 9 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 13, legal: invalid
        ddi->diagonals.emplace_back(13, -1);

        this->diagonal_data[8].push_back(ddi);


    }
    // generating for white man 10 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal white (move man d 6 e 7))
        ddi->diagonals.emplace_back(7, 95);

        // to position 3, legal: (legal white (move man d 6 f 8))
        ddi->diagonals.emplace_back(3, 97);

        this->diagonal_data[9].push_back(ddi);


        this->reverse_legal_lookup_white[95] = ReverseLegalLookup(Role::White, Piece::Man, 10, 7, NE);
        this->reverse_legal_lookup_white[97] = ReverseLegalLookup(Role::White, Piece::Man, 10, 3, NE);
    }
    // generating for white man 10 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 6, legal: (legal white (move man d 6 c 7))
        ddi->diagonals.emplace_back(6, 96);

        // to position 1, legal: (legal white (move man d 6 b 8))
        ddi->diagonals.emplace_back(1, 98);

        this->diagonal_data[9].push_back(ddi);


        this->reverse_legal_lookup_white[96] = ReverseLegalLookup(Role::White, Piece::Man, 10, 6, NW);
        this->reverse_legal_lookup_white[98] = ReverseLegalLookup(Role::White, Piece::Man, 10, 1, NW);
    }
    // generating for white man 10 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 15, legal: invalid
        ddi->diagonals.emplace_back(15, -1);

        // to position 19, legal: (legal white (move man d 6 f 4))
        ddi->diagonals.emplace_back(19, 99);

        this->diagonal_data[9].push_back(ddi);


        this->reverse_legal_lookup_white[99] = ReverseLegalLookup(Role::White, Piece::Man, 10, 19, SE);
    }
    // generating for white man 10 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 14, legal: invalid
        ddi->diagonals.emplace_back(14, -1);

        // to position 17, legal: (legal white (move man d 6 b 4))
        ddi->diagonals.emplace_back(17, 100);

        this->diagonal_data[9].push_back(ddi);


        this->reverse_legal_lookup_white[100] = ReverseLegalLookup(Role::White, Piece::Man, 10, 17, SW);
    }
    // generating for white man 11 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal white (move man f 6 g 7))
        ddi->diagonals.emplace_back(8, 112);

        // to position 4, legal: (legal white (move man f 6 h 8))
        ddi->diagonals.emplace_back(4, 114);

        this->diagonal_data[10].push_back(ddi);


        this->reverse_legal_lookup_white[112] = ReverseLegalLookup(Role::White, Piece::Man, 11, 8, NE);
        this->reverse_legal_lookup_white[114] = ReverseLegalLookup(Role::White, Piece::Man, 11, 4, NE);
    }
    // generating for white man 11 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal white (move man f 6 e 7))
        ddi->diagonals.emplace_back(7, 113);

        // to position 2, legal: (legal white (move man f 6 d 8))
        ddi->diagonals.emplace_back(2, 115);

        this->diagonal_data[10].push_back(ddi);


        this->reverse_legal_lookup_white[113] = ReverseLegalLookup(Role::White, Piece::Man, 11, 7, NW);
        this->reverse_legal_lookup_white[115] = ReverseLegalLookup(Role::White, Piece::Man, 11, 2, NW);
    }
    // generating for white man 11 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 16, legal: invalid
        ddi->diagonals.emplace_back(16, -1);

        // to position 20, legal: (legal white (move man f 6 h 4))
        ddi->diagonals.emplace_back(20, 116);

        this->diagonal_data[10].push_back(ddi);


        this->reverse_legal_lookup_white[116] = ReverseLegalLookup(Role::White, Piece::Man, 11, 20, SE);
    }
    // generating for white man 11 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 15, legal: invalid
        ddi->diagonals.emplace_back(15, -1);

        // to position 18, legal: (legal white (move man f 6 d 4))
        ddi->diagonals.emplace_back(18, 117);

        this->diagonal_data[10].push_back(ddi);


        this->reverse_legal_lookup_white[117] = ReverseLegalLookup(Role::White, Piece::Man, 11, 18, SW);
    }
    // generating for white man 12 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal white (move man h 6 g 7))
        ddi->diagonals.emplace_back(8, 129);

        // to position 3, legal: (legal white (move man h 6 f 8))
        ddi->diagonals.emplace_back(3, 130);

        this->diagonal_data[11].push_back(ddi);


        this->reverse_legal_lookup_white[129] = ReverseLegalLookup(Role::White, Piece::Man, 12, 8, NW);
        this->reverse_legal_lookup_white[130] = ReverseLegalLookup(Role::White, Piece::Man, 12, 3, NW);
    }
    // generating for white man 12 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 16, legal: invalid
        ddi->diagonals.emplace_back(16, -1);

        // to position 19, legal: (legal white (move man h 6 f 4))
        ddi->diagonals.emplace_back(19, 131);

        this->diagonal_data[11].push_back(ddi);


        this->reverse_legal_lookup_white[131] = ReverseLegalLookup(Role::White, Piece::Man, 12, 19, SW);
    }
    // generating for white man 13 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal white (move man a 5 b 6))
        ddi->diagonals.emplace_back(9, 139);

        // to position 6, legal: (legal white (move man a 5 c 7))
        ddi->diagonals.emplace_back(6, 140);

        this->diagonal_data[12].push_back(ddi);


        this->reverse_legal_lookup_white[139] = ReverseLegalLookup(Role::White, Piece::Man, 13, 9, NE);
        this->reverse_legal_lookup_white[140] = ReverseLegalLookup(Role::White, Piece::Man, 13, 6, NE);
    }
    // generating for white man 13 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 17, legal: invalid
        ddi->diagonals.emplace_back(17, -1);

        // to position 22, legal: (legal white (move man a 5 c 3))
        ddi->diagonals.emplace_back(22, 141);

        this->diagonal_data[12].push_back(ddi);


        this->reverse_legal_lookup_white[141] = ReverseLegalLookup(Role::White, Piece::Man, 13, 22, SE);
    }
    // generating for white man 14 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal white (move man c 5 d 6))
        ddi->diagonals.emplace_back(10, 149);

        // to position 7, legal: (legal white (move man c 5 e 7))
        ddi->diagonals.emplace_back(7, 151);

        this->diagonal_data[13].push_back(ddi);


        this->reverse_legal_lookup_white[149] = ReverseLegalLookup(Role::White, Piece::Man, 14, 10, NE);
        this->reverse_legal_lookup_white[151] = ReverseLegalLookup(Role::White, Piece::Man, 14, 7, NE);
    }
    // generating for white man 14 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal white (move man c 5 b 6))
        ddi->diagonals.emplace_back(9, 150);

        // to position 5, legal: (legal white (move man c 5 a 7))
        ddi->diagonals.emplace_back(5, 152);

        this->diagonal_data[13].push_back(ddi);


        this->reverse_legal_lookup_white[150] = ReverseLegalLookup(Role::White, Piece::Man, 14, 9, NW);
        this->reverse_legal_lookup_white[152] = ReverseLegalLookup(Role::White, Piece::Man, 14, 5, NW);
    }
    // generating for white man 14 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 18, legal: invalid
        ddi->diagonals.emplace_back(18, -1);

        // to position 23, legal: (legal white (move man c 5 e 3))
        ddi->diagonals.emplace_back(23, 153);

        this->diagonal_data[13].push_back(ddi);


        this->reverse_legal_lookup_white[153] = ReverseLegalLookup(Role::White, Piece::Man, 14, 23, SE);
    }
    // generating for white man 14 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: invalid
        ddi->diagonals.emplace_back(17, -1);

        // to position 21, legal: (legal white (move man c 5 a 3))
        ddi->diagonals.emplace_back(21, 154);

        this->diagonal_data[13].push_back(ddi);


        this->reverse_legal_lookup_white[154] = ReverseLegalLookup(Role::White, Piece::Man, 14, 21, SW);
    }
    // generating for white man 15 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal white (move man e 5 f 6))
        ddi->diagonals.emplace_back(11, 166);

        // to position 8, legal: (legal white (move man e 5 g 7))
        ddi->diagonals.emplace_back(8, 168);

        this->diagonal_data[14].push_back(ddi);


        this->reverse_legal_lookup_white[166] = ReverseLegalLookup(Role::White, Piece::Man, 15, 11, NE);
        this->reverse_legal_lookup_white[168] = ReverseLegalLookup(Role::White, Piece::Man, 15, 8, NE);
    }
    // generating for white man 15 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal white (move man e 5 d 6))
        ddi->diagonals.emplace_back(10, 167);

        // to position 6, legal: (legal white (move man e 5 c 7))
        ddi->diagonals.emplace_back(6, 169);

        this->diagonal_data[14].push_back(ddi);


        this->reverse_legal_lookup_white[167] = ReverseLegalLookup(Role::White, Piece::Man, 15, 10, NW);
        this->reverse_legal_lookup_white[169] = ReverseLegalLookup(Role::White, Piece::Man, 15, 6, NW);
    }
    // generating for white man 15 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 19, legal: invalid
        ddi->diagonals.emplace_back(19, -1);

        // to position 24, legal: (legal white (move man e 5 g 3))
        ddi->diagonals.emplace_back(24, 170);

        this->diagonal_data[14].push_back(ddi);


        this->reverse_legal_lookup_white[170] = ReverseLegalLookup(Role::White, Piece::Man, 15, 24, SE);
    }
    // generating for white man 15 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 18, legal: invalid
        ddi->diagonals.emplace_back(18, -1);

        // to position 22, legal: (legal white (move man e 5 c 3))
        ddi->diagonals.emplace_back(22, 171);

        this->diagonal_data[14].push_back(ddi);


        this->reverse_legal_lookup_white[171] = ReverseLegalLookup(Role::White, Piece::Man, 15, 22, SW);
    }
    // generating for white man 16 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 12, legal: (legal white (move man g 5 h 6))
        ddi->diagonals.emplace_back(12, 185);

        this->diagonal_data[15].push_back(ddi);


        this->reverse_legal_lookup_white[185] = ReverseLegalLookup(Role::White, Piece::Man, 16, 12, NE);
    }
    // generating for white man 16 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal white (move man g 5 f 6))
        ddi->diagonals.emplace_back(11, 186);

        // to position 7, legal: (legal white (move man g 5 e 7))
        ddi->diagonals.emplace_back(7, 187);

        this->diagonal_data[15].push_back(ddi);


        this->reverse_legal_lookup_white[186] = ReverseLegalLookup(Role::White, Piece::Man, 16, 11, NW);
        this->reverse_legal_lookup_white[187] = ReverseLegalLookup(Role::White, Piece::Man, 16, 7, NW);
    }
    // generating for white man 16 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 20, legal: invalid
        ddi->diagonals.emplace_back(20, -1);

        this->diagonal_data[15].push_back(ddi);


    }
    // generating for white man 16 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 19, legal: invalid
        ddi->diagonals.emplace_back(19, -1);

        // to position 23, legal: (legal white (move man g 5 e 3))
        ddi->diagonals.emplace_back(23, 188);

        this->diagonal_data[15].push_back(ddi);


        this->reverse_legal_lookup_white[188] = ReverseLegalLookup(Role::White, Piece::Man, 16, 23, SW);
    }
    // generating for white man 17 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 14, legal: (legal white (move man b 4 c 5))
        ddi->diagonals.emplace_back(14, 198);

        // to position 10, legal: (legal white (move man b 4 d 6))
        ddi->diagonals.emplace_back(10, 200);

        this->diagonal_data[16].push_back(ddi);


        this->reverse_legal_lookup_white[198] = ReverseLegalLookup(Role::White, Piece::Man, 17, 14, NE);
        this->reverse_legal_lookup_white[200] = ReverseLegalLookup(Role::White, Piece::Man, 17, 10, NE);
    }
    // generating for white man 17 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 13, legal: (legal white (move man b 4 a 5))
        ddi->diagonals.emplace_back(13, 199);

        this->diagonal_data[16].push_back(ddi);


        this->reverse_legal_lookup_white[199] = ReverseLegalLookup(Role::White, Piece::Man, 17, 13, NW);
    }
    // generating for white man 17 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 22, legal: invalid
        ddi->diagonals.emplace_back(22, -1);

        // to position 26, legal: (legal white (move man b 4 d 2))
        ddi->diagonals.emplace_back(26, 201);

        this->diagonal_data[16].push_back(ddi);


        this->reverse_legal_lookup_white[201] = ReverseLegalLookup(Role::White, Piece::Man, 17, 26, SE);
    }
    // generating for white man 17 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 21, legal: invalid
        ddi->diagonals.emplace_back(21, -1);

        this->diagonal_data[16].push_back(ddi);


    }
    // generating for white man 18 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 15, legal: (legal white (move man d 4 e 5))
        ddi->diagonals.emplace_back(15, 211);

        // to position 11, legal: (legal white (move man d 4 f 6))
        ddi->diagonals.emplace_back(11, 213);

        this->diagonal_data[17].push_back(ddi);


        this->reverse_legal_lookup_white[211] = ReverseLegalLookup(Role::White, Piece::Man, 18, 15, NE);
        this->reverse_legal_lookup_white[213] = ReverseLegalLookup(Role::White, Piece::Man, 18, 11, NE);
    }
    // generating for white man 18 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 14, legal: (legal white (move man d 4 c 5))
        ddi->diagonals.emplace_back(14, 212);

        // to position 9, legal: (legal white (move man d 4 b 6))
        ddi->diagonals.emplace_back(9, 214);

        this->diagonal_data[17].push_back(ddi);


        this->reverse_legal_lookup_white[212] = ReverseLegalLookup(Role::White, Piece::Man, 18, 14, NW);
        this->reverse_legal_lookup_white[214] = ReverseLegalLookup(Role::White, Piece::Man, 18, 9, NW);
    }
    // generating for white man 18 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 23, legal: invalid
        ddi->diagonals.emplace_back(23, -1);

        // to position 27, legal: (legal white (move man d 4 f 2))
        ddi->diagonals.emplace_back(27, 215);

        this->diagonal_data[17].push_back(ddi);


        this->reverse_legal_lookup_white[215] = ReverseLegalLookup(Role::White, Piece::Man, 18, 27, SE);
    }
    // generating for white man 18 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 22, legal: invalid
        ddi->diagonals.emplace_back(22, -1);

        // to position 25, legal: (legal white (move man d 4 b 2))
        ddi->diagonals.emplace_back(25, 216);

        this->diagonal_data[17].push_back(ddi);


        this->reverse_legal_lookup_white[216] = ReverseLegalLookup(Role::White, Piece::Man, 18, 25, SW);
    }
    // generating for white man 19 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 16, legal: (legal white (move man f 4 g 5))
        ddi->diagonals.emplace_back(16, 230);

        // to position 12, legal: (legal white (move man f 4 h 6))
        ddi->diagonals.emplace_back(12, 232);

        this->diagonal_data[18].push_back(ddi);


        this->reverse_legal_lookup_white[230] = ReverseLegalLookup(Role::White, Piece::Man, 19, 16, NE);
        this->reverse_legal_lookup_white[232] = ReverseLegalLookup(Role::White, Piece::Man, 19, 12, NE);
    }
    // generating for white man 19 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 15, legal: (legal white (move man f 4 e 5))
        ddi->diagonals.emplace_back(15, 231);

        // to position 10, legal: (legal white (move man f 4 d 6))
        ddi->diagonals.emplace_back(10, 233);

        this->diagonal_data[18].push_back(ddi);


        this->reverse_legal_lookup_white[231] = ReverseLegalLookup(Role::White, Piece::Man, 19, 15, NW);
        this->reverse_legal_lookup_white[233] = ReverseLegalLookup(Role::White, Piece::Man, 19, 10, NW);
    }
    // generating for white man 19 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: invalid
        ddi->diagonals.emplace_back(24, -1);

        // to position 28, legal: (legal white (move man f 4 h 2))
        ddi->diagonals.emplace_back(28, 234);

        this->diagonal_data[18].push_back(ddi);


        this->reverse_legal_lookup_white[234] = ReverseLegalLookup(Role::White, Piece::Man, 19, 28, SE);
    }
    // generating for white man 19 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 23, legal: invalid
        ddi->diagonals.emplace_back(23, -1);

        // to position 26, legal: (legal white (move man f 4 d 2))
        ddi->diagonals.emplace_back(26, 235);

        this->diagonal_data[18].push_back(ddi);


        this->reverse_legal_lookup_white[235] = ReverseLegalLookup(Role::White, Piece::Man, 19, 26, SW);
    }
    // generating for white man 20 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 16, legal: (legal white (move man h 4 g 5))
        ddi->diagonals.emplace_back(16, 247);

        // to position 11, legal: (legal white (move man h 4 f 6))
        ddi->diagonals.emplace_back(11, 248);

        this->diagonal_data[19].push_back(ddi);


        this->reverse_legal_lookup_white[247] = ReverseLegalLookup(Role::White, Piece::Man, 20, 16, NW);
        this->reverse_legal_lookup_white[248] = ReverseLegalLookup(Role::White, Piece::Man, 20, 11, NW);
    }
    // generating for white man 20 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 24, legal: invalid
        ddi->diagonals.emplace_back(24, -1);

        // to position 27, legal: (legal white (move man h 4 f 2))
        ddi->diagonals.emplace_back(27, 249);

        this->diagonal_data[19].push_back(ddi);


        this->reverse_legal_lookup_white[249] = ReverseLegalLookup(Role::White, Piece::Man, 20, 27, SW);
    }
    // generating for white man 21 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal white (move man a 3 b 4))
        ddi->diagonals.emplace_back(17, 257);

        // to position 14, legal: (legal white (move man a 3 c 5))
        ddi->diagonals.emplace_back(14, 258);

        this->diagonal_data[20].push_back(ddi);


        this->reverse_legal_lookup_white[257] = ReverseLegalLookup(Role::White, Piece::Man, 21, 17, NE);
        this->reverse_legal_lookup_white[258] = ReverseLegalLookup(Role::White, Piece::Man, 21, 14, NE);
    }
    // generating for white man 21 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 25, legal: invalid
        ddi->diagonals.emplace_back(25, -1);

        // to position 30, legal: (legal white (move man a 3 c 1))
        ddi->diagonals.emplace_back(30, 259);

        this->diagonal_data[20].push_back(ddi);


        this->reverse_legal_lookup_white[259] = ReverseLegalLookup(Role::White, Piece::Man, 21, 30, SE);
    }
    // generating for white man 22 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 18, legal: (legal white (move man c 3 d 4))
        ddi->diagonals.emplace_back(18, 267);

        // to position 15, legal: (legal white (move man c 3 e 5))
        ddi->diagonals.emplace_back(15, 269);

        this->diagonal_data[21].push_back(ddi);


        this->reverse_legal_lookup_white[267] = ReverseLegalLookup(Role::White, Piece::Man, 22, 18, NE);
        this->reverse_legal_lookup_white[269] = ReverseLegalLookup(Role::White, Piece::Man, 22, 15, NE);
    }
    // generating for white man 22 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal white (move man c 3 b 4))
        ddi->diagonals.emplace_back(17, 268);

        // to position 13, legal: (legal white (move man c 3 a 5))
        ddi->diagonals.emplace_back(13, 270);

        this->diagonal_data[21].push_back(ddi);


        this->reverse_legal_lookup_white[268] = ReverseLegalLookup(Role::White, Piece::Man, 22, 17, NW);
        this->reverse_legal_lookup_white[270] = ReverseLegalLookup(Role::White, Piece::Man, 22, 13, NW);
    }
    // generating for white man 22 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 26, legal: invalid
        ddi->diagonals.emplace_back(26, -1);

        // to position 31, legal: (legal white (move man c 3 e 1))
        ddi->diagonals.emplace_back(31, 271);

        this->diagonal_data[21].push_back(ddi);


        this->reverse_legal_lookup_white[271] = ReverseLegalLookup(Role::White, Piece::Man, 22, 31, SE);
    }
    // generating for white man 22 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 25, legal: invalid
        ddi->diagonals.emplace_back(25, -1);

        // to position 29, legal: (legal white (move man c 3 a 1))
        ddi->diagonals.emplace_back(29, 272);

        this->diagonal_data[21].push_back(ddi);


        this->reverse_legal_lookup_white[272] = ReverseLegalLookup(Role::White, Piece::Man, 22, 29, SW);
    }
    // generating for white man 23 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 19, legal: (legal white (move man e 3 f 4))
        ddi->diagonals.emplace_back(19, 284);

        // to position 16, legal: (legal white (move man e 3 g 5))
        ddi->diagonals.emplace_back(16, 286);

        this->diagonal_data[22].push_back(ddi);


        this->reverse_legal_lookup_white[284] = ReverseLegalLookup(Role::White, Piece::Man, 23, 19, NE);
        this->reverse_legal_lookup_white[286] = ReverseLegalLookup(Role::White, Piece::Man, 23, 16, NE);
    }
    // generating for white man 23 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 18, legal: (legal white (move man e 3 d 4))
        ddi->diagonals.emplace_back(18, 285);

        // to position 14, legal: (legal white (move man e 3 c 5))
        ddi->diagonals.emplace_back(14, 287);

        this->diagonal_data[22].push_back(ddi);


        this->reverse_legal_lookup_white[285] = ReverseLegalLookup(Role::White, Piece::Man, 23, 18, NW);
        this->reverse_legal_lookup_white[287] = ReverseLegalLookup(Role::White, Piece::Man, 23, 14, NW);
    }
    // generating for white man 23 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 27, legal: invalid
        ddi->diagonals.emplace_back(27, -1);

        // to position 32, legal: (legal white (move man e 3 g 1))
        ddi->diagonals.emplace_back(32, 288);

        this->diagonal_data[22].push_back(ddi);


        this->reverse_legal_lookup_white[288] = ReverseLegalLookup(Role::White, Piece::Man, 23, 32, SE);
    }
    // generating for white man 23 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 26, legal: invalid
        ddi->diagonals.emplace_back(26, -1);

        // to position 30, legal: (legal white (move man e 3 c 1))
        ddi->diagonals.emplace_back(30, 289);

        this->diagonal_data[22].push_back(ddi);


        this->reverse_legal_lookup_white[289] = ReverseLegalLookup(Role::White, Piece::Man, 23, 30, SW);
    }
    // generating for white man 24 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 20, legal: (legal white (move man g 3 h 4))
        ddi->diagonals.emplace_back(20, 301);

        this->diagonal_data[23].push_back(ddi);


        this->reverse_legal_lookup_white[301] = ReverseLegalLookup(Role::White, Piece::Man, 24, 20, NE);
    }
    // generating for white man 24 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 19, legal: (legal white (move man g 3 f 4))
        ddi->diagonals.emplace_back(19, 302);

        // to position 15, legal: (legal white (move man g 3 e 5))
        ddi->diagonals.emplace_back(15, 303);

        this->diagonal_data[23].push_back(ddi);


        this->reverse_legal_lookup_white[302] = ReverseLegalLookup(Role::White, Piece::Man, 24, 19, NW);
        this->reverse_legal_lookup_white[303] = ReverseLegalLookup(Role::White, Piece::Man, 24, 15, NW);
    }
    // generating for white man 24 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 28, legal: invalid
        ddi->diagonals.emplace_back(28, -1);

        this->diagonal_data[23].push_back(ddi);


    }
    // generating for white man 24 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 27, legal: invalid
        ddi->diagonals.emplace_back(27, -1);

        // to position 31, legal: (legal white (move man g 3 e 1))
        ddi->diagonals.emplace_back(31, 304);

        this->diagonal_data[23].push_back(ddi);


        this->reverse_legal_lookup_white[304] = ReverseLegalLookup(Role::White, Piece::Man, 24, 31, SW);
    }
    // generating for white man 25 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 22, legal: (legal white (move man b 2 c 3))
        ddi->diagonals.emplace_back(22, 314);

        // to position 18, legal: (legal white (move man b 2 d 4))
        ddi->diagonals.emplace_back(18, 316);

        this->diagonal_data[24].push_back(ddi);


        this->reverse_legal_lookup_white[314] = ReverseLegalLookup(Role::White, Piece::Man, 25, 22, NE);
        this->reverse_legal_lookup_white[316] = ReverseLegalLookup(Role::White, Piece::Man, 25, 18, NE);
    }
    // generating for white man 25 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 21, legal: (legal white (move man b 2 a 3))
        ddi->diagonals.emplace_back(21, 315);

        this->diagonal_data[24].push_back(ddi);


        this->reverse_legal_lookup_white[315] = ReverseLegalLookup(Role::White, Piece::Man, 25, 21, NW);
    }
    // generating for white man 25 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 30, legal: invalid
        ddi->diagonals.emplace_back(30, -1);

        this->diagonal_data[24].push_back(ddi);


    }
    // generating for white man 25 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 29, legal: invalid
        ddi->diagonals.emplace_back(29, -1);

        this->diagonal_data[24].push_back(ddi);


    }
    // generating for white man 26 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 23, legal: (legal white (move man d 2 e 3))
        ddi->diagonals.emplace_back(23, 326);

        // to position 19, legal: (legal white (move man d 2 f 4))
        ddi->diagonals.emplace_back(19, 328);

        this->diagonal_data[25].push_back(ddi);


        this->reverse_legal_lookup_white[326] = ReverseLegalLookup(Role::White, Piece::Man, 26, 23, NE);
        this->reverse_legal_lookup_white[328] = ReverseLegalLookup(Role::White, Piece::Man, 26, 19, NE);
    }
    // generating for white man 26 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 22, legal: (legal white (move man d 2 c 3))
        ddi->diagonals.emplace_back(22, 327);

        // to position 17, legal: (legal white (move man d 2 b 4))
        ddi->diagonals.emplace_back(17, 329);

        this->diagonal_data[25].push_back(ddi);


        this->reverse_legal_lookup_white[327] = ReverseLegalLookup(Role::White, Piece::Man, 26, 22, NW);
        this->reverse_legal_lookup_white[329] = ReverseLegalLookup(Role::White, Piece::Man, 26, 17, NW);
    }
    // generating for white man 26 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 31, legal: invalid
        ddi->diagonals.emplace_back(31, -1);

        this->diagonal_data[25].push_back(ddi);


    }
    // generating for white man 26 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 30, legal: invalid
        ddi->diagonals.emplace_back(30, -1);

        this->diagonal_data[25].push_back(ddi);


    }
    // generating for white man 27 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal white (move man f 2 g 3))
        ddi->diagonals.emplace_back(24, 339);

        // to position 20, legal: (legal white (move man f 2 h 4))
        ddi->diagonals.emplace_back(20, 341);

        this->diagonal_data[26].push_back(ddi);


        this->reverse_legal_lookup_white[339] = ReverseLegalLookup(Role::White, Piece::Man, 27, 24, NE);
        this->reverse_legal_lookup_white[341] = ReverseLegalLookup(Role::White, Piece::Man, 27, 20, NE);
    }
    // generating for white man 27 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 23, legal: (legal white (move man f 2 e 3))
        ddi->diagonals.emplace_back(23, 340);

        // to position 18, legal: (legal white (move man f 2 d 4))
        ddi->diagonals.emplace_back(18, 342);

        this->diagonal_data[26].push_back(ddi);


        this->reverse_legal_lookup_white[340] = ReverseLegalLookup(Role::White, Piece::Man, 27, 23, NW);
        this->reverse_legal_lookup_white[342] = ReverseLegalLookup(Role::White, Piece::Man, 27, 18, NW);
    }
    // generating for white man 27 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 32, legal: invalid
        ddi->diagonals.emplace_back(32, -1);

        this->diagonal_data[26].push_back(ddi);


    }
    // generating for white man 27 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 31, legal: invalid
        ddi->diagonals.emplace_back(31, -1);

        this->diagonal_data[26].push_back(ddi);


    }
    // generating for white man 28 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal white (move man h 2 g 3))
        ddi->diagonals.emplace_back(24, 352);

        // to position 19, legal: (legal white (move man h 2 f 4))
        ddi->diagonals.emplace_back(19, 353);

        this->diagonal_data[27].push_back(ddi);


        this->reverse_legal_lookup_white[352] = ReverseLegalLookup(Role::White, Piece::Man, 28, 24, NW);
        this->reverse_legal_lookup_white[353] = ReverseLegalLookup(Role::White, Piece::Man, 28, 19, NW);
    }
    // generating for white man 28 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 32, legal: invalid
        ddi->diagonals.emplace_back(32, -1);

        this->diagonal_data[27].push_back(ddi);


    }
    // generating for white man 29 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 25, legal: (legal white (move man a 1 b 2))
        ddi->diagonals.emplace_back(25, 361);

        // to position 22, legal: (legal white (move man a 1 c 3))
        ddi->diagonals.emplace_back(22, 362);

        this->diagonal_data[28].push_back(ddi);


        this->reverse_legal_lookup_white[361] = ReverseLegalLookup(Role::White, Piece::Man, 29, 25, NE);
        this->reverse_legal_lookup_white[362] = ReverseLegalLookup(Role::White, Piece::Man, 29, 22, NE);
    }
    // generating for white man 30 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 26, legal: (legal white (move man c 1 d 2))
        ddi->diagonals.emplace_back(26, 370);

        // to position 23, legal: (legal white (move man c 1 e 3))
        ddi->diagonals.emplace_back(23, 372);

        this->diagonal_data[29].push_back(ddi);


        this->reverse_legal_lookup_white[370] = ReverseLegalLookup(Role::White, Piece::Man, 30, 26, NE);
        this->reverse_legal_lookup_white[372] = ReverseLegalLookup(Role::White, Piece::Man, 30, 23, NE);
    }
    // generating for white man 30 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 25, legal: (legal white (move man c 1 b 2))
        ddi->diagonals.emplace_back(25, 371);

        // to position 21, legal: (legal white (move man c 1 a 3))
        ddi->diagonals.emplace_back(21, 373);

        this->diagonal_data[29].push_back(ddi);


        this->reverse_legal_lookup_white[371] = ReverseLegalLookup(Role::White, Piece::Man, 30, 25, NW);
        this->reverse_legal_lookup_white[373] = ReverseLegalLookup(Role::White, Piece::Man, 30, 21, NW);
    }
    // generating for white man 31 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal white (move man e 1 f 2))
        ddi->diagonals.emplace_back(27, 381);

        // to position 24, legal: (legal white (move man e 1 g 3))
        ddi->diagonals.emplace_back(24, 383);

        this->diagonal_data[30].push_back(ddi);


        this->reverse_legal_lookup_white[381] = ReverseLegalLookup(Role::White, Piece::Man, 31, 27, NE);
        this->reverse_legal_lookup_white[383] = ReverseLegalLookup(Role::White, Piece::Man, 31, 24, NE);
    }
    // generating for white man 31 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 26, legal: (legal white (move man e 1 d 2))
        ddi->diagonals.emplace_back(26, 382);

        // to position 22, legal: (legal white (move man e 1 c 3))
        ddi->diagonals.emplace_back(22, 384);

        this->diagonal_data[30].push_back(ddi);


        this->reverse_legal_lookup_white[382] = ReverseLegalLookup(Role::White, Piece::Man, 31, 26, NW);
        this->reverse_legal_lookup_white[384] = ReverseLegalLookup(Role::White, Piece::Man, 31, 22, NW);
    }
    // generating for white man 32 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 28, legal: (legal white (move man g 1 h 2))
        ddi->diagonals.emplace_back(28, 392);

        this->diagonal_data[31].push_back(ddi);


        this->reverse_legal_lookup_white[392] = ReverseLegalLookup(Role::White, Piece::Man, 32, 28, NE);
    }
    // generating for white man 32 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal white (move man g 1 f 2))
        ddi->diagonals.emplace_back(27, 393);

        // to position 23, legal: (legal white (move man g 1 e 3))
        ddi->diagonals.emplace_back(23, 394);

        this->diagonal_data[31].push_back(ddi);


        this->reverse_legal_lookup_white[393] = ReverseLegalLookup(Role::White, Piece::Man, 32, 27, NW);
        this->reverse_legal_lookup_white[394] = ReverseLegalLookup(Role::White, Piece::Man, 32, 23, NW);
    }
    // generating for white king 1 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 6, legal: (legal white (move king b 8 c 7))
        ddi->diagonals.emplace_back(6, 2);

        // to position 10, legal: (legal white (move king b 8 d 6))
        ddi->diagonals.emplace_back(10, 3);

        // to position 15, legal: (legal white (move king b 8 e 5))
        ddi->diagonals.emplace_back(15, 4);

        // to position 19, legal: (legal white (move king b 8 f 4))
        ddi->diagonals.emplace_back(19, 5);

        // to position 24, legal: (legal white (move king b 8 g 3))
        ddi->diagonals.emplace_back(24, 6);

        // to position 28, legal: (legal white (move king b 8 h 2))
        ddi->diagonals.emplace_back(28, 7);

        this->diagonal_data[32].push_back(ddi);


        this->reverse_legal_lookup_white[2] = ReverseLegalLookup(Role::White, Piece::King, 1, 6, SE);
        this->reverse_legal_lookup_white[3] = ReverseLegalLookup(Role::White, Piece::King, 1, 10, SE);
        this->reverse_legal_lookup_white[4] = ReverseLegalLookup(Role::White, Piece::King, 1, 15, SE);
        this->reverse_legal_lookup_white[5] = ReverseLegalLookup(Role::White, Piece::King, 1, 19, SE);
        this->reverse_legal_lookup_white[6] = ReverseLegalLookup(Role::White, Piece::King, 1, 24, SE);
        this->reverse_legal_lookup_white[7] = ReverseLegalLookup(Role::White, Piece::King, 1, 28, SE);
    }
    // generating for white king 1 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 5, legal: (legal white (move king b 8 a 7))
        ddi->diagonals.emplace_back(5, 8);

        this->diagonal_data[32].push_back(ddi);


        this->reverse_legal_lookup_white[8] = ReverseLegalLookup(Role::White, Piece::King, 1, 5, SW);
    }
    // generating for white king 2 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 7, legal: (legal white (move king d 8 e 7))
        ddi->diagonals.emplace_back(7, 11);

        // to position 11, legal: (legal white (move king d 8 f 6))
        ddi->diagonals.emplace_back(11, 12);

        // to position 16, legal: (legal white (move king d 8 g 5))
        ddi->diagonals.emplace_back(16, 13);

        // to position 20, legal: (legal white (move king d 8 h 4))
        ddi->diagonals.emplace_back(20, 14);

        this->diagonal_data[33].push_back(ddi);


        this->reverse_legal_lookup_white[11] = ReverseLegalLookup(Role::White, Piece::King, 2, 7, SE);
        this->reverse_legal_lookup_white[12] = ReverseLegalLookup(Role::White, Piece::King, 2, 11, SE);
        this->reverse_legal_lookup_white[13] = ReverseLegalLookup(Role::White, Piece::King, 2, 16, SE);
        this->reverse_legal_lookup_white[14] = ReverseLegalLookup(Role::White, Piece::King, 2, 20, SE);
    }
    // generating for white king 2 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 6, legal: (legal white (move king d 8 c 7))
        ddi->diagonals.emplace_back(6, 15);

        // to position 9, legal: (legal white (move king d 8 b 6))
        ddi->diagonals.emplace_back(9, 16);

        // to position 13, legal: (legal white (move king d 8 a 5))
        ddi->diagonals.emplace_back(13, 17);

        this->diagonal_data[33].push_back(ddi);


        this->reverse_legal_lookup_white[15] = ReverseLegalLookup(Role::White, Piece::King, 2, 6, SW);
        this->reverse_legal_lookup_white[16] = ReverseLegalLookup(Role::White, Piece::King, 2, 9, SW);
        this->reverse_legal_lookup_white[17] = ReverseLegalLookup(Role::White, Piece::King, 2, 13, SW);
    }
    // generating for white king 3 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal white (move king f 8 g 7))
        ddi->diagonals.emplace_back(8, 20);

        // to position 12, legal: (legal white (move king f 8 h 6))
        ddi->diagonals.emplace_back(12, 21);

        this->diagonal_data[34].push_back(ddi);


        this->reverse_legal_lookup_white[20] = ReverseLegalLookup(Role::White, Piece::King, 3, 8, SE);
        this->reverse_legal_lookup_white[21] = ReverseLegalLookup(Role::White, Piece::King, 3, 12, SE);
    }
    // generating for white king 3 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 7, legal: (legal white (move king f 8 e 7))
        ddi->diagonals.emplace_back(7, 22);

        // to position 10, legal: (legal white (move king f 8 d 6))
        ddi->diagonals.emplace_back(10, 23);

        // to position 14, legal: (legal white (move king f 8 c 5))
        ddi->diagonals.emplace_back(14, 24);

        // to position 17, legal: (legal white (move king f 8 b 4))
        ddi->diagonals.emplace_back(17, 25);

        // to position 21, legal: (legal white (move king f 8 a 3))
        ddi->diagonals.emplace_back(21, 26);

        this->diagonal_data[34].push_back(ddi);


        this->reverse_legal_lookup_white[22] = ReverseLegalLookup(Role::White, Piece::King, 3, 7, SW);
        this->reverse_legal_lookup_white[23] = ReverseLegalLookup(Role::White, Piece::King, 3, 10, SW);
        this->reverse_legal_lookup_white[24] = ReverseLegalLookup(Role::White, Piece::King, 3, 14, SW);
        this->reverse_legal_lookup_white[25] = ReverseLegalLookup(Role::White, Piece::King, 3, 17, SW);
        this->reverse_legal_lookup_white[26] = ReverseLegalLookup(Role::White, Piece::King, 3, 21, SW);
    }
    // generating for white king 4 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(7);
        // to position 8, legal: (legal white (move king h 8 g 7))
        ddi->diagonals.emplace_back(8, 28);

        // to position 11, legal: (legal white (move king h 8 f 6))
        ddi->diagonals.emplace_back(11, 29);

        // to position 15, legal: (legal white (move king h 8 e 5))
        ddi->diagonals.emplace_back(15, 30);

        // to position 18, legal: (legal white (move king h 8 d 4))
        ddi->diagonals.emplace_back(18, 31);

        // to position 22, legal: (legal white (move king h 8 c 3))
        ddi->diagonals.emplace_back(22, 32);

        // to position 25, legal: (legal white (move king h 8 b 2))
        ddi->diagonals.emplace_back(25, 33);

        // to position 29, legal: (legal white (move king h 8 a 1))
        ddi->diagonals.emplace_back(29, 34);

        this->diagonal_data[35].push_back(ddi);


        this->reverse_legal_lookup_white[28] = ReverseLegalLookup(Role::White, Piece::King, 4, 8, SW);
        this->reverse_legal_lookup_white[29] = ReverseLegalLookup(Role::White, Piece::King, 4, 11, SW);
        this->reverse_legal_lookup_white[30] = ReverseLegalLookup(Role::White, Piece::King, 4, 15, SW);
        this->reverse_legal_lookup_white[31] = ReverseLegalLookup(Role::White, Piece::King, 4, 18, SW);
        this->reverse_legal_lookup_white[32] = ReverseLegalLookup(Role::White, Piece::King, 4, 22, SW);
        this->reverse_legal_lookup_white[33] = ReverseLegalLookup(Role::White, Piece::King, 4, 25, SW);
        this->reverse_legal_lookup_white[34] = ReverseLegalLookup(Role::White, Piece::King, 4, 29, SW);
    }
    // generating for white king 5 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal white (move king a 7 b 8))
        ddi->diagonals.emplace_back(1, 37);

        this->diagonal_data[36].push_back(ddi);


        this->reverse_legal_lookup_white[37] = ReverseLegalLookup(Role::White, Piece::King, 5, 1, NE);
    }
    // generating for white king 5 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 9, legal: (legal white (move king a 7 b 6))
        ddi->diagonals.emplace_back(9, 38);

        // to position 14, legal: (legal white (move king a 7 c 5))
        ddi->diagonals.emplace_back(14, 39);

        // to position 18, legal: (legal white (move king a 7 d 4))
        ddi->diagonals.emplace_back(18, 40);

        // to position 23, legal: (legal white (move king a 7 e 3))
        ddi->diagonals.emplace_back(23, 41);

        // to position 27, legal: (legal white (move king a 7 f 2))
        ddi->diagonals.emplace_back(27, 42);

        // to position 32, legal: (legal white (move king a 7 g 1))
        ddi->diagonals.emplace_back(32, 43);

        this->diagonal_data[36].push_back(ddi);


        this->reverse_legal_lookup_white[38] = ReverseLegalLookup(Role::White, Piece::King, 5, 9, SE);
        this->reverse_legal_lookup_white[39] = ReverseLegalLookup(Role::White, Piece::King, 5, 14, SE);
        this->reverse_legal_lookup_white[40] = ReverseLegalLookup(Role::White, Piece::King, 5, 18, SE);
        this->reverse_legal_lookup_white[41] = ReverseLegalLookup(Role::White, Piece::King, 5, 23, SE);
        this->reverse_legal_lookup_white[42] = ReverseLegalLookup(Role::White, Piece::King, 5, 27, SE);
        this->reverse_legal_lookup_white[43] = ReverseLegalLookup(Role::White, Piece::King, 5, 32, SE);
    }
    // generating for white king 6 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal white (move king c 7 d 8))
        ddi->diagonals.emplace_back(2, 48);

        this->diagonal_data[37].push_back(ddi);


        this->reverse_legal_lookup_white[48] = ReverseLegalLookup(Role::White, Piece::King, 6, 2, NE);
    }
    // generating for white king 6 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal white (move king c 7 b 8))
        ddi->diagonals.emplace_back(1, 49);

        this->diagonal_data[37].push_back(ddi);


        this->reverse_legal_lookup_white[49] = ReverseLegalLookup(Role::White, Piece::King, 6, 1, NW);
    }
    // generating for white king 6 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 10, legal: (legal white (move king c 7 d 6))
        ddi->diagonals.emplace_back(10, 50);

        // to position 15, legal: (legal white (move king c 7 e 5))
        ddi->diagonals.emplace_back(15, 51);

        // to position 19, legal: (legal white (move king c 7 f 4))
        ddi->diagonals.emplace_back(19, 52);

        // to position 24, legal: (legal white (move king c 7 g 3))
        ddi->diagonals.emplace_back(24, 53);

        // to position 28, legal: (legal white (move king c 7 h 2))
        ddi->diagonals.emplace_back(28, 54);

        this->diagonal_data[37].push_back(ddi);


        this->reverse_legal_lookup_white[50] = ReverseLegalLookup(Role::White, Piece::King, 6, 10, SE);
        this->reverse_legal_lookup_white[51] = ReverseLegalLookup(Role::White, Piece::King, 6, 15, SE);
        this->reverse_legal_lookup_white[52] = ReverseLegalLookup(Role::White, Piece::King, 6, 19, SE);
        this->reverse_legal_lookup_white[53] = ReverseLegalLookup(Role::White, Piece::King, 6, 24, SE);
        this->reverse_legal_lookup_white[54] = ReverseLegalLookup(Role::White, Piece::King, 6, 28, SE);
    }
    // generating for white king 6 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal white (move king c 7 b 6))
        ddi->diagonals.emplace_back(9, 55);

        // to position 13, legal: (legal white (move king c 7 a 5))
        ddi->diagonals.emplace_back(13, 56);

        this->diagonal_data[37].push_back(ddi);


        this->reverse_legal_lookup_white[55] = ReverseLegalLookup(Role::White, Piece::King, 6, 9, SW);
        this->reverse_legal_lookup_white[56] = ReverseLegalLookup(Role::White, Piece::King, 6, 13, SW);
    }
    // generating for white king 7 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal white (move king e 7 f 8))
        ddi->diagonals.emplace_back(3, 61);

        this->diagonal_data[38].push_back(ddi);


        this->reverse_legal_lookup_white[61] = ReverseLegalLookup(Role::White, Piece::King, 7, 3, NE);
    }
    // generating for white king 7 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal white (move king e 7 d 8))
        ddi->diagonals.emplace_back(2, 62);

        this->diagonal_data[38].push_back(ddi);


        this->reverse_legal_lookup_white[62] = ReverseLegalLookup(Role::White, Piece::King, 7, 2, NW);
    }
    // generating for white king 7 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 11, legal: (legal white (move king e 7 f 6))
        ddi->diagonals.emplace_back(11, 63);

        // to position 16, legal: (legal white (move king e 7 g 5))
        ddi->diagonals.emplace_back(16, 64);

        // to position 20, legal: (legal white (move king e 7 h 4))
        ddi->diagonals.emplace_back(20, 65);

        this->diagonal_data[38].push_back(ddi);


        this->reverse_legal_lookup_white[63] = ReverseLegalLookup(Role::White, Piece::King, 7, 11, SE);
        this->reverse_legal_lookup_white[64] = ReverseLegalLookup(Role::White, Piece::King, 7, 16, SE);
        this->reverse_legal_lookup_white[65] = ReverseLegalLookup(Role::White, Piece::King, 7, 20, SE);
    }
    // generating for white king 7 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 10, legal: (legal white (move king e 7 d 6))
        ddi->diagonals.emplace_back(10, 66);

        // to position 14, legal: (legal white (move king e 7 c 5))
        ddi->diagonals.emplace_back(14, 67);

        // to position 17, legal: (legal white (move king e 7 b 4))
        ddi->diagonals.emplace_back(17, 68);

        // to position 21, legal: (legal white (move king e 7 a 3))
        ddi->diagonals.emplace_back(21, 69);

        this->diagonal_data[38].push_back(ddi);


        this->reverse_legal_lookup_white[66] = ReverseLegalLookup(Role::White, Piece::King, 7, 10, SW);
        this->reverse_legal_lookup_white[67] = ReverseLegalLookup(Role::White, Piece::King, 7, 14, SW);
        this->reverse_legal_lookup_white[68] = ReverseLegalLookup(Role::White, Piece::King, 7, 17, SW);
        this->reverse_legal_lookup_white[69] = ReverseLegalLookup(Role::White, Piece::King, 7, 21, SW);
    }
    // generating for white king 8 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 4, legal: (legal white (move king g 7 h 8))
        ddi->diagonals.emplace_back(4, 73);

        this->diagonal_data[39].push_back(ddi);


        this->reverse_legal_lookup_white[73] = ReverseLegalLookup(Role::White, Piece::King, 8, 4, NE);
    }
    // generating for white king 8 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal white (move king g 7 f 8))
        ddi->diagonals.emplace_back(3, 74);

        this->diagonal_data[39].push_back(ddi);


        this->reverse_legal_lookup_white[74] = ReverseLegalLookup(Role::White, Piece::King, 8, 3, NW);
    }
    // generating for white king 8 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 12, legal: (legal white (move king g 7 h 6))
        ddi->diagonals.emplace_back(12, 75);

        this->diagonal_data[39].push_back(ddi);


        this->reverse_legal_lookup_white[75] = ReverseLegalLookup(Role::White, Piece::King, 8, 12, SE);
    }
    // generating for white king 8 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(6);
        // to position 11, legal: (legal white (move king g 7 f 6))
        ddi->diagonals.emplace_back(11, 76);

        // to position 15, legal: (legal white (move king g 7 e 5))
        ddi->diagonals.emplace_back(15, 77);

        // to position 18, legal: (legal white (move king g 7 d 4))
        ddi->diagonals.emplace_back(18, 78);

        // to position 22, legal: (legal white (move king g 7 c 3))
        ddi->diagonals.emplace_back(22, 79);

        // to position 25, legal: (legal white (move king g 7 b 2))
        ddi->diagonals.emplace_back(25, 80);

        // to position 29, legal: (legal white (move king g 7 a 1))
        ddi->diagonals.emplace_back(29, 81);

        this->diagonal_data[39].push_back(ddi);


        this->reverse_legal_lookup_white[76] = ReverseLegalLookup(Role::White, Piece::King, 8, 11, SW);
        this->reverse_legal_lookup_white[77] = ReverseLegalLookup(Role::White, Piece::King, 8, 15, SW);
        this->reverse_legal_lookup_white[78] = ReverseLegalLookup(Role::White, Piece::King, 8, 18, SW);
        this->reverse_legal_lookup_white[79] = ReverseLegalLookup(Role::White, Piece::King, 8, 22, SW);
        this->reverse_legal_lookup_white[80] = ReverseLegalLookup(Role::White, Piece::King, 8, 25, SW);
        this->reverse_legal_lookup_white[81] = ReverseLegalLookup(Role::White, Piece::King, 8, 29, SW);
    }
    // generating for white king 9 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 6, legal: (legal white (move king b 6 c 7))
        ddi->diagonals.emplace_back(6, 86);

        // to position 2, legal: (legal white (move king b 6 d 8))
        ddi->diagonals.emplace_back(2, 87);

        this->diagonal_data[40].push_back(ddi);


        this->reverse_legal_lookup_white[86] = ReverseLegalLookup(Role::White, Piece::King, 9, 6, NE);
        this->reverse_legal_lookup_white[87] = ReverseLegalLookup(Role::White, Piece::King, 9, 2, NE);
    }
    // generating for white king 9 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 5, legal: (legal white (move king b 6 a 7))
        ddi->diagonals.emplace_back(5, 88);

        this->diagonal_data[40].push_back(ddi);


        this->reverse_legal_lookup_white[88] = ReverseLegalLookup(Role::White, Piece::King, 9, 5, NW);
    }
    // generating for white king 9 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 14, legal: (legal white (move king b 6 c 5))
        ddi->diagonals.emplace_back(14, 89);

        // to position 18, legal: (legal white (move king b 6 d 4))
        ddi->diagonals.emplace_back(18, 90);

        // to position 23, legal: (legal white (move king b 6 e 3))
        ddi->diagonals.emplace_back(23, 91);

        // to position 27, legal: (legal white (move king b 6 f 2))
        ddi->diagonals.emplace_back(27, 92);

        // to position 32, legal: (legal white (move king b 6 g 1))
        ddi->diagonals.emplace_back(32, 93);

        this->diagonal_data[40].push_back(ddi);


        this->reverse_legal_lookup_white[89] = ReverseLegalLookup(Role::White, Piece::King, 9, 14, SE);
        this->reverse_legal_lookup_white[90] = ReverseLegalLookup(Role::White, Piece::King, 9, 18, SE);
        this->reverse_legal_lookup_white[91] = ReverseLegalLookup(Role::White, Piece::King, 9, 23, SE);
        this->reverse_legal_lookup_white[92] = ReverseLegalLookup(Role::White, Piece::King, 9, 27, SE);
        this->reverse_legal_lookup_white[93] = ReverseLegalLookup(Role::White, Piece::King, 9, 32, SE);
    }
    // generating for white king 9 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 13, legal: (legal white (move king b 6 a 5))
        ddi->diagonals.emplace_back(13, 94);

        this->diagonal_data[40].push_back(ddi);


        this->reverse_legal_lookup_white[94] = ReverseLegalLookup(Role::White, Piece::King, 9, 13, SW);
    }
    // generating for white king 10 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal white (move king d 6 e 7))
        ddi->diagonals.emplace_back(7, 101);

        // to position 3, legal: (legal white (move king d 6 f 8))
        ddi->diagonals.emplace_back(3, 102);

        this->diagonal_data[41].push_back(ddi);


        this->reverse_legal_lookup_white[101] = ReverseLegalLookup(Role::White, Piece::King, 10, 7, NE);
        this->reverse_legal_lookup_white[102] = ReverseLegalLookup(Role::White, Piece::King, 10, 3, NE);
    }
    // generating for white king 10 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 6, legal: (legal white (move king d 6 c 7))
        ddi->diagonals.emplace_back(6, 103);

        // to position 1, legal: (legal white (move king d 6 b 8))
        ddi->diagonals.emplace_back(1, 104);

        this->diagonal_data[41].push_back(ddi);


        this->reverse_legal_lookup_white[103] = ReverseLegalLookup(Role::White, Piece::King, 10, 6, NW);
        this->reverse_legal_lookup_white[104] = ReverseLegalLookup(Role::White, Piece::King, 10, 1, NW);
    }
    // generating for white king 10 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 15, legal: (legal white (move king d 6 e 5))
        ddi->diagonals.emplace_back(15, 105);

        // to position 19, legal: (legal white (move king d 6 f 4))
        ddi->diagonals.emplace_back(19, 106);

        // to position 24, legal: (legal white (move king d 6 g 3))
        ddi->diagonals.emplace_back(24, 107);

        // to position 28, legal: (legal white (move king d 6 h 2))
        ddi->diagonals.emplace_back(28, 108);

        this->diagonal_data[41].push_back(ddi);


        this->reverse_legal_lookup_white[105] = ReverseLegalLookup(Role::White, Piece::King, 10, 15, SE);
        this->reverse_legal_lookup_white[106] = ReverseLegalLookup(Role::White, Piece::King, 10, 19, SE);
        this->reverse_legal_lookup_white[107] = ReverseLegalLookup(Role::White, Piece::King, 10, 24, SE);
        this->reverse_legal_lookup_white[108] = ReverseLegalLookup(Role::White, Piece::King, 10, 28, SE);
    }
    // generating for white king 10 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 14, legal: (legal white (move king d 6 c 5))
        ddi->diagonals.emplace_back(14, 109);

        // to position 17, legal: (legal white (move king d 6 b 4))
        ddi->diagonals.emplace_back(17, 110);

        // to position 21, legal: (legal white (move king d 6 a 3))
        ddi->diagonals.emplace_back(21, 111);

        this->diagonal_data[41].push_back(ddi);


        this->reverse_legal_lookup_white[109] = ReverseLegalLookup(Role::White, Piece::King, 10, 14, SW);
        this->reverse_legal_lookup_white[110] = ReverseLegalLookup(Role::White, Piece::King, 10, 17, SW);
        this->reverse_legal_lookup_white[111] = ReverseLegalLookup(Role::White, Piece::King, 10, 21, SW);
    }
    // generating for white king 11 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal white (move king f 6 g 7))
        ddi->diagonals.emplace_back(8, 118);

        // to position 4, legal: (legal white (move king f 6 h 8))
        ddi->diagonals.emplace_back(4, 119);

        this->diagonal_data[42].push_back(ddi);


        this->reverse_legal_lookup_white[118] = ReverseLegalLookup(Role::White, Piece::King, 11, 8, NE);
        this->reverse_legal_lookup_white[119] = ReverseLegalLookup(Role::White, Piece::King, 11, 4, NE);
    }
    // generating for white king 11 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal white (move king f 6 e 7))
        ddi->diagonals.emplace_back(7, 120);

        // to position 2, legal: (legal white (move king f 6 d 8))
        ddi->diagonals.emplace_back(2, 121);

        this->diagonal_data[42].push_back(ddi);


        this->reverse_legal_lookup_white[120] = ReverseLegalLookup(Role::White, Piece::King, 11, 7, NW);
        this->reverse_legal_lookup_white[121] = ReverseLegalLookup(Role::White, Piece::King, 11, 2, NW);
    }
    // generating for white king 11 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 16, legal: (legal white (move king f 6 g 5))
        ddi->diagonals.emplace_back(16, 122);

        // to position 20, legal: (legal white (move king f 6 h 4))
        ddi->diagonals.emplace_back(20, 123);

        this->diagonal_data[42].push_back(ddi);


        this->reverse_legal_lookup_white[122] = ReverseLegalLookup(Role::White, Piece::King, 11, 16, SE);
        this->reverse_legal_lookup_white[123] = ReverseLegalLookup(Role::White, Piece::King, 11, 20, SE);
    }
    // generating for white king 11 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 15, legal: (legal white (move king f 6 e 5))
        ddi->diagonals.emplace_back(15, 124);

        // to position 18, legal: (legal white (move king f 6 d 4))
        ddi->diagonals.emplace_back(18, 125);

        // to position 22, legal: (legal white (move king f 6 c 3))
        ddi->diagonals.emplace_back(22, 126);

        // to position 25, legal: (legal white (move king f 6 b 2))
        ddi->diagonals.emplace_back(25, 127);

        // to position 29, legal: (legal white (move king f 6 a 1))
        ddi->diagonals.emplace_back(29, 128);

        this->diagonal_data[42].push_back(ddi);


        this->reverse_legal_lookup_white[124] = ReverseLegalLookup(Role::White, Piece::King, 11, 15, SW);
        this->reverse_legal_lookup_white[125] = ReverseLegalLookup(Role::White, Piece::King, 11, 18, SW);
        this->reverse_legal_lookup_white[126] = ReverseLegalLookup(Role::White, Piece::King, 11, 22, SW);
        this->reverse_legal_lookup_white[127] = ReverseLegalLookup(Role::White, Piece::King, 11, 25, SW);
        this->reverse_legal_lookup_white[128] = ReverseLegalLookup(Role::White, Piece::King, 11, 29, SW);
    }
    // generating for white king 12 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal white (move king h 6 g 7))
        ddi->diagonals.emplace_back(8, 132);

        // to position 3, legal: (legal white (move king h 6 f 8))
        ddi->diagonals.emplace_back(3, 133);

        this->diagonal_data[43].push_back(ddi);


        this->reverse_legal_lookup_white[132] = ReverseLegalLookup(Role::White, Piece::King, 12, 8, NW);
        this->reverse_legal_lookup_white[133] = ReverseLegalLookup(Role::White, Piece::King, 12, 3, NW);
    }
    // generating for white king 12 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 16, legal: (legal white (move king h 6 g 5))
        ddi->diagonals.emplace_back(16, 134);

        // to position 19, legal: (legal white (move king h 6 f 4))
        ddi->diagonals.emplace_back(19, 135);

        // to position 23, legal: (legal white (move king h 6 e 3))
        ddi->diagonals.emplace_back(23, 136);

        // to position 26, legal: (legal white (move king h 6 d 2))
        ddi->diagonals.emplace_back(26, 137);

        // to position 30, legal: (legal white (move king h 6 c 1))
        ddi->diagonals.emplace_back(30, 138);

        this->diagonal_data[43].push_back(ddi);


        this->reverse_legal_lookup_white[134] = ReverseLegalLookup(Role::White, Piece::King, 12, 16, SW);
        this->reverse_legal_lookup_white[135] = ReverseLegalLookup(Role::White, Piece::King, 12, 19, SW);
        this->reverse_legal_lookup_white[136] = ReverseLegalLookup(Role::White, Piece::King, 12, 23, SW);
        this->reverse_legal_lookup_white[137] = ReverseLegalLookup(Role::White, Piece::King, 12, 26, SW);
        this->reverse_legal_lookup_white[138] = ReverseLegalLookup(Role::White, Piece::King, 12, 30, SW);
    }
    // generating for white king 13 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 9, legal: (legal white (move king a 5 b 6))
        ddi->diagonals.emplace_back(9, 142);

        // to position 6, legal: (legal white (move king a 5 c 7))
        ddi->diagonals.emplace_back(6, 143);

        // to position 2, legal: (legal white (move king a 5 d 8))
        ddi->diagonals.emplace_back(2, 144);

        this->diagonal_data[44].push_back(ddi);


        this->reverse_legal_lookup_white[142] = ReverseLegalLookup(Role::White, Piece::King, 13, 9, NE);
        this->reverse_legal_lookup_white[143] = ReverseLegalLookup(Role::White, Piece::King, 13, 6, NE);
        this->reverse_legal_lookup_white[144] = ReverseLegalLookup(Role::White, Piece::King, 13, 2, NE);
    }
    // generating for white king 13 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 17, legal: (legal white (move king a 5 b 4))
        ddi->diagonals.emplace_back(17, 145);

        // to position 22, legal: (legal white (move king a 5 c 3))
        ddi->diagonals.emplace_back(22, 146);

        // to position 26, legal: (legal white (move king a 5 d 2))
        ddi->diagonals.emplace_back(26, 147);

        // to position 31, legal: (legal white (move king a 5 e 1))
        ddi->diagonals.emplace_back(31, 148);

        this->diagonal_data[44].push_back(ddi);


        this->reverse_legal_lookup_white[145] = ReverseLegalLookup(Role::White, Piece::King, 13, 17, SE);
        this->reverse_legal_lookup_white[146] = ReverseLegalLookup(Role::White, Piece::King, 13, 22, SE);
        this->reverse_legal_lookup_white[147] = ReverseLegalLookup(Role::White, Piece::King, 13, 26, SE);
        this->reverse_legal_lookup_white[148] = ReverseLegalLookup(Role::White, Piece::King, 13, 31, SE);
    }
    // generating for white king 14 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 10, legal: (legal white (move king c 5 d 6))
        ddi->diagonals.emplace_back(10, 155);

        // to position 7, legal: (legal white (move king c 5 e 7))
        ddi->diagonals.emplace_back(7, 156);

        // to position 3, legal: (legal white (move king c 5 f 8))
        ddi->diagonals.emplace_back(3, 157);

        this->diagonal_data[45].push_back(ddi);


        this->reverse_legal_lookup_white[155] = ReverseLegalLookup(Role::White, Piece::King, 14, 10, NE);
        this->reverse_legal_lookup_white[156] = ReverseLegalLookup(Role::White, Piece::King, 14, 7, NE);
        this->reverse_legal_lookup_white[157] = ReverseLegalLookup(Role::White, Piece::King, 14, 3, NE);
    }
    // generating for white king 14 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal white (move king c 5 b 6))
        ddi->diagonals.emplace_back(9, 158);

        // to position 5, legal: (legal white (move king c 5 a 7))
        ddi->diagonals.emplace_back(5, 159);

        this->diagonal_data[45].push_back(ddi);


        this->reverse_legal_lookup_white[158] = ReverseLegalLookup(Role::White, Piece::King, 14, 9, NW);
        this->reverse_legal_lookup_white[159] = ReverseLegalLookup(Role::White, Piece::King, 14, 5, NW);
    }
    // generating for white king 14 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 18, legal: (legal white (move king c 5 d 4))
        ddi->diagonals.emplace_back(18, 160);

        // to position 23, legal: (legal white (move king c 5 e 3))
        ddi->diagonals.emplace_back(23, 161);

        // to position 27, legal: (legal white (move king c 5 f 2))
        ddi->diagonals.emplace_back(27, 162);

        // to position 32, legal: (legal white (move king c 5 g 1))
        ddi->diagonals.emplace_back(32, 163);

        this->diagonal_data[45].push_back(ddi);


        this->reverse_legal_lookup_white[160] = ReverseLegalLookup(Role::White, Piece::King, 14, 18, SE);
        this->reverse_legal_lookup_white[161] = ReverseLegalLookup(Role::White, Piece::King, 14, 23, SE);
        this->reverse_legal_lookup_white[162] = ReverseLegalLookup(Role::White, Piece::King, 14, 27, SE);
        this->reverse_legal_lookup_white[163] = ReverseLegalLookup(Role::White, Piece::King, 14, 32, SE);
    }
    // generating for white king 14 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal white (move king c 5 b 4))
        ddi->diagonals.emplace_back(17, 164);

        // to position 21, legal: (legal white (move king c 5 a 3))
        ddi->diagonals.emplace_back(21, 165);

        this->diagonal_data[45].push_back(ddi);


        this->reverse_legal_lookup_white[164] = ReverseLegalLookup(Role::White, Piece::King, 14, 17, SW);
        this->reverse_legal_lookup_white[165] = ReverseLegalLookup(Role::White, Piece::King, 14, 21, SW);
    }
    // generating for white king 15 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 11, legal: (legal white (move king e 5 f 6))
        ddi->diagonals.emplace_back(11, 172);

        // to position 8, legal: (legal white (move king e 5 g 7))
        ddi->diagonals.emplace_back(8, 173);

        // to position 4, legal: (legal white (move king e 5 h 8))
        ddi->diagonals.emplace_back(4, 174);

        this->diagonal_data[46].push_back(ddi);


        this->reverse_legal_lookup_white[172] = ReverseLegalLookup(Role::White, Piece::King, 15, 11, NE);
        this->reverse_legal_lookup_white[173] = ReverseLegalLookup(Role::White, Piece::King, 15, 8, NE);
        this->reverse_legal_lookup_white[174] = ReverseLegalLookup(Role::White, Piece::King, 15, 4, NE);
    }
    // generating for white king 15 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 10, legal: (legal white (move king e 5 d 6))
        ddi->diagonals.emplace_back(10, 175);

        // to position 6, legal: (legal white (move king e 5 c 7))
        ddi->diagonals.emplace_back(6, 176);

        // to position 1, legal: (legal white (move king e 5 b 8))
        ddi->diagonals.emplace_back(1, 177);

        this->diagonal_data[46].push_back(ddi);


        this->reverse_legal_lookup_white[175] = ReverseLegalLookup(Role::White, Piece::King, 15, 10, NW);
        this->reverse_legal_lookup_white[176] = ReverseLegalLookup(Role::White, Piece::King, 15, 6, NW);
        this->reverse_legal_lookup_white[177] = ReverseLegalLookup(Role::White, Piece::King, 15, 1, NW);
    }
    // generating for white king 15 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 19, legal: (legal white (move king e 5 f 4))
        ddi->diagonals.emplace_back(19, 178);

        // to position 24, legal: (legal white (move king e 5 g 3))
        ddi->diagonals.emplace_back(24, 179);

        // to position 28, legal: (legal white (move king e 5 h 2))
        ddi->diagonals.emplace_back(28, 180);

        this->diagonal_data[46].push_back(ddi);


        this->reverse_legal_lookup_white[178] = ReverseLegalLookup(Role::White, Piece::King, 15, 19, SE);
        this->reverse_legal_lookup_white[179] = ReverseLegalLookup(Role::White, Piece::King, 15, 24, SE);
        this->reverse_legal_lookup_white[180] = ReverseLegalLookup(Role::White, Piece::King, 15, 28, SE);
    }
    // generating for white king 15 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 18, legal: (legal white (move king e 5 d 4))
        ddi->diagonals.emplace_back(18, 181);

        // to position 22, legal: (legal white (move king e 5 c 3))
        ddi->diagonals.emplace_back(22, 182);

        // to position 25, legal: (legal white (move king e 5 b 2))
        ddi->diagonals.emplace_back(25, 183);

        // to position 29, legal: (legal white (move king e 5 a 1))
        ddi->diagonals.emplace_back(29, 184);

        this->diagonal_data[46].push_back(ddi);


        this->reverse_legal_lookup_white[181] = ReverseLegalLookup(Role::White, Piece::King, 15, 18, SW);
        this->reverse_legal_lookup_white[182] = ReverseLegalLookup(Role::White, Piece::King, 15, 22, SW);
        this->reverse_legal_lookup_white[183] = ReverseLegalLookup(Role::White, Piece::King, 15, 25, SW);
        this->reverse_legal_lookup_white[184] = ReverseLegalLookup(Role::White, Piece::King, 15, 29, SW);
    }
    // generating for white king 16 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 12, legal: (legal white (move king g 5 h 6))
        ddi->diagonals.emplace_back(12, 189);

        this->diagonal_data[47].push_back(ddi);


        this->reverse_legal_lookup_white[189] = ReverseLegalLookup(Role::White, Piece::King, 16, 12, NE);
    }
    // generating for white king 16 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 11, legal: (legal white (move king g 5 f 6))
        ddi->diagonals.emplace_back(11, 190);

        // to position 7, legal: (legal white (move king g 5 e 7))
        ddi->diagonals.emplace_back(7, 191);

        // to position 2, legal: (legal white (move king g 5 d 8))
        ddi->diagonals.emplace_back(2, 192);

        this->diagonal_data[47].push_back(ddi);


        this->reverse_legal_lookup_white[190] = ReverseLegalLookup(Role::White, Piece::King, 16, 11, NW);
        this->reverse_legal_lookup_white[191] = ReverseLegalLookup(Role::White, Piece::King, 16, 7, NW);
        this->reverse_legal_lookup_white[192] = ReverseLegalLookup(Role::White, Piece::King, 16, 2, NW);
    }
    // generating for white king 16 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 20, legal: (legal white (move king g 5 h 4))
        ddi->diagonals.emplace_back(20, 193);

        this->diagonal_data[47].push_back(ddi);


        this->reverse_legal_lookup_white[193] = ReverseLegalLookup(Role::White, Piece::King, 16, 20, SE);
    }
    // generating for white king 16 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 19, legal: (legal white (move king g 5 f 4))
        ddi->diagonals.emplace_back(19, 194);

        // to position 23, legal: (legal white (move king g 5 e 3))
        ddi->diagonals.emplace_back(23, 195);

        // to position 26, legal: (legal white (move king g 5 d 2))
        ddi->diagonals.emplace_back(26, 196);

        // to position 30, legal: (legal white (move king g 5 c 1))
        ddi->diagonals.emplace_back(30, 197);

        this->diagonal_data[47].push_back(ddi);


        this->reverse_legal_lookup_white[194] = ReverseLegalLookup(Role::White, Piece::King, 16, 19, SW);
        this->reverse_legal_lookup_white[195] = ReverseLegalLookup(Role::White, Piece::King, 16, 23, SW);
        this->reverse_legal_lookup_white[196] = ReverseLegalLookup(Role::White, Piece::King, 16, 26, SW);
        this->reverse_legal_lookup_white[197] = ReverseLegalLookup(Role::White, Piece::King, 16, 30, SW);
    }
    // generating for white king 17 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 14, legal: (legal white (move king b 4 c 5))
        ddi->diagonals.emplace_back(14, 202);

        // to position 10, legal: (legal white (move king b 4 d 6))
        ddi->diagonals.emplace_back(10, 203);

        // to position 7, legal: (legal white (move king b 4 e 7))
        ddi->diagonals.emplace_back(7, 204);

        // to position 3, legal: (legal white (move king b 4 f 8))
        ddi->diagonals.emplace_back(3, 205);

        this->diagonal_data[48].push_back(ddi);


        this->reverse_legal_lookup_white[202] = ReverseLegalLookup(Role::White, Piece::King, 17, 14, NE);
        this->reverse_legal_lookup_white[203] = ReverseLegalLookup(Role::White, Piece::King, 17, 10, NE);
        this->reverse_legal_lookup_white[204] = ReverseLegalLookup(Role::White, Piece::King, 17, 7, NE);
        this->reverse_legal_lookup_white[205] = ReverseLegalLookup(Role::White, Piece::King, 17, 3, NE);
    }
    // generating for white king 17 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 13, legal: (legal white (move king b 4 a 5))
        ddi->diagonals.emplace_back(13, 206);

        this->diagonal_data[48].push_back(ddi);


        this->reverse_legal_lookup_white[206] = ReverseLegalLookup(Role::White, Piece::King, 17, 13, NW);
    }
    // generating for white king 17 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 22, legal: (legal white (move king b 4 c 3))
        ddi->diagonals.emplace_back(22, 207);

        // to position 26, legal: (legal white (move king b 4 d 2))
        ddi->diagonals.emplace_back(26, 208);

        // to position 31, legal: (legal white (move king b 4 e 1))
        ddi->diagonals.emplace_back(31, 209);

        this->diagonal_data[48].push_back(ddi);


        this->reverse_legal_lookup_white[207] = ReverseLegalLookup(Role::White, Piece::King, 17, 22, SE);
        this->reverse_legal_lookup_white[208] = ReverseLegalLookup(Role::White, Piece::King, 17, 26, SE);
        this->reverse_legal_lookup_white[209] = ReverseLegalLookup(Role::White, Piece::King, 17, 31, SE);
    }
    // generating for white king 17 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 21, legal: (legal white (move king b 4 a 3))
        ddi->diagonals.emplace_back(21, 210);

        this->diagonal_data[48].push_back(ddi);


        this->reverse_legal_lookup_white[210] = ReverseLegalLookup(Role::White, Piece::King, 17, 21, SW);
    }
    // generating for white king 18 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 15, legal: (legal white (move king d 4 e 5))
        ddi->diagonals.emplace_back(15, 217);

        // to position 11, legal: (legal white (move king d 4 f 6))
        ddi->diagonals.emplace_back(11, 218);

        // to position 8, legal: (legal white (move king d 4 g 7))
        ddi->diagonals.emplace_back(8, 219);

        // to position 4, legal: (legal white (move king d 4 h 8))
        ddi->diagonals.emplace_back(4, 220);

        this->diagonal_data[49].push_back(ddi);


        this->reverse_legal_lookup_white[217] = ReverseLegalLookup(Role::White, Piece::King, 18, 15, NE);
        this->reverse_legal_lookup_white[218] = ReverseLegalLookup(Role::White, Piece::King, 18, 11, NE);
        this->reverse_legal_lookup_white[219] = ReverseLegalLookup(Role::White, Piece::King, 18, 8, NE);
        this->reverse_legal_lookup_white[220] = ReverseLegalLookup(Role::White, Piece::King, 18, 4, NE);
    }
    // generating for white king 18 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 14, legal: (legal white (move king d 4 c 5))
        ddi->diagonals.emplace_back(14, 221);

        // to position 9, legal: (legal white (move king d 4 b 6))
        ddi->diagonals.emplace_back(9, 222);

        // to position 5, legal: (legal white (move king d 4 a 7))
        ddi->diagonals.emplace_back(5, 223);

        this->diagonal_data[49].push_back(ddi);


        this->reverse_legal_lookup_white[221] = ReverseLegalLookup(Role::White, Piece::King, 18, 14, NW);
        this->reverse_legal_lookup_white[222] = ReverseLegalLookup(Role::White, Piece::King, 18, 9, NW);
        this->reverse_legal_lookup_white[223] = ReverseLegalLookup(Role::White, Piece::King, 18, 5, NW);
    }
    // generating for white king 18 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 23, legal: (legal white (move king d 4 e 3))
        ddi->diagonals.emplace_back(23, 224);

        // to position 27, legal: (legal white (move king d 4 f 2))
        ddi->diagonals.emplace_back(27, 225);

        // to position 32, legal: (legal white (move king d 4 g 1))
        ddi->diagonals.emplace_back(32, 226);

        this->diagonal_data[49].push_back(ddi);


        this->reverse_legal_lookup_white[224] = ReverseLegalLookup(Role::White, Piece::King, 18, 23, SE);
        this->reverse_legal_lookup_white[225] = ReverseLegalLookup(Role::White, Piece::King, 18, 27, SE);
        this->reverse_legal_lookup_white[226] = ReverseLegalLookup(Role::White, Piece::King, 18, 32, SE);
    }
    // generating for white king 18 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 22, legal: (legal white (move king d 4 c 3))
        ddi->diagonals.emplace_back(22, 227);

        // to position 25, legal: (legal white (move king d 4 b 2))
        ddi->diagonals.emplace_back(25, 228);

        // to position 29, legal: (legal white (move king d 4 a 1))
        ddi->diagonals.emplace_back(29, 229);

        this->diagonal_data[49].push_back(ddi);


        this->reverse_legal_lookup_white[227] = ReverseLegalLookup(Role::White, Piece::King, 18, 22, SW);
        this->reverse_legal_lookup_white[228] = ReverseLegalLookup(Role::White, Piece::King, 18, 25, SW);
        this->reverse_legal_lookup_white[229] = ReverseLegalLookup(Role::White, Piece::King, 18, 29, SW);
    }
    // generating for white king 19 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 16, legal: (legal white (move king f 4 g 5))
        ddi->diagonals.emplace_back(16, 236);

        // to position 12, legal: (legal white (move king f 4 h 6))
        ddi->diagonals.emplace_back(12, 237);

        this->diagonal_data[50].push_back(ddi);


        this->reverse_legal_lookup_white[236] = ReverseLegalLookup(Role::White, Piece::King, 19, 16, NE);
        this->reverse_legal_lookup_white[237] = ReverseLegalLookup(Role::White, Piece::King, 19, 12, NE);
    }
    // generating for white king 19 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 15, legal: (legal white (move king f 4 e 5))
        ddi->diagonals.emplace_back(15, 238);

        // to position 10, legal: (legal white (move king f 4 d 6))
        ddi->diagonals.emplace_back(10, 239);

        // to position 6, legal: (legal white (move king f 4 c 7))
        ddi->diagonals.emplace_back(6, 240);

        // to position 1, legal: (legal white (move king f 4 b 8))
        ddi->diagonals.emplace_back(1, 241);

        this->diagonal_data[50].push_back(ddi);


        this->reverse_legal_lookup_white[238] = ReverseLegalLookup(Role::White, Piece::King, 19, 15, NW);
        this->reverse_legal_lookup_white[239] = ReverseLegalLookup(Role::White, Piece::King, 19, 10, NW);
        this->reverse_legal_lookup_white[240] = ReverseLegalLookup(Role::White, Piece::King, 19, 6, NW);
        this->reverse_legal_lookup_white[241] = ReverseLegalLookup(Role::White, Piece::King, 19, 1, NW);
    }
    // generating for white king 19 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal white (move king f 4 g 3))
        ddi->diagonals.emplace_back(24, 242);

        // to position 28, legal: (legal white (move king f 4 h 2))
        ddi->diagonals.emplace_back(28, 243);

        this->diagonal_data[50].push_back(ddi);


        this->reverse_legal_lookup_white[242] = ReverseLegalLookup(Role::White, Piece::King, 19, 24, SE);
        this->reverse_legal_lookup_white[243] = ReverseLegalLookup(Role::White, Piece::King, 19, 28, SE);
    }
    // generating for white king 19 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 23, legal: (legal white (move king f 4 e 3))
        ddi->diagonals.emplace_back(23, 244);

        // to position 26, legal: (legal white (move king f 4 d 2))
        ddi->diagonals.emplace_back(26, 245);

        // to position 30, legal: (legal white (move king f 4 c 1))
        ddi->diagonals.emplace_back(30, 246);

        this->diagonal_data[50].push_back(ddi);


        this->reverse_legal_lookup_white[244] = ReverseLegalLookup(Role::White, Piece::King, 19, 23, SW);
        this->reverse_legal_lookup_white[245] = ReverseLegalLookup(Role::White, Piece::King, 19, 26, SW);
        this->reverse_legal_lookup_white[246] = ReverseLegalLookup(Role::White, Piece::King, 19, 30, SW);
    }
    // generating for white king 20 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 16, legal: (legal white (move king h 4 g 5))
        ddi->diagonals.emplace_back(16, 250);

        // to position 11, legal: (legal white (move king h 4 f 6))
        ddi->diagonals.emplace_back(11, 251);

        // to position 7, legal: (legal white (move king h 4 e 7))
        ddi->diagonals.emplace_back(7, 252);

        // to position 2, legal: (legal white (move king h 4 d 8))
        ddi->diagonals.emplace_back(2, 253);

        this->diagonal_data[51].push_back(ddi);


        this->reverse_legal_lookup_white[250] = ReverseLegalLookup(Role::White, Piece::King, 20, 16, NW);
        this->reverse_legal_lookup_white[251] = ReverseLegalLookup(Role::White, Piece::King, 20, 11, NW);
        this->reverse_legal_lookup_white[252] = ReverseLegalLookup(Role::White, Piece::King, 20, 7, NW);
        this->reverse_legal_lookup_white[253] = ReverseLegalLookup(Role::White, Piece::King, 20, 2, NW);
    }
    // generating for white king 20 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 24, legal: (legal white (move king h 4 g 3))
        ddi->diagonals.emplace_back(24, 254);

        // to position 27, legal: (legal white (move king h 4 f 2))
        ddi->diagonals.emplace_back(27, 255);

        // to position 31, legal: (legal white (move king h 4 e 1))
        ddi->diagonals.emplace_back(31, 256);

        this->diagonal_data[51].push_back(ddi);


        this->reverse_legal_lookup_white[254] = ReverseLegalLookup(Role::White, Piece::King, 20, 24, SW);
        this->reverse_legal_lookup_white[255] = ReverseLegalLookup(Role::White, Piece::King, 20, 27, SW);
        this->reverse_legal_lookup_white[256] = ReverseLegalLookup(Role::White, Piece::King, 20, 31, SW);
    }
    // generating for white king 21 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 17, legal: (legal white (move king a 3 b 4))
        ddi->diagonals.emplace_back(17, 260);

        // to position 14, legal: (legal white (move king a 3 c 5))
        ddi->diagonals.emplace_back(14, 261);

        // to position 10, legal: (legal white (move king a 3 d 6))
        ddi->diagonals.emplace_back(10, 262);

        // to position 7, legal: (legal white (move king a 3 e 7))
        ddi->diagonals.emplace_back(7, 263);

        // to position 3, legal: (legal white (move king a 3 f 8))
        ddi->diagonals.emplace_back(3, 264);

        this->diagonal_data[52].push_back(ddi);


        this->reverse_legal_lookup_white[260] = ReverseLegalLookup(Role::White, Piece::King, 21, 17, NE);
        this->reverse_legal_lookup_white[261] = ReverseLegalLookup(Role::White, Piece::King, 21, 14, NE);
        this->reverse_legal_lookup_white[262] = ReverseLegalLookup(Role::White, Piece::King, 21, 10, NE);
        this->reverse_legal_lookup_white[263] = ReverseLegalLookup(Role::White, Piece::King, 21, 7, NE);
        this->reverse_legal_lookup_white[264] = ReverseLegalLookup(Role::White, Piece::King, 21, 3, NE);
    }
    // generating for white king 21 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 25, legal: (legal white (move king a 3 b 2))
        ddi->diagonals.emplace_back(25, 265);

        // to position 30, legal: (legal white (move king a 3 c 1))
        ddi->diagonals.emplace_back(30, 266);

        this->diagonal_data[52].push_back(ddi);


        this->reverse_legal_lookup_white[265] = ReverseLegalLookup(Role::White, Piece::King, 21, 25, SE);
        this->reverse_legal_lookup_white[266] = ReverseLegalLookup(Role::White, Piece::King, 21, 30, SE);
    }
    // generating for white king 22 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 18, legal: (legal white (move king c 3 d 4))
        ddi->diagonals.emplace_back(18, 273);

        // to position 15, legal: (legal white (move king c 3 e 5))
        ddi->diagonals.emplace_back(15, 274);

        // to position 11, legal: (legal white (move king c 3 f 6))
        ddi->diagonals.emplace_back(11, 275);

        // to position 8, legal: (legal white (move king c 3 g 7))
        ddi->diagonals.emplace_back(8, 276);

        // to position 4, legal: (legal white (move king c 3 h 8))
        ddi->diagonals.emplace_back(4, 277);

        this->diagonal_data[53].push_back(ddi);


        this->reverse_legal_lookup_white[273] = ReverseLegalLookup(Role::White, Piece::King, 22, 18, NE);
        this->reverse_legal_lookup_white[274] = ReverseLegalLookup(Role::White, Piece::King, 22, 15, NE);
        this->reverse_legal_lookup_white[275] = ReverseLegalLookup(Role::White, Piece::King, 22, 11, NE);
        this->reverse_legal_lookup_white[276] = ReverseLegalLookup(Role::White, Piece::King, 22, 8, NE);
        this->reverse_legal_lookup_white[277] = ReverseLegalLookup(Role::White, Piece::King, 22, 4, NE);
    }
    // generating for white king 22 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal white (move king c 3 b 4))
        ddi->diagonals.emplace_back(17, 278);

        // to position 13, legal: (legal white (move king c 3 a 5))
        ddi->diagonals.emplace_back(13, 279);

        this->diagonal_data[53].push_back(ddi);


        this->reverse_legal_lookup_white[278] = ReverseLegalLookup(Role::White, Piece::King, 22, 17, NW);
        this->reverse_legal_lookup_white[279] = ReverseLegalLookup(Role::White, Piece::King, 22, 13, NW);
    }
    // generating for white king 22 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 26, legal: (legal white (move king c 3 d 2))
        ddi->diagonals.emplace_back(26, 280);

        // to position 31, legal: (legal white (move king c 3 e 1))
        ddi->diagonals.emplace_back(31, 281);

        this->diagonal_data[53].push_back(ddi);


        this->reverse_legal_lookup_white[280] = ReverseLegalLookup(Role::White, Piece::King, 22, 26, SE);
        this->reverse_legal_lookup_white[281] = ReverseLegalLookup(Role::White, Piece::King, 22, 31, SE);
    }
    // generating for white king 22 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 25, legal: (legal white (move king c 3 b 2))
        ddi->diagonals.emplace_back(25, 282);

        // to position 29, legal: (legal white (move king c 3 a 1))
        ddi->diagonals.emplace_back(29, 283);

        this->diagonal_data[53].push_back(ddi);


        this->reverse_legal_lookup_white[282] = ReverseLegalLookup(Role::White, Piece::King, 22, 25, SW);
        this->reverse_legal_lookup_white[283] = ReverseLegalLookup(Role::White, Piece::King, 22, 29, SW);
    }
    // generating for white king 23 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 19, legal: (legal white (move king e 3 f 4))
        ddi->diagonals.emplace_back(19, 290);

        // to position 16, legal: (legal white (move king e 3 g 5))
        ddi->diagonals.emplace_back(16, 291);

        // to position 12, legal: (legal white (move king e 3 h 6))
        ddi->diagonals.emplace_back(12, 292);

        this->diagonal_data[54].push_back(ddi);


        this->reverse_legal_lookup_white[290] = ReverseLegalLookup(Role::White, Piece::King, 23, 19, NE);
        this->reverse_legal_lookup_white[291] = ReverseLegalLookup(Role::White, Piece::King, 23, 16, NE);
        this->reverse_legal_lookup_white[292] = ReverseLegalLookup(Role::White, Piece::King, 23, 12, NE);
    }
    // generating for white king 23 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 18, legal: (legal white (move king e 3 d 4))
        ddi->diagonals.emplace_back(18, 293);

        // to position 14, legal: (legal white (move king e 3 c 5))
        ddi->diagonals.emplace_back(14, 294);

        // to position 9, legal: (legal white (move king e 3 b 6))
        ddi->diagonals.emplace_back(9, 295);

        // to position 5, legal: (legal white (move king e 3 a 7))
        ddi->diagonals.emplace_back(5, 296);

        this->diagonal_data[54].push_back(ddi);


        this->reverse_legal_lookup_white[293] = ReverseLegalLookup(Role::White, Piece::King, 23, 18, NW);
        this->reverse_legal_lookup_white[294] = ReverseLegalLookup(Role::White, Piece::King, 23, 14, NW);
        this->reverse_legal_lookup_white[295] = ReverseLegalLookup(Role::White, Piece::King, 23, 9, NW);
        this->reverse_legal_lookup_white[296] = ReverseLegalLookup(Role::White, Piece::King, 23, 5, NW);
    }
    // generating for white king 23 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal white (move king e 3 f 2))
        ddi->diagonals.emplace_back(27, 297);

        // to position 32, legal: (legal white (move king e 3 g 1))
        ddi->diagonals.emplace_back(32, 298);

        this->diagonal_data[54].push_back(ddi);


        this->reverse_legal_lookup_white[297] = ReverseLegalLookup(Role::White, Piece::King, 23, 27, SE);
        this->reverse_legal_lookup_white[298] = ReverseLegalLookup(Role::White, Piece::King, 23, 32, SE);
    }
    // generating for white king 23 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 26, legal: (legal white (move king e 3 d 2))
        ddi->diagonals.emplace_back(26, 299);

        // to position 30, legal: (legal white (move king e 3 c 1))
        ddi->diagonals.emplace_back(30, 300);

        this->diagonal_data[54].push_back(ddi);


        this->reverse_legal_lookup_white[299] = ReverseLegalLookup(Role::White, Piece::King, 23, 26, SW);
        this->reverse_legal_lookup_white[300] = ReverseLegalLookup(Role::White, Piece::King, 23, 30, SW);
    }
    // generating for white king 24 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 20, legal: (legal white (move king g 3 h 4))
        ddi->diagonals.emplace_back(20, 305);

        this->diagonal_data[55].push_back(ddi);


        this->reverse_legal_lookup_white[305] = ReverseLegalLookup(Role::White, Piece::King, 24, 20, NE);
    }
    // generating for white king 24 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 19, legal: (legal white (move king g 3 f 4))
        ddi->diagonals.emplace_back(19, 306);

        // to position 15, legal: (legal white (move king g 3 e 5))
        ddi->diagonals.emplace_back(15, 307);

        // to position 10, legal: (legal white (move king g 3 d 6))
        ddi->diagonals.emplace_back(10, 308);

        // to position 6, legal: (legal white (move king g 3 c 7))
        ddi->diagonals.emplace_back(6, 309);

        // to position 1, legal: (legal white (move king g 3 b 8))
        ddi->diagonals.emplace_back(1, 310);

        this->diagonal_data[55].push_back(ddi);


        this->reverse_legal_lookup_white[306] = ReverseLegalLookup(Role::White, Piece::King, 24, 19, NW);
        this->reverse_legal_lookup_white[307] = ReverseLegalLookup(Role::White, Piece::King, 24, 15, NW);
        this->reverse_legal_lookup_white[308] = ReverseLegalLookup(Role::White, Piece::King, 24, 10, NW);
        this->reverse_legal_lookup_white[309] = ReverseLegalLookup(Role::White, Piece::King, 24, 6, NW);
        this->reverse_legal_lookup_white[310] = ReverseLegalLookup(Role::White, Piece::King, 24, 1, NW);
    }
    // generating for white king 24 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 28, legal: (legal white (move king g 3 h 2))
        ddi->diagonals.emplace_back(28, 311);

        this->diagonal_data[55].push_back(ddi);


        this->reverse_legal_lookup_white[311] = ReverseLegalLookup(Role::White, Piece::King, 24, 28, SE);
    }
    // generating for white king 24 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal white (move king g 3 f 2))
        ddi->diagonals.emplace_back(27, 312);

        // to position 31, legal: (legal white (move king g 3 e 1))
        ddi->diagonals.emplace_back(31, 313);

        this->diagonal_data[55].push_back(ddi);


        this->reverse_legal_lookup_white[312] = ReverseLegalLookup(Role::White, Piece::King, 24, 27, SW);
        this->reverse_legal_lookup_white[313] = ReverseLegalLookup(Role::White, Piece::King, 24, 31, SW);
    }
    // generating for white king 25 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(6);
        // to position 22, legal: (legal white (move king b 2 c 3))
        ddi->diagonals.emplace_back(22, 317);

        // to position 18, legal: (legal white (move king b 2 d 4))
        ddi->diagonals.emplace_back(18, 318);

        // to position 15, legal: (legal white (move king b 2 e 5))
        ddi->diagonals.emplace_back(15, 319);

        // to position 11, legal: (legal white (move king b 2 f 6))
        ddi->diagonals.emplace_back(11, 320);

        // to position 8, legal: (legal white (move king b 2 g 7))
        ddi->diagonals.emplace_back(8, 321);

        // to position 4, legal: (legal white (move king b 2 h 8))
        ddi->diagonals.emplace_back(4, 322);

        this->diagonal_data[56].push_back(ddi);


        this->reverse_legal_lookup_white[317] = ReverseLegalLookup(Role::White, Piece::King, 25, 22, NE);
        this->reverse_legal_lookup_white[318] = ReverseLegalLookup(Role::White, Piece::King, 25, 18, NE);
        this->reverse_legal_lookup_white[319] = ReverseLegalLookup(Role::White, Piece::King, 25, 15, NE);
        this->reverse_legal_lookup_white[320] = ReverseLegalLookup(Role::White, Piece::King, 25, 11, NE);
        this->reverse_legal_lookup_white[321] = ReverseLegalLookup(Role::White, Piece::King, 25, 8, NE);
        this->reverse_legal_lookup_white[322] = ReverseLegalLookup(Role::White, Piece::King, 25, 4, NE);
    }
    // generating for white king 25 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 21, legal: (legal white (move king b 2 a 3))
        ddi->diagonals.emplace_back(21, 323);

        this->diagonal_data[56].push_back(ddi);


        this->reverse_legal_lookup_white[323] = ReverseLegalLookup(Role::White, Piece::King, 25, 21, NW);
    }
    // generating for white king 25 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 30, legal: (legal white (move king b 2 c 1))
        ddi->diagonals.emplace_back(30, 324);

        this->diagonal_data[56].push_back(ddi);


        this->reverse_legal_lookup_white[324] = ReverseLegalLookup(Role::White, Piece::King, 25, 30, SE);
    }
    // generating for white king 25 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 29, legal: (legal white (move king b 2 a 1))
        ddi->diagonals.emplace_back(29, 325);

        this->diagonal_data[56].push_back(ddi);


        this->reverse_legal_lookup_white[325] = ReverseLegalLookup(Role::White, Piece::King, 25, 29, SW);
    }
    // generating for white king 26 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 23, legal: (legal white (move king d 2 e 3))
        ddi->diagonals.emplace_back(23, 330);

        // to position 19, legal: (legal white (move king d 2 f 4))
        ddi->diagonals.emplace_back(19, 331);

        // to position 16, legal: (legal white (move king d 2 g 5))
        ddi->diagonals.emplace_back(16, 332);

        // to position 12, legal: (legal white (move king d 2 h 6))
        ddi->diagonals.emplace_back(12, 333);

        this->diagonal_data[57].push_back(ddi);


        this->reverse_legal_lookup_white[330] = ReverseLegalLookup(Role::White, Piece::King, 26, 23, NE);
        this->reverse_legal_lookup_white[331] = ReverseLegalLookup(Role::White, Piece::King, 26, 19, NE);
        this->reverse_legal_lookup_white[332] = ReverseLegalLookup(Role::White, Piece::King, 26, 16, NE);
        this->reverse_legal_lookup_white[333] = ReverseLegalLookup(Role::White, Piece::King, 26, 12, NE);
    }
    // generating for white king 26 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 22, legal: (legal white (move king d 2 c 3))
        ddi->diagonals.emplace_back(22, 334);

        // to position 17, legal: (legal white (move king d 2 b 4))
        ddi->diagonals.emplace_back(17, 335);

        // to position 13, legal: (legal white (move king d 2 a 5))
        ddi->diagonals.emplace_back(13, 336);

        this->diagonal_data[57].push_back(ddi);


        this->reverse_legal_lookup_white[334] = ReverseLegalLookup(Role::White, Piece::King, 26, 22, NW);
        this->reverse_legal_lookup_white[335] = ReverseLegalLookup(Role::White, Piece::King, 26, 17, NW);
        this->reverse_legal_lookup_white[336] = ReverseLegalLookup(Role::White, Piece::King, 26, 13, NW);
    }
    // generating for white king 26 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 31, legal: (legal white (move king d 2 e 1))
        ddi->diagonals.emplace_back(31, 337);

        this->diagonal_data[57].push_back(ddi);


        this->reverse_legal_lookup_white[337] = ReverseLegalLookup(Role::White, Piece::King, 26, 31, SE);
    }
    // generating for white king 26 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 30, legal: (legal white (move king d 2 c 1))
        ddi->diagonals.emplace_back(30, 338);

        this->diagonal_data[57].push_back(ddi);


        this->reverse_legal_lookup_white[338] = ReverseLegalLookup(Role::White, Piece::King, 26, 30, SW);
    }
    // generating for white king 27 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal white (move king f 2 g 3))
        ddi->diagonals.emplace_back(24, 343);

        // to position 20, legal: (legal white (move king f 2 h 4))
        ddi->diagonals.emplace_back(20, 344);

        this->diagonal_data[58].push_back(ddi);


        this->reverse_legal_lookup_white[343] = ReverseLegalLookup(Role::White, Piece::King, 27, 24, NE);
        this->reverse_legal_lookup_white[344] = ReverseLegalLookup(Role::White, Piece::King, 27, 20, NE);
    }
    // generating for white king 27 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 23, legal: (legal white (move king f 2 e 3))
        ddi->diagonals.emplace_back(23, 345);

        // to position 18, legal: (legal white (move king f 2 d 4))
        ddi->diagonals.emplace_back(18, 346);

        // to position 14, legal: (legal white (move king f 2 c 5))
        ddi->diagonals.emplace_back(14, 347);

        // to position 9, legal: (legal white (move king f 2 b 6))
        ddi->diagonals.emplace_back(9, 348);

        // to position 5, legal: (legal white (move king f 2 a 7))
        ddi->diagonals.emplace_back(5, 349);

        this->diagonal_data[58].push_back(ddi);


        this->reverse_legal_lookup_white[345] = ReverseLegalLookup(Role::White, Piece::King, 27, 23, NW);
        this->reverse_legal_lookup_white[346] = ReverseLegalLookup(Role::White, Piece::King, 27, 18, NW);
        this->reverse_legal_lookup_white[347] = ReverseLegalLookup(Role::White, Piece::King, 27, 14, NW);
        this->reverse_legal_lookup_white[348] = ReverseLegalLookup(Role::White, Piece::King, 27, 9, NW);
        this->reverse_legal_lookup_white[349] = ReverseLegalLookup(Role::White, Piece::King, 27, 5, NW);
    }
    // generating for white king 27 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 32, legal: (legal white (move king f 2 g 1))
        ddi->diagonals.emplace_back(32, 350);

        this->diagonal_data[58].push_back(ddi);


        this->reverse_legal_lookup_white[350] = ReverseLegalLookup(Role::White, Piece::King, 27, 32, SE);
    }
    // generating for white king 27 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 31, legal: (legal white (move king f 2 e 1))
        ddi->diagonals.emplace_back(31, 351);

        this->diagonal_data[58].push_back(ddi);


        this->reverse_legal_lookup_white[351] = ReverseLegalLookup(Role::White, Piece::King, 27, 31, SW);
    }
    // generating for white king 28 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 24, legal: (legal white (move king h 2 g 3))
        ddi->diagonals.emplace_back(24, 354);

        // to position 19, legal: (legal white (move king h 2 f 4))
        ddi->diagonals.emplace_back(19, 355);

        // to position 15, legal: (legal white (move king h 2 e 5))
        ddi->diagonals.emplace_back(15, 356);

        // to position 10, legal: (legal white (move king h 2 d 6))
        ddi->diagonals.emplace_back(10, 357);

        // to position 6, legal: (legal white (move king h 2 c 7))
        ddi->diagonals.emplace_back(6, 358);

        // to position 1, legal: (legal white (move king h 2 b 8))
        ddi->diagonals.emplace_back(1, 359);

        this->diagonal_data[59].push_back(ddi);


        this->reverse_legal_lookup_white[354] = ReverseLegalLookup(Role::White, Piece::King, 28, 24, NW);
        this->reverse_legal_lookup_white[355] = ReverseLegalLookup(Role::White, Piece::King, 28, 19, NW);
        this->reverse_legal_lookup_white[356] = ReverseLegalLookup(Role::White, Piece::King, 28, 15, NW);
        this->reverse_legal_lookup_white[357] = ReverseLegalLookup(Role::White, Piece::King, 28, 10, NW);
        this->reverse_legal_lookup_white[358] = ReverseLegalLookup(Role::White, Piece::King, 28, 6, NW);
        this->reverse_legal_lookup_white[359] = ReverseLegalLookup(Role::White, Piece::King, 28, 1, NW);
    }
    // generating for white king 28 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 32, legal: (legal white (move king h 2 g 1))
        ddi->diagonals.emplace_back(32, 360);

        this->diagonal_data[59].push_back(ddi);


        this->reverse_legal_lookup_white[360] = ReverseLegalLookup(Role::White, Piece::King, 28, 32, SW);
    }
    // generating for white king 29 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(7);
        // to position 25, legal: (legal white (move king a 1 b 2))
        ddi->diagonals.emplace_back(25, 363);

        // to position 22, legal: (legal white (move king a 1 c 3))
        ddi->diagonals.emplace_back(22, 364);

        // to position 18, legal: (legal white (move king a 1 d 4))
        ddi->diagonals.emplace_back(18, 365);

        // to position 15, legal: (legal white (move king a 1 e 5))
        ddi->diagonals.emplace_back(15, 366);

        // to position 11, legal: (legal white (move king a 1 f 6))
        ddi->diagonals.emplace_back(11, 367);

        // to position 8, legal: (legal white (move king a 1 g 7))
        ddi->diagonals.emplace_back(8, 368);

        // to position 4, legal: (legal white (move king a 1 h 8))
        ddi->diagonals.emplace_back(4, 369);

        this->diagonal_data[60].push_back(ddi);


        this->reverse_legal_lookup_white[363] = ReverseLegalLookup(Role::White, Piece::King, 29, 25, NE);
        this->reverse_legal_lookup_white[364] = ReverseLegalLookup(Role::White, Piece::King, 29, 22, NE);
        this->reverse_legal_lookup_white[365] = ReverseLegalLookup(Role::White, Piece::King, 29, 18, NE);
        this->reverse_legal_lookup_white[366] = ReverseLegalLookup(Role::White, Piece::King, 29, 15, NE);
        this->reverse_legal_lookup_white[367] = ReverseLegalLookup(Role::White, Piece::King, 29, 11, NE);
        this->reverse_legal_lookup_white[368] = ReverseLegalLookup(Role::White, Piece::King, 29, 8, NE);
        this->reverse_legal_lookup_white[369] = ReverseLegalLookup(Role::White, Piece::King, 29, 4, NE);
    }
    // generating for white king 30 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 26, legal: (legal white (move king c 1 d 2))
        ddi->diagonals.emplace_back(26, 374);

        // to position 23, legal: (legal white (move king c 1 e 3))
        ddi->diagonals.emplace_back(23, 375);

        // to position 19, legal: (legal white (move king c 1 f 4))
        ddi->diagonals.emplace_back(19, 376);

        // to position 16, legal: (legal white (move king c 1 g 5))
        ddi->diagonals.emplace_back(16, 377);

        // to position 12, legal: (legal white (move king c 1 h 6))
        ddi->diagonals.emplace_back(12, 378);

        this->diagonal_data[61].push_back(ddi);


        this->reverse_legal_lookup_white[374] = ReverseLegalLookup(Role::White, Piece::King, 30, 26, NE);
        this->reverse_legal_lookup_white[375] = ReverseLegalLookup(Role::White, Piece::King, 30, 23, NE);
        this->reverse_legal_lookup_white[376] = ReverseLegalLookup(Role::White, Piece::King, 30, 19, NE);
        this->reverse_legal_lookup_white[377] = ReverseLegalLookup(Role::White, Piece::King, 30, 16, NE);
        this->reverse_legal_lookup_white[378] = ReverseLegalLookup(Role::White, Piece::King, 30, 12, NE);
    }
    // generating for white king 30 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 25, legal: (legal white (move king c 1 b 2))
        ddi->diagonals.emplace_back(25, 379);

        // to position 21, legal: (legal white (move king c 1 a 3))
        ddi->diagonals.emplace_back(21, 380);

        this->diagonal_data[61].push_back(ddi);


        this->reverse_legal_lookup_white[379] = ReverseLegalLookup(Role::White, Piece::King, 30, 25, NW);
        this->reverse_legal_lookup_white[380] = ReverseLegalLookup(Role::White, Piece::King, 30, 21, NW);
    }
    // generating for white king 31 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 27, legal: (legal white (move king e 1 f 2))
        ddi->diagonals.emplace_back(27, 385);

        // to position 24, legal: (legal white (move king e 1 g 3))
        ddi->diagonals.emplace_back(24, 386);

        // to position 20, legal: (legal white (move king e 1 h 4))
        ddi->diagonals.emplace_back(20, 387);

        this->diagonal_data[62].push_back(ddi);


        this->reverse_legal_lookup_white[385] = ReverseLegalLookup(Role::White, Piece::King, 31, 27, NE);
        this->reverse_legal_lookup_white[386] = ReverseLegalLookup(Role::White, Piece::King, 31, 24, NE);
        this->reverse_legal_lookup_white[387] = ReverseLegalLookup(Role::White, Piece::King, 31, 20, NE);
    }
    // generating for white king 31 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 26, legal: (legal white (move king e 1 d 2))
        ddi->diagonals.emplace_back(26, 388);

        // to position 22, legal: (legal white (move king e 1 c 3))
        ddi->diagonals.emplace_back(22, 389);

        // to position 17, legal: (legal white (move king e 1 b 4))
        ddi->diagonals.emplace_back(17, 390);

        // to position 13, legal: (legal white (move king e 1 a 5))
        ddi->diagonals.emplace_back(13, 391);

        this->diagonal_data[62].push_back(ddi);


        this->reverse_legal_lookup_white[388] = ReverseLegalLookup(Role::White, Piece::King, 31, 26, NW);
        this->reverse_legal_lookup_white[389] = ReverseLegalLookup(Role::White, Piece::King, 31, 22, NW);
        this->reverse_legal_lookup_white[390] = ReverseLegalLookup(Role::White, Piece::King, 31, 17, NW);
        this->reverse_legal_lookup_white[391] = ReverseLegalLookup(Role::White, Piece::King, 31, 13, NW);
    }
    // generating for white king 32 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 28, legal: (legal white (move king g 1 h 2))
        ddi->diagonals.emplace_back(28, 395);

        this->diagonal_data[63].push_back(ddi);


        this->reverse_legal_lookup_white[395] = ReverseLegalLookup(Role::White, Piece::King, 32, 28, NE);
    }
    // generating for white king 32 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 27, legal: (legal white (move king g 1 f 2))
        ddi->diagonals.emplace_back(27, 396);

        // to position 23, legal: (legal white (move king g 1 e 3))
        ddi->diagonals.emplace_back(23, 397);

        // to position 18, legal: (legal white (move king g 1 d 4))
        ddi->diagonals.emplace_back(18, 398);

        // to position 14, legal: (legal white (move king g 1 c 5))
        ddi->diagonals.emplace_back(14, 399);

        // to position 9, legal: (legal white (move king g 1 b 6))
        ddi->diagonals.emplace_back(9, 400);

        // to position 5, legal: (legal white (move king g 1 a 7))
        ddi->diagonals.emplace_back(5, 401);

        this->diagonal_data[63].push_back(ddi);


        this->reverse_legal_lookup_white[396] = ReverseLegalLookup(Role::White, Piece::King, 32, 27, NW);
        this->reverse_legal_lookup_white[397] = ReverseLegalLookup(Role::White, Piece::King, 32, 23, NW);
        this->reverse_legal_lookup_white[398] = ReverseLegalLookup(Role::White, Piece::King, 32, 18, NW);
        this->reverse_legal_lookup_white[399] = ReverseLegalLookup(Role::White, Piece::King, 32, 14, NW);
        this->reverse_legal_lookup_white[400] = ReverseLegalLookup(Role::White, Piece::King, 32, 9, NW);
        this->reverse_legal_lookup_white[401] = ReverseLegalLookup(Role::White, Piece::King, 32, 5, NW);
    }

    // generating promotion line for black
    this->black_promotion_line = {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true};


    // generating moves for black
    this->black_legal_moves = {"noop", "(move man b 8 c 7)", "(move man b 8 a 7)", "(move man b 8 d 6)", "(move king b 8 c 7)", "(move king b 8 d 6)", "(move king b 8 e 5)", "(move king b 8 f 4)", "(move king b 8 g 3)", "(move king b 8 h 2)", "(move king b 8 a 7)", "(move man d 8 e 7)", "(move man d 8 c 7)", "(move man d 8 f 6)", "(move man d 8 b 6)", "(move king d 8 e 7)", "(move king d 8 f 6)", "(move king d 8 g 5)", "(move king d 8 h 4)", "(move king d 8 c 7)", "(move king d 8 b 6)", "(move king d 8 a 5)", "(move man f 8 g 7)", "(move man f 8 e 7)", "(move man f 8 h 6)", "(move man f 8 d 6)", "(move king f 8 g 7)", "(move king f 8 h 6)", "(move king f 8 e 7)", "(move king f 8 d 6)", "(move king f 8 c 5)", "(move king f 8 b 4)", "(move king f 8 a 3)", "(move man h 8 g 7)", "(move man h 8 f 6)", "(move king h 8 g 7)", "(move king h 8 f 6)", "(move king h 8 e 5)", "(move king h 8 d 4)", "(move king h 8 c 3)", "(move king h 8 b 2)", "(move king h 8 a 1)", "(move man a 7 b 6)", "(move man a 7 c 5)", "(move king a 7 b 8)", "(move king a 7 b 6)", "(move king a 7 c 5)", "(move king a 7 d 4)", "(move king a 7 e 3)", "(move king a 7 f 2)", "(move king a 7 g 1)", "(move man c 7 d 6)", "(move man c 7 b 6)", "(move man c 7 e 5)", "(move man c 7 a 5)", "(move king c 7 d 8)", "(move king c 7 b 8)", "(move king c 7 d 6)", "(move king c 7 e 5)", "(move king c 7 f 4)", "(move king c 7 g 3)", "(move king c 7 h 2)", "(move king c 7 b 6)", "(move king c 7 a 5)", "(move man e 7 f 6)", "(move man e 7 d 6)", "(move man e 7 g 5)", "(move man e 7 c 5)", "(move king e 7 f 8)", "(move king e 7 d 8)", "(move king e 7 f 6)", "(move king e 7 g 5)", "(move king e 7 h 4)", "(move king e 7 d 6)", "(move king e 7 c 5)", "(move king e 7 b 4)", "(move king e 7 a 3)", "(move man g 7 h 6)", "(move man g 7 f 6)", "(move man g 7 e 5)", "(move king g 7 h 8)", "(move king g 7 f 8)", "(move king g 7 h 6)", "(move king g 7 f 6)", "(move king g 7 e 5)", "(move king g 7 d 4)", "(move king g 7 c 3)", "(move king g 7 b 2)", "(move king g 7 a 1)", "(move man b 6 c 5)", "(move man b 6 a 5)", "(move man b 6 d 8)", "(move man b 6 d 4)", "(move king b 6 c 7)", "(move king b 6 d 8)", "(move king b 6 a 7)", "(move king b 6 c 5)", "(move king b 6 d 4)", "(move king b 6 e 3)", "(move king b 6 f 2)", "(move king b 6 g 1)", "(move king b 6 a 5)", "(move man d 6 e 5)", "(move man d 6 c 5)", "(move man d 6 f 8)", "(move man d 6 b 8)", "(move man d 6 f 4)", "(move man d 6 b 4)", "(move king d 6 e 7)", "(move king d 6 f 8)", "(move king d 6 c 7)", "(move king d 6 b 8)", "(move king d 6 e 5)", "(move king d 6 f 4)", "(move king d 6 g 3)", "(move king d 6 h 2)", "(move king d 6 c 5)", "(move king d 6 b 4)", "(move king d 6 a 3)", "(move man f 6 g 5)", "(move man f 6 e 5)", "(move man f 6 h 8)", "(move man f 6 d 8)", "(move man f 6 h 4)", "(move man f 6 d 4)", "(move king f 6 g 7)", "(move king f 6 h 8)", "(move king f 6 e 7)", "(move king f 6 d 8)", "(move king f 6 g 5)", "(move king f 6 h 4)", "(move king f 6 e 5)", "(move king f 6 d 4)", "(move king f 6 c 3)", "(move king f 6 b 2)", "(move king f 6 a 1)", "(move man h 6 g 5)", "(move man h 6 f 8)", "(move man h 6 f 4)", "(move king h 6 g 7)", "(move king h 6 f 8)", "(move king h 6 g 5)", "(move king h 6 f 4)", "(move king h 6 e 3)", "(move king h 6 d 2)", "(move king h 6 c 1)", "(move man a 5 b 4)", "(move man a 5 c 7)", "(move man a 5 c 3)", "(move king a 5 b 6)", "(move king a 5 c 7)", "(move king a 5 d 8)", "(move king a 5 b 4)", "(move king a 5 c 3)", "(move king a 5 d 2)", "(move king a 5 e 1)", "(move man c 5 d 4)", "(move man c 5 b 4)", "(move man c 5 e 7)", "(move man c 5 a 7)", "(move man c 5 e 3)", "(move man c 5 a 3)", "(move king c 5 d 6)", "(move king c 5 e 7)", "(move king c 5 f 8)", "(move king c 5 b 6)", "(move king c 5 a 7)", "(move king c 5 d 4)", "(move king c 5 e 3)", "(move king c 5 f 2)", "(move king c 5 g 1)", "(move king c 5 b 4)", "(move king c 5 a 3)", "(move man e 5 f 4)", "(move man e 5 d 4)", "(move man e 5 g 7)", "(move man e 5 c 7)", "(move man e 5 g 3)", "(move man e 5 c 3)", "(move king e 5 f 6)", "(move king e 5 g 7)", "(move king e 5 h 8)", "(move king e 5 d 6)", "(move king e 5 c 7)", "(move king e 5 b 8)", "(move king e 5 f 4)", "(move king e 5 g 3)", "(move king e 5 h 2)", "(move king e 5 d 4)", "(move king e 5 c 3)", "(move king e 5 b 2)", "(move king e 5 a 1)", "(move man g 5 h 4)", "(move man g 5 f 4)", "(move man g 5 e 7)", "(move man g 5 e 3)", "(move king g 5 h 6)", "(move king g 5 f 6)", "(move king g 5 e 7)", "(move king g 5 d 8)", "(move king g 5 h 4)", "(move king g 5 f 4)", "(move king g 5 e 3)", "(move king g 5 d 2)", "(move king g 5 c 1)", "(move man b 4 c 3)", "(move man b 4 a 3)", "(move man b 4 d 6)", "(move man b 4 d 2)", "(move king b 4 c 5)", "(move king b 4 d 6)", "(move king b 4 e 7)", "(move king b 4 f 8)", "(move king b 4 a 5)", "(move king b 4 c 3)", "(move king b 4 d 2)", "(move king b 4 e 1)", "(move king b 4 a 3)", "(move man d 4 e 3)", "(move man d 4 c 3)", "(move man d 4 f 6)", "(move man d 4 b 6)", "(move man d 4 f 2)", "(move man d 4 b 2)", "(move king d 4 e 5)", "(move king d 4 f 6)", "(move king d 4 g 7)", "(move king d 4 h 8)", "(move king d 4 c 5)", "(move king d 4 b 6)", "(move king d 4 a 7)", "(move king d 4 e 3)", "(move king d 4 f 2)", "(move king d 4 g 1)", "(move king d 4 c 3)", "(move king d 4 b 2)", "(move king d 4 a 1)", "(move man f 4 g 3)", "(move man f 4 e 3)", "(move man f 4 h 6)", "(move man f 4 d 6)", "(move man f 4 h 2)", "(move man f 4 d 2)", "(move king f 4 g 5)", "(move king f 4 h 6)", "(move king f 4 e 5)", "(move king f 4 d 6)", "(move king f 4 c 7)", "(move king f 4 b 8)", "(move king f 4 g 3)", "(move king f 4 h 2)", "(move king f 4 e 3)", "(move king f 4 d 2)", "(move king f 4 c 1)", "(move man h 4 g 3)", "(move man h 4 f 6)", "(move man h 4 f 2)", "(move king h 4 g 5)", "(move king h 4 f 6)", "(move king h 4 e 7)", "(move king h 4 d 8)", "(move king h 4 g 3)", "(move king h 4 f 2)", "(move king h 4 e 1)", "(move man a 3 b 2)", "(move man a 3 c 5)", "(move man a 3 c 1)", "(move king a 3 b 4)", "(move king a 3 c 5)", "(move king a 3 d 6)", "(move king a 3 e 7)", "(move king a 3 f 8)", "(move king a 3 b 2)", "(move king a 3 c 1)", "(move man c 3 d 2)", "(move man c 3 b 2)", "(move man c 3 e 5)", "(move man c 3 a 5)", "(move man c 3 e 1)", "(move man c 3 a 1)", "(move king c 3 d 4)", "(move king c 3 e 5)", "(move king c 3 f 6)", "(move king c 3 g 7)", "(move king c 3 h 8)", "(move king c 3 b 4)", "(move king c 3 a 5)", "(move king c 3 d 2)", "(move king c 3 e 1)", "(move king c 3 b 2)", "(move king c 3 a 1)", "(move man e 3 f 2)", "(move man e 3 d 2)", "(move man e 3 g 5)", "(move man e 3 c 5)", "(move man e 3 g 1)", "(move man e 3 c 1)", "(move king e 3 f 4)", "(move king e 3 g 5)", "(move king e 3 h 6)", "(move king e 3 d 4)", "(move king e 3 c 5)", "(move king e 3 b 6)", "(move king e 3 a 7)", "(move king e 3 f 2)", "(move king e 3 g 1)", "(move king e 3 d 2)", "(move king e 3 c 1)", "(move man g 3 h 2)", "(move man g 3 f 2)", "(move man g 3 e 5)", "(move man g 3 e 1)", "(move king g 3 h 4)", "(move king g 3 f 4)", "(move king g 3 e 5)", "(move king g 3 d 6)", "(move king g 3 c 7)", "(move king g 3 b 8)", "(move king g 3 h 2)", "(move king g 3 f 2)", "(move king g 3 e 1)", "(move man b 2 c 1)", "(move man b 2 a 1)", "(move man b 2 d 4)", "(move king b 2 c 3)", "(move king b 2 d 4)", "(move king b 2 e 5)", "(move king b 2 f 6)", "(move king b 2 g 7)", "(move king b 2 h 8)", "(move king b 2 a 3)", "(move king b 2 c 1)", "(move king b 2 a 1)", "(move man d 2 e 1)", "(move man d 2 c 1)", "(move man d 2 f 4)", "(move man d 2 b 4)", "(move king d 2 e 3)", "(move king d 2 f 4)", "(move king d 2 g 5)", "(move king d 2 h 6)", "(move king d 2 c 3)", "(move king d 2 b 4)", "(move king d 2 a 5)", "(move king d 2 e 1)", "(move king d 2 c 1)", "(move man f 2 g 1)", "(move man f 2 e 1)", "(move man f 2 h 4)", "(move man f 2 d 4)", "(move king f 2 g 3)", "(move king f 2 h 4)", "(move king f 2 e 3)", "(move king f 2 d 4)", "(move king f 2 c 5)", "(move king f 2 b 6)", "(move king f 2 a 7)", "(move king f 2 g 1)", "(move king f 2 e 1)", "(move man h 2 g 1)", "(move man h 2 f 4)", "(move king h 2 g 3)", "(move king h 2 f 4)", "(move king h 2 e 5)", "(move king h 2 d 6)", "(move king h 2 c 7)", "(move king h 2 b 8)", "(move king h 2 g 1)", "(move man a 1 c 3)", "(move king a 1 b 2)", "(move king a 1 c 3)", "(move king a 1 d 4)", "(move king a 1 e 5)", "(move king a 1 f 6)", "(move king a 1 g 7)", "(move king a 1 h 8)", "(move man c 1 e 3)", "(move man c 1 a 3)", "(move king c 1 d 2)", "(move king c 1 e 3)", "(move king c 1 f 4)", "(move king c 1 g 5)", "(move king c 1 h 6)", "(move king c 1 b 2)", "(move king c 1 a 3)", "(move man e 1 g 3)", "(move man e 1 c 3)", "(move king e 1 f 2)", "(move king e 1 g 3)", "(move king e 1 h 4)", "(move king e 1 d 2)", "(move king e 1 c 3)", "(move king e 1 b 4)", "(move king e 1 a 5)", "(move man g 1 e 3)", "(move king g 1 h 2)", "(move king g 1 f 2)", "(move king g 1 e 3)", "(move king g 1 d 4)", "(move king g 1 c 5)", "(move king g 1 b 6)", "(move king g 1 a 7)"};
    // generating for black man 1 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 6, legal: (legal black (move man b 8 c 7))
        ddi->diagonals.emplace_back(6, 1);

        // to position 10, legal: (legal black (move man b 8 d 6))
        ddi->diagonals.emplace_back(10, 3);

        this->diagonal_data[64].push_back(ddi);


        this->reverse_legal_lookup_black[1] = ReverseLegalLookup(Role::Black, Piece::Man, 1, 6, SE);
        this->reverse_legal_lookup_black[3] = ReverseLegalLookup(Role::Black, Piece::Man, 1, 10, SE);
    }
    // generating for black man 1 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 5, legal: (legal black (move man b 8 a 7))
        ddi->diagonals.emplace_back(5, 2);

        this->diagonal_data[64].push_back(ddi);


        this->reverse_legal_lookup_black[2] = ReverseLegalLookup(Role::Black, Piece::Man, 1, 5, SW);
    }
    // generating for black man 2 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal black (move man d 8 e 7))
        ddi->diagonals.emplace_back(7, 11);

        // to position 11, legal: (legal black (move man d 8 f 6))
        ddi->diagonals.emplace_back(11, 13);

        this->diagonal_data[65].push_back(ddi);


        this->reverse_legal_lookup_black[11] = ReverseLegalLookup(Role::Black, Piece::Man, 2, 7, SE);
        this->reverse_legal_lookup_black[13] = ReverseLegalLookup(Role::Black, Piece::Man, 2, 11, SE);
    }
    // generating for black man 2 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 6, legal: (legal black (move man d 8 c 7))
        ddi->diagonals.emplace_back(6, 12);

        // to position 9, legal: (legal black (move man d 8 b 6))
        ddi->diagonals.emplace_back(9, 14);

        this->diagonal_data[65].push_back(ddi);


        this->reverse_legal_lookup_black[12] = ReverseLegalLookup(Role::Black, Piece::Man, 2, 6, SW);
        this->reverse_legal_lookup_black[14] = ReverseLegalLookup(Role::Black, Piece::Man, 2, 9, SW);
    }
    // generating for black man 3 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal black (move man f 8 g 7))
        ddi->diagonals.emplace_back(8, 22);

        // to position 12, legal: (legal black (move man f 8 h 6))
        ddi->diagonals.emplace_back(12, 24);

        this->diagonal_data[66].push_back(ddi);


        this->reverse_legal_lookup_black[22] = ReverseLegalLookup(Role::Black, Piece::Man, 3, 8, SE);
        this->reverse_legal_lookup_black[24] = ReverseLegalLookup(Role::Black, Piece::Man, 3, 12, SE);
    }
    // generating for black man 3 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal black (move man f 8 e 7))
        ddi->diagonals.emplace_back(7, 23);

        // to position 10, legal: (legal black (move man f 8 d 6))
        ddi->diagonals.emplace_back(10, 25);

        this->diagonal_data[66].push_back(ddi);


        this->reverse_legal_lookup_black[23] = ReverseLegalLookup(Role::Black, Piece::Man, 3, 7, SW);
        this->reverse_legal_lookup_black[25] = ReverseLegalLookup(Role::Black, Piece::Man, 3, 10, SW);
    }
    // generating for black man 4 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal black (move man h 8 g 7))
        ddi->diagonals.emplace_back(8, 33);

        // to position 11, legal: (legal black (move man h 8 f 6))
        ddi->diagonals.emplace_back(11, 34);

        this->diagonal_data[67].push_back(ddi);


        this->reverse_legal_lookup_black[33] = ReverseLegalLookup(Role::Black, Piece::Man, 4, 8, SW);
        this->reverse_legal_lookup_black[34] = ReverseLegalLookup(Role::Black, Piece::Man, 4, 11, SW);
    }
    // generating for black man 5 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 1, legal: invalid
        ddi->diagonals.emplace_back(1, -403);

        this->diagonal_data[68].push_back(ddi);


    }
    // generating for black man 5 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal black (move man a 7 b 6))
        ddi->diagonals.emplace_back(9, 42);

        // to position 14, legal: (legal black (move man a 7 c 5))
        ddi->diagonals.emplace_back(14, 43);

        this->diagonal_data[68].push_back(ddi);


        this->reverse_legal_lookup_black[42] = ReverseLegalLookup(Role::Black, Piece::Man, 5, 9, SE);
        this->reverse_legal_lookup_black[43] = ReverseLegalLookup(Role::Black, Piece::Man, 5, 14, SE);
    }
    // generating for black man 6 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 2, legal: invalid
        ddi->diagonals.emplace_back(2, -403);

        this->diagonal_data[69].push_back(ddi);


    }
    // generating for black man 6 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 1, legal: invalid
        ddi->diagonals.emplace_back(1, -403);

        this->diagonal_data[69].push_back(ddi);


    }
    // generating for black man 6 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal black (move man c 7 d 6))
        ddi->diagonals.emplace_back(10, 51);

        // to position 15, legal: (legal black (move man c 7 e 5))
        ddi->diagonals.emplace_back(15, 53);

        this->diagonal_data[69].push_back(ddi);


        this->reverse_legal_lookup_black[51] = ReverseLegalLookup(Role::Black, Piece::Man, 6, 10, SE);
        this->reverse_legal_lookup_black[53] = ReverseLegalLookup(Role::Black, Piece::Man, 6, 15, SE);
    }
    // generating for black man 6 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal black (move man c 7 b 6))
        ddi->diagonals.emplace_back(9, 52);

        // to position 13, legal: (legal black (move man c 7 a 5))
        ddi->diagonals.emplace_back(13, 54);

        this->diagonal_data[69].push_back(ddi);


        this->reverse_legal_lookup_black[52] = ReverseLegalLookup(Role::Black, Piece::Man, 6, 9, SW);
        this->reverse_legal_lookup_black[54] = ReverseLegalLookup(Role::Black, Piece::Man, 6, 13, SW);
    }
    // generating for black man 7 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 3, legal: invalid
        ddi->diagonals.emplace_back(3, -403);

        this->diagonal_data[70].push_back(ddi);


    }
    // generating for black man 7 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 2, legal: invalid
        ddi->diagonals.emplace_back(2, -403);

        this->diagonal_data[70].push_back(ddi);


    }
    // generating for black man 7 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal black (move man e 7 f 6))
        ddi->diagonals.emplace_back(11, 64);

        // to position 16, legal: (legal black (move man e 7 g 5))
        ddi->diagonals.emplace_back(16, 66);

        this->diagonal_data[70].push_back(ddi);


        this->reverse_legal_lookup_black[64] = ReverseLegalLookup(Role::Black, Piece::Man, 7, 11, SE);
        this->reverse_legal_lookup_black[66] = ReverseLegalLookup(Role::Black, Piece::Man, 7, 16, SE);
    }
    // generating for black man 7 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal black (move man e 7 d 6))
        ddi->diagonals.emplace_back(10, 65);

        // to position 14, legal: (legal black (move man e 7 c 5))
        ddi->diagonals.emplace_back(14, 67);

        this->diagonal_data[70].push_back(ddi);


        this->reverse_legal_lookup_black[65] = ReverseLegalLookup(Role::Black, Piece::Man, 7, 10, SW);
        this->reverse_legal_lookup_black[67] = ReverseLegalLookup(Role::Black, Piece::Man, 7, 14, SW);
    }
    // generating for black man 8 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 4, legal: invalid
        ddi->diagonals.emplace_back(4, -403);

        this->diagonal_data[71].push_back(ddi);


    }
    // generating for black man 8 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 3, legal: invalid
        ddi->diagonals.emplace_back(3, -403);

        this->diagonal_data[71].push_back(ddi);


    }
    // generating for black man 8 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 12, legal: (legal black (move man g 7 h 6))
        ddi->diagonals.emplace_back(12, 77);

        this->diagonal_data[71].push_back(ddi);


        this->reverse_legal_lookup_black[77] = ReverseLegalLookup(Role::Black, Piece::Man, 8, 12, SE);
    }
    // generating for black man 8 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal black (move man g 7 f 6))
        ddi->diagonals.emplace_back(11, 78);

        // to position 15, legal: (legal black (move man g 7 e 5))
        ddi->diagonals.emplace_back(15, 79);

        this->diagonal_data[71].push_back(ddi);


        this->reverse_legal_lookup_black[78] = ReverseLegalLookup(Role::Black, Piece::Man, 8, 11, SW);
        this->reverse_legal_lookup_black[79] = ReverseLegalLookup(Role::Black, Piece::Man, 8, 15, SW);
    }
    // generating for black man 9 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 6, legal: invalid
        ddi->diagonals.emplace_back(6, -403);

        // to position 2, legal: (legal black (move man b 6 d 8))
        ddi->diagonals.emplace_back(2, 91);

        this->diagonal_data[72].push_back(ddi);


        this->reverse_legal_lookup_black[91] = ReverseLegalLookup(Role::Black, Piece::Man, 9, 2, NE);
    }
    // generating for black man 9 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 5, legal: invalid
        ddi->diagonals.emplace_back(5, -403);

        this->diagonal_data[72].push_back(ddi);


    }
    // generating for black man 9 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 14, legal: (legal black (move man b 6 c 5))
        ddi->diagonals.emplace_back(14, 89);

        // to position 18, legal: (legal black (move man b 6 d 4))
        ddi->diagonals.emplace_back(18, 92);

        this->diagonal_data[72].push_back(ddi);


        this->reverse_legal_lookup_black[89] = ReverseLegalLookup(Role::Black, Piece::Man, 9, 14, SE);
        this->reverse_legal_lookup_black[92] = ReverseLegalLookup(Role::Black, Piece::Man, 9, 18, SE);
    }
    // generating for black man 9 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 13, legal: (legal black (move man b 6 a 5))
        ddi->diagonals.emplace_back(13, 90);

        this->diagonal_data[72].push_back(ddi);


        this->reverse_legal_lookup_black[90] = ReverseLegalLookup(Role::Black, Piece::Man, 9, 13, SW);
    }
    // generating for black man 10 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: invalid
        ddi->diagonals.emplace_back(7, -403);

        // to position 3, legal: (legal black (move man d 6 f 8))
        ddi->diagonals.emplace_back(3, 104);

        this->diagonal_data[73].push_back(ddi);


        this->reverse_legal_lookup_black[104] = ReverseLegalLookup(Role::Black, Piece::Man, 10, 3, NE);
    }
    // generating for black man 10 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 6, legal: invalid
        ddi->diagonals.emplace_back(6, -403);

        // to position 1, legal: (legal black (move man d 6 b 8))
        ddi->diagonals.emplace_back(1, 105);

        this->diagonal_data[73].push_back(ddi);


        this->reverse_legal_lookup_black[105] = ReverseLegalLookup(Role::Black, Piece::Man, 10, 1, NW);
    }
    // generating for black man 10 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 15, legal: (legal black (move man d 6 e 5))
        ddi->diagonals.emplace_back(15, 102);

        // to position 19, legal: (legal black (move man d 6 f 4))
        ddi->diagonals.emplace_back(19, 106);

        this->diagonal_data[73].push_back(ddi);


        this->reverse_legal_lookup_black[102] = ReverseLegalLookup(Role::Black, Piece::Man, 10, 15, SE);
        this->reverse_legal_lookup_black[106] = ReverseLegalLookup(Role::Black, Piece::Man, 10, 19, SE);
    }
    // generating for black man 10 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 14, legal: (legal black (move man d 6 c 5))
        ddi->diagonals.emplace_back(14, 103);

        // to position 17, legal: (legal black (move man d 6 b 4))
        ddi->diagonals.emplace_back(17, 107);

        this->diagonal_data[73].push_back(ddi);


        this->reverse_legal_lookup_black[103] = ReverseLegalLookup(Role::Black, Piece::Man, 10, 14, SW);
        this->reverse_legal_lookup_black[107] = ReverseLegalLookup(Role::Black, Piece::Man, 10, 17, SW);
    }
    // generating for black man 11 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: invalid
        ddi->diagonals.emplace_back(8, -403);

        // to position 4, legal: (legal black (move man f 6 h 8))
        ddi->diagonals.emplace_back(4, 121);

        this->diagonal_data[74].push_back(ddi);


        this->reverse_legal_lookup_black[121] = ReverseLegalLookup(Role::Black, Piece::Man, 11, 4, NE);
    }
    // generating for black man 11 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: invalid
        ddi->diagonals.emplace_back(7, -403);

        // to position 2, legal: (legal black (move man f 6 d 8))
        ddi->diagonals.emplace_back(2, 122);

        this->diagonal_data[74].push_back(ddi);


        this->reverse_legal_lookup_black[122] = ReverseLegalLookup(Role::Black, Piece::Man, 11, 2, NW);
    }
    // generating for black man 11 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 16, legal: (legal black (move man f 6 g 5))
        ddi->diagonals.emplace_back(16, 119);

        // to position 20, legal: (legal black (move man f 6 h 4))
        ddi->diagonals.emplace_back(20, 123);

        this->diagonal_data[74].push_back(ddi);


        this->reverse_legal_lookup_black[119] = ReverseLegalLookup(Role::Black, Piece::Man, 11, 16, SE);
        this->reverse_legal_lookup_black[123] = ReverseLegalLookup(Role::Black, Piece::Man, 11, 20, SE);
    }
    // generating for black man 11 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 15, legal: (legal black (move man f 6 e 5))
        ddi->diagonals.emplace_back(15, 120);

        // to position 18, legal: (legal black (move man f 6 d 4))
        ddi->diagonals.emplace_back(18, 124);

        this->diagonal_data[74].push_back(ddi);


        this->reverse_legal_lookup_black[120] = ReverseLegalLookup(Role::Black, Piece::Man, 11, 15, SW);
        this->reverse_legal_lookup_black[124] = ReverseLegalLookup(Role::Black, Piece::Man, 11, 18, SW);
    }
    // generating for black man 12 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: invalid
        ddi->diagonals.emplace_back(8, -403);

        // to position 3, legal: (legal black (move man h 6 f 8))
        ddi->diagonals.emplace_back(3, 137);

        this->diagonal_data[75].push_back(ddi);


        this->reverse_legal_lookup_black[137] = ReverseLegalLookup(Role::Black, Piece::Man, 12, 3, NW);
    }
    // generating for black man 12 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 16, legal: (legal black (move man h 6 g 5))
        ddi->diagonals.emplace_back(16, 136);

        // to position 19, legal: (legal black (move man h 6 f 4))
        ddi->diagonals.emplace_back(19, 138);

        this->diagonal_data[75].push_back(ddi);


        this->reverse_legal_lookup_black[136] = ReverseLegalLookup(Role::Black, Piece::Man, 12, 16, SW);
        this->reverse_legal_lookup_black[138] = ReverseLegalLookup(Role::Black, Piece::Man, 12, 19, SW);
    }
    // generating for black man 13 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 9, legal: invalid
        ddi->diagonals.emplace_back(9, -403);

        // to position 6, legal: (legal black (move man a 5 c 7))
        ddi->diagonals.emplace_back(6, 147);

        this->diagonal_data[76].push_back(ddi);


        this->reverse_legal_lookup_black[147] = ReverseLegalLookup(Role::Black, Piece::Man, 13, 6, NE);
    }
    // generating for black man 13 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal black (move man a 5 b 4))
        ddi->diagonals.emplace_back(17, 146);

        // to position 22, legal: (legal black (move man a 5 c 3))
        ddi->diagonals.emplace_back(22, 148);

        this->diagonal_data[76].push_back(ddi);


        this->reverse_legal_lookup_black[146] = ReverseLegalLookup(Role::Black, Piece::Man, 13, 17, SE);
        this->reverse_legal_lookup_black[148] = ReverseLegalLookup(Role::Black, Piece::Man, 13, 22, SE);
    }
    // generating for black man 14 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: invalid
        ddi->diagonals.emplace_back(10, -403);

        // to position 7, legal: (legal black (move man c 5 e 7))
        ddi->diagonals.emplace_back(7, 158);

        this->diagonal_data[77].push_back(ddi);


        this->reverse_legal_lookup_black[158] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 7, NE);
    }
    // generating for black man 14 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: invalid
        ddi->diagonals.emplace_back(9, -403);

        // to position 5, legal: (legal black (move man c 5 a 7))
        ddi->diagonals.emplace_back(5, 159);

        this->diagonal_data[77].push_back(ddi);


        this->reverse_legal_lookup_black[159] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 5, NW);
    }
    // generating for black man 14 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 18, legal: (legal black (move man c 5 d 4))
        ddi->diagonals.emplace_back(18, 156);

        // to position 23, legal: (legal black (move man c 5 e 3))
        ddi->diagonals.emplace_back(23, 160);

        this->diagonal_data[77].push_back(ddi);


        this->reverse_legal_lookup_black[156] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 18, SE);
        this->reverse_legal_lookup_black[160] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 23, SE);
    }
    // generating for black man 14 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal black (move man c 5 b 4))
        ddi->diagonals.emplace_back(17, 157);

        // to position 21, legal: (legal black (move man c 5 a 3))
        ddi->diagonals.emplace_back(21, 161);

        this->diagonal_data[77].push_back(ddi);


        this->reverse_legal_lookup_black[157] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 17, SW);
        this->reverse_legal_lookup_black[161] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 21, SW);
    }
    // generating for black man 15 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 11, legal: invalid
        ddi->diagonals.emplace_back(11, -403);

        // to position 8, legal: (legal black (move man e 5 g 7))
        ddi->diagonals.emplace_back(8, 175);

        this->diagonal_data[78].push_back(ddi);


        this->reverse_legal_lookup_black[175] = ReverseLegalLookup(Role::Black, Piece::Man, 15, 8, NE);
    }
    // generating for black man 15 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 10, legal: invalid
        ddi->diagonals.emplace_back(10, -403);

        // to position 6, legal: (legal black (move man e 5 c 7))
        ddi->diagonals.emplace_back(6, 176);

        this->diagonal_data[78].push_back(ddi);


        this->reverse_legal_lookup_black[176] = ReverseLegalLookup(Role::Black, Piece::Man, 15, 6, NW);
    }
    // generating for black man 15 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 19, legal: (legal black (move man e 5 f 4))
        ddi->diagonals.emplace_back(19, 173);

        // to position 24, legal: (legal black (move man e 5 g 3))
        ddi->diagonals.emplace_back(24, 177);

        this->diagonal_data[78].push_back(ddi);


        this->reverse_legal_lookup_black[173] = ReverseLegalLookup(Role::Black, Piece::Man, 15, 19, SE);
        this->reverse_legal_lookup_black[177] = ReverseLegalLookup(Role::Black, Piece::Man, 15, 24, SE);
    }
    // generating for black man 15 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 18, legal: (legal black (move man e 5 d 4))
        ddi->diagonals.emplace_back(18, 174);

        // to position 22, legal: (legal black (move man e 5 c 3))
        ddi->diagonals.emplace_back(22, 178);

        this->diagonal_data[78].push_back(ddi);


        this->reverse_legal_lookup_black[174] = ReverseLegalLookup(Role::Black, Piece::Man, 15, 18, SW);
        this->reverse_legal_lookup_black[178] = ReverseLegalLookup(Role::Black, Piece::Man, 15, 22, SW);
    }
    // generating for black man 16 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 12, legal: invalid
        ddi->diagonals.emplace_back(12, -403);

        this->diagonal_data[79].push_back(ddi);


    }
    // generating for black man 16 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: invalid
        ddi->diagonals.emplace_back(11, -403);

        // to position 7, legal: (legal black (move man g 5 e 7))
        ddi->diagonals.emplace_back(7, 194);

        this->diagonal_data[79].push_back(ddi);


        this->reverse_legal_lookup_black[194] = ReverseLegalLookup(Role::Black, Piece::Man, 16, 7, NW);
    }
    // generating for black man 16 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 20, legal: (legal black (move man g 5 h 4))
        ddi->diagonals.emplace_back(20, 192);

        this->diagonal_data[79].push_back(ddi);


        this->reverse_legal_lookup_black[192] = ReverseLegalLookup(Role::Black, Piece::Man, 16, 20, SE);
    }
    // generating for black man 16 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 19, legal: (legal black (move man g 5 f 4))
        ddi->diagonals.emplace_back(19, 193);

        // to position 23, legal: (legal black (move man g 5 e 3))
        ddi->diagonals.emplace_back(23, 195);

        this->diagonal_data[79].push_back(ddi);


        this->reverse_legal_lookup_black[193] = ReverseLegalLookup(Role::Black, Piece::Man, 16, 19, SW);
        this->reverse_legal_lookup_black[195] = ReverseLegalLookup(Role::Black, Piece::Man, 16, 23, SW);
    }
    // generating for black man 17 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 14, legal: invalid
        ddi->diagonals.emplace_back(14, -403);

        // to position 10, legal: (legal black (move man b 4 d 6))
        ddi->diagonals.emplace_back(10, 207);

        this->diagonal_data[80].push_back(ddi);


        this->reverse_legal_lookup_black[207] = ReverseLegalLookup(Role::Black, Piece::Man, 17, 10, NE);
    }
    // generating for black man 17 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 13, legal: invalid
        ddi->diagonals.emplace_back(13, -403);

        this->diagonal_data[80].push_back(ddi);


    }
    // generating for black man 17 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 22, legal: (legal black (move man b 4 c 3))
        ddi->diagonals.emplace_back(22, 205);

        // to position 26, legal: (legal black (move man b 4 d 2))
        ddi->diagonals.emplace_back(26, 208);

        this->diagonal_data[80].push_back(ddi);


        this->reverse_legal_lookup_black[205] = ReverseLegalLookup(Role::Black, Piece::Man, 17, 22, SE);
        this->reverse_legal_lookup_black[208] = ReverseLegalLookup(Role::Black, Piece::Man, 17, 26, SE);
    }
    // generating for black man 17 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 21, legal: (legal black (move man b 4 a 3))
        ddi->diagonals.emplace_back(21, 206);

        this->diagonal_data[80].push_back(ddi);


        this->reverse_legal_lookup_black[206] = ReverseLegalLookup(Role::Black, Piece::Man, 17, 21, SW);
    }
    // generating for black man 18 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 15, legal: invalid
        ddi->diagonals.emplace_back(15, -403);

        // to position 11, legal: (legal black (move man d 4 f 6))
        ddi->diagonals.emplace_back(11, 220);

        this->diagonal_data[81].push_back(ddi);


        this->reverse_legal_lookup_black[220] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 11, NE);
    }
    // generating for black man 18 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 14, legal: invalid
        ddi->diagonals.emplace_back(14, -403);

        // to position 9, legal: (legal black (move man d 4 b 6))
        ddi->diagonals.emplace_back(9, 221);

        this->diagonal_data[81].push_back(ddi);


        this->reverse_legal_lookup_black[221] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 9, NW);
    }
    // generating for black man 18 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 23, legal: (legal black (move man d 4 e 3))
        ddi->diagonals.emplace_back(23, 218);

        // to position 27, legal: (legal black (move man d 4 f 2))
        ddi->diagonals.emplace_back(27, 222);

        this->diagonal_data[81].push_back(ddi);


        this->reverse_legal_lookup_black[218] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 23, SE);
        this->reverse_legal_lookup_black[222] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 27, SE);
    }
    // generating for black man 18 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 22, legal: (legal black (move man d 4 c 3))
        ddi->diagonals.emplace_back(22, 219);

        // to position 25, legal: (legal black (move man d 4 b 2))
        ddi->diagonals.emplace_back(25, 223);

        this->diagonal_data[81].push_back(ddi);


        this->reverse_legal_lookup_black[219] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 22, SW);
        this->reverse_legal_lookup_black[223] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 25, SW);
    }
    // generating for black man 19 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 16, legal: invalid
        ddi->diagonals.emplace_back(16, -403);

        // to position 12, legal: (legal black (move man f 4 h 6))
        ddi->diagonals.emplace_back(12, 239);

        this->diagonal_data[82].push_back(ddi);


        this->reverse_legal_lookup_black[239] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 12, NE);
    }
    // generating for black man 19 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 15, legal: invalid
        ddi->diagonals.emplace_back(15, -403);

        // to position 10, legal: (legal black (move man f 4 d 6))
        ddi->diagonals.emplace_back(10, 240);

        this->diagonal_data[82].push_back(ddi);


        this->reverse_legal_lookup_black[240] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 10, NW);
    }
    // generating for black man 19 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal black (move man f 4 g 3))
        ddi->diagonals.emplace_back(24, 237);

        // to position 28, legal: (legal black (move man f 4 h 2))
        ddi->diagonals.emplace_back(28, 241);

        this->diagonal_data[82].push_back(ddi);


        this->reverse_legal_lookup_black[237] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 24, SE);
        this->reverse_legal_lookup_black[241] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 28, SE);
    }
    // generating for black man 19 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 23, legal: (legal black (move man f 4 e 3))
        ddi->diagonals.emplace_back(23, 238);

        // to position 26, legal: (legal black (move man f 4 d 2))
        ddi->diagonals.emplace_back(26, 242);

        this->diagonal_data[82].push_back(ddi);


        this->reverse_legal_lookup_black[238] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 23, SW);
        this->reverse_legal_lookup_black[242] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 26, SW);
    }
    // generating for black man 20 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 16, legal: invalid
        ddi->diagonals.emplace_back(16, -403);

        // to position 11, legal: (legal black (move man h 4 f 6))
        ddi->diagonals.emplace_back(11, 255);

        this->diagonal_data[83].push_back(ddi);


        this->reverse_legal_lookup_black[255] = ReverseLegalLookup(Role::Black, Piece::Man, 20, 11, NW);
    }
    // generating for black man 20 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal black (move man h 4 g 3))
        ddi->diagonals.emplace_back(24, 254);

        // to position 27, legal: (legal black (move man h 4 f 2))
        ddi->diagonals.emplace_back(27, 256);

        this->diagonal_data[83].push_back(ddi);


        this->reverse_legal_lookup_black[254] = ReverseLegalLookup(Role::Black, Piece::Man, 20, 24, SW);
        this->reverse_legal_lookup_black[256] = ReverseLegalLookup(Role::Black, Piece::Man, 20, 27, SW);
    }
    // generating for black man 21 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 17, legal: invalid
        ddi->diagonals.emplace_back(17, -403);

        // to position 14, legal: (legal black (move man a 3 c 5))
        ddi->diagonals.emplace_back(14, 265);

        this->diagonal_data[84].push_back(ddi);


        this->reverse_legal_lookup_black[265] = ReverseLegalLookup(Role::Black, Piece::Man, 21, 14, NE);
    }
    // generating for black man 21 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 25, legal: (legal black (move man a 3 b 2))
        ddi->diagonals.emplace_back(25, 264);

        // to position 30, legal: (legal black (move man a 3 c 1))
        ddi->diagonals.emplace_back(30, 266);

        this->diagonal_data[84].push_back(ddi);


        this->reverse_legal_lookup_black[264] = ReverseLegalLookup(Role::Black, Piece::Man, 21, 25, SE);
        this->reverse_legal_lookup_black[266] = ReverseLegalLookup(Role::Black, Piece::Man, 21, 30, SE);
    }
    // generating for black man 22 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 18, legal: invalid
        ddi->diagonals.emplace_back(18, -403);

        // to position 15, legal: (legal black (move man c 3 e 5))
        ddi->diagonals.emplace_back(15, 276);

        this->diagonal_data[85].push_back(ddi);


        this->reverse_legal_lookup_black[276] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 15, NE);
    }
    // generating for black man 22 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: invalid
        ddi->diagonals.emplace_back(17, -403);

        // to position 13, legal: (legal black (move man c 3 a 5))
        ddi->diagonals.emplace_back(13, 277);

        this->diagonal_data[85].push_back(ddi);


        this->reverse_legal_lookup_black[277] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 13, NW);
    }
    // generating for black man 22 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 26, legal: (legal black (move man c 3 d 2))
        ddi->diagonals.emplace_back(26, 274);

        // to position 31, legal: (legal black (move man c 3 e 1))
        ddi->diagonals.emplace_back(31, 278);

        this->diagonal_data[85].push_back(ddi);


        this->reverse_legal_lookup_black[274] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 26, SE);
        this->reverse_legal_lookup_black[278] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 31, SE);
    }
    // generating for black man 22 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 25, legal: (legal black (move man c 3 b 2))
        ddi->diagonals.emplace_back(25, 275);

        // to position 29, legal: (legal black (move man c 3 a 1))
        ddi->diagonals.emplace_back(29, 279);

        this->diagonal_data[85].push_back(ddi);


        this->reverse_legal_lookup_black[275] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 25, SW);
        this->reverse_legal_lookup_black[279] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 29, SW);
    }
    // generating for black man 23 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 19, legal: invalid
        ddi->diagonals.emplace_back(19, -403);

        // to position 16, legal: (legal black (move man e 3 g 5))
        ddi->diagonals.emplace_back(16, 293);

        this->diagonal_data[86].push_back(ddi);


        this->reverse_legal_lookup_black[293] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 16, NE);
    }
    // generating for black man 23 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 18, legal: invalid
        ddi->diagonals.emplace_back(18, -403);

        // to position 14, legal: (legal black (move man e 3 c 5))
        ddi->diagonals.emplace_back(14, 294);

        this->diagonal_data[86].push_back(ddi);


        this->reverse_legal_lookup_black[294] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 14, NW);
    }
    // generating for black man 23 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal black (move man e 3 f 2))
        ddi->diagonals.emplace_back(27, 291);

        // to position 32, legal: (legal black (move man e 3 g 1))
        ddi->diagonals.emplace_back(32, 295);

        this->diagonal_data[86].push_back(ddi);


        this->reverse_legal_lookup_black[291] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 27, SE);
        this->reverse_legal_lookup_black[295] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 32, SE);
    }
    // generating for black man 23 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 26, legal: (legal black (move man e 3 d 2))
        ddi->diagonals.emplace_back(26, 292);

        // to position 30, legal: (legal black (move man e 3 c 1))
        ddi->diagonals.emplace_back(30, 296);

        this->diagonal_data[86].push_back(ddi);


        this->reverse_legal_lookup_black[292] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 26, SW);
        this->reverse_legal_lookup_black[296] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 30, SW);
    }
    // generating for black man 24 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 20, legal: invalid
        ddi->diagonals.emplace_back(20, -403);

        this->diagonal_data[87].push_back(ddi);


    }
    // generating for black man 24 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 19, legal: invalid
        ddi->diagonals.emplace_back(19, -403);

        // to position 15, legal: (legal black (move man g 3 e 5))
        ddi->diagonals.emplace_back(15, 310);

        this->diagonal_data[87].push_back(ddi);


        this->reverse_legal_lookup_black[310] = ReverseLegalLookup(Role::Black, Piece::Man, 24, 15, NW);
    }
    // generating for black man 24 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 28, legal: (legal black (move man g 3 h 2))
        ddi->diagonals.emplace_back(28, 308);

        this->diagonal_data[87].push_back(ddi);


        this->reverse_legal_lookup_black[308] = ReverseLegalLookup(Role::Black, Piece::Man, 24, 28, SE);
    }
    // generating for black man 24 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal black (move man g 3 f 2))
        ddi->diagonals.emplace_back(27, 309);

        // to position 31, legal: (legal black (move man g 3 e 1))
        ddi->diagonals.emplace_back(31, 311);

        this->diagonal_data[87].push_back(ddi);


        this->reverse_legal_lookup_black[309] = ReverseLegalLookup(Role::Black, Piece::Man, 24, 27, SW);
        this->reverse_legal_lookup_black[311] = ReverseLegalLookup(Role::Black, Piece::Man, 24, 31, SW);
    }
    // generating for black man 25 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 22, legal: invalid
        ddi->diagonals.emplace_back(22, -403);

        // to position 18, legal: (legal black (move man b 2 d 4))
        ddi->diagonals.emplace_back(18, 323);

        this->diagonal_data[88].push_back(ddi);


        this->reverse_legal_lookup_black[323] = ReverseLegalLookup(Role::Black, Piece::Man, 25, 18, NE);
    }
    // generating for black man 25 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 21, legal: invalid
        ddi->diagonals.emplace_back(21, -403);

        this->diagonal_data[88].push_back(ddi);


    }
    // generating for black man 25 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 30, legal: (legal black (move man b 2 c 1))
        ddi->diagonals.emplace_back(30, 321);

        this->diagonal_data[88].push_back(ddi);


        this->reverse_legal_lookup_black[321] = ReverseLegalLookup(Role::Black, Piece::Man, 25, 30, SE);
    }
    // generating for black man 25 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 29, legal: (legal black (move man b 2 a 1))
        ddi->diagonals.emplace_back(29, 322);

        this->diagonal_data[88].push_back(ddi);


        this->reverse_legal_lookup_black[322] = ReverseLegalLookup(Role::Black, Piece::Man, 25, 29, SW);
    }
    // generating for black man 26 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 23, legal: invalid
        ddi->diagonals.emplace_back(23, -403);

        // to position 19, legal: (legal black (move man d 2 f 4))
        ddi->diagonals.emplace_back(19, 335);

        this->diagonal_data[89].push_back(ddi);


        this->reverse_legal_lookup_black[335] = ReverseLegalLookup(Role::Black, Piece::Man, 26, 19, NE);
    }
    // generating for black man 26 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 22, legal: invalid
        ddi->diagonals.emplace_back(22, -403);

        // to position 17, legal: (legal black (move man d 2 b 4))
        ddi->diagonals.emplace_back(17, 336);

        this->diagonal_data[89].push_back(ddi);


        this->reverse_legal_lookup_black[336] = ReverseLegalLookup(Role::Black, Piece::Man, 26, 17, NW);
    }
    // generating for black man 26 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 31, legal: (legal black (move man d 2 e 1))
        ddi->diagonals.emplace_back(31, 333);

        this->diagonal_data[89].push_back(ddi);


        this->reverse_legal_lookup_black[333] = ReverseLegalLookup(Role::Black, Piece::Man, 26, 31, SE);
    }
    // generating for black man 26 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 30, legal: (legal black (move man d 2 c 1))
        ddi->diagonals.emplace_back(30, 334);

        this->diagonal_data[89].push_back(ddi);


        this->reverse_legal_lookup_black[334] = ReverseLegalLookup(Role::Black, Piece::Man, 26, 30, SW);
    }
    // generating for black man 27 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: invalid
        ddi->diagonals.emplace_back(24, -403);

        // to position 20, legal: (legal black (move man f 2 h 4))
        ddi->diagonals.emplace_back(20, 348);

        this->diagonal_data[90].push_back(ddi);


        this->reverse_legal_lookup_black[348] = ReverseLegalLookup(Role::Black, Piece::Man, 27, 20, NE);
    }
    // generating for black man 27 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 23, legal: invalid
        ddi->diagonals.emplace_back(23, -403);

        // to position 18, legal: (legal black (move man f 2 d 4))
        ddi->diagonals.emplace_back(18, 349);

        this->diagonal_data[90].push_back(ddi);


        this->reverse_legal_lookup_black[349] = ReverseLegalLookup(Role::Black, Piece::Man, 27, 18, NW);
    }
    // generating for black man 27 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 32, legal: (legal black (move man f 2 g 1))
        ddi->diagonals.emplace_back(32, 346);

        this->diagonal_data[90].push_back(ddi);


        this->reverse_legal_lookup_black[346] = ReverseLegalLookup(Role::Black, Piece::Man, 27, 32, SE);
    }
    // generating for black man 27 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 31, legal: (legal black (move man f 2 e 1))
        ddi->diagonals.emplace_back(31, 347);

        this->diagonal_data[90].push_back(ddi);


        this->reverse_legal_lookup_black[347] = ReverseLegalLookup(Role::Black, Piece::Man, 27, 31, SW);
    }
    // generating for black man 28 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 24, legal: invalid
        ddi->diagonals.emplace_back(24, -403);

        // to position 19, legal: (legal black (move man h 2 f 4))
        ddi->diagonals.emplace_back(19, 360);

        this->diagonal_data[91].push_back(ddi);


        this->reverse_legal_lookup_black[360] = ReverseLegalLookup(Role::Black, Piece::Man, 28, 19, NW);
    }
    // generating for black man 28 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 32, legal: (legal black (move man h 2 g 1))
        ddi->diagonals.emplace_back(32, 359);

        this->diagonal_data[91].push_back(ddi);


        this->reverse_legal_lookup_black[359] = ReverseLegalLookup(Role::Black, Piece::Man, 28, 32, SW);
    }
    // generating for black man 29 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 25, legal: invalid
        ddi->diagonals.emplace_back(25, -403);

        // to position 22, legal: (legal black (move man a 1 c 3))
        ddi->diagonals.emplace_back(22, 368);

        this->diagonal_data[92].push_back(ddi);


        this->reverse_legal_lookup_black[368] = ReverseLegalLookup(Role::Black, Piece::Man, 29, 22, NE);
    }
    // generating for black man 30 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 26, legal: invalid
        ddi->diagonals.emplace_back(26, -403);

        // to position 23, legal: (legal black (move man c 1 e 3))
        ddi->diagonals.emplace_back(23, 376);

        this->diagonal_data[93].push_back(ddi);


        this->reverse_legal_lookup_black[376] = ReverseLegalLookup(Role::Black, Piece::Man, 30, 23, NE);
    }
    // generating for black man 30 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 25, legal: invalid
        ddi->diagonals.emplace_back(25, -403);

        // to position 21, legal: (legal black (move man c 1 a 3))
        ddi->diagonals.emplace_back(21, 377);

        this->diagonal_data[93].push_back(ddi);


        this->reverse_legal_lookup_black[377] = ReverseLegalLookup(Role::Black, Piece::Man, 30, 21, NW);
    }
    // generating for black man 31 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 27, legal: invalid
        ddi->diagonals.emplace_back(27, -403);

        // to position 24, legal: (legal black (move man e 1 g 3))
        ddi->diagonals.emplace_back(24, 385);

        this->diagonal_data[94].push_back(ddi);


        this->reverse_legal_lookup_black[385] = ReverseLegalLookup(Role::Black, Piece::Man, 31, 24, NE);
    }
    // generating for black man 31 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 26, legal: invalid
        ddi->diagonals.emplace_back(26, -403);

        // to position 22, legal: (legal black (move man e 1 c 3))
        ddi->diagonals.emplace_back(22, 386);

        this->diagonal_data[94].push_back(ddi);


        this->reverse_legal_lookup_black[386] = ReverseLegalLookup(Role::Black, Piece::Man, 31, 22, NW);
    }
    // generating for black man 32 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 28, legal: invalid
        ddi->diagonals.emplace_back(28, -403);

        this->diagonal_data[95].push_back(ddi);


    }
    // generating for black man 32 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 27, legal: invalid
        ddi->diagonals.emplace_back(27, -403);

        // to position 23, legal: (legal black (move man g 1 e 3))
        ddi->diagonals.emplace_back(23, 394);

        this->diagonal_data[95].push_back(ddi);


        this->reverse_legal_lookup_black[394] = ReverseLegalLookup(Role::Black, Piece::Man, 32, 23, NW);
    }
    // generating for black king 1 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 6, legal: (legal black (move king b 8 c 7))
        ddi->diagonals.emplace_back(6, 4);

        // to position 10, legal: (legal black (move king b 8 d 6))
        ddi->diagonals.emplace_back(10, 5);

        // to position 15, legal: (legal black (move king b 8 e 5))
        ddi->diagonals.emplace_back(15, 6);

        // to position 19, legal: (legal black (move king b 8 f 4))
        ddi->diagonals.emplace_back(19, 7);

        // to position 24, legal: (legal black (move king b 8 g 3))
        ddi->diagonals.emplace_back(24, 8);

        // to position 28, legal: (legal black (move king b 8 h 2))
        ddi->diagonals.emplace_back(28, 9);

        this->diagonal_data[96].push_back(ddi);


        this->reverse_legal_lookup_black[4] = ReverseLegalLookup(Role::Black, Piece::King, 1, 6, SE);
        this->reverse_legal_lookup_black[5] = ReverseLegalLookup(Role::Black, Piece::King, 1, 10, SE);
        this->reverse_legal_lookup_black[6] = ReverseLegalLookup(Role::Black, Piece::King, 1, 15, SE);
        this->reverse_legal_lookup_black[7] = ReverseLegalLookup(Role::Black, Piece::King, 1, 19, SE);
        this->reverse_legal_lookup_black[8] = ReverseLegalLookup(Role::Black, Piece::King, 1, 24, SE);
        this->reverse_legal_lookup_black[9] = ReverseLegalLookup(Role::Black, Piece::King, 1, 28, SE);
    }
    // generating for black king 1 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 5, legal: (legal black (move king b 8 a 7))
        ddi->diagonals.emplace_back(5, 10);

        this->diagonal_data[96].push_back(ddi);


        this->reverse_legal_lookup_black[10] = ReverseLegalLookup(Role::Black, Piece::King, 1, 5, SW);
    }
    // generating for black king 2 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 7, legal: (legal black (move king d 8 e 7))
        ddi->diagonals.emplace_back(7, 15);

        // to position 11, legal: (legal black (move king d 8 f 6))
        ddi->diagonals.emplace_back(11, 16);

        // to position 16, legal: (legal black (move king d 8 g 5))
        ddi->diagonals.emplace_back(16, 17);

        // to position 20, legal: (legal black (move king d 8 h 4))
        ddi->diagonals.emplace_back(20, 18);

        this->diagonal_data[97].push_back(ddi);


        this->reverse_legal_lookup_black[15] = ReverseLegalLookup(Role::Black, Piece::King, 2, 7, SE);
        this->reverse_legal_lookup_black[16] = ReverseLegalLookup(Role::Black, Piece::King, 2, 11, SE);
        this->reverse_legal_lookup_black[17] = ReverseLegalLookup(Role::Black, Piece::King, 2, 16, SE);
        this->reverse_legal_lookup_black[18] = ReverseLegalLookup(Role::Black, Piece::King, 2, 20, SE);
    }
    // generating for black king 2 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 6, legal: (legal black (move king d 8 c 7))
        ddi->diagonals.emplace_back(6, 19);

        // to position 9, legal: (legal black (move king d 8 b 6))
        ddi->diagonals.emplace_back(9, 20);

        // to position 13, legal: (legal black (move king d 8 a 5))
        ddi->diagonals.emplace_back(13, 21);

        this->diagonal_data[97].push_back(ddi);


        this->reverse_legal_lookup_black[19] = ReverseLegalLookup(Role::Black, Piece::King, 2, 6, SW);
        this->reverse_legal_lookup_black[20] = ReverseLegalLookup(Role::Black, Piece::King, 2, 9, SW);
        this->reverse_legal_lookup_black[21] = ReverseLegalLookup(Role::Black, Piece::King, 2, 13, SW);
    }
    // generating for black king 3 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal black (move king f 8 g 7))
        ddi->diagonals.emplace_back(8, 26);

        // to position 12, legal: (legal black (move king f 8 h 6))
        ddi->diagonals.emplace_back(12, 27);

        this->diagonal_data[98].push_back(ddi);


        this->reverse_legal_lookup_black[26] = ReverseLegalLookup(Role::Black, Piece::King, 3, 8, SE);
        this->reverse_legal_lookup_black[27] = ReverseLegalLookup(Role::Black, Piece::King, 3, 12, SE);
    }
    // generating for black king 3 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 7, legal: (legal black (move king f 8 e 7))
        ddi->diagonals.emplace_back(7, 28);

        // to position 10, legal: (legal black (move king f 8 d 6))
        ddi->diagonals.emplace_back(10, 29);

        // to position 14, legal: (legal black (move king f 8 c 5))
        ddi->diagonals.emplace_back(14, 30);

        // to position 17, legal: (legal black (move king f 8 b 4))
        ddi->diagonals.emplace_back(17, 31);

        // to position 21, legal: (legal black (move king f 8 a 3))
        ddi->diagonals.emplace_back(21, 32);

        this->diagonal_data[98].push_back(ddi);


        this->reverse_legal_lookup_black[28] = ReverseLegalLookup(Role::Black, Piece::King, 3, 7, SW);
        this->reverse_legal_lookup_black[29] = ReverseLegalLookup(Role::Black, Piece::King, 3, 10, SW);
        this->reverse_legal_lookup_black[30] = ReverseLegalLookup(Role::Black, Piece::King, 3, 14, SW);
        this->reverse_legal_lookup_black[31] = ReverseLegalLookup(Role::Black, Piece::King, 3, 17, SW);
        this->reverse_legal_lookup_black[32] = ReverseLegalLookup(Role::Black, Piece::King, 3, 21, SW);
    }
    // generating for black king 4 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(7);
        // to position 8, legal: (legal black (move king h 8 g 7))
        ddi->diagonals.emplace_back(8, 35);

        // to position 11, legal: (legal black (move king h 8 f 6))
        ddi->diagonals.emplace_back(11, 36);

        // to position 15, legal: (legal black (move king h 8 e 5))
        ddi->diagonals.emplace_back(15, 37);

        // to position 18, legal: (legal black (move king h 8 d 4))
        ddi->diagonals.emplace_back(18, 38);

        // to position 22, legal: (legal black (move king h 8 c 3))
        ddi->diagonals.emplace_back(22, 39);

        // to position 25, legal: (legal black (move king h 8 b 2))
        ddi->diagonals.emplace_back(25, 40);

        // to position 29, legal: (legal black (move king h 8 a 1))
        ddi->diagonals.emplace_back(29, 41);

        this->diagonal_data[99].push_back(ddi);


        this->reverse_legal_lookup_black[35] = ReverseLegalLookup(Role::Black, Piece::King, 4, 8, SW);
        this->reverse_legal_lookup_black[36] = ReverseLegalLookup(Role::Black, Piece::King, 4, 11, SW);
        this->reverse_legal_lookup_black[37] = ReverseLegalLookup(Role::Black, Piece::King, 4, 15, SW);
        this->reverse_legal_lookup_black[38] = ReverseLegalLookup(Role::Black, Piece::King, 4, 18, SW);
        this->reverse_legal_lookup_black[39] = ReverseLegalLookup(Role::Black, Piece::King, 4, 22, SW);
        this->reverse_legal_lookup_black[40] = ReverseLegalLookup(Role::Black, Piece::King, 4, 25, SW);
        this->reverse_legal_lookup_black[41] = ReverseLegalLookup(Role::Black, Piece::King, 4, 29, SW);
    }
    // generating for black king 5 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal black (move king a 7 b 8))
        ddi->diagonals.emplace_back(1, 44);

        this->diagonal_data[100].push_back(ddi);


        this->reverse_legal_lookup_black[44] = ReverseLegalLookup(Role::Black, Piece::King, 5, 1, NE);
    }
    // generating for black king 5 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 9, legal: (legal black (move king a 7 b 6))
        ddi->diagonals.emplace_back(9, 45);

        // to position 14, legal: (legal black (move king a 7 c 5))
        ddi->diagonals.emplace_back(14, 46);

        // to position 18, legal: (legal black (move king a 7 d 4))
        ddi->diagonals.emplace_back(18, 47);

        // to position 23, legal: (legal black (move king a 7 e 3))
        ddi->diagonals.emplace_back(23, 48);

        // to position 27, legal: (legal black (move king a 7 f 2))
        ddi->diagonals.emplace_back(27, 49);

        // to position 32, legal: (legal black (move king a 7 g 1))
        ddi->diagonals.emplace_back(32, 50);

        this->diagonal_data[100].push_back(ddi);


        this->reverse_legal_lookup_black[45] = ReverseLegalLookup(Role::Black, Piece::King, 5, 9, SE);
        this->reverse_legal_lookup_black[46] = ReverseLegalLookup(Role::Black, Piece::King, 5, 14, SE);
        this->reverse_legal_lookup_black[47] = ReverseLegalLookup(Role::Black, Piece::King, 5, 18, SE);
        this->reverse_legal_lookup_black[48] = ReverseLegalLookup(Role::Black, Piece::King, 5, 23, SE);
        this->reverse_legal_lookup_black[49] = ReverseLegalLookup(Role::Black, Piece::King, 5, 27, SE);
        this->reverse_legal_lookup_black[50] = ReverseLegalLookup(Role::Black, Piece::King, 5, 32, SE);
    }
    // generating for black king 6 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal black (move king c 7 d 8))
        ddi->diagonals.emplace_back(2, 55);

        this->diagonal_data[101].push_back(ddi);


        this->reverse_legal_lookup_black[55] = ReverseLegalLookup(Role::Black, Piece::King, 6, 2, NE);
    }
    // generating for black king 6 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal black (move king c 7 b 8))
        ddi->diagonals.emplace_back(1, 56);

        this->diagonal_data[101].push_back(ddi);


        this->reverse_legal_lookup_black[56] = ReverseLegalLookup(Role::Black, Piece::King, 6, 1, NW);
    }
    // generating for black king 6 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 10, legal: (legal black (move king c 7 d 6))
        ddi->diagonals.emplace_back(10, 57);

        // to position 15, legal: (legal black (move king c 7 e 5))
        ddi->diagonals.emplace_back(15, 58);

        // to position 19, legal: (legal black (move king c 7 f 4))
        ddi->diagonals.emplace_back(19, 59);

        // to position 24, legal: (legal black (move king c 7 g 3))
        ddi->diagonals.emplace_back(24, 60);

        // to position 28, legal: (legal black (move king c 7 h 2))
        ddi->diagonals.emplace_back(28, 61);

        this->diagonal_data[101].push_back(ddi);


        this->reverse_legal_lookup_black[57] = ReverseLegalLookup(Role::Black, Piece::King, 6, 10, SE);
        this->reverse_legal_lookup_black[58] = ReverseLegalLookup(Role::Black, Piece::King, 6, 15, SE);
        this->reverse_legal_lookup_black[59] = ReverseLegalLookup(Role::Black, Piece::King, 6, 19, SE);
        this->reverse_legal_lookup_black[60] = ReverseLegalLookup(Role::Black, Piece::King, 6, 24, SE);
        this->reverse_legal_lookup_black[61] = ReverseLegalLookup(Role::Black, Piece::King, 6, 28, SE);
    }
    // generating for black king 6 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal black (move king c 7 b 6))
        ddi->diagonals.emplace_back(9, 62);

        // to position 13, legal: (legal black (move king c 7 a 5))
        ddi->diagonals.emplace_back(13, 63);

        this->diagonal_data[101].push_back(ddi);


        this->reverse_legal_lookup_black[62] = ReverseLegalLookup(Role::Black, Piece::King, 6, 9, SW);
        this->reverse_legal_lookup_black[63] = ReverseLegalLookup(Role::Black, Piece::King, 6, 13, SW);
    }
    // generating for black king 7 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal black (move king e 7 f 8))
        ddi->diagonals.emplace_back(3, 68);

        this->diagonal_data[102].push_back(ddi);


        this->reverse_legal_lookup_black[68] = ReverseLegalLookup(Role::Black, Piece::King, 7, 3, NE);
    }
    // generating for black king 7 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal black (move king e 7 d 8))
        ddi->diagonals.emplace_back(2, 69);

        this->diagonal_data[102].push_back(ddi);


        this->reverse_legal_lookup_black[69] = ReverseLegalLookup(Role::Black, Piece::King, 7, 2, NW);
    }
    // generating for black king 7 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 11, legal: (legal black (move king e 7 f 6))
        ddi->diagonals.emplace_back(11, 70);

        // to position 16, legal: (legal black (move king e 7 g 5))
        ddi->diagonals.emplace_back(16, 71);

        // to position 20, legal: (legal black (move king e 7 h 4))
        ddi->diagonals.emplace_back(20, 72);

        this->diagonal_data[102].push_back(ddi);


        this->reverse_legal_lookup_black[70] = ReverseLegalLookup(Role::Black, Piece::King, 7, 11, SE);
        this->reverse_legal_lookup_black[71] = ReverseLegalLookup(Role::Black, Piece::King, 7, 16, SE);
        this->reverse_legal_lookup_black[72] = ReverseLegalLookup(Role::Black, Piece::King, 7, 20, SE);
    }
    // generating for black king 7 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 10, legal: (legal black (move king e 7 d 6))
        ddi->diagonals.emplace_back(10, 73);

        // to position 14, legal: (legal black (move king e 7 c 5))
        ddi->diagonals.emplace_back(14, 74);

        // to position 17, legal: (legal black (move king e 7 b 4))
        ddi->diagonals.emplace_back(17, 75);

        // to position 21, legal: (legal black (move king e 7 a 3))
        ddi->diagonals.emplace_back(21, 76);

        this->diagonal_data[102].push_back(ddi);


        this->reverse_legal_lookup_black[73] = ReverseLegalLookup(Role::Black, Piece::King, 7, 10, SW);
        this->reverse_legal_lookup_black[74] = ReverseLegalLookup(Role::Black, Piece::King, 7, 14, SW);
        this->reverse_legal_lookup_black[75] = ReverseLegalLookup(Role::Black, Piece::King, 7, 17, SW);
        this->reverse_legal_lookup_black[76] = ReverseLegalLookup(Role::Black, Piece::King, 7, 21, SW);
    }
    // generating for black king 8 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 4, legal: (legal black (move king g 7 h 8))
        ddi->diagonals.emplace_back(4, 80);

        this->diagonal_data[103].push_back(ddi);


        this->reverse_legal_lookup_black[80] = ReverseLegalLookup(Role::Black, Piece::King, 8, 4, NE);
    }
    // generating for black king 8 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal black (move king g 7 f 8))
        ddi->diagonals.emplace_back(3, 81);

        this->diagonal_data[103].push_back(ddi);


        this->reverse_legal_lookup_black[81] = ReverseLegalLookup(Role::Black, Piece::King, 8, 3, NW);
    }
    // generating for black king 8 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 12, legal: (legal black (move king g 7 h 6))
        ddi->diagonals.emplace_back(12, 82);

        this->diagonal_data[103].push_back(ddi);


        this->reverse_legal_lookup_black[82] = ReverseLegalLookup(Role::Black, Piece::King, 8, 12, SE);
    }
    // generating for black king 8 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(6);
        // to position 11, legal: (legal black (move king g 7 f 6))
        ddi->diagonals.emplace_back(11, 83);

        // to position 15, legal: (legal black (move king g 7 e 5))
        ddi->diagonals.emplace_back(15, 84);

        // to position 18, legal: (legal black (move king g 7 d 4))
        ddi->diagonals.emplace_back(18, 85);

        // to position 22, legal: (legal black (move king g 7 c 3))
        ddi->diagonals.emplace_back(22, 86);

        // to position 25, legal: (legal black (move king g 7 b 2))
        ddi->diagonals.emplace_back(25, 87);

        // to position 29, legal: (legal black (move king g 7 a 1))
        ddi->diagonals.emplace_back(29, 88);

        this->diagonal_data[103].push_back(ddi);


        this->reverse_legal_lookup_black[83] = ReverseLegalLookup(Role::Black, Piece::King, 8, 11, SW);
        this->reverse_legal_lookup_black[84] = ReverseLegalLookup(Role::Black, Piece::King, 8, 15, SW);
        this->reverse_legal_lookup_black[85] = ReverseLegalLookup(Role::Black, Piece::King, 8, 18, SW);
        this->reverse_legal_lookup_black[86] = ReverseLegalLookup(Role::Black, Piece::King, 8, 22, SW);
        this->reverse_legal_lookup_black[87] = ReverseLegalLookup(Role::Black, Piece::King, 8, 25, SW);
        this->reverse_legal_lookup_black[88] = ReverseLegalLookup(Role::Black, Piece::King, 8, 29, SW);
    }
    // generating for black king 9 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 6, legal: (legal black (move king b 6 c 7))
        ddi->diagonals.emplace_back(6, 93);

        // to position 2, legal: (legal black (move king b 6 d 8))
        ddi->diagonals.emplace_back(2, 94);

        this->diagonal_data[104].push_back(ddi);


        this->reverse_legal_lookup_black[93] = ReverseLegalLookup(Role::Black, Piece::King, 9, 6, NE);
        this->reverse_legal_lookup_black[94] = ReverseLegalLookup(Role::Black, Piece::King, 9, 2, NE);
    }
    // generating for black king 9 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 5, legal: (legal black (move king b 6 a 7))
        ddi->diagonals.emplace_back(5, 95);

        this->diagonal_data[104].push_back(ddi);


        this->reverse_legal_lookup_black[95] = ReverseLegalLookup(Role::Black, Piece::King, 9, 5, NW);
    }
    // generating for black king 9 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 14, legal: (legal black (move king b 6 c 5))
        ddi->diagonals.emplace_back(14, 96);

        // to position 18, legal: (legal black (move king b 6 d 4))
        ddi->diagonals.emplace_back(18, 97);

        // to position 23, legal: (legal black (move king b 6 e 3))
        ddi->diagonals.emplace_back(23, 98);

        // to position 27, legal: (legal black (move king b 6 f 2))
        ddi->diagonals.emplace_back(27, 99);

        // to position 32, legal: (legal black (move king b 6 g 1))
        ddi->diagonals.emplace_back(32, 100);

        this->diagonal_data[104].push_back(ddi);


        this->reverse_legal_lookup_black[96] = ReverseLegalLookup(Role::Black, Piece::King, 9, 14, SE);
        this->reverse_legal_lookup_black[97] = ReverseLegalLookup(Role::Black, Piece::King, 9, 18, SE);
        this->reverse_legal_lookup_black[98] = ReverseLegalLookup(Role::Black, Piece::King, 9, 23, SE);
        this->reverse_legal_lookup_black[99] = ReverseLegalLookup(Role::Black, Piece::King, 9, 27, SE);
        this->reverse_legal_lookup_black[100] = ReverseLegalLookup(Role::Black, Piece::King, 9, 32, SE);
    }
    // generating for black king 9 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 13, legal: (legal black (move king b 6 a 5))
        ddi->diagonals.emplace_back(13, 101);

        this->diagonal_data[104].push_back(ddi);


        this->reverse_legal_lookup_black[101] = ReverseLegalLookup(Role::Black, Piece::King, 9, 13, SW);
    }
    // generating for black king 10 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal black (move king d 6 e 7))
        ddi->diagonals.emplace_back(7, 108);

        // to position 3, legal: (legal black (move king d 6 f 8))
        ddi->diagonals.emplace_back(3, 109);

        this->diagonal_data[105].push_back(ddi);


        this->reverse_legal_lookup_black[108] = ReverseLegalLookup(Role::Black, Piece::King, 10, 7, NE);
        this->reverse_legal_lookup_black[109] = ReverseLegalLookup(Role::Black, Piece::King, 10, 3, NE);
    }
    // generating for black king 10 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 6, legal: (legal black (move king d 6 c 7))
        ddi->diagonals.emplace_back(6, 110);

        // to position 1, legal: (legal black (move king d 6 b 8))
        ddi->diagonals.emplace_back(1, 111);

        this->diagonal_data[105].push_back(ddi);


        this->reverse_legal_lookup_black[110] = ReverseLegalLookup(Role::Black, Piece::King, 10, 6, NW);
        this->reverse_legal_lookup_black[111] = ReverseLegalLookup(Role::Black, Piece::King, 10, 1, NW);
    }
    // generating for black king 10 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 15, legal: (legal black (move king d 6 e 5))
        ddi->diagonals.emplace_back(15, 112);

        // to position 19, legal: (legal black (move king d 6 f 4))
        ddi->diagonals.emplace_back(19, 113);

        // to position 24, legal: (legal black (move king d 6 g 3))
        ddi->diagonals.emplace_back(24, 114);

        // to position 28, legal: (legal black (move king d 6 h 2))
        ddi->diagonals.emplace_back(28, 115);

        this->diagonal_data[105].push_back(ddi);


        this->reverse_legal_lookup_black[112] = ReverseLegalLookup(Role::Black, Piece::King, 10, 15, SE);
        this->reverse_legal_lookup_black[113] = ReverseLegalLookup(Role::Black, Piece::King, 10, 19, SE);
        this->reverse_legal_lookup_black[114] = ReverseLegalLookup(Role::Black, Piece::King, 10, 24, SE);
        this->reverse_legal_lookup_black[115] = ReverseLegalLookup(Role::Black, Piece::King, 10, 28, SE);
    }
    // generating for black king 10 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 14, legal: (legal black (move king d 6 c 5))
        ddi->diagonals.emplace_back(14, 116);

        // to position 17, legal: (legal black (move king d 6 b 4))
        ddi->diagonals.emplace_back(17, 117);

        // to position 21, legal: (legal black (move king d 6 a 3))
        ddi->diagonals.emplace_back(21, 118);

        this->diagonal_data[105].push_back(ddi);


        this->reverse_legal_lookup_black[116] = ReverseLegalLookup(Role::Black, Piece::King, 10, 14, SW);
        this->reverse_legal_lookup_black[117] = ReverseLegalLookup(Role::Black, Piece::King, 10, 17, SW);
        this->reverse_legal_lookup_black[118] = ReverseLegalLookup(Role::Black, Piece::King, 10, 21, SW);
    }
    // generating for black king 11 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal black (move king f 6 g 7))
        ddi->diagonals.emplace_back(8, 125);

        // to position 4, legal: (legal black (move king f 6 h 8))
        ddi->diagonals.emplace_back(4, 126);

        this->diagonal_data[106].push_back(ddi);


        this->reverse_legal_lookup_black[125] = ReverseLegalLookup(Role::Black, Piece::King, 11, 8, NE);
        this->reverse_legal_lookup_black[126] = ReverseLegalLookup(Role::Black, Piece::King, 11, 4, NE);
    }
    // generating for black king 11 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal black (move king f 6 e 7))
        ddi->diagonals.emplace_back(7, 127);

        // to position 2, legal: (legal black (move king f 6 d 8))
        ddi->diagonals.emplace_back(2, 128);

        this->diagonal_data[106].push_back(ddi);


        this->reverse_legal_lookup_black[127] = ReverseLegalLookup(Role::Black, Piece::King, 11, 7, NW);
        this->reverse_legal_lookup_black[128] = ReverseLegalLookup(Role::Black, Piece::King, 11, 2, NW);
    }
    // generating for black king 11 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 16, legal: (legal black (move king f 6 g 5))
        ddi->diagonals.emplace_back(16, 129);

        // to position 20, legal: (legal black (move king f 6 h 4))
        ddi->diagonals.emplace_back(20, 130);

        this->diagonal_data[106].push_back(ddi);


        this->reverse_legal_lookup_black[129] = ReverseLegalLookup(Role::Black, Piece::King, 11, 16, SE);
        this->reverse_legal_lookup_black[130] = ReverseLegalLookup(Role::Black, Piece::King, 11, 20, SE);
    }
    // generating for black king 11 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 15, legal: (legal black (move king f 6 e 5))
        ddi->diagonals.emplace_back(15, 131);

        // to position 18, legal: (legal black (move king f 6 d 4))
        ddi->diagonals.emplace_back(18, 132);

        // to position 22, legal: (legal black (move king f 6 c 3))
        ddi->diagonals.emplace_back(22, 133);

        // to position 25, legal: (legal black (move king f 6 b 2))
        ddi->diagonals.emplace_back(25, 134);

        // to position 29, legal: (legal black (move king f 6 a 1))
        ddi->diagonals.emplace_back(29, 135);

        this->diagonal_data[106].push_back(ddi);


        this->reverse_legal_lookup_black[131] = ReverseLegalLookup(Role::Black, Piece::King, 11, 15, SW);
        this->reverse_legal_lookup_black[132] = ReverseLegalLookup(Role::Black, Piece::King, 11, 18, SW);
        this->reverse_legal_lookup_black[133] = ReverseLegalLookup(Role::Black, Piece::King, 11, 22, SW);
        this->reverse_legal_lookup_black[134] = ReverseLegalLookup(Role::Black, Piece::King, 11, 25, SW);
        this->reverse_legal_lookup_black[135] = ReverseLegalLookup(Role::Black, Piece::King, 11, 29, SW);
    }
    // generating for black king 12 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal black (move king h 6 g 7))
        ddi->diagonals.emplace_back(8, 139);

        // to position 3, legal: (legal black (move king h 6 f 8))
        ddi->diagonals.emplace_back(3, 140);

        this->diagonal_data[107].push_back(ddi);


        this->reverse_legal_lookup_black[139] = ReverseLegalLookup(Role::Black, Piece::King, 12, 8, NW);
        this->reverse_legal_lookup_black[140] = ReverseLegalLookup(Role::Black, Piece::King, 12, 3, NW);
    }
    // generating for black king 12 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 16, legal: (legal black (move king h 6 g 5))
        ddi->diagonals.emplace_back(16, 141);

        // to position 19, legal: (legal black (move king h 6 f 4))
        ddi->diagonals.emplace_back(19, 142);

        // to position 23, legal: (legal black (move king h 6 e 3))
        ddi->diagonals.emplace_back(23, 143);

        // to position 26, legal: (legal black (move king h 6 d 2))
        ddi->diagonals.emplace_back(26, 144);

        // to position 30, legal: (legal black (move king h 6 c 1))
        ddi->diagonals.emplace_back(30, 145);

        this->diagonal_data[107].push_back(ddi);


        this->reverse_legal_lookup_black[141] = ReverseLegalLookup(Role::Black, Piece::King, 12, 16, SW);
        this->reverse_legal_lookup_black[142] = ReverseLegalLookup(Role::Black, Piece::King, 12, 19, SW);
        this->reverse_legal_lookup_black[143] = ReverseLegalLookup(Role::Black, Piece::King, 12, 23, SW);
        this->reverse_legal_lookup_black[144] = ReverseLegalLookup(Role::Black, Piece::King, 12, 26, SW);
        this->reverse_legal_lookup_black[145] = ReverseLegalLookup(Role::Black, Piece::King, 12, 30, SW);
    }
    // generating for black king 13 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 9, legal: (legal black (move king a 5 b 6))
        ddi->diagonals.emplace_back(9, 149);

        // to position 6, legal: (legal black (move king a 5 c 7))
        ddi->diagonals.emplace_back(6, 150);

        // to position 2, legal: (legal black (move king a 5 d 8))
        ddi->diagonals.emplace_back(2, 151);

        this->diagonal_data[108].push_back(ddi);


        this->reverse_legal_lookup_black[149] = ReverseLegalLookup(Role::Black, Piece::King, 13, 9, NE);
        this->reverse_legal_lookup_black[150] = ReverseLegalLookup(Role::Black, Piece::King, 13, 6, NE);
        this->reverse_legal_lookup_black[151] = ReverseLegalLookup(Role::Black, Piece::King, 13, 2, NE);
    }
    // generating for black king 13 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 17, legal: (legal black (move king a 5 b 4))
        ddi->diagonals.emplace_back(17, 152);

        // to position 22, legal: (legal black (move king a 5 c 3))
        ddi->diagonals.emplace_back(22, 153);

        // to position 26, legal: (legal black (move king a 5 d 2))
        ddi->diagonals.emplace_back(26, 154);

        // to position 31, legal: (legal black (move king a 5 e 1))
        ddi->diagonals.emplace_back(31, 155);

        this->diagonal_data[108].push_back(ddi);


        this->reverse_legal_lookup_black[152] = ReverseLegalLookup(Role::Black, Piece::King, 13, 17, SE);
        this->reverse_legal_lookup_black[153] = ReverseLegalLookup(Role::Black, Piece::King, 13, 22, SE);
        this->reverse_legal_lookup_black[154] = ReverseLegalLookup(Role::Black, Piece::King, 13, 26, SE);
        this->reverse_legal_lookup_black[155] = ReverseLegalLookup(Role::Black, Piece::King, 13, 31, SE);
    }
    // generating for black king 14 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 10, legal: (legal black (move king c 5 d 6))
        ddi->diagonals.emplace_back(10, 162);

        // to position 7, legal: (legal black (move king c 5 e 7))
        ddi->diagonals.emplace_back(7, 163);

        // to position 3, legal: (legal black (move king c 5 f 8))
        ddi->diagonals.emplace_back(3, 164);

        this->diagonal_data[109].push_back(ddi);


        this->reverse_legal_lookup_black[162] = ReverseLegalLookup(Role::Black, Piece::King, 14, 10, NE);
        this->reverse_legal_lookup_black[163] = ReverseLegalLookup(Role::Black, Piece::King, 14, 7, NE);
        this->reverse_legal_lookup_black[164] = ReverseLegalLookup(Role::Black, Piece::King, 14, 3, NE);
    }
    // generating for black king 14 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal black (move king c 5 b 6))
        ddi->diagonals.emplace_back(9, 165);

        // to position 5, legal: (legal black (move king c 5 a 7))
        ddi->diagonals.emplace_back(5, 166);

        this->diagonal_data[109].push_back(ddi);


        this->reverse_legal_lookup_black[165] = ReverseLegalLookup(Role::Black, Piece::King, 14, 9, NW);
        this->reverse_legal_lookup_black[166] = ReverseLegalLookup(Role::Black, Piece::King, 14, 5, NW);
    }
    // generating for black king 14 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 18, legal: (legal black (move king c 5 d 4))
        ddi->diagonals.emplace_back(18, 167);

        // to position 23, legal: (legal black (move king c 5 e 3))
        ddi->diagonals.emplace_back(23, 168);

        // to position 27, legal: (legal black (move king c 5 f 2))
        ddi->diagonals.emplace_back(27, 169);

        // to position 32, legal: (legal black (move king c 5 g 1))
        ddi->diagonals.emplace_back(32, 170);

        this->diagonal_data[109].push_back(ddi);


        this->reverse_legal_lookup_black[167] = ReverseLegalLookup(Role::Black, Piece::King, 14, 18, SE);
        this->reverse_legal_lookup_black[168] = ReverseLegalLookup(Role::Black, Piece::King, 14, 23, SE);
        this->reverse_legal_lookup_black[169] = ReverseLegalLookup(Role::Black, Piece::King, 14, 27, SE);
        this->reverse_legal_lookup_black[170] = ReverseLegalLookup(Role::Black, Piece::King, 14, 32, SE);
    }
    // generating for black king 14 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal black (move king c 5 b 4))
        ddi->diagonals.emplace_back(17, 171);

        // to position 21, legal: (legal black (move king c 5 a 3))
        ddi->diagonals.emplace_back(21, 172);

        this->diagonal_data[109].push_back(ddi);


        this->reverse_legal_lookup_black[171] = ReverseLegalLookup(Role::Black, Piece::King, 14, 17, SW);
        this->reverse_legal_lookup_black[172] = ReverseLegalLookup(Role::Black, Piece::King, 14, 21, SW);
    }
    // generating for black king 15 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 11, legal: (legal black (move king e 5 f 6))
        ddi->diagonals.emplace_back(11, 179);

        // to position 8, legal: (legal black (move king e 5 g 7))
        ddi->diagonals.emplace_back(8, 180);

        // to position 4, legal: (legal black (move king e 5 h 8))
        ddi->diagonals.emplace_back(4, 181);

        this->diagonal_data[110].push_back(ddi);


        this->reverse_legal_lookup_black[179] = ReverseLegalLookup(Role::Black, Piece::King, 15, 11, NE);
        this->reverse_legal_lookup_black[180] = ReverseLegalLookup(Role::Black, Piece::King, 15, 8, NE);
        this->reverse_legal_lookup_black[181] = ReverseLegalLookup(Role::Black, Piece::King, 15, 4, NE);
    }
    // generating for black king 15 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 10, legal: (legal black (move king e 5 d 6))
        ddi->diagonals.emplace_back(10, 182);

        // to position 6, legal: (legal black (move king e 5 c 7))
        ddi->diagonals.emplace_back(6, 183);

        // to position 1, legal: (legal black (move king e 5 b 8))
        ddi->diagonals.emplace_back(1, 184);

        this->diagonal_data[110].push_back(ddi);


        this->reverse_legal_lookup_black[182] = ReverseLegalLookup(Role::Black, Piece::King, 15, 10, NW);
        this->reverse_legal_lookup_black[183] = ReverseLegalLookup(Role::Black, Piece::King, 15, 6, NW);
        this->reverse_legal_lookup_black[184] = ReverseLegalLookup(Role::Black, Piece::King, 15, 1, NW);
    }
    // generating for black king 15 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 19, legal: (legal black (move king e 5 f 4))
        ddi->diagonals.emplace_back(19, 185);

        // to position 24, legal: (legal black (move king e 5 g 3))
        ddi->diagonals.emplace_back(24, 186);

        // to position 28, legal: (legal black (move king e 5 h 2))
        ddi->diagonals.emplace_back(28, 187);

        this->diagonal_data[110].push_back(ddi);


        this->reverse_legal_lookup_black[185] = ReverseLegalLookup(Role::Black, Piece::King, 15, 19, SE);
        this->reverse_legal_lookup_black[186] = ReverseLegalLookup(Role::Black, Piece::King, 15, 24, SE);
        this->reverse_legal_lookup_black[187] = ReverseLegalLookup(Role::Black, Piece::King, 15, 28, SE);
    }
    // generating for black king 15 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 18, legal: (legal black (move king e 5 d 4))
        ddi->diagonals.emplace_back(18, 188);

        // to position 22, legal: (legal black (move king e 5 c 3))
        ddi->diagonals.emplace_back(22, 189);

        // to position 25, legal: (legal black (move king e 5 b 2))
        ddi->diagonals.emplace_back(25, 190);

        // to position 29, legal: (legal black (move king e 5 a 1))
        ddi->diagonals.emplace_back(29, 191);

        this->diagonal_data[110].push_back(ddi);


        this->reverse_legal_lookup_black[188] = ReverseLegalLookup(Role::Black, Piece::King, 15, 18, SW);
        this->reverse_legal_lookup_black[189] = ReverseLegalLookup(Role::Black, Piece::King, 15, 22, SW);
        this->reverse_legal_lookup_black[190] = ReverseLegalLookup(Role::Black, Piece::King, 15, 25, SW);
        this->reverse_legal_lookup_black[191] = ReverseLegalLookup(Role::Black, Piece::King, 15, 29, SW);
    }
    // generating for black king 16 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 12, legal: (legal black (move king g 5 h 6))
        ddi->diagonals.emplace_back(12, 196);

        this->diagonal_data[111].push_back(ddi);


        this->reverse_legal_lookup_black[196] = ReverseLegalLookup(Role::Black, Piece::King, 16, 12, NE);
    }
    // generating for black king 16 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 11, legal: (legal black (move king g 5 f 6))
        ddi->diagonals.emplace_back(11, 197);

        // to position 7, legal: (legal black (move king g 5 e 7))
        ddi->diagonals.emplace_back(7, 198);

        // to position 2, legal: (legal black (move king g 5 d 8))
        ddi->diagonals.emplace_back(2, 199);

        this->diagonal_data[111].push_back(ddi);


        this->reverse_legal_lookup_black[197] = ReverseLegalLookup(Role::Black, Piece::King, 16, 11, NW);
        this->reverse_legal_lookup_black[198] = ReverseLegalLookup(Role::Black, Piece::King, 16, 7, NW);
        this->reverse_legal_lookup_black[199] = ReverseLegalLookup(Role::Black, Piece::King, 16, 2, NW);
    }
    // generating for black king 16 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 20, legal: (legal black (move king g 5 h 4))
        ddi->diagonals.emplace_back(20, 200);

        this->diagonal_data[111].push_back(ddi);


        this->reverse_legal_lookup_black[200] = ReverseLegalLookup(Role::Black, Piece::King, 16, 20, SE);
    }
    // generating for black king 16 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 19, legal: (legal black (move king g 5 f 4))
        ddi->diagonals.emplace_back(19, 201);

        // to position 23, legal: (legal black (move king g 5 e 3))
        ddi->diagonals.emplace_back(23, 202);

        // to position 26, legal: (legal black (move king g 5 d 2))
        ddi->diagonals.emplace_back(26, 203);

        // to position 30, legal: (legal black (move king g 5 c 1))
        ddi->diagonals.emplace_back(30, 204);

        this->diagonal_data[111].push_back(ddi);


        this->reverse_legal_lookup_black[201] = ReverseLegalLookup(Role::Black, Piece::King, 16, 19, SW);
        this->reverse_legal_lookup_black[202] = ReverseLegalLookup(Role::Black, Piece::King, 16, 23, SW);
        this->reverse_legal_lookup_black[203] = ReverseLegalLookup(Role::Black, Piece::King, 16, 26, SW);
        this->reverse_legal_lookup_black[204] = ReverseLegalLookup(Role::Black, Piece::King, 16, 30, SW);
    }
    // generating for black king 17 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 14, legal: (legal black (move king b 4 c 5))
        ddi->diagonals.emplace_back(14, 209);

        // to position 10, legal: (legal black (move king b 4 d 6))
        ddi->diagonals.emplace_back(10, 210);

        // to position 7, legal: (legal black (move king b 4 e 7))
        ddi->diagonals.emplace_back(7, 211);

        // to position 3, legal: (legal black (move king b 4 f 8))
        ddi->diagonals.emplace_back(3, 212);

        this->diagonal_data[112].push_back(ddi);


        this->reverse_legal_lookup_black[209] = ReverseLegalLookup(Role::Black, Piece::King, 17, 14, NE);
        this->reverse_legal_lookup_black[210] = ReverseLegalLookup(Role::Black, Piece::King, 17, 10, NE);
        this->reverse_legal_lookup_black[211] = ReverseLegalLookup(Role::Black, Piece::King, 17, 7, NE);
        this->reverse_legal_lookup_black[212] = ReverseLegalLookup(Role::Black, Piece::King, 17, 3, NE);
    }
    // generating for black king 17 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 13, legal: (legal black (move king b 4 a 5))
        ddi->diagonals.emplace_back(13, 213);

        this->diagonal_data[112].push_back(ddi);


        this->reverse_legal_lookup_black[213] = ReverseLegalLookup(Role::Black, Piece::King, 17, 13, NW);
    }
    // generating for black king 17 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 22, legal: (legal black (move king b 4 c 3))
        ddi->diagonals.emplace_back(22, 214);

        // to position 26, legal: (legal black (move king b 4 d 2))
        ddi->diagonals.emplace_back(26, 215);

        // to position 31, legal: (legal black (move king b 4 e 1))
        ddi->diagonals.emplace_back(31, 216);

        this->diagonal_data[112].push_back(ddi);


        this->reverse_legal_lookup_black[214] = ReverseLegalLookup(Role::Black, Piece::King, 17, 22, SE);
        this->reverse_legal_lookup_black[215] = ReverseLegalLookup(Role::Black, Piece::King, 17, 26, SE);
        this->reverse_legal_lookup_black[216] = ReverseLegalLookup(Role::Black, Piece::King, 17, 31, SE);
    }
    // generating for black king 17 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 21, legal: (legal black (move king b 4 a 3))
        ddi->diagonals.emplace_back(21, 217);

        this->diagonal_data[112].push_back(ddi);


        this->reverse_legal_lookup_black[217] = ReverseLegalLookup(Role::Black, Piece::King, 17, 21, SW);
    }
    // generating for black king 18 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 15, legal: (legal black (move king d 4 e 5))
        ddi->diagonals.emplace_back(15, 224);

        // to position 11, legal: (legal black (move king d 4 f 6))
        ddi->diagonals.emplace_back(11, 225);

        // to position 8, legal: (legal black (move king d 4 g 7))
        ddi->diagonals.emplace_back(8, 226);

        // to position 4, legal: (legal black (move king d 4 h 8))
        ddi->diagonals.emplace_back(4, 227);

        this->diagonal_data[113].push_back(ddi);


        this->reverse_legal_lookup_black[224] = ReverseLegalLookup(Role::Black, Piece::King, 18, 15, NE);
        this->reverse_legal_lookup_black[225] = ReverseLegalLookup(Role::Black, Piece::King, 18, 11, NE);
        this->reverse_legal_lookup_black[226] = ReverseLegalLookup(Role::Black, Piece::King, 18, 8, NE);
        this->reverse_legal_lookup_black[227] = ReverseLegalLookup(Role::Black, Piece::King, 18, 4, NE);
    }
    // generating for black king 18 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 14, legal: (legal black (move king d 4 c 5))
        ddi->diagonals.emplace_back(14, 228);

        // to position 9, legal: (legal black (move king d 4 b 6))
        ddi->diagonals.emplace_back(9, 229);

        // to position 5, legal: (legal black (move king d 4 a 7))
        ddi->diagonals.emplace_back(5, 230);

        this->diagonal_data[113].push_back(ddi);


        this->reverse_legal_lookup_black[228] = ReverseLegalLookup(Role::Black, Piece::King, 18, 14, NW);
        this->reverse_legal_lookup_black[229] = ReverseLegalLookup(Role::Black, Piece::King, 18, 9, NW);
        this->reverse_legal_lookup_black[230] = ReverseLegalLookup(Role::Black, Piece::King, 18, 5, NW);
    }
    // generating for black king 18 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 23, legal: (legal black (move king d 4 e 3))
        ddi->diagonals.emplace_back(23, 231);

        // to position 27, legal: (legal black (move king d 4 f 2))
        ddi->diagonals.emplace_back(27, 232);

        // to position 32, legal: (legal black (move king d 4 g 1))
        ddi->diagonals.emplace_back(32, 233);

        this->diagonal_data[113].push_back(ddi);


        this->reverse_legal_lookup_black[231] = ReverseLegalLookup(Role::Black, Piece::King, 18, 23, SE);
        this->reverse_legal_lookup_black[232] = ReverseLegalLookup(Role::Black, Piece::King, 18, 27, SE);
        this->reverse_legal_lookup_black[233] = ReverseLegalLookup(Role::Black, Piece::King, 18, 32, SE);
    }
    // generating for black king 18 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 22, legal: (legal black (move king d 4 c 3))
        ddi->diagonals.emplace_back(22, 234);

        // to position 25, legal: (legal black (move king d 4 b 2))
        ddi->diagonals.emplace_back(25, 235);

        // to position 29, legal: (legal black (move king d 4 a 1))
        ddi->diagonals.emplace_back(29, 236);

        this->diagonal_data[113].push_back(ddi);


        this->reverse_legal_lookup_black[234] = ReverseLegalLookup(Role::Black, Piece::King, 18, 22, SW);
        this->reverse_legal_lookup_black[235] = ReverseLegalLookup(Role::Black, Piece::King, 18, 25, SW);
        this->reverse_legal_lookup_black[236] = ReverseLegalLookup(Role::Black, Piece::King, 18, 29, SW);
    }
    // generating for black king 19 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 16, legal: (legal black (move king f 4 g 5))
        ddi->diagonals.emplace_back(16, 243);

        // to position 12, legal: (legal black (move king f 4 h 6))
        ddi->diagonals.emplace_back(12, 244);

        this->diagonal_data[114].push_back(ddi);


        this->reverse_legal_lookup_black[243] = ReverseLegalLookup(Role::Black, Piece::King, 19, 16, NE);
        this->reverse_legal_lookup_black[244] = ReverseLegalLookup(Role::Black, Piece::King, 19, 12, NE);
    }
    // generating for black king 19 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 15, legal: (legal black (move king f 4 e 5))
        ddi->diagonals.emplace_back(15, 245);

        // to position 10, legal: (legal black (move king f 4 d 6))
        ddi->diagonals.emplace_back(10, 246);

        // to position 6, legal: (legal black (move king f 4 c 7))
        ddi->diagonals.emplace_back(6, 247);

        // to position 1, legal: (legal black (move king f 4 b 8))
        ddi->diagonals.emplace_back(1, 248);

        this->diagonal_data[114].push_back(ddi);


        this->reverse_legal_lookup_black[245] = ReverseLegalLookup(Role::Black, Piece::King, 19, 15, NW);
        this->reverse_legal_lookup_black[246] = ReverseLegalLookup(Role::Black, Piece::King, 19, 10, NW);
        this->reverse_legal_lookup_black[247] = ReverseLegalLookup(Role::Black, Piece::King, 19, 6, NW);
        this->reverse_legal_lookup_black[248] = ReverseLegalLookup(Role::Black, Piece::King, 19, 1, NW);
    }
    // generating for black king 19 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal black (move king f 4 g 3))
        ddi->diagonals.emplace_back(24, 249);

        // to position 28, legal: (legal black (move king f 4 h 2))
        ddi->diagonals.emplace_back(28, 250);

        this->diagonal_data[114].push_back(ddi);


        this->reverse_legal_lookup_black[249] = ReverseLegalLookup(Role::Black, Piece::King, 19, 24, SE);
        this->reverse_legal_lookup_black[250] = ReverseLegalLookup(Role::Black, Piece::King, 19, 28, SE);
    }
    // generating for black king 19 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 23, legal: (legal black (move king f 4 e 3))
        ddi->diagonals.emplace_back(23, 251);

        // to position 26, legal: (legal black (move king f 4 d 2))
        ddi->diagonals.emplace_back(26, 252);

        // to position 30, legal: (legal black (move king f 4 c 1))
        ddi->diagonals.emplace_back(30, 253);

        this->diagonal_data[114].push_back(ddi);


        this->reverse_legal_lookup_black[251] = ReverseLegalLookup(Role::Black, Piece::King, 19, 23, SW);
        this->reverse_legal_lookup_black[252] = ReverseLegalLookup(Role::Black, Piece::King, 19, 26, SW);
        this->reverse_legal_lookup_black[253] = ReverseLegalLookup(Role::Black, Piece::King, 19, 30, SW);
    }
    // generating for black king 20 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 16, legal: (legal black (move king h 4 g 5))
        ddi->diagonals.emplace_back(16, 257);

        // to position 11, legal: (legal black (move king h 4 f 6))
        ddi->diagonals.emplace_back(11, 258);

        // to position 7, legal: (legal black (move king h 4 e 7))
        ddi->diagonals.emplace_back(7, 259);

        // to position 2, legal: (legal black (move king h 4 d 8))
        ddi->diagonals.emplace_back(2, 260);

        this->diagonal_data[115].push_back(ddi);


        this->reverse_legal_lookup_black[257] = ReverseLegalLookup(Role::Black, Piece::King, 20, 16, NW);
        this->reverse_legal_lookup_black[258] = ReverseLegalLookup(Role::Black, Piece::King, 20, 11, NW);
        this->reverse_legal_lookup_black[259] = ReverseLegalLookup(Role::Black, Piece::King, 20, 7, NW);
        this->reverse_legal_lookup_black[260] = ReverseLegalLookup(Role::Black, Piece::King, 20, 2, NW);
    }
    // generating for black king 20 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 24, legal: (legal black (move king h 4 g 3))
        ddi->diagonals.emplace_back(24, 261);

        // to position 27, legal: (legal black (move king h 4 f 2))
        ddi->diagonals.emplace_back(27, 262);

        // to position 31, legal: (legal black (move king h 4 e 1))
        ddi->diagonals.emplace_back(31, 263);

        this->diagonal_data[115].push_back(ddi);


        this->reverse_legal_lookup_black[261] = ReverseLegalLookup(Role::Black, Piece::King, 20, 24, SW);
        this->reverse_legal_lookup_black[262] = ReverseLegalLookup(Role::Black, Piece::King, 20, 27, SW);
        this->reverse_legal_lookup_black[263] = ReverseLegalLookup(Role::Black, Piece::King, 20, 31, SW);
    }
    // generating for black king 21 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 17, legal: (legal black (move king a 3 b 4))
        ddi->diagonals.emplace_back(17, 267);

        // to position 14, legal: (legal black (move king a 3 c 5))
        ddi->diagonals.emplace_back(14, 268);

        // to position 10, legal: (legal black (move king a 3 d 6))
        ddi->diagonals.emplace_back(10, 269);

        // to position 7, legal: (legal black (move king a 3 e 7))
        ddi->diagonals.emplace_back(7, 270);

        // to position 3, legal: (legal black (move king a 3 f 8))
        ddi->diagonals.emplace_back(3, 271);

        this->diagonal_data[116].push_back(ddi);


        this->reverse_legal_lookup_black[267] = ReverseLegalLookup(Role::Black, Piece::King, 21, 17, NE);
        this->reverse_legal_lookup_black[268] = ReverseLegalLookup(Role::Black, Piece::King, 21, 14, NE);
        this->reverse_legal_lookup_black[269] = ReverseLegalLookup(Role::Black, Piece::King, 21, 10, NE);
        this->reverse_legal_lookup_black[270] = ReverseLegalLookup(Role::Black, Piece::King, 21, 7, NE);
        this->reverse_legal_lookup_black[271] = ReverseLegalLookup(Role::Black, Piece::King, 21, 3, NE);
    }
    // generating for black king 21 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 25, legal: (legal black (move king a 3 b 2))
        ddi->diagonals.emplace_back(25, 272);

        // to position 30, legal: (legal black (move king a 3 c 1))
        ddi->diagonals.emplace_back(30, 273);

        this->diagonal_data[116].push_back(ddi);


        this->reverse_legal_lookup_black[272] = ReverseLegalLookup(Role::Black, Piece::King, 21, 25, SE);
        this->reverse_legal_lookup_black[273] = ReverseLegalLookup(Role::Black, Piece::King, 21, 30, SE);
    }
    // generating for black king 22 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 18, legal: (legal black (move king c 3 d 4))
        ddi->diagonals.emplace_back(18, 280);

        // to position 15, legal: (legal black (move king c 3 e 5))
        ddi->diagonals.emplace_back(15, 281);

        // to position 11, legal: (legal black (move king c 3 f 6))
        ddi->diagonals.emplace_back(11, 282);

        // to position 8, legal: (legal black (move king c 3 g 7))
        ddi->diagonals.emplace_back(8, 283);

        // to position 4, legal: (legal black (move king c 3 h 8))
        ddi->diagonals.emplace_back(4, 284);

        this->diagonal_data[117].push_back(ddi);


        this->reverse_legal_lookup_black[280] = ReverseLegalLookup(Role::Black, Piece::King, 22, 18, NE);
        this->reverse_legal_lookup_black[281] = ReverseLegalLookup(Role::Black, Piece::King, 22, 15, NE);
        this->reverse_legal_lookup_black[282] = ReverseLegalLookup(Role::Black, Piece::King, 22, 11, NE);
        this->reverse_legal_lookup_black[283] = ReverseLegalLookup(Role::Black, Piece::King, 22, 8, NE);
        this->reverse_legal_lookup_black[284] = ReverseLegalLookup(Role::Black, Piece::King, 22, 4, NE);
    }
    // generating for black king 22 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal black (move king c 3 b 4))
        ddi->diagonals.emplace_back(17, 285);

        // to position 13, legal: (legal black (move king c 3 a 5))
        ddi->diagonals.emplace_back(13, 286);

        this->diagonal_data[117].push_back(ddi);


        this->reverse_legal_lookup_black[285] = ReverseLegalLookup(Role::Black, Piece::King, 22, 17, NW);
        this->reverse_legal_lookup_black[286] = ReverseLegalLookup(Role::Black, Piece::King, 22, 13, NW);
    }
    // generating for black king 22 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 26, legal: (legal black (move king c 3 d 2))
        ddi->diagonals.emplace_back(26, 287);

        // to position 31, legal: (legal black (move king c 3 e 1))
        ddi->diagonals.emplace_back(31, 288);

        this->diagonal_data[117].push_back(ddi);


        this->reverse_legal_lookup_black[287] = ReverseLegalLookup(Role::Black, Piece::King, 22, 26, SE);
        this->reverse_legal_lookup_black[288] = ReverseLegalLookup(Role::Black, Piece::King, 22, 31, SE);
    }
    // generating for black king 22 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 25, legal: (legal black (move king c 3 b 2))
        ddi->diagonals.emplace_back(25, 289);

        // to position 29, legal: (legal black (move king c 3 a 1))
        ddi->diagonals.emplace_back(29, 290);

        this->diagonal_data[117].push_back(ddi);


        this->reverse_legal_lookup_black[289] = ReverseLegalLookup(Role::Black, Piece::King, 22, 25, SW);
        this->reverse_legal_lookup_black[290] = ReverseLegalLookup(Role::Black, Piece::King, 22, 29, SW);
    }
    // generating for black king 23 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 19, legal: (legal black (move king e 3 f 4))
        ddi->diagonals.emplace_back(19, 297);

        // to position 16, legal: (legal black (move king e 3 g 5))
        ddi->diagonals.emplace_back(16, 298);

        // to position 12, legal: (legal black (move king e 3 h 6))
        ddi->diagonals.emplace_back(12, 299);

        this->diagonal_data[118].push_back(ddi);


        this->reverse_legal_lookup_black[297] = ReverseLegalLookup(Role::Black, Piece::King, 23, 19, NE);
        this->reverse_legal_lookup_black[298] = ReverseLegalLookup(Role::Black, Piece::King, 23, 16, NE);
        this->reverse_legal_lookup_black[299] = ReverseLegalLookup(Role::Black, Piece::King, 23, 12, NE);
    }
    // generating for black king 23 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 18, legal: (legal black (move king e 3 d 4))
        ddi->diagonals.emplace_back(18, 300);

        // to position 14, legal: (legal black (move king e 3 c 5))
        ddi->diagonals.emplace_back(14, 301);

        // to position 9, legal: (legal black (move king e 3 b 6))
        ddi->diagonals.emplace_back(9, 302);

        // to position 5, legal: (legal black (move king e 3 a 7))
        ddi->diagonals.emplace_back(5, 303);

        this->diagonal_data[118].push_back(ddi);


        this->reverse_legal_lookup_black[300] = ReverseLegalLookup(Role::Black, Piece::King, 23, 18, NW);
        this->reverse_legal_lookup_black[301] = ReverseLegalLookup(Role::Black, Piece::King, 23, 14, NW);
        this->reverse_legal_lookup_black[302] = ReverseLegalLookup(Role::Black, Piece::King, 23, 9, NW);
        this->reverse_legal_lookup_black[303] = ReverseLegalLookup(Role::Black, Piece::King, 23, 5, NW);
    }
    // generating for black king 23 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal black (move king e 3 f 2))
        ddi->diagonals.emplace_back(27, 304);

        // to position 32, legal: (legal black (move king e 3 g 1))
        ddi->diagonals.emplace_back(32, 305);

        this->diagonal_data[118].push_back(ddi);


        this->reverse_legal_lookup_black[304] = ReverseLegalLookup(Role::Black, Piece::King, 23, 27, SE);
        this->reverse_legal_lookup_black[305] = ReverseLegalLookup(Role::Black, Piece::King, 23, 32, SE);
    }
    // generating for black king 23 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 26, legal: (legal black (move king e 3 d 2))
        ddi->diagonals.emplace_back(26, 306);

        // to position 30, legal: (legal black (move king e 3 c 1))
        ddi->diagonals.emplace_back(30, 307);

        this->diagonal_data[118].push_back(ddi);


        this->reverse_legal_lookup_black[306] = ReverseLegalLookup(Role::Black, Piece::King, 23, 26, SW);
        this->reverse_legal_lookup_black[307] = ReverseLegalLookup(Role::Black, Piece::King, 23, 30, SW);
    }
    // generating for black king 24 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 20, legal: (legal black (move king g 3 h 4))
        ddi->diagonals.emplace_back(20, 312);

        this->diagonal_data[119].push_back(ddi);


        this->reverse_legal_lookup_black[312] = ReverseLegalLookup(Role::Black, Piece::King, 24, 20, NE);
    }
    // generating for black king 24 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 19, legal: (legal black (move king g 3 f 4))
        ddi->diagonals.emplace_back(19, 313);

        // to position 15, legal: (legal black (move king g 3 e 5))
        ddi->diagonals.emplace_back(15, 314);

        // to position 10, legal: (legal black (move king g 3 d 6))
        ddi->diagonals.emplace_back(10, 315);

        // to position 6, legal: (legal black (move king g 3 c 7))
        ddi->diagonals.emplace_back(6, 316);

        // to position 1, legal: (legal black (move king g 3 b 8))
        ddi->diagonals.emplace_back(1, 317);

        this->diagonal_data[119].push_back(ddi);


        this->reverse_legal_lookup_black[313] = ReverseLegalLookup(Role::Black, Piece::King, 24, 19, NW);
        this->reverse_legal_lookup_black[314] = ReverseLegalLookup(Role::Black, Piece::King, 24, 15, NW);
        this->reverse_legal_lookup_black[315] = ReverseLegalLookup(Role::Black, Piece::King, 24, 10, NW);
        this->reverse_legal_lookup_black[316] = ReverseLegalLookup(Role::Black, Piece::King, 24, 6, NW);
        this->reverse_legal_lookup_black[317] = ReverseLegalLookup(Role::Black, Piece::King, 24, 1, NW);
    }
    // generating for black king 24 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 28, legal: (legal black (move king g 3 h 2))
        ddi->diagonals.emplace_back(28, 318);

        this->diagonal_data[119].push_back(ddi);


        this->reverse_legal_lookup_black[318] = ReverseLegalLookup(Role::Black, Piece::King, 24, 28, SE);
    }
    // generating for black king 24 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal black (move king g 3 f 2))
        ddi->diagonals.emplace_back(27, 319);

        // to position 31, legal: (legal black (move king g 3 e 1))
        ddi->diagonals.emplace_back(31, 320);

        this->diagonal_data[119].push_back(ddi);


        this->reverse_legal_lookup_black[319] = ReverseLegalLookup(Role::Black, Piece::King, 24, 27, SW);
        this->reverse_legal_lookup_black[320] = ReverseLegalLookup(Role::Black, Piece::King, 24, 31, SW);
    }
    // generating for black king 25 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(6);
        // to position 22, legal: (legal black (move king b 2 c 3))
        ddi->diagonals.emplace_back(22, 324);

        // to position 18, legal: (legal black (move king b 2 d 4))
        ddi->diagonals.emplace_back(18, 325);

        // to position 15, legal: (legal black (move king b 2 e 5))
        ddi->diagonals.emplace_back(15, 326);

        // to position 11, legal: (legal black (move king b 2 f 6))
        ddi->diagonals.emplace_back(11, 327);

        // to position 8, legal: (legal black (move king b 2 g 7))
        ddi->diagonals.emplace_back(8, 328);

        // to position 4, legal: (legal black (move king b 2 h 8))
        ddi->diagonals.emplace_back(4, 329);

        this->diagonal_data[120].push_back(ddi);


        this->reverse_legal_lookup_black[324] = ReverseLegalLookup(Role::Black, Piece::King, 25, 22, NE);
        this->reverse_legal_lookup_black[325] = ReverseLegalLookup(Role::Black, Piece::King, 25, 18, NE);
        this->reverse_legal_lookup_black[326] = ReverseLegalLookup(Role::Black, Piece::King, 25, 15, NE);
        this->reverse_legal_lookup_black[327] = ReverseLegalLookup(Role::Black, Piece::King, 25, 11, NE);
        this->reverse_legal_lookup_black[328] = ReverseLegalLookup(Role::Black, Piece::King, 25, 8, NE);
        this->reverse_legal_lookup_black[329] = ReverseLegalLookup(Role::Black, Piece::King, 25, 4, NE);
    }
    // generating for black king 25 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 21, legal: (legal black (move king b 2 a 3))
        ddi->diagonals.emplace_back(21, 330);

        this->diagonal_data[120].push_back(ddi);


        this->reverse_legal_lookup_black[330] = ReverseLegalLookup(Role::Black, Piece::King, 25, 21, NW);
    }
    // generating for black king 25 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 30, legal: (legal black (move king b 2 c 1))
        ddi->diagonals.emplace_back(30, 331);

        this->diagonal_data[120].push_back(ddi);


        this->reverse_legal_lookup_black[331] = ReverseLegalLookup(Role::Black, Piece::King, 25, 30, SE);
    }
    // generating for black king 25 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 29, legal: (legal black (move king b 2 a 1))
        ddi->diagonals.emplace_back(29, 332);

        this->diagonal_data[120].push_back(ddi);


        this->reverse_legal_lookup_black[332] = ReverseLegalLookup(Role::Black, Piece::King, 25, 29, SW);
    }
    // generating for black king 26 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 23, legal: (legal black (move king d 2 e 3))
        ddi->diagonals.emplace_back(23, 337);

        // to position 19, legal: (legal black (move king d 2 f 4))
        ddi->diagonals.emplace_back(19, 338);

        // to position 16, legal: (legal black (move king d 2 g 5))
        ddi->diagonals.emplace_back(16, 339);

        // to position 12, legal: (legal black (move king d 2 h 6))
        ddi->diagonals.emplace_back(12, 340);

        this->diagonal_data[121].push_back(ddi);


        this->reverse_legal_lookup_black[337] = ReverseLegalLookup(Role::Black, Piece::King, 26, 23, NE);
        this->reverse_legal_lookup_black[338] = ReverseLegalLookup(Role::Black, Piece::King, 26, 19, NE);
        this->reverse_legal_lookup_black[339] = ReverseLegalLookup(Role::Black, Piece::King, 26, 16, NE);
        this->reverse_legal_lookup_black[340] = ReverseLegalLookup(Role::Black, Piece::King, 26, 12, NE);
    }
    // generating for black king 26 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 22, legal: (legal black (move king d 2 c 3))
        ddi->diagonals.emplace_back(22, 341);

        // to position 17, legal: (legal black (move king d 2 b 4))
        ddi->diagonals.emplace_back(17, 342);

        // to position 13, legal: (legal black (move king d 2 a 5))
        ddi->diagonals.emplace_back(13, 343);

        this->diagonal_data[121].push_back(ddi);


        this->reverse_legal_lookup_black[341] = ReverseLegalLookup(Role::Black, Piece::King, 26, 22, NW);
        this->reverse_legal_lookup_black[342] = ReverseLegalLookup(Role::Black, Piece::King, 26, 17, NW);
        this->reverse_legal_lookup_black[343] = ReverseLegalLookup(Role::Black, Piece::King, 26, 13, NW);
    }
    // generating for black king 26 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 31, legal: (legal black (move king d 2 e 1))
        ddi->diagonals.emplace_back(31, 344);

        this->diagonal_data[121].push_back(ddi);


        this->reverse_legal_lookup_black[344] = ReverseLegalLookup(Role::Black, Piece::King, 26, 31, SE);
    }
    // generating for black king 26 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 30, legal: (legal black (move king d 2 c 1))
        ddi->diagonals.emplace_back(30, 345);

        this->diagonal_data[121].push_back(ddi);


        this->reverse_legal_lookup_black[345] = ReverseLegalLookup(Role::Black, Piece::King, 26, 30, SW);
    }
    // generating for black king 27 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal black (move king f 2 g 3))
        ddi->diagonals.emplace_back(24, 350);

        // to position 20, legal: (legal black (move king f 2 h 4))
        ddi->diagonals.emplace_back(20, 351);

        this->diagonal_data[122].push_back(ddi);


        this->reverse_legal_lookup_black[350] = ReverseLegalLookup(Role::Black, Piece::King, 27, 24, NE);
        this->reverse_legal_lookup_black[351] = ReverseLegalLookup(Role::Black, Piece::King, 27, 20, NE);
    }
    // generating for black king 27 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 23, legal: (legal black (move king f 2 e 3))
        ddi->diagonals.emplace_back(23, 352);

        // to position 18, legal: (legal black (move king f 2 d 4))
        ddi->diagonals.emplace_back(18, 353);

        // to position 14, legal: (legal black (move king f 2 c 5))
        ddi->diagonals.emplace_back(14, 354);

        // to position 9, legal: (legal black (move king f 2 b 6))
        ddi->diagonals.emplace_back(9, 355);

        // to position 5, legal: (legal black (move king f 2 a 7))
        ddi->diagonals.emplace_back(5, 356);

        this->diagonal_data[122].push_back(ddi);


        this->reverse_legal_lookup_black[352] = ReverseLegalLookup(Role::Black, Piece::King, 27, 23, NW);
        this->reverse_legal_lookup_black[353] = ReverseLegalLookup(Role::Black, Piece::King, 27, 18, NW);
        this->reverse_legal_lookup_black[354] = ReverseLegalLookup(Role::Black, Piece::King, 27, 14, NW);
        this->reverse_legal_lookup_black[355] = ReverseLegalLookup(Role::Black, Piece::King, 27, 9, NW);
        this->reverse_legal_lookup_black[356] = ReverseLegalLookup(Role::Black, Piece::King, 27, 5, NW);
    }
    // generating for black king 27 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 32, legal: (legal black (move king f 2 g 1))
        ddi->diagonals.emplace_back(32, 357);

        this->diagonal_data[122].push_back(ddi);


        this->reverse_legal_lookup_black[357] = ReverseLegalLookup(Role::Black, Piece::King, 27, 32, SE);
    }
    // generating for black king 27 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 31, legal: (legal black (move king f 2 e 1))
        ddi->diagonals.emplace_back(31, 358);

        this->diagonal_data[122].push_back(ddi);


        this->reverse_legal_lookup_black[358] = ReverseLegalLookup(Role::Black, Piece::King, 27, 31, SW);
    }
    // generating for black king 28 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 24, legal: (legal black (move king h 2 g 3))
        ddi->diagonals.emplace_back(24, 361);

        // to position 19, legal: (legal black (move king h 2 f 4))
        ddi->diagonals.emplace_back(19, 362);

        // to position 15, legal: (legal black (move king h 2 e 5))
        ddi->diagonals.emplace_back(15, 363);

        // to position 10, legal: (legal black (move king h 2 d 6))
        ddi->diagonals.emplace_back(10, 364);

        // to position 6, legal: (legal black (move king h 2 c 7))
        ddi->diagonals.emplace_back(6, 365);

        // to position 1, legal: (legal black (move king h 2 b 8))
        ddi->diagonals.emplace_back(1, 366);

        this->diagonal_data[123].push_back(ddi);


        this->reverse_legal_lookup_black[361] = ReverseLegalLookup(Role::Black, Piece::King, 28, 24, NW);
        this->reverse_legal_lookup_black[362] = ReverseLegalLookup(Role::Black, Piece::King, 28, 19, NW);
        this->reverse_legal_lookup_black[363] = ReverseLegalLookup(Role::Black, Piece::King, 28, 15, NW);
        this->reverse_legal_lookup_black[364] = ReverseLegalLookup(Role::Black, Piece::King, 28, 10, NW);
        this->reverse_legal_lookup_black[365] = ReverseLegalLookup(Role::Black, Piece::King, 28, 6, NW);
        this->reverse_legal_lookup_black[366] = ReverseLegalLookup(Role::Black, Piece::King, 28, 1, NW);
    }
    // generating for black king 28 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 32, legal: (legal black (move king h 2 g 1))
        ddi->diagonals.emplace_back(32, 367);

        this->diagonal_data[123].push_back(ddi);


        this->reverse_legal_lookup_black[367] = ReverseLegalLookup(Role::Black, Piece::King, 28, 32, SW);
    }
    // generating for black king 29 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(7);
        // to position 25, legal: (legal black (move king a 1 b 2))
        ddi->diagonals.emplace_back(25, 369);

        // to position 22, legal: (legal black (move king a 1 c 3))
        ddi->diagonals.emplace_back(22, 370);

        // to position 18, legal: (legal black (move king a 1 d 4))
        ddi->diagonals.emplace_back(18, 371);

        // to position 15, legal: (legal black (move king a 1 e 5))
        ddi->diagonals.emplace_back(15, 372);

        // to position 11, legal: (legal black (move king a 1 f 6))
        ddi->diagonals.emplace_back(11, 373);

        // to position 8, legal: (legal black (move king a 1 g 7))
        ddi->diagonals.emplace_back(8, 374);

        // to position 4, legal: (legal black (move king a 1 h 8))
        ddi->diagonals.emplace_back(4, 375);

        this->diagonal_data[124].push_back(ddi);


        this->reverse_legal_lookup_black[369] = ReverseLegalLookup(Role::Black, Piece::King, 29, 25, NE);
        this->reverse_legal_lookup_black[370] = ReverseLegalLookup(Role::Black, Piece::King, 29, 22, NE);
        this->reverse_legal_lookup_black[371] = ReverseLegalLookup(Role::Black, Piece::King, 29, 18, NE);
        this->reverse_legal_lookup_black[372] = ReverseLegalLookup(Role::Black, Piece::King, 29, 15, NE);
        this->reverse_legal_lookup_black[373] = ReverseLegalLookup(Role::Black, Piece::King, 29, 11, NE);
        this->reverse_legal_lookup_black[374] = ReverseLegalLookup(Role::Black, Piece::King, 29, 8, NE);
        this->reverse_legal_lookup_black[375] = ReverseLegalLookup(Role::Black, Piece::King, 29, 4, NE);
    }
    // generating for black king 30 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 26, legal: (legal black (move king c 1 d 2))
        ddi->diagonals.emplace_back(26, 378);

        // to position 23, legal: (legal black (move king c 1 e 3))
        ddi->diagonals.emplace_back(23, 379);

        // to position 19, legal: (legal black (move king c 1 f 4))
        ddi->diagonals.emplace_back(19, 380);

        // to position 16, legal: (legal black (move king c 1 g 5))
        ddi->diagonals.emplace_back(16, 381);

        // to position 12, legal: (legal black (move king c 1 h 6))
        ddi->diagonals.emplace_back(12, 382);

        this->diagonal_data[125].push_back(ddi);


        this->reverse_legal_lookup_black[378] = ReverseLegalLookup(Role::Black, Piece::King, 30, 26, NE);
        this->reverse_legal_lookup_black[379] = ReverseLegalLookup(Role::Black, Piece::King, 30, 23, NE);
        this->reverse_legal_lookup_black[380] = ReverseLegalLookup(Role::Black, Piece::King, 30, 19, NE);
        this->reverse_legal_lookup_black[381] = ReverseLegalLookup(Role::Black, Piece::King, 30, 16, NE);
        this->reverse_legal_lookup_black[382] = ReverseLegalLookup(Role::Black, Piece::King, 30, 12, NE);
    }
    // generating for black king 30 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 25, legal: (legal black (move king c 1 b 2))
        ddi->diagonals.emplace_back(25, 383);

        // to position 21, legal: (legal black (move king c 1 a 3))
        ddi->diagonals.emplace_back(21, 384);

        this->diagonal_data[125].push_back(ddi);


        this->reverse_legal_lookup_black[383] = ReverseLegalLookup(Role::Black, Piece::King, 30, 25, NW);
        this->reverse_legal_lookup_black[384] = ReverseLegalLookup(Role::Black, Piece::King, 30, 21, NW);
    }
    // generating for black king 31 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 27, legal: (legal black (move king e 1 f 2))
        ddi->diagonals.emplace_back(27, 387);

        // to position 24, legal: (legal black (move king e 1 g 3))
        ddi->diagonals.emplace_back(24, 388);

        // to position 20, legal: (legal black (move king e 1 h 4))
        ddi->diagonals.emplace_back(20, 389);

        this->diagonal_data[126].push_back(ddi);


        this->reverse_legal_lookup_black[387] = ReverseLegalLookup(Role::Black, Piece::King, 31, 27, NE);
        this->reverse_legal_lookup_black[388] = ReverseLegalLookup(Role::Black, Piece::King, 31, 24, NE);
        this->reverse_legal_lookup_black[389] = ReverseLegalLookup(Role::Black, Piece::King, 31, 20, NE);
    }
    // generating for black king 31 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 26, legal: (legal black (move king e 1 d 2))
        ddi->diagonals.emplace_back(26, 390);

        // to position 22, legal: (legal black (move king e 1 c 3))
        ddi->diagonals.emplace_back(22, 391);

        // to position 17, legal: (legal black (move king e 1 b 4))
        ddi->diagonals.emplace_back(17, 392);

        // to position 13, legal: (legal black (move king e 1 a 5))
        ddi->diagonals.emplace_back(13, 393);

        this->diagonal_data[126].push_back(ddi);


        this->reverse_legal_lookup_black[390] = ReverseLegalLookup(Role::Black, Piece::King, 31, 26, NW);
        this->reverse_legal_lookup_black[391] = ReverseLegalLookup(Role::Black, Piece::King, 31, 22, NW);
        this->reverse_legal_lookup_black[392] = ReverseLegalLookup(Role::Black, Piece::King, 31, 17, NW);
        this->reverse_legal_lookup_black[393] = ReverseLegalLookup(Role::Black, Piece::King, 31, 13, NW);
    }
    // generating for black king 32 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 28, legal: (legal black (move king g 1 h 2))
        ddi->diagonals.emplace_back(28, 395);

        this->diagonal_data[127].push_back(ddi);


        this->reverse_legal_lookup_black[395] = ReverseLegalLookup(Role::Black, Piece::King, 32, 28, NE);
    }
    // generating for black king 32 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 27, legal: (legal black (move king g 1 f 2))
        ddi->diagonals.emplace_back(27, 396);

        // to position 23, legal: (legal black (move king g 1 e 3))
        ddi->diagonals.emplace_back(23, 397);

        // to position 18, legal: (legal black (move king g 1 d 4))
        ddi->diagonals.emplace_back(18, 398);

        // to position 14, legal: (legal black (move king g 1 c 5))
        ddi->diagonals.emplace_back(14, 399);

        // to position 9, legal: (legal black (move king g 1 b 6))
        ddi->diagonals.emplace_back(9, 400);

        // to position 5, legal: (legal black (move king g 1 a 7))
        ddi->diagonals.emplace_back(5, 401);

        this->diagonal_data[127].push_back(ddi);


        this->reverse_legal_lookup_black[396] = ReverseLegalLookup(Role::Black, Piece::King, 32, 27, NW);
        this->reverse_legal_lookup_black[397] = ReverseLegalLookup(Role::Black, Piece::King, 32, 23, NW);
        this->reverse_legal_lookup_black[398] = ReverseLegalLookup(Role::Black, Piece::King, 32, 18, NW);
        this->reverse_legal_lookup_black[399] = ReverseLegalLookup(Role::Black, Piece::King, 32, 14, NW);
        this->reverse_legal_lookup_black[400] = ReverseLegalLookup(Role::Black, Piece::King, 32, 9, NW);
        this->reverse_legal_lookup_black[401] = ReverseLegalLookup(Role::Black, Piece::King, 32, 5, NW);
    }


} // end of BoardDescription::initBoard_10x10
void Description::initBoard_10x10() {

    this->num_positions = 50;
    this->white_noop = 0;
    this->black_noop = 0;
    this->diagonal_data.resize(200);
    // Reserve the map size upfront, hopefully memory will be contiguous (XXX check)
    this->reverse_legal_lookup_white.resize(780);
    this->reverse_legal_lookup_black.resize(780);

    // Initial state
    this->initial_state = {false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false};



    // generating promotion line for white
    this->white_promotion_line = {true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false};


    // generating moves for white
    this->white_legal_moves = {"noop", "(move man b 10 d 8)", "(move king b 10 c 9)", "(move king b 10 d 8)", "(move king b 10 e 7)", "(move king b 10 f 6)", "(move king b 10 g 5)", "(move king b 10 h 4)", "(move king b 10 i 3)", "(move king b 10 j 2)", "(move king b 10 a 9)", "(move man d 10 f 8)", "(move man d 10 b 8)", "(move king d 10 e 9)", "(move king d 10 f 8)", "(move king d 10 g 7)", "(move king d 10 h 6)", "(move king d 10 i 5)", "(move king d 10 j 4)", "(move king d 10 c 9)", "(move king d 10 b 8)", "(move king d 10 a 7)", "(move man f 10 h 8)", "(move man f 10 d 8)", "(move king f 10 g 9)", "(move king f 10 h 8)", "(move king f 10 i 7)", "(move king f 10 j 6)", "(move king f 10 e 9)", "(move king f 10 d 8)", "(move king f 10 c 7)", "(move king f 10 b 6)", "(move king f 10 a 5)", "(move man h 10 j 8)", "(move man h 10 f 8)", "(move king h 10 i 9)", "(move king h 10 j 8)", "(move king h 10 g 9)", "(move king h 10 f 8)", "(move king h 10 e 7)", "(move king h 10 d 6)", "(move king h 10 c 5)", "(move king h 10 b 4)", "(move king h 10 a 3)", "(move man j 10 h 8)", "(move king j 10 i 9)", "(move king j 10 h 8)", "(move king j 10 g 7)", "(move king j 10 f 6)", "(move king j 10 e 5)", "(move king j 10 d 4)", "(move king j 10 c 3)", "(move king j 10 b 2)", "(move king j 10 a 1)", "(move man a 9 b 10)", "(move man a 9 c 7)", "(move king a 9 b 10)", "(move king a 9 b 8)", "(move king a 9 c 7)", "(move king a 9 d 6)", "(move king a 9 e 5)", "(move king a 9 f 4)", "(move king a 9 g 3)", "(move king a 9 h 2)", "(move king a 9 i 1)", "(move man c 9 d 10)", "(move man c 9 b 10)", "(move man c 9 e 7)", "(move man c 9 a 7)", "(move king c 9 d 10)", "(move king c 9 b 10)", "(move king c 9 d 8)", "(move king c 9 e 7)", "(move king c 9 f 6)", "(move king c 9 g 5)", "(move king c 9 h 4)", "(move king c 9 i 3)", "(move king c 9 j 2)", "(move king c 9 b 8)", "(move king c 9 a 7)", "(move man e 9 f 10)", "(move man e 9 d 10)", "(move man e 9 g 7)", "(move man e 9 c 7)", "(move king e 9 f 10)", "(move king e 9 d 10)", "(move king e 9 f 8)", "(move king e 9 g 7)", "(move king e 9 h 6)", "(move king e 9 i 5)", "(move king e 9 j 4)", "(move king e 9 d 8)", "(move king e 9 c 7)", "(move king e 9 b 6)", "(move king e 9 a 5)", "(move man g 9 h 10)", "(move man g 9 f 10)", "(move man g 9 i 7)", "(move man g 9 e 7)", "(move king g 9 h 10)", "(move king g 9 f 10)", "(move king g 9 h 8)", "(move king g 9 i 7)", "(move king g 9 j 6)", "(move king g 9 f 8)", "(move king g 9 e 7)", "(move king g 9 d 6)", "(move king g 9 c 5)", "(move king g 9 b 4)", "(move king g 9 a 3)", "(move man i 9 j 10)", "(move man i 9 h 10)", "(move man i 9 g 7)", "(move king i 9 j 10)", "(move king i 9 h 10)", "(move king i 9 j 8)", "(move king i 9 h 8)", "(move king i 9 g 7)", "(move king i 9 f 6)", "(move king i 9 e 5)", "(move king i 9 d 4)", "(move king i 9 c 3)", "(move king i 9 b 2)", "(move king i 9 a 1)", "(move man b 8 c 9)", "(move man b 8 a 9)", "(move man b 8 d 10)", "(move man b 8 d 6)", "(move king b 8 c 9)", "(move king b 8 d 10)", "(move king b 8 a 9)", "(move king b 8 c 7)", "(move king b 8 d 6)", "(move king b 8 e 5)", "(move king b 8 f 4)", "(move king b 8 g 3)", "(move king b 8 h 2)", "(move king b 8 i 1)", "(move king b 8 a 7)", "(move man d 8 e 9)", "(move man d 8 c 9)", "(move man d 8 f 10)", "(move man d 8 b 10)", "(move man d 8 f 6)", "(move man d 8 b 6)", "(move king d 8 e 9)", "(move king d 8 f 10)", "(move king d 8 c 9)", "(move king d 8 b 10)", "(move king d 8 e 7)", "(move king d 8 f 6)", "(move king d 8 g 5)", "(move king d 8 h 4)", "(move king d 8 i 3)", "(move king d 8 j 2)", "(move king d 8 c 7)", "(move king d 8 b 6)", "(move king d 8 a 5)", "(move man f 8 g 9)", "(move man f 8 e 9)", "(move man f 8 h 10)", "(move man f 8 d 10)", "(move man f 8 h 6)", "(move man f 8 d 6)", "(move king f 8 g 9)", "(move king f 8 h 10)", "(move king f 8 e 9)", "(move king f 8 d 10)", "(move king f 8 g 7)", "(move king f 8 h 6)", "(move king f 8 i 5)", "(move king f 8 j 4)", "(move king f 8 e 7)", "(move king f 8 d 6)", "(move king f 8 c 5)", "(move king f 8 b 4)", "(move king f 8 a 3)", "(move man h 8 i 9)", "(move man h 8 g 9)", "(move man h 8 j 10)", "(move man h 8 f 10)", "(move man h 8 j 6)", "(move man h 8 f 6)", "(move king h 8 i 9)", "(move king h 8 j 10)", "(move king h 8 g 9)", "(move king h 8 f 10)", "(move king h 8 i 7)", "(move king h 8 j 6)", "(move king h 8 g 7)", "(move king h 8 f 6)", "(move king h 8 e 5)", "(move king h 8 d 4)", "(move king h 8 c 3)", "(move king h 8 b 2)", "(move king h 8 a 1)", "(move man j 8 i 9)", "(move man j 8 h 10)", "(move man j 8 h 6)", "(move king j 8 i 9)", "(move king j 8 h 10)", "(move king j 8 i 7)", "(move king j 8 h 6)", "(move king j 8 g 5)", "(move king j 8 f 4)", "(move king j 8 e 3)", "(move king j 8 d 2)", "(move king j 8 c 1)", "(move man a 7 b 8)", "(move man a 7 c 9)", "(move man a 7 c 5)", "(move king a 7 b 8)", "(move king a 7 c 9)", "(move king a 7 d 10)", "(move king a 7 b 6)", "(move king a 7 c 5)", "(move king a 7 d 4)", "(move king a 7 e 3)", "(move king a 7 f 2)", "(move king a 7 g 1)", "(move man c 7 d 8)", "(move man c 7 b 8)", "(move man c 7 e 9)", "(move man c 7 a 9)", "(move man c 7 e 5)", "(move man c 7 a 5)", "(move king c 7 d 8)", "(move king c 7 e 9)", "(move king c 7 f 10)", "(move king c 7 b 8)", "(move king c 7 a 9)", "(move king c 7 d 6)", "(move king c 7 e 5)", "(move king c 7 f 4)", "(move king c 7 g 3)", "(move king c 7 h 2)", "(move king c 7 i 1)", "(move king c 7 b 6)", "(move king c 7 a 5)", "(move man e 7 f 8)", "(move man e 7 d 8)", "(move man e 7 g 9)", "(move man e 7 c 9)", "(move man e 7 g 5)", "(move man e 7 c 5)", "(move king e 7 f 8)", "(move king e 7 g 9)", "(move king e 7 h 10)", "(move king e 7 d 8)", "(move king e 7 c 9)", "(move king e 7 b 10)", "(move king e 7 f 6)", "(move king e 7 g 5)", "(move king e 7 h 4)", "(move king e 7 i 3)", "(move king e 7 j 2)", "(move king e 7 d 6)", "(move king e 7 c 5)", "(move king e 7 b 4)", "(move king e 7 a 3)", "(move man g 7 h 8)", "(move man g 7 f 8)", "(move man g 7 i 9)", "(move man g 7 e 9)", "(move man g 7 i 5)", "(move man g 7 e 5)", "(move king g 7 h 8)", "(move king g 7 i 9)", "(move king g 7 j 10)", "(move king g 7 f 8)", "(move king g 7 e 9)", "(move king g 7 d 10)", "(move king g 7 h 6)", "(move king g 7 i 5)", "(move king g 7 j 4)", "(move king g 7 f 6)", "(move king g 7 e 5)", "(move king g 7 d 4)", "(move king g 7 c 3)", "(move king g 7 b 2)", "(move king g 7 a 1)", "(move man i 7 j 8)", "(move man i 7 h 8)", "(move man i 7 g 9)", "(move man i 7 g 5)", "(move king i 7 j 8)", "(move king i 7 h 8)", "(move king i 7 g 9)", "(move king i 7 f 10)", "(move king i 7 j 6)", "(move king i 7 h 6)", "(move king i 7 g 5)", "(move king i 7 f 4)", "(move king i 7 e 3)", "(move king i 7 d 2)", "(move king i 7 c 1)", "(move man b 6 c 7)", "(move man b 6 a 7)", "(move man b 6 d 8)", "(move man b 6 d 4)", "(move king b 6 c 7)", "(move king b 6 d 8)", "(move king b 6 e 9)", "(move king b 6 f 10)", "(move king b 6 a 7)", "(move king b 6 c 5)", "(move king b 6 d 4)", "(move king b 6 e 3)", "(move king b 6 f 2)", "(move king b 6 g 1)", "(move king b 6 a 5)", "(move man d 6 e 7)", "(move man d 6 c 7)", "(move man d 6 f 8)", "(move man d 6 b 8)", "(move man d 6 f 4)", "(move man d 6 b 4)", "(move king d 6 e 7)", "(move king d 6 f 8)", "(move king d 6 g 9)", "(move king d 6 h 10)", "(move king d 6 c 7)", "(move king d 6 b 8)", "(move king d 6 a 9)", "(move king d 6 e 5)", "(move king d 6 f 4)", "(move king d 6 g 3)", "(move king d 6 h 2)", "(move king d 6 i 1)", "(move king d 6 c 5)", "(move king d 6 b 4)", "(move king d 6 a 3)", "(move man f 6 g 7)", "(move man f 6 e 7)", "(move man f 6 h 8)", "(move man f 6 d 8)", "(move man f 6 h 4)", "(move man f 6 d 4)", "(move king f 6 g 7)", "(move king f 6 h 8)", "(move king f 6 i 9)", "(move king f 6 j 10)", "(move king f 6 e 7)", "(move king f 6 d 8)", "(move king f 6 c 9)", "(move king f 6 b 10)", "(move king f 6 g 5)", "(move king f 6 h 4)", "(move king f 6 i 3)", "(move king f 6 j 2)", "(move king f 6 e 5)", "(move king f 6 d 4)", "(move king f 6 c 3)", "(move king f 6 b 2)", "(move king f 6 a 1)", "(move man h 6 i 7)", "(move man h 6 g 7)", "(move man h 6 j 8)", "(move man h 6 f 8)", "(move man h 6 j 4)", "(move man h 6 f 4)", "(move king h 6 i 7)", "(move king h 6 j 8)", "(move king h 6 g 7)", "(move king h 6 f 8)", "(move king h 6 e 9)", "(move king h 6 d 10)", "(move king h 6 i 5)", "(move king h 6 j 4)", "(move king h 6 g 5)", "(move king h 6 f 4)", "(move king h 6 e 3)", "(move king h 6 d 2)", "(move king h 6 c 1)", "(move man j 6 i 7)", "(move man j 6 h 8)", "(move man j 6 h 4)", "(move king j 6 i 7)", "(move king j 6 h 8)", "(move king j 6 g 9)", "(move king j 6 f 10)", "(move king j 6 i 5)", "(move king j 6 h 4)", "(move king j 6 g 3)", "(move king j 6 f 2)", "(move king j 6 e 1)", "(move man a 5 b 6)", "(move man a 5 c 7)", "(move man a 5 c 3)", "(move king a 5 b 6)", "(move king a 5 c 7)", "(move king a 5 d 8)", "(move king a 5 e 9)", "(move king a 5 f 10)", "(move king a 5 b 4)", "(move king a 5 c 3)", "(move king a 5 d 2)", "(move king a 5 e 1)", "(move man c 5 d 6)", "(move man c 5 b 6)", "(move man c 5 e 7)", "(move man c 5 a 7)", "(move man c 5 e 3)", "(move man c 5 a 3)", "(move king c 5 d 6)", "(move king c 5 e 7)", "(move king c 5 f 8)", "(move king c 5 g 9)", "(move king c 5 h 10)", "(move king c 5 b 6)", "(move king c 5 a 7)", "(move king c 5 d 4)", "(move king c 5 e 3)", "(move king c 5 f 2)", "(move king c 5 g 1)", "(move king c 5 b 4)", "(move king c 5 a 3)", "(move man e 5 f 6)", "(move man e 5 d 6)", "(move man e 5 g 7)", "(move man e 5 c 7)", "(move man e 5 g 3)", "(move man e 5 c 3)", "(move king e 5 f 6)", "(move king e 5 g 7)", "(move king e 5 h 8)", "(move king e 5 i 9)", "(move king e 5 j 10)", "(move king e 5 d 6)", "(move king e 5 c 7)", "(move king e 5 b 8)", "(move king e 5 a 9)", "(move king e 5 f 4)", "(move king e 5 g 3)", "(move king e 5 h 2)", "(move king e 5 i 1)", "(move king e 5 d 4)", "(move king e 5 c 3)", "(move king e 5 b 2)", "(move king e 5 a 1)", "(move man g 5 h 6)", "(move man g 5 f 6)", "(move man g 5 i 7)", "(move man g 5 e 7)", "(move man g 5 i 3)", "(move man g 5 e 3)", "(move king g 5 h 6)", "(move king g 5 i 7)", "(move king g 5 j 8)", "(move king g 5 f 6)", "(move king g 5 e 7)", "(move king g 5 d 8)", "(move king g 5 c 9)", "(move king g 5 b 10)", "(move king g 5 h 4)", "(move king g 5 i 3)", "(move king g 5 j 2)", "(move king g 5 f 4)", "(move king g 5 e 3)", "(move king g 5 d 2)", "(move king g 5 c 1)", "(move man i 5 j 6)", "(move man i 5 h 6)", "(move man i 5 g 7)", "(move man i 5 g 3)", "(move king i 5 j 6)", "(move king i 5 h 6)", "(move king i 5 g 7)", "(move king i 5 f 8)", "(move king i 5 e 9)", "(move king i 5 d 10)", "(move king i 5 j 4)", "(move king i 5 h 4)", "(move king i 5 g 3)", "(move king i 5 f 2)", "(move king i 5 e 1)", "(move man b 4 c 5)", "(move man b 4 a 5)", "(move man b 4 d 6)", "(move man b 4 d 2)", "(move king b 4 c 5)", "(move king b 4 d 6)", "(move king b 4 e 7)", "(move king b 4 f 8)", "(move king b 4 g 9)", "(move king b 4 h 10)", "(move king b 4 a 5)", "(move king b 4 c 3)", "(move king b 4 d 2)", "(move king b 4 e 1)", "(move king b 4 a 3)", "(move man d 4 e 5)", "(move man d 4 c 5)", "(move man d 4 f 6)", "(move man d 4 b 6)", "(move man d 4 f 2)", "(move man d 4 b 2)", "(move king d 4 e 5)", "(move king d 4 f 6)", "(move king d 4 g 7)", "(move king d 4 h 8)", "(move king d 4 i 9)", "(move king d 4 j 10)", "(move king d 4 c 5)", "(move king d 4 b 6)", "(move king d 4 a 7)", "(move king d 4 e 3)", "(move king d 4 f 2)", "(move king d 4 g 1)", "(move king d 4 c 3)", "(move king d 4 b 2)", "(move king d 4 a 1)", "(move man f 4 g 5)", "(move man f 4 e 5)", "(move man f 4 h 6)", "(move man f 4 d 6)", "(move man f 4 h 2)", "(move man f 4 d 2)", "(move king f 4 g 5)", "(move king f 4 h 6)", "(move king f 4 i 7)", "(move king f 4 j 8)", "(move king f 4 e 5)", "(move king f 4 d 6)", "(move king f 4 c 7)", "(move king f 4 b 8)", "(move king f 4 a 9)", "(move king f 4 g 3)", "(move king f 4 h 2)", "(move king f 4 i 1)", "(move king f 4 e 3)", "(move king f 4 d 2)", "(move king f 4 c 1)", "(move man h 4 i 5)", "(move man h 4 g 5)", "(move man h 4 j 6)", "(move man h 4 f 6)", "(move man h 4 j 2)", "(move man h 4 f 2)", "(move king h 4 i 5)", "(move king h 4 j 6)", "(move king h 4 g 5)", "(move king h 4 f 6)", "(move king h 4 e 7)", "(move king h 4 d 8)", "(move king h 4 c 9)", "(move king h 4 b 10)", "(move king h 4 i 3)", "(move king h 4 j 2)", "(move king h 4 g 3)", "(move king h 4 f 2)", "(move king h 4 e 1)", "(move man j 4 i 5)", "(move man j 4 h 6)", "(move man j 4 h 2)", "(move king j 4 i 5)", "(move king j 4 h 6)", "(move king j 4 g 7)", "(move king j 4 f 8)", "(move king j 4 e 9)", "(move king j 4 d 10)", "(move king j 4 i 3)", "(move king j 4 h 2)", "(move king j 4 g 1)", "(move man a 3 b 4)", "(move man a 3 c 5)", "(move man a 3 c 1)", "(move king a 3 b 4)", "(move king a 3 c 5)", "(move king a 3 d 6)", "(move king a 3 e 7)", "(move king a 3 f 8)", "(move king a 3 g 9)", "(move king a 3 h 10)", "(move king a 3 b 2)", "(move king a 3 c 1)", "(move man c 3 d 4)", "(move man c 3 b 4)", "(move man c 3 e 5)", "(move man c 3 a 5)", "(move man c 3 e 1)", "(move man c 3 a 1)", "(move king c 3 d 4)", "(move king c 3 e 5)", "(move king c 3 f 6)", "(move king c 3 g 7)", "(move king c 3 h 8)", "(move king c 3 i 9)", "(move king c 3 j 10)", "(move king c 3 b 4)", "(move king c 3 a 5)", "(move king c 3 d 2)", "(move king c 3 e 1)", "(move king c 3 b 2)", "(move king c 3 a 1)", "(move man e 3 f 4)", "(move man e 3 d 4)", "(move man e 3 g 5)", "(move man e 3 c 5)", "(move man e 3 g 1)", "(move man e 3 c 1)", "(move king e 3 f 4)", "(move king e 3 g 5)", "(move king e 3 h 6)", "(move king e 3 i 7)", "(move king e 3 j 8)", "(move king e 3 d 4)", "(move king e 3 c 5)", "(move king e 3 b 6)", "(move king e 3 a 7)", "(move king e 3 f 2)", "(move king e 3 g 1)", "(move king e 3 d 2)", "(move king e 3 c 1)", "(move man g 3 h 4)", "(move man g 3 f 4)", "(move man g 3 i 5)", "(move man g 3 e 5)", "(move man g 3 i 1)", "(move man g 3 e 1)", "(move king g 3 h 4)", "(move king g 3 i 5)", "(move king g 3 j 6)", "(move king g 3 f 4)", "(move king g 3 e 5)", "(move king g 3 d 6)", "(move king g 3 c 7)", "(move king g 3 b 8)", "(move king g 3 a 9)", "(move king g 3 h 2)", "(move king g 3 i 1)", "(move king g 3 f 2)", "(move king g 3 e 1)", "(move man i 3 j 4)", "(move man i 3 h 4)", "(move man i 3 g 5)", "(move man i 3 g 1)", "(move king i 3 j 4)", "(move king i 3 h 4)", "(move king i 3 g 5)", "(move king i 3 f 6)", "(move king i 3 e 7)", "(move king i 3 d 8)", "(move king i 3 c 9)", "(move king i 3 b 10)", "(move king i 3 j 2)", "(move king i 3 h 2)", "(move king i 3 g 1)", "(move man b 2 c 3)", "(move man b 2 a 3)", "(move man b 2 d 4)", "(move king b 2 c 3)", "(move king b 2 d 4)", "(move king b 2 e 5)", "(move king b 2 f 6)", "(move king b 2 g 7)", "(move king b 2 h 8)", "(move king b 2 i 9)", "(move king b 2 j 10)", "(move king b 2 a 3)", "(move king b 2 c 1)", "(move king b 2 a 1)", "(move man d 2 e 3)", "(move man d 2 c 3)", "(move man d 2 f 4)", "(move man d 2 b 4)", "(move king d 2 e 3)", "(move king d 2 f 4)", "(move king d 2 g 5)", "(move king d 2 h 6)", "(move king d 2 i 7)", "(move king d 2 j 8)", "(move king d 2 c 3)", "(move king d 2 b 4)", "(move king d 2 a 5)", "(move king d 2 e 1)", "(move king d 2 c 1)", "(move man f 2 g 3)", "(move man f 2 e 3)", "(move man f 2 h 4)", "(move man f 2 d 4)", "(move king f 2 g 3)", "(move king f 2 h 4)", "(move king f 2 i 5)", "(move king f 2 j 6)", "(move king f 2 e 3)", "(move king f 2 d 4)", "(move king f 2 c 5)", "(move king f 2 b 6)", "(move king f 2 a 7)", "(move king f 2 g 1)", "(move king f 2 e 1)", "(move man h 2 i 3)", "(move man h 2 g 3)", "(move man h 2 j 4)", "(move man h 2 f 4)", "(move king h 2 i 3)", "(move king h 2 j 4)", "(move king h 2 g 3)", "(move king h 2 f 4)", "(move king h 2 e 5)", "(move king h 2 d 6)", "(move king h 2 c 7)", "(move king h 2 b 8)", "(move king h 2 a 9)", "(move king h 2 i 1)", "(move king h 2 g 1)", "(move man j 2 i 3)", "(move man j 2 h 4)", "(move king j 2 i 3)", "(move king j 2 h 4)", "(move king j 2 g 5)", "(move king j 2 f 6)", "(move king j 2 e 7)", "(move king j 2 d 8)", "(move king j 2 c 9)", "(move king j 2 b 10)", "(move king j 2 i 1)", "(move man a 1 b 2)", "(move man a 1 c 3)", "(move king a 1 b 2)", "(move king a 1 c 3)", "(move king a 1 d 4)", "(move king a 1 e 5)", "(move king a 1 f 6)", "(move king a 1 g 7)", "(move king a 1 h 8)", "(move king a 1 i 9)", "(move king a 1 j 10)", "(move man c 1 d 2)", "(move man c 1 b 2)", "(move man c 1 e 3)", "(move man c 1 a 3)", "(move king c 1 d 2)", "(move king c 1 e 3)", "(move king c 1 f 4)", "(move king c 1 g 5)", "(move king c 1 h 6)", "(move king c 1 i 7)", "(move king c 1 j 8)", "(move king c 1 b 2)", "(move king c 1 a 3)", "(move man e 1 f 2)", "(move man e 1 d 2)", "(move man e 1 g 3)", "(move man e 1 c 3)", "(move king e 1 f 2)", "(move king e 1 g 3)", "(move king e 1 h 4)", "(move king e 1 i 5)", "(move king e 1 j 6)", "(move king e 1 d 2)", "(move king e 1 c 3)", "(move king e 1 b 4)", "(move king e 1 a 5)", "(move man g 1 h 2)", "(move man g 1 f 2)", "(move man g 1 i 3)", "(move man g 1 e 3)", "(move king g 1 h 2)", "(move king g 1 i 3)", "(move king g 1 j 4)", "(move king g 1 f 2)", "(move king g 1 e 3)", "(move king g 1 d 4)", "(move king g 1 c 5)", "(move king g 1 b 6)", "(move king g 1 a 7)", "(move man i 1 j 2)", "(move man i 1 h 2)", "(move man i 1 g 3)", "(move king i 1 j 2)", "(move king i 1 h 2)", "(move king i 1 g 3)", "(move king i 1 f 4)", "(move king i 1 e 5)", "(move king i 1 d 6)", "(move king i 1 c 7)", "(move king i 1 b 8)", "(move king i 1 a 9)"};
    // generating for white man 1 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: invalid
        ddi->diagonals.emplace_back(7, -1);

        // to position 12, legal: (legal white (move man b 10 d 8))
        ddi->diagonals.emplace_back(12, 1);

        this->diagonal_data[0].push_back(ddi);


        this->reverse_legal_lookup_white[1] = ReverseLegalLookup(Role::White, Piece::Man, 1, 12, SE);
    }
    // generating for white man 1 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 6, legal: invalid
        ddi->diagonals.emplace_back(6, -1);

        this->diagonal_data[0].push_back(ddi);


    }
    // generating for white man 2 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: invalid
        ddi->diagonals.emplace_back(8, -1);

        // to position 13, legal: (legal white (move man d 10 f 8))
        ddi->diagonals.emplace_back(13, 11);

        this->diagonal_data[1].push_back(ddi);


        this->reverse_legal_lookup_white[11] = ReverseLegalLookup(Role::White, Piece::Man, 2, 13, SE);
    }
    // generating for white man 2 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: invalid
        ddi->diagonals.emplace_back(7, -1);

        // to position 11, legal: (legal white (move man d 10 b 8))
        ddi->diagonals.emplace_back(11, 12);

        this->diagonal_data[1].push_back(ddi);


        this->reverse_legal_lookup_white[12] = ReverseLegalLookup(Role::White, Piece::Man, 2, 11, SW);
    }
    // generating for white man 3 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 9, legal: invalid
        ddi->diagonals.emplace_back(9, -1);

        // to position 14, legal: (legal white (move man f 10 h 8))
        ddi->diagonals.emplace_back(14, 22);

        this->diagonal_data[2].push_back(ddi);


        this->reverse_legal_lookup_white[22] = ReverseLegalLookup(Role::White, Piece::Man, 3, 14, SE);
    }
    // generating for white man 3 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: invalid
        ddi->diagonals.emplace_back(8, -1);

        // to position 12, legal: (legal white (move man f 10 d 8))
        ddi->diagonals.emplace_back(12, 23);

        this->diagonal_data[2].push_back(ddi);


        this->reverse_legal_lookup_white[23] = ReverseLegalLookup(Role::White, Piece::Man, 3, 12, SW);
    }
    // generating for white man 4 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: invalid
        ddi->diagonals.emplace_back(10, -1);

        // to position 15, legal: (legal white (move man h 10 j 8))
        ddi->diagonals.emplace_back(15, 33);

        this->diagonal_data[3].push_back(ddi);


        this->reverse_legal_lookup_white[33] = ReverseLegalLookup(Role::White, Piece::Man, 4, 15, SE);
    }
    // generating for white man 4 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: invalid
        ddi->diagonals.emplace_back(9, -1);

        // to position 13, legal: (legal white (move man h 10 f 8))
        ddi->diagonals.emplace_back(13, 34);

        this->diagonal_data[3].push_back(ddi);


        this->reverse_legal_lookup_white[34] = ReverseLegalLookup(Role::White, Piece::Man, 4, 13, SW);
    }
    // generating for white man 5 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 10, legal: invalid
        ddi->diagonals.emplace_back(10, -1);

        // to position 14, legal: (legal white (move man j 10 h 8))
        ddi->diagonals.emplace_back(14, 44);

        this->diagonal_data[4].push_back(ddi);


        this->reverse_legal_lookup_white[44] = ReverseLegalLookup(Role::White, Piece::Man, 5, 14, SW);
    }
    // generating for white man 6 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal white (move man a 9 b 10))
        ddi->diagonals.emplace_back(1, 54);

        this->diagonal_data[5].push_back(ddi);


        this->reverse_legal_lookup_white[54] = ReverseLegalLookup(Role::White, Piece::Man, 6, 1, NE);
    }
    // generating for white man 6 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 11, legal: invalid
        ddi->diagonals.emplace_back(11, -1);

        // to position 17, legal: (legal white (move man a 9 c 7))
        ddi->diagonals.emplace_back(17, 55);

        this->diagonal_data[5].push_back(ddi);


        this->reverse_legal_lookup_white[55] = ReverseLegalLookup(Role::White, Piece::Man, 6, 17, SE);
    }
    // generating for white man 7 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal white (move man c 9 d 10))
        ddi->diagonals.emplace_back(2, 65);

        this->diagonal_data[6].push_back(ddi);


        this->reverse_legal_lookup_white[65] = ReverseLegalLookup(Role::White, Piece::Man, 7, 2, NE);
    }
    // generating for white man 7 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal white (move man c 9 b 10))
        ddi->diagonals.emplace_back(1, 66);

        this->diagonal_data[6].push_back(ddi);


        this->reverse_legal_lookup_white[66] = ReverseLegalLookup(Role::White, Piece::Man, 7, 1, NW);
    }
    // generating for white man 7 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 12, legal: invalid
        ddi->diagonals.emplace_back(12, -1);

        // to position 18, legal: (legal white (move man c 9 e 7))
        ddi->diagonals.emplace_back(18, 67);

        this->diagonal_data[6].push_back(ddi);


        this->reverse_legal_lookup_white[67] = ReverseLegalLookup(Role::White, Piece::Man, 7, 18, SE);
    }
    // generating for white man 7 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: invalid
        ddi->diagonals.emplace_back(11, -1);

        // to position 16, legal: (legal white (move man c 9 a 7))
        ddi->diagonals.emplace_back(16, 68);

        this->diagonal_data[6].push_back(ddi);


        this->reverse_legal_lookup_white[68] = ReverseLegalLookup(Role::White, Piece::Man, 7, 16, SW);
    }
    // generating for white man 8 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal white (move man e 9 f 10))
        ddi->diagonals.emplace_back(3, 80);

        this->diagonal_data[7].push_back(ddi);


        this->reverse_legal_lookup_white[80] = ReverseLegalLookup(Role::White, Piece::Man, 8, 3, NE);
    }
    // generating for white man 8 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal white (move man e 9 d 10))
        ddi->diagonals.emplace_back(2, 81);

        this->diagonal_data[7].push_back(ddi);


        this->reverse_legal_lookup_white[81] = ReverseLegalLookup(Role::White, Piece::Man, 8, 2, NW);
    }
    // generating for white man 8 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 13, legal: invalid
        ddi->diagonals.emplace_back(13, -1);

        // to position 19, legal: (legal white (move man e 9 g 7))
        ddi->diagonals.emplace_back(19, 82);

        this->diagonal_data[7].push_back(ddi);


        this->reverse_legal_lookup_white[82] = ReverseLegalLookup(Role::White, Piece::Man, 8, 19, SE);
    }
    // generating for white man 8 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 12, legal: invalid
        ddi->diagonals.emplace_back(12, -1);

        // to position 17, legal: (legal white (move man e 9 c 7))
        ddi->diagonals.emplace_back(17, 83);

        this->diagonal_data[7].push_back(ddi);


        this->reverse_legal_lookup_white[83] = ReverseLegalLookup(Role::White, Piece::Man, 8, 17, SW);
    }
    // generating for white man 9 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 4, legal: (legal white (move man g 9 h 10))
        ddi->diagonals.emplace_back(4, 95);

        this->diagonal_data[8].push_back(ddi);


        this->reverse_legal_lookup_white[95] = ReverseLegalLookup(Role::White, Piece::Man, 9, 4, NE);
    }
    // generating for white man 9 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal white (move man g 9 f 10))
        ddi->diagonals.emplace_back(3, 96);

        this->diagonal_data[8].push_back(ddi);


        this->reverse_legal_lookup_white[96] = ReverseLegalLookup(Role::White, Piece::Man, 9, 3, NW);
    }
    // generating for white man 9 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 14, legal: invalid
        ddi->diagonals.emplace_back(14, -1);

        // to position 20, legal: (legal white (move man g 9 i 7))
        ddi->diagonals.emplace_back(20, 97);

        this->diagonal_data[8].push_back(ddi);


        this->reverse_legal_lookup_white[97] = ReverseLegalLookup(Role::White, Piece::Man, 9, 20, SE);
    }
    // generating for white man 9 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 13, legal: invalid
        ddi->diagonals.emplace_back(13, -1);

        // to position 18, legal: (legal white (move man g 9 e 7))
        ddi->diagonals.emplace_back(18, 98);

        this->diagonal_data[8].push_back(ddi);


        this->reverse_legal_lookup_white[98] = ReverseLegalLookup(Role::White, Piece::Man, 9, 18, SW);
    }
    // generating for white man 10 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 5, legal: (legal white (move man i 9 j 10))
        ddi->diagonals.emplace_back(5, 110);

        this->diagonal_data[9].push_back(ddi);


        this->reverse_legal_lookup_white[110] = ReverseLegalLookup(Role::White, Piece::Man, 10, 5, NE);
    }
    // generating for white man 10 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 4, legal: (legal white (move man i 9 h 10))
        ddi->diagonals.emplace_back(4, 111);

        this->diagonal_data[9].push_back(ddi);


        this->reverse_legal_lookup_white[111] = ReverseLegalLookup(Role::White, Piece::Man, 10, 4, NW);
    }
    // generating for white man 10 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 15, legal: invalid
        ddi->diagonals.emplace_back(15, -1);

        this->diagonal_data[9].push_back(ddi);


    }
    // generating for white man 10 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 14, legal: invalid
        ddi->diagonals.emplace_back(14, -1);

        // to position 19, legal: (legal white (move man i 9 g 7))
        ddi->diagonals.emplace_back(19, 112);

        this->diagonal_data[9].push_back(ddi);


        this->reverse_legal_lookup_white[112] = ReverseLegalLookup(Role::White, Piece::Man, 10, 19, SW);
    }
    // generating for white man 11 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal white (move man b 8 c 9))
        ddi->diagonals.emplace_back(7, 124);

        // to position 2, legal: (legal white (move man b 8 d 10))
        ddi->diagonals.emplace_back(2, 126);

        this->diagonal_data[10].push_back(ddi);


        this->reverse_legal_lookup_white[124] = ReverseLegalLookup(Role::White, Piece::Man, 11, 7, NE);
        this->reverse_legal_lookup_white[126] = ReverseLegalLookup(Role::White, Piece::Man, 11, 2, NE);
    }
    // generating for white man 11 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 6, legal: (legal white (move man b 8 a 9))
        ddi->diagonals.emplace_back(6, 125);

        this->diagonal_data[10].push_back(ddi);


        this->reverse_legal_lookup_white[125] = ReverseLegalLookup(Role::White, Piece::Man, 11, 6, NW);
    }
    // generating for white man 11 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 17, legal: invalid
        ddi->diagonals.emplace_back(17, -1);

        // to position 22, legal: (legal white (move man b 8 d 6))
        ddi->diagonals.emplace_back(22, 127);

        this->diagonal_data[10].push_back(ddi);


        this->reverse_legal_lookup_white[127] = ReverseLegalLookup(Role::White, Piece::Man, 11, 22, SE);
    }
    // generating for white man 11 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 16, legal: invalid
        ddi->diagonals.emplace_back(16, -1);

        this->diagonal_data[10].push_back(ddi);


    }
    // generating for white man 12 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal white (move man d 8 e 9))
        ddi->diagonals.emplace_back(8, 139);

        // to position 3, legal: (legal white (move man d 8 f 10))
        ddi->diagonals.emplace_back(3, 141);

        this->diagonal_data[11].push_back(ddi);


        this->reverse_legal_lookup_white[139] = ReverseLegalLookup(Role::White, Piece::Man, 12, 8, NE);
        this->reverse_legal_lookup_white[141] = ReverseLegalLookup(Role::White, Piece::Man, 12, 3, NE);
    }
    // generating for white man 12 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal white (move man d 8 c 9))
        ddi->diagonals.emplace_back(7, 140);

        // to position 1, legal: (legal white (move man d 8 b 10))
        ddi->diagonals.emplace_back(1, 142);

        this->diagonal_data[11].push_back(ddi);


        this->reverse_legal_lookup_white[140] = ReverseLegalLookup(Role::White, Piece::Man, 12, 7, NW);
        this->reverse_legal_lookup_white[142] = ReverseLegalLookup(Role::White, Piece::Man, 12, 1, NW);
    }
    // generating for white man 12 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 18, legal: invalid
        ddi->diagonals.emplace_back(18, -1);

        // to position 23, legal: (legal white (move man d 8 f 6))
        ddi->diagonals.emplace_back(23, 143);

        this->diagonal_data[11].push_back(ddi);


        this->reverse_legal_lookup_white[143] = ReverseLegalLookup(Role::White, Piece::Man, 12, 23, SE);
    }
    // generating for white man 12 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: invalid
        ddi->diagonals.emplace_back(17, -1);

        // to position 21, legal: (legal white (move man d 8 b 6))
        ddi->diagonals.emplace_back(21, 144);

        this->diagonal_data[11].push_back(ddi);


        this->reverse_legal_lookup_white[144] = ReverseLegalLookup(Role::White, Piece::Man, 12, 21, SW);
    }
    // generating for white man 13 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal white (move man f 8 g 9))
        ddi->diagonals.emplace_back(9, 158);

        // to position 4, legal: (legal white (move man f 8 h 10))
        ddi->diagonals.emplace_back(4, 160);

        this->diagonal_data[12].push_back(ddi);


        this->reverse_legal_lookup_white[158] = ReverseLegalLookup(Role::White, Piece::Man, 13, 9, NE);
        this->reverse_legal_lookup_white[160] = ReverseLegalLookup(Role::White, Piece::Man, 13, 4, NE);
    }
    // generating for white man 13 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal white (move man f 8 e 9))
        ddi->diagonals.emplace_back(8, 159);

        // to position 2, legal: (legal white (move man f 8 d 10))
        ddi->diagonals.emplace_back(2, 161);

        this->diagonal_data[12].push_back(ddi);


        this->reverse_legal_lookup_white[159] = ReverseLegalLookup(Role::White, Piece::Man, 13, 8, NW);
        this->reverse_legal_lookup_white[161] = ReverseLegalLookup(Role::White, Piece::Man, 13, 2, NW);
    }
    // generating for white man 13 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 19, legal: invalid
        ddi->diagonals.emplace_back(19, -1);

        // to position 24, legal: (legal white (move man f 8 h 6))
        ddi->diagonals.emplace_back(24, 162);

        this->diagonal_data[12].push_back(ddi);


        this->reverse_legal_lookup_white[162] = ReverseLegalLookup(Role::White, Piece::Man, 13, 24, SE);
    }
    // generating for white man 13 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 18, legal: invalid
        ddi->diagonals.emplace_back(18, -1);

        // to position 22, legal: (legal white (move man f 8 d 6))
        ddi->diagonals.emplace_back(22, 163);

        this->diagonal_data[12].push_back(ddi);


        this->reverse_legal_lookup_white[163] = ReverseLegalLookup(Role::White, Piece::Man, 13, 22, SW);
    }
    // generating for white man 14 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal white (move man h 8 i 9))
        ddi->diagonals.emplace_back(10, 177);

        // to position 5, legal: (legal white (move man h 8 j 10))
        ddi->diagonals.emplace_back(5, 179);

        this->diagonal_data[13].push_back(ddi);


        this->reverse_legal_lookup_white[177] = ReverseLegalLookup(Role::White, Piece::Man, 14, 10, NE);
        this->reverse_legal_lookup_white[179] = ReverseLegalLookup(Role::White, Piece::Man, 14, 5, NE);
    }
    // generating for white man 14 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal white (move man h 8 g 9))
        ddi->diagonals.emplace_back(9, 178);

        // to position 3, legal: (legal white (move man h 8 f 10))
        ddi->diagonals.emplace_back(3, 180);

        this->diagonal_data[13].push_back(ddi);


        this->reverse_legal_lookup_white[178] = ReverseLegalLookup(Role::White, Piece::Man, 14, 9, NW);
        this->reverse_legal_lookup_white[180] = ReverseLegalLookup(Role::White, Piece::Man, 14, 3, NW);
    }
    // generating for white man 14 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 20, legal: invalid
        ddi->diagonals.emplace_back(20, -1);

        // to position 25, legal: (legal white (move man h 8 j 6))
        ddi->diagonals.emplace_back(25, 181);

        this->diagonal_data[13].push_back(ddi);


        this->reverse_legal_lookup_white[181] = ReverseLegalLookup(Role::White, Piece::Man, 14, 25, SE);
    }
    // generating for white man 14 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 19, legal: invalid
        ddi->diagonals.emplace_back(19, -1);

        // to position 23, legal: (legal white (move man h 8 f 6))
        ddi->diagonals.emplace_back(23, 182);

        this->diagonal_data[13].push_back(ddi);


        this->reverse_legal_lookup_white[182] = ReverseLegalLookup(Role::White, Piece::Man, 14, 23, SW);
    }
    // generating for white man 15 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal white (move man j 8 i 9))
        ddi->diagonals.emplace_back(10, 196);

        // to position 4, legal: (legal white (move man j 8 h 10))
        ddi->diagonals.emplace_back(4, 197);

        this->diagonal_data[14].push_back(ddi);


        this->reverse_legal_lookup_white[196] = ReverseLegalLookup(Role::White, Piece::Man, 15, 10, NW);
        this->reverse_legal_lookup_white[197] = ReverseLegalLookup(Role::White, Piece::Man, 15, 4, NW);
    }
    // generating for white man 15 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 20, legal: invalid
        ddi->diagonals.emplace_back(20, -1);

        // to position 24, legal: (legal white (move man j 8 h 6))
        ddi->diagonals.emplace_back(24, 198);

        this->diagonal_data[14].push_back(ddi);


        this->reverse_legal_lookup_white[198] = ReverseLegalLookup(Role::White, Piece::Man, 15, 24, SW);
    }
    // generating for white man 16 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal white (move man a 7 b 8))
        ddi->diagonals.emplace_back(11, 208);

        // to position 7, legal: (legal white (move man a 7 c 9))
        ddi->diagonals.emplace_back(7, 209);

        this->diagonal_data[15].push_back(ddi);


        this->reverse_legal_lookup_white[208] = ReverseLegalLookup(Role::White, Piece::Man, 16, 11, NE);
        this->reverse_legal_lookup_white[209] = ReverseLegalLookup(Role::White, Piece::Man, 16, 7, NE);
    }
    // generating for white man 16 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 21, legal: invalid
        ddi->diagonals.emplace_back(21, -1);

        // to position 27, legal: (legal white (move man a 7 c 5))
        ddi->diagonals.emplace_back(27, 210);

        this->diagonal_data[15].push_back(ddi);


        this->reverse_legal_lookup_white[210] = ReverseLegalLookup(Role::White, Piece::Man, 16, 27, SE);
    }
    // generating for white man 17 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 12, legal: (legal white (move man c 7 d 8))
        ddi->diagonals.emplace_back(12, 220);

        // to position 8, legal: (legal white (move man c 7 e 9))
        ddi->diagonals.emplace_back(8, 222);

        this->diagonal_data[16].push_back(ddi);


        this->reverse_legal_lookup_white[220] = ReverseLegalLookup(Role::White, Piece::Man, 17, 12, NE);
        this->reverse_legal_lookup_white[222] = ReverseLegalLookup(Role::White, Piece::Man, 17, 8, NE);
    }
    // generating for white man 17 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal white (move man c 7 b 8))
        ddi->diagonals.emplace_back(11, 221);

        // to position 6, legal: (legal white (move man c 7 a 9))
        ddi->diagonals.emplace_back(6, 223);

        this->diagonal_data[16].push_back(ddi);


        this->reverse_legal_lookup_white[221] = ReverseLegalLookup(Role::White, Piece::Man, 17, 11, NW);
        this->reverse_legal_lookup_white[223] = ReverseLegalLookup(Role::White, Piece::Man, 17, 6, NW);
    }
    // generating for white man 17 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 22, legal: invalid
        ddi->diagonals.emplace_back(22, -1);

        // to position 28, legal: (legal white (move man c 7 e 5))
        ddi->diagonals.emplace_back(28, 224);

        this->diagonal_data[16].push_back(ddi);


        this->reverse_legal_lookup_white[224] = ReverseLegalLookup(Role::White, Piece::Man, 17, 28, SE);
    }
    // generating for white man 17 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 21, legal: invalid
        ddi->diagonals.emplace_back(21, -1);

        // to position 26, legal: (legal white (move man c 7 a 5))
        ddi->diagonals.emplace_back(26, 225);

        this->diagonal_data[16].push_back(ddi);


        this->reverse_legal_lookup_white[225] = ReverseLegalLookup(Role::White, Piece::Man, 17, 26, SW);
    }
    // generating for white man 18 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 13, legal: (legal white (move man e 7 f 8))
        ddi->diagonals.emplace_back(13, 239);

        // to position 9, legal: (legal white (move man e 7 g 9))
        ddi->diagonals.emplace_back(9, 241);

        this->diagonal_data[17].push_back(ddi);


        this->reverse_legal_lookup_white[239] = ReverseLegalLookup(Role::White, Piece::Man, 18, 13, NE);
        this->reverse_legal_lookup_white[241] = ReverseLegalLookup(Role::White, Piece::Man, 18, 9, NE);
    }
    // generating for white man 18 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 12, legal: (legal white (move man e 7 d 8))
        ddi->diagonals.emplace_back(12, 240);

        // to position 7, legal: (legal white (move man e 7 c 9))
        ddi->diagonals.emplace_back(7, 242);

        this->diagonal_data[17].push_back(ddi);


        this->reverse_legal_lookup_white[240] = ReverseLegalLookup(Role::White, Piece::Man, 18, 12, NW);
        this->reverse_legal_lookup_white[242] = ReverseLegalLookup(Role::White, Piece::Man, 18, 7, NW);
    }
    // generating for white man 18 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 23, legal: invalid
        ddi->diagonals.emplace_back(23, -1);

        // to position 29, legal: (legal white (move man e 7 g 5))
        ddi->diagonals.emplace_back(29, 243);

        this->diagonal_data[17].push_back(ddi);


        this->reverse_legal_lookup_white[243] = ReverseLegalLookup(Role::White, Piece::Man, 18, 29, SE);
    }
    // generating for white man 18 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 22, legal: invalid
        ddi->diagonals.emplace_back(22, -1);

        // to position 27, legal: (legal white (move man e 7 c 5))
        ddi->diagonals.emplace_back(27, 244);

        this->diagonal_data[17].push_back(ddi);


        this->reverse_legal_lookup_white[244] = ReverseLegalLookup(Role::White, Piece::Man, 18, 27, SW);
    }
    // generating for white man 19 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 14, legal: (legal white (move man g 7 h 8))
        ddi->diagonals.emplace_back(14, 260);

        // to position 10, legal: (legal white (move man g 7 i 9))
        ddi->diagonals.emplace_back(10, 262);

        this->diagonal_data[18].push_back(ddi);


        this->reverse_legal_lookup_white[260] = ReverseLegalLookup(Role::White, Piece::Man, 19, 14, NE);
        this->reverse_legal_lookup_white[262] = ReverseLegalLookup(Role::White, Piece::Man, 19, 10, NE);
    }
    // generating for white man 19 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 13, legal: (legal white (move man g 7 f 8))
        ddi->diagonals.emplace_back(13, 261);

        // to position 8, legal: (legal white (move man g 7 e 9))
        ddi->diagonals.emplace_back(8, 263);

        this->diagonal_data[18].push_back(ddi);


        this->reverse_legal_lookup_white[261] = ReverseLegalLookup(Role::White, Piece::Man, 19, 13, NW);
        this->reverse_legal_lookup_white[263] = ReverseLegalLookup(Role::White, Piece::Man, 19, 8, NW);
    }
    // generating for white man 19 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: invalid
        ddi->diagonals.emplace_back(24, -1);

        // to position 30, legal: (legal white (move man g 7 i 5))
        ddi->diagonals.emplace_back(30, 264);

        this->diagonal_data[18].push_back(ddi);


        this->reverse_legal_lookup_white[264] = ReverseLegalLookup(Role::White, Piece::Man, 19, 30, SE);
    }
    // generating for white man 19 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 23, legal: invalid
        ddi->diagonals.emplace_back(23, -1);

        // to position 28, legal: (legal white (move man g 7 e 5))
        ddi->diagonals.emplace_back(28, 265);

        this->diagonal_data[18].push_back(ddi);


        this->reverse_legal_lookup_white[265] = ReverseLegalLookup(Role::White, Piece::Man, 19, 28, SW);
    }
    // generating for white man 20 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 15, legal: (legal white (move man i 7 j 8))
        ddi->diagonals.emplace_back(15, 281);

        this->diagonal_data[19].push_back(ddi);


        this->reverse_legal_lookup_white[281] = ReverseLegalLookup(Role::White, Piece::Man, 20, 15, NE);
    }
    // generating for white man 20 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 14, legal: (legal white (move man i 7 h 8))
        ddi->diagonals.emplace_back(14, 282);

        // to position 9, legal: (legal white (move man i 7 g 9))
        ddi->diagonals.emplace_back(9, 283);

        this->diagonal_data[19].push_back(ddi);


        this->reverse_legal_lookup_white[282] = ReverseLegalLookup(Role::White, Piece::Man, 20, 14, NW);
        this->reverse_legal_lookup_white[283] = ReverseLegalLookup(Role::White, Piece::Man, 20, 9, NW);
    }
    // generating for white man 20 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 25, legal: invalid
        ddi->diagonals.emplace_back(25, -1);

        this->diagonal_data[19].push_back(ddi);


    }
    // generating for white man 20 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 24, legal: invalid
        ddi->diagonals.emplace_back(24, -1);

        // to position 29, legal: (legal white (move man i 7 g 5))
        ddi->diagonals.emplace_back(29, 284);

        this->diagonal_data[19].push_back(ddi);


        this->reverse_legal_lookup_white[284] = ReverseLegalLookup(Role::White, Piece::Man, 20, 29, SW);
    }
    // generating for white man 21 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal white (move man b 6 c 7))
        ddi->diagonals.emplace_back(17, 296);

        // to position 12, legal: (legal white (move man b 6 d 8))
        ddi->diagonals.emplace_back(12, 298);

        this->diagonal_data[20].push_back(ddi);


        this->reverse_legal_lookup_white[296] = ReverseLegalLookup(Role::White, Piece::Man, 21, 17, NE);
        this->reverse_legal_lookup_white[298] = ReverseLegalLookup(Role::White, Piece::Man, 21, 12, NE);
    }
    // generating for white man 21 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 16, legal: (legal white (move man b 6 a 7))
        ddi->diagonals.emplace_back(16, 297);

        this->diagonal_data[20].push_back(ddi);


        this->reverse_legal_lookup_white[297] = ReverseLegalLookup(Role::White, Piece::Man, 21, 16, NW);
    }
    // generating for white man 21 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 27, legal: invalid
        ddi->diagonals.emplace_back(27, -1);

        // to position 32, legal: (legal white (move man b 6 d 4))
        ddi->diagonals.emplace_back(32, 299);

        this->diagonal_data[20].push_back(ddi);


        this->reverse_legal_lookup_white[299] = ReverseLegalLookup(Role::White, Piece::Man, 21, 32, SE);
    }
    // generating for white man 21 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 26, legal: invalid
        ddi->diagonals.emplace_back(26, -1);

        this->diagonal_data[20].push_back(ddi);


    }
    // generating for white man 22 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 18, legal: (legal white (move man d 6 e 7))
        ddi->diagonals.emplace_back(18, 311);

        // to position 13, legal: (legal white (move man d 6 f 8))
        ddi->diagonals.emplace_back(13, 313);

        this->diagonal_data[21].push_back(ddi);


        this->reverse_legal_lookup_white[311] = ReverseLegalLookup(Role::White, Piece::Man, 22, 18, NE);
        this->reverse_legal_lookup_white[313] = ReverseLegalLookup(Role::White, Piece::Man, 22, 13, NE);
    }
    // generating for white man 22 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal white (move man d 6 c 7))
        ddi->diagonals.emplace_back(17, 312);

        // to position 11, legal: (legal white (move man d 6 b 8))
        ddi->diagonals.emplace_back(11, 314);

        this->diagonal_data[21].push_back(ddi);


        this->reverse_legal_lookup_white[312] = ReverseLegalLookup(Role::White, Piece::Man, 22, 17, NW);
        this->reverse_legal_lookup_white[314] = ReverseLegalLookup(Role::White, Piece::Man, 22, 11, NW);
    }
    // generating for white man 22 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 28, legal: invalid
        ddi->diagonals.emplace_back(28, -1);

        // to position 33, legal: (legal white (move man d 6 f 4))
        ddi->diagonals.emplace_back(33, 315);

        this->diagonal_data[21].push_back(ddi);


        this->reverse_legal_lookup_white[315] = ReverseLegalLookup(Role::White, Piece::Man, 22, 33, SE);
    }
    // generating for white man 22 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 27, legal: invalid
        ddi->diagonals.emplace_back(27, -1);

        // to position 31, legal: (legal white (move man d 6 b 4))
        ddi->diagonals.emplace_back(31, 316);

        this->diagonal_data[21].push_back(ddi);


        this->reverse_legal_lookup_white[316] = ReverseLegalLookup(Role::White, Piece::Man, 22, 31, SW);
    }
    // generating for white man 23 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 19, legal: (legal white (move man f 6 g 7))
        ddi->diagonals.emplace_back(19, 332);

        // to position 14, legal: (legal white (move man f 6 h 8))
        ddi->diagonals.emplace_back(14, 334);

        this->diagonal_data[22].push_back(ddi);


        this->reverse_legal_lookup_white[332] = ReverseLegalLookup(Role::White, Piece::Man, 23, 19, NE);
        this->reverse_legal_lookup_white[334] = ReverseLegalLookup(Role::White, Piece::Man, 23, 14, NE);
    }
    // generating for white man 23 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 18, legal: (legal white (move man f 6 e 7))
        ddi->diagonals.emplace_back(18, 333);

        // to position 12, legal: (legal white (move man f 6 d 8))
        ddi->diagonals.emplace_back(12, 335);

        this->diagonal_data[22].push_back(ddi);


        this->reverse_legal_lookup_white[333] = ReverseLegalLookup(Role::White, Piece::Man, 23, 18, NW);
        this->reverse_legal_lookup_white[335] = ReverseLegalLookup(Role::White, Piece::Man, 23, 12, NW);
    }
    // generating for white man 23 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 29, legal: invalid
        ddi->diagonals.emplace_back(29, -1);

        // to position 34, legal: (legal white (move man f 6 h 4))
        ddi->diagonals.emplace_back(34, 336);

        this->diagonal_data[22].push_back(ddi);


        this->reverse_legal_lookup_white[336] = ReverseLegalLookup(Role::White, Piece::Man, 23, 34, SE);
    }
    // generating for white man 23 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 28, legal: invalid
        ddi->diagonals.emplace_back(28, -1);

        // to position 32, legal: (legal white (move man f 6 d 4))
        ddi->diagonals.emplace_back(32, 337);

        this->diagonal_data[22].push_back(ddi);


        this->reverse_legal_lookup_white[337] = ReverseLegalLookup(Role::White, Piece::Man, 23, 32, SW);
    }
    // generating for white man 24 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 20, legal: (legal white (move man h 6 i 7))
        ddi->diagonals.emplace_back(20, 355);

        // to position 15, legal: (legal white (move man h 6 j 8))
        ddi->diagonals.emplace_back(15, 357);

        this->diagonal_data[23].push_back(ddi);


        this->reverse_legal_lookup_white[355] = ReverseLegalLookup(Role::White, Piece::Man, 24, 20, NE);
        this->reverse_legal_lookup_white[357] = ReverseLegalLookup(Role::White, Piece::Man, 24, 15, NE);
    }
    // generating for white man 24 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 19, legal: (legal white (move man h 6 g 7))
        ddi->diagonals.emplace_back(19, 356);

        // to position 13, legal: (legal white (move man h 6 f 8))
        ddi->diagonals.emplace_back(13, 358);

        this->diagonal_data[23].push_back(ddi);


        this->reverse_legal_lookup_white[356] = ReverseLegalLookup(Role::White, Piece::Man, 24, 19, NW);
        this->reverse_legal_lookup_white[358] = ReverseLegalLookup(Role::White, Piece::Man, 24, 13, NW);
    }
    // generating for white man 24 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 30, legal: invalid
        ddi->diagonals.emplace_back(30, -1);

        // to position 35, legal: (legal white (move man h 6 j 4))
        ddi->diagonals.emplace_back(35, 359);

        this->diagonal_data[23].push_back(ddi);


        this->reverse_legal_lookup_white[359] = ReverseLegalLookup(Role::White, Piece::Man, 24, 35, SE);
    }
    // generating for white man 24 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 29, legal: invalid
        ddi->diagonals.emplace_back(29, -1);

        // to position 33, legal: (legal white (move man h 6 f 4))
        ddi->diagonals.emplace_back(33, 360);

        this->diagonal_data[23].push_back(ddi);


        this->reverse_legal_lookup_white[360] = ReverseLegalLookup(Role::White, Piece::Man, 24, 33, SW);
    }
    // generating for white man 25 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 20, legal: (legal white (move man j 6 i 7))
        ddi->diagonals.emplace_back(20, 374);

        // to position 14, legal: (legal white (move man j 6 h 8))
        ddi->diagonals.emplace_back(14, 375);

        this->diagonal_data[24].push_back(ddi);


        this->reverse_legal_lookup_white[374] = ReverseLegalLookup(Role::White, Piece::Man, 25, 20, NW);
        this->reverse_legal_lookup_white[375] = ReverseLegalLookup(Role::White, Piece::Man, 25, 14, NW);
    }
    // generating for white man 25 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 30, legal: invalid
        ddi->diagonals.emplace_back(30, -1);

        // to position 34, legal: (legal white (move man j 6 h 4))
        ddi->diagonals.emplace_back(34, 376);

        this->diagonal_data[24].push_back(ddi);


        this->reverse_legal_lookup_white[376] = ReverseLegalLookup(Role::White, Piece::Man, 25, 34, SW);
    }
    // generating for white man 26 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 21, legal: (legal white (move man a 5 b 6))
        ddi->diagonals.emplace_back(21, 386);

        // to position 17, legal: (legal white (move man a 5 c 7))
        ddi->diagonals.emplace_back(17, 387);

        this->diagonal_data[25].push_back(ddi);


        this->reverse_legal_lookup_white[386] = ReverseLegalLookup(Role::White, Piece::Man, 26, 21, NE);
        this->reverse_legal_lookup_white[387] = ReverseLegalLookup(Role::White, Piece::Man, 26, 17, NE);
    }
    // generating for white man 26 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 31, legal: invalid
        ddi->diagonals.emplace_back(31, -1);

        // to position 37, legal: (legal white (move man a 5 c 3))
        ddi->diagonals.emplace_back(37, 388);

        this->diagonal_data[25].push_back(ddi);


        this->reverse_legal_lookup_white[388] = ReverseLegalLookup(Role::White, Piece::Man, 26, 37, SE);
    }
    // generating for white man 27 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 22, legal: (legal white (move man c 5 d 6))
        ddi->diagonals.emplace_back(22, 398);

        // to position 18, legal: (legal white (move man c 5 e 7))
        ddi->diagonals.emplace_back(18, 400);

        this->diagonal_data[26].push_back(ddi);


        this->reverse_legal_lookup_white[398] = ReverseLegalLookup(Role::White, Piece::Man, 27, 22, NE);
        this->reverse_legal_lookup_white[400] = ReverseLegalLookup(Role::White, Piece::Man, 27, 18, NE);
    }
    // generating for white man 27 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 21, legal: (legal white (move man c 5 b 6))
        ddi->diagonals.emplace_back(21, 399);

        // to position 16, legal: (legal white (move man c 5 a 7))
        ddi->diagonals.emplace_back(16, 401);

        this->diagonal_data[26].push_back(ddi);


        this->reverse_legal_lookup_white[399] = ReverseLegalLookup(Role::White, Piece::Man, 27, 21, NW);
        this->reverse_legal_lookup_white[401] = ReverseLegalLookup(Role::White, Piece::Man, 27, 16, NW);
    }
    // generating for white man 27 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 32, legal: invalid
        ddi->diagonals.emplace_back(32, -1);

        // to position 38, legal: (legal white (move man c 5 e 3))
        ddi->diagonals.emplace_back(38, 402);

        this->diagonal_data[26].push_back(ddi);


        this->reverse_legal_lookup_white[402] = ReverseLegalLookup(Role::White, Piece::Man, 27, 38, SE);
    }
    // generating for white man 27 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 31, legal: invalid
        ddi->diagonals.emplace_back(31, -1);

        // to position 36, legal: (legal white (move man c 5 a 3))
        ddi->diagonals.emplace_back(36, 403);

        this->diagonal_data[26].push_back(ddi);


        this->reverse_legal_lookup_white[403] = ReverseLegalLookup(Role::White, Piece::Man, 27, 36, SW);
    }
    // generating for white man 28 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 23, legal: (legal white (move man e 5 f 6))
        ddi->diagonals.emplace_back(23, 417);

        // to position 19, legal: (legal white (move man e 5 g 7))
        ddi->diagonals.emplace_back(19, 419);

        this->diagonal_data[27].push_back(ddi);


        this->reverse_legal_lookup_white[417] = ReverseLegalLookup(Role::White, Piece::Man, 28, 23, NE);
        this->reverse_legal_lookup_white[419] = ReverseLegalLookup(Role::White, Piece::Man, 28, 19, NE);
    }
    // generating for white man 28 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 22, legal: (legal white (move man e 5 d 6))
        ddi->diagonals.emplace_back(22, 418);

        // to position 17, legal: (legal white (move man e 5 c 7))
        ddi->diagonals.emplace_back(17, 420);

        this->diagonal_data[27].push_back(ddi);


        this->reverse_legal_lookup_white[418] = ReverseLegalLookup(Role::White, Piece::Man, 28, 22, NW);
        this->reverse_legal_lookup_white[420] = ReverseLegalLookup(Role::White, Piece::Man, 28, 17, NW);
    }
    // generating for white man 28 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 33, legal: invalid
        ddi->diagonals.emplace_back(33, -1);

        // to position 39, legal: (legal white (move man e 5 g 3))
        ddi->diagonals.emplace_back(39, 421);

        this->diagonal_data[27].push_back(ddi);


        this->reverse_legal_lookup_white[421] = ReverseLegalLookup(Role::White, Piece::Man, 28, 39, SE);
    }
    // generating for white man 28 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 32, legal: invalid
        ddi->diagonals.emplace_back(32, -1);

        // to position 37, legal: (legal white (move man e 5 c 3))
        ddi->diagonals.emplace_back(37, 422);

        this->diagonal_data[27].push_back(ddi);


        this->reverse_legal_lookup_white[422] = ReverseLegalLookup(Role::White, Piece::Man, 28, 37, SW);
    }
    // generating for white man 29 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal white (move man g 5 h 6))
        ddi->diagonals.emplace_back(24, 440);

        // to position 20, legal: (legal white (move man g 5 i 7))
        ddi->diagonals.emplace_back(20, 442);

        this->diagonal_data[28].push_back(ddi);


        this->reverse_legal_lookup_white[440] = ReverseLegalLookup(Role::White, Piece::Man, 29, 24, NE);
        this->reverse_legal_lookup_white[442] = ReverseLegalLookup(Role::White, Piece::Man, 29, 20, NE);
    }
    // generating for white man 29 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 23, legal: (legal white (move man g 5 f 6))
        ddi->diagonals.emplace_back(23, 441);

        // to position 18, legal: (legal white (move man g 5 e 7))
        ddi->diagonals.emplace_back(18, 443);

        this->diagonal_data[28].push_back(ddi);


        this->reverse_legal_lookup_white[441] = ReverseLegalLookup(Role::White, Piece::Man, 29, 23, NW);
        this->reverse_legal_lookup_white[443] = ReverseLegalLookup(Role::White, Piece::Man, 29, 18, NW);
    }
    // generating for white man 29 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 34, legal: invalid
        ddi->diagonals.emplace_back(34, -1);

        // to position 40, legal: (legal white (move man g 5 i 3))
        ddi->diagonals.emplace_back(40, 444);

        this->diagonal_data[28].push_back(ddi);


        this->reverse_legal_lookup_white[444] = ReverseLegalLookup(Role::White, Piece::Man, 29, 40, SE);
    }
    // generating for white man 29 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 33, legal: invalid
        ddi->diagonals.emplace_back(33, -1);

        // to position 38, legal: (legal white (move man g 5 e 3))
        ddi->diagonals.emplace_back(38, 445);

        this->diagonal_data[28].push_back(ddi);


        this->reverse_legal_lookup_white[445] = ReverseLegalLookup(Role::White, Piece::Man, 29, 38, SW);
    }
    // generating for white man 30 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 25, legal: (legal white (move man i 5 j 6))
        ddi->diagonals.emplace_back(25, 461);

        this->diagonal_data[29].push_back(ddi);


        this->reverse_legal_lookup_white[461] = ReverseLegalLookup(Role::White, Piece::Man, 30, 25, NE);
    }
    // generating for white man 30 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal white (move man i 5 h 6))
        ddi->diagonals.emplace_back(24, 462);

        // to position 19, legal: (legal white (move man i 5 g 7))
        ddi->diagonals.emplace_back(19, 463);

        this->diagonal_data[29].push_back(ddi);


        this->reverse_legal_lookup_white[462] = ReverseLegalLookup(Role::White, Piece::Man, 30, 24, NW);
        this->reverse_legal_lookup_white[463] = ReverseLegalLookup(Role::White, Piece::Man, 30, 19, NW);
    }
    // generating for white man 30 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 35, legal: invalid
        ddi->diagonals.emplace_back(35, -1);

        this->diagonal_data[29].push_back(ddi);


    }
    // generating for white man 30 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 34, legal: invalid
        ddi->diagonals.emplace_back(34, -1);

        // to position 39, legal: (legal white (move man i 5 g 3))
        ddi->diagonals.emplace_back(39, 464);

        this->diagonal_data[29].push_back(ddi);


        this->reverse_legal_lookup_white[464] = ReverseLegalLookup(Role::White, Piece::Man, 30, 39, SW);
    }
    // generating for white man 31 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal white (move man b 4 c 5))
        ddi->diagonals.emplace_back(27, 476);

        // to position 22, legal: (legal white (move man b 4 d 6))
        ddi->diagonals.emplace_back(22, 478);

        this->diagonal_data[30].push_back(ddi);


        this->reverse_legal_lookup_white[476] = ReverseLegalLookup(Role::White, Piece::Man, 31, 27, NE);
        this->reverse_legal_lookup_white[478] = ReverseLegalLookup(Role::White, Piece::Man, 31, 22, NE);
    }
    // generating for white man 31 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 26, legal: (legal white (move man b 4 a 5))
        ddi->diagonals.emplace_back(26, 477);

        this->diagonal_data[30].push_back(ddi);


        this->reverse_legal_lookup_white[477] = ReverseLegalLookup(Role::White, Piece::Man, 31, 26, NW);
    }
    // generating for white man 31 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 37, legal: invalid
        ddi->diagonals.emplace_back(37, -1);

        // to position 42, legal: (legal white (move man b 4 d 2))
        ddi->diagonals.emplace_back(42, 479);

        this->diagonal_data[30].push_back(ddi);


        this->reverse_legal_lookup_white[479] = ReverseLegalLookup(Role::White, Piece::Man, 31, 42, SE);
    }
    // generating for white man 31 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 36, legal: invalid
        ddi->diagonals.emplace_back(36, -1);

        this->diagonal_data[30].push_back(ddi);


    }
    // generating for white man 32 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 28, legal: (legal white (move man d 4 e 5))
        ddi->diagonals.emplace_back(28, 491);

        // to position 23, legal: (legal white (move man d 4 f 6))
        ddi->diagonals.emplace_back(23, 493);

        this->diagonal_data[31].push_back(ddi);


        this->reverse_legal_lookup_white[491] = ReverseLegalLookup(Role::White, Piece::Man, 32, 28, NE);
        this->reverse_legal_lookup_white[493] = ReverseLegalLookup(Role::White, Piece::Man, 32, 23, NE);
    }
    // generating for white man 32 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal white (move man d 4 c 5))
        ddi->diagonals.emplace_back(27, 492);

        // to position 21, legal: (legal white (move man d 4 b 6))
        ddi->diagonals.emplace_back(21, 494);

        this->diagonal_data[31].push_back(ddi);


        this->reverse_legal_lookup_white[492] = ReverseLegalLookup(Role::White, Piece::Man, 32, 27, NW);
        this->reverse_legal_lookup_white[494] = ReverseLegalLookup(Role::White, Piece::Man, 32, 21, NW);
    }
    // generating for white man 32 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 38, legal: invalid
        ddi->diagonals.emplace_back(38, -1);

        // to position 43, legal: (legal white (move man d 4 f 2))
        ddi->diagonals.emplace_back(43, 495);

        this->diagonal_data[31].push_back(ddi);


        this->reverse_legal_lookup_white[495] = ReverseLegalLookup(Role::White, Piece::Man, 32, 43, SE);
    }
    // generating for white man 32 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 37, legal: invalid
        ddi->diagonals.emplace_back(37, -1);

        // to position 41, legal: (legal white (move man d 4 b 2))
        ddi->diagonals.emplace_back(41, 496);

        this->diagonal_data[31].push_back(ddi);


        this->reverse_legal_lookup_white[496] = ReverseLegalLookup(Role::White, Piece::Man, 32, 41, SW);
    }
    // generating for white man 33 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 29, legal: (legal white (move man f 4 g 5))
        ddi->diagonals.emplace_back(29, 512);

        // to position 24, legal: (legal white (move man f 4 h 6))
        ddi->diagonals.emplace_back(24, 514);

        this->diagonal_data[32].push_back(ddi);


        this->reverse_legal_lookup_white[512] = ReverseLegalLookup(Role::White, Piece::Man, 33, 29, NE);
        this->reverse_legal_lookup_white[514] = ReverseLegalLookup(Role::White, Piece::Man, 33, 24, NE);
    }
    // generating for white man 33 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 28, legal: (legal white (move man f 4 e 5))
        ddi->diagonals.emplace_back(28, 513);

        // to position 22, legal: (legal white (move man f 4 d 6))
        ddi->diagonals.emplace_back(22, 515);

        this->diagonal_data[32].push_back(ddi);


        this->reverse_legal_lookup_white[513] = ReverseLegalLookup(Role::White, Piece::Man, 33, 28, NW);
        this->reverse_legal_lookup_white[515] = ReverseLegalLookup(Role::White, Piece::Man, 33, 22, NW);
    }
    // generating for white man 33 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 39, legal: invalid
        ddi->diagonals.emplace_back(39, -1);

        // to position 44, legal: (legal white (move man f 4 h 2))
        ddi->diagonals.emplace_back(44, 516);

        this->diagonal_data[32].push_back(ddi);


        this->reverse_legal_lookup_white[516] = ReverseLegalLookup(Role::White, Piece::Man, 33, 44, SE);
    }
    // generating for white man 33 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 38, legal: invalid
        ddi->diagonals.emplace_back(38, -1);

        // to position 42, legal: (legal white (move man f 4 d 2))
        ddi->diagonals.emplace_back(42, 517);

        this->diagonal_data[32].push_back(ddi);


        this->reverse_legal_lookup_white[517] = ReverseLegalLookup(Role::White, Piece::Man, 33, 42, SW);
    }
    // generating for white man 34 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 30, legal: (legal white (move man h 4 i 5))
        ddi->diagonals.emplace_back(30, 533);

        // to position 25, legal: (legal white (move man h 4 j 6))
        ddi->diagonals.emplace_back(25, 535);

        this->diagonal_data[33].push_back(ddi);


        this->reverse_legal_lookup_white[533] = ReverseLegalLookup(Role::White, Piece::Man, 34, 30, NE);
        this->reverse_legal_lookup_white[535] = ReverseLegalLookup(Role::White, Piece::Man, 34, 25, NE);
    }
    // generating for white man 34 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 29, legal: (legal white (move man h 4 g 5))
        ddi->diagonals.emplace_back(29, 534);

        // to position 23, legal: (legal white (move man h 4 f 6))
        ddi->diagonals.emplace_back(23, 536);

        this->diagonal_data[33].push_back(ddi);


        this->reverse_legal_lookup_white[534] = ReverseLegalLookup(Role::White, Piece::Man, 34, 29, NW);
        this->reverse_legal_lookup_white[536] = ReverseLegalLookup(Role::White, Piece::Man, 34, 23, NW);
    }
    // generating for white man 34 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 40, legal: invalid
        ddi->diagonals.emplace_back(40, -1);

        // to position 45, legal: (legal white (move man h 4 j 2))
        ddi->diagonals.emplace_back(45, 537);

        this->diagonal_data[33].push_back(ddi);


        this->reverse_legal_lookup_white[537] = ReverseLegalLookup(Role::White, Piece::Man, 34, 45, SE);
    }
    // generating for white man 34 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 39, legal: invalid
        ddi->diagonals.emplace_back(39, -1);

        // to position 43, legal: (legal white (move man h 4 f 2))
        ddi->diagonals.emplace_back(43, 538);

        this->diagonal_data[33].push_back(ddi);


        this->reverse_legal_lookup_white[538] = ReverseLegalLookup(Role::White, Piece::Man, 34, 43, SW);
    }
    // generating for white man 35 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 30, legal: (legal white (move man j 4 i 5))
        ddi->diagonals.emplace_back(30, 552);

        // to position 24, legal: (legal white (move man j 4 h 6))
        ddi->diagonals.emplace_back(24, 553);

        this->diagonal_data[34].push_back(ddi);


        this->reverse_legal_lookup_white[552] = ReverseLegalLookup(Role::White, Piece::Man, 35, 30, NW);
        this->reverse_legal_lookup_white[553] = ReverseLegalLookup(Role::White, Piece::Man, 35, 24, NW);
    }
    // generating for white man 35 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 40, legal: invalid
        ddi->diagonals.emplace_back(40, -1);

        // to position 44, legal: (legal white (move man j 4 h 2))
        ddi->diagonals.emplace_back(44, 554);

        this->diagonal_data[34].push_back(ddi);


        this->reverse_legal_lookup_white[554] = ReverseLegalLookup(Role::White, Piece::Man, 35, 44, SW);
    }
    // generating for white man 36 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 31, legal: (legal white (move man a 3 b 4))
        ddi->diagonals.emplace_back(31, 564);

        // to position 27, legal: (legal white (move man a 3 c 5))
        ddi->diagonals.emplace_back(27, 565);

        this->diagonal_data[35].push_back(ddi);


        this->reverse_legal_lookup_white[564] = ReverseLegalLookup(Role::White, Piece::Man, 36, 31, NE);
        this->reverse_legal_lookup_white[565] = ReverseLegalLookup(Role::White, Piece::Man, 36, 27, NE);
    }
    // generating for white man 36 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 41, legal: invalid
        ddi->diagonals.emplace_back(41, -1);

        // to position 47, legal: (legal white (move man a 3 c 1))
        ddi->diagonals.emplace_back(47, 566);

        this->diagonal_data[35].push_back(ddi);


        this->reverse_legal_lookup_white[566] = ReverseLegalLookup(Role::White, Piece::Man, 36, 47, SE);
    }
    // generating for white man 37 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 32, legal: (legal white (move man c 3 d 4))
        ddi->diagonals.emplace_back(32, 576);

        // to position 28, legal: (legal white (move man c 3 e 5))
        ddi->diagonals.emplace_back(28, 578);

        this->diagonal_data[36].push_back(ddi);


        this->reverse_legal_lookup_white[576] = ReverseLegalLookup(Role::White, Piece::Man, 37, 32, NE);
        this->reverse_legal_lookup_white[578] = ReverseLegalLookup(Role::White, Piece::Man, 37, 28, NE);
    }
    // generating for white man 37 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 31, legal: (legal white (move man c 3 b 4))
        ddi->diagonals.emplace_back(31, 577);

        // to position 26, legal: (legal white (move man c 3 a 5))
        ddi->diagonals.emplace_back(26, 579);

        this->diagonal_data[36].push_back(ddi);


        this->reverse_legal_lookup_white[577] = ReverseLegalLookup(Role::White, Piece::Man, 37, 31, NW);
        this->reverse_legal_lookup_white[579] = ReverseLegalLookup(Role::White, Piece::Man, 37, 26, NW);
    }
    // generating for white man 37 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 42, legal: invalid
        ddi->diagonals.emplace_back(42, -1);

        // to position 48, legal: (legal white (move man c 3 e 1))
        ddi->diagonals.emplace_back(48, 580);

        this->diagonal_data[36].push_back(ddi);


        this->reverse_legal_lookup_white[580] = ReverseLegalLookup(Role::White, Piece::Man, 37, 48, SE);
    }
    // generating for white man 37 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 41, legal: invalid
        ddi->diagonals.emplace_back(41, -1);

        // to position 46, legal: (legal white (move man c 3 a 1))
        ddi->diagonals.emplace_back(46, 581);

        this->diagonal_data[36].push_back(ddi);


        this->reverse_legal_lookup_white[581] = ReverseLegalLookup(Role::White, Piece::Man, 37, 46, SW);
    }
    // generating for white man 38 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 33, legal: (legal white (move man e 3 f 4))
        ddi->diagonals.emplace_back(33, 595);

        // to position 29, legal: (legal white (move man e 3 g 5))
        ddi->diagonals.emplace_back(29, 597);

        this->diagonal_data[37].push_back(ddi);


        this->reverse_legal_lookup_white[595] = ReverseLegalLookup(Role::White, Piece::Man, 38, 33, NE);
        this->reverse_legal_lookup_white[597] = ReverseLegalLookup(Role::White, Piece::Man, 38, 29, NE);
    }
    // generating for white man 38 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 32, legal: (legal white (move man e 3 d 4))
        ddi->diagonals.emplace_back(32, 596);

        // to position 27, legal: (legal white (move man e 3 c 5))
        ddi->diagonals.emplace_back(27, 598);

        this->diagonal_data[37].push_back(ddi);


        this->reverse_legal_lookup_white[596] = ReverseLegalLookup(Role::White, Piece::Man, 38, 32, NW);
        this->reverse_legal_lookup_white[598] = ReverseLegalLookup(Role::White, Piece::Man, 38, 27, NW);
    }
    // generating for white man 38 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 43, legal: invalid
        ddi->diagonals.emplace_back(43, -1);

        // to position 49, legal: (legal white (move man e 3 g 1))
        ddi->diagonals.emplace_back(49, 599);

        this->diagonal_data[37].push_back(ddi);


        this->reverse_legal_lookup_white[599] = ReverseLegalLookup(Role::White, Piece::Man, 38, 49, SE);
    }
    // generating for white man 38 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 42, legal: invalid
        ddi->diagonals.emplace_back(42, -1);

        // to position 47, legal: (legal white (move man e 3 c 1))
        ddi->diagonals.emplace_back(47, 600);

        this->diagonal_data[37].push_back(ddi);


        this->reverse_legal_lookup_white[600] = ReverseLegalLookup(Role::White, Piece::Man, 38, 47, SW);
    }
    // generating for white man 39 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 34, legal: (legal white (move man g 3 h 4))
        ddi->diagonals.emplace_back(34, 614);

        // to position 30, legal: (legal white (move man g 3 i 5))
        ddi->diagonals.emplace_back(30, 616);

        this->diagonal_data[38].push_back(ddi);


        this->reverse_legal_lookup_white[614] = ReverseLegalLookup(Role::White, Piece::Man, 39, 34, NE);
        this->reverse_legal_lookup_white[616] = ReverseLegalLookup(Role::White, Piece::Man, 39, 30, NE);
    }
    // generating for white man 39 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 33, legal: (legal white (move man g 3 f 4))
        ddi->diagonals.emplace_back(33, 615);

        // to position 28, legal: (legal white (move man g 3 e 5))
        ddi->diagonals.emplace_back(28, 617);

        this->diagonal_data[38].push_back(ddi);


        this->reverse_legal_lookup_white[615] = ReverseLegalLookup(Role::White, Piece::Man, 39, 33, NW);
        this->reverse_legal_lookup_white[617] = ReverseLegalLookup(Role::White, Piece::Man, 39, 28, NW);
    }
    // generating for white man 39 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 44, legal: invalid
        ddi->diagonals.emplace_back(44, -1);

        // to position 50, legal: (legal white (move man g 3 i 1))
        ddi->diagonals.emplace_back(50, 618);

        this->diagonal_data[38].push_back(ddi);


        this->reverse_legal_lookup_white[618] = ReverseLegalLookup(Role::White, Piece::Man, 39, 50, SE);
    }
    // generating for white man 39 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 43, legal: invalid
        ddi->diagonals.emplace_back(43, -1);

        // to position 48, legal: (legal white (move man g 3 e 1))
        ddi->diagonals.emplace_back(48, 619);

        this->diagonal_data[38].push_back(ddi);


        this->reverse_legal_lookup_white[619] = ReverseLegalLookup(Role::White, Piece::Man, 39, 48, SW);
    }
    // generating for white man 40 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 35, legal: (legal white (move man i 3 j 4))
        ddi->diagonals.emplace_back(35, 633);

        this->diagonal_data[39].push_back(ddi);


        this->reverse_legal_lookup_white[633] = ReverseLegalLookup(Role::White, Piece::Man, 40, 35, NE);
    }
    // generating for white man 40 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 34, legal: (legal white (move man i 3 h 4))
        ddi->diagonals.emplace_back(34, 634);

        // to position 29, legal: (legal white (move man i 3 g 5))
        ddi->diagonals.emplace_back(29, 635);

        this->diagonal_data[39].push_back(ddi);


        this->reverse_legal_lookup_white[634] = ReverseLegalLookup(Role::White, Piece::Man, 40, 34, NW);
        this->reverse_legal_lookup_white[635] = ReverseLegalLookup(Role::White, Piece::Man, 40, 29, NW);
    }
    // generating for white man 40 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 45, legal: invalid
        ddi->diagonals.emplace_back(45, -1);

        this->diagonal_data[39].push_back(ddi);


    }
    // generating for white man 40 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 44, legal: invalid
        ddi->diagonals.emplace_back(44, -1);

        // to position 49, legal: (legal white (move man i 3 g 1))
        ddi->diagonals.emplace_back(49, 636);

        this->diagonal_data[39].push_back(ddi);


        this->reverse_legal_lookup_white[636] = ReverseLegalLookup(Role::White, Piece::Man, 40, 49, SW);
    }
    // generating for white man 41 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 37, legal: (legal white (move man b 2 c 3))
        ddi->diagonals.emplace_back(37, 648);

        // to position 32, legal: (legal white (move man b 2 d 4))
        ddi->diagonals.emplace_back(32, 650);

        this->diagonal_data[40].push_back(ddi);


        this->reverse_legal_lookup_white[648] = ReverseLegalLookup(Role::White, Piece::Man, 41, 37, NE);
        this->reverse_legal_lookup_white[650] = ReverseLegalLookup(Role::White, Piece::Man, 41, 32, NE);
    }
    // generating for white man 41 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 36, legal: (legal white (move man b 2 a 3))
        ddi->diagonals.emplace_back(36, 649);

        this->diagonal_data[40].push_back(ddi);


        this->reverse_legal_lookup_white[649] = ReverseLegalLookup(Role::White, Piece::Man, 41, 36, NW);
    }
    // generating for white man 41 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 47, legal: invalid
        ddi->diagonals.emplace_back(47, -1);

        this->diagonal_data[40].push_back(ddi);


    }
    // generating for white man 41 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 46, legal: invalid
        ddi->diagonals.emplace_back(46, -1);

        this->diagonal_data[40].push_back(ddi);


    }
    // generating for white man 42 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 38, legal: (legal white (move man d 2 e 3))
        ddi->diagonals.emplace_back(38, 662);

        // to position 33, legal: (legal white (move man d 2 f 4))
        ddi->diagonals.emplace_back(33, 664);

        this->diagonal_data[41].push_back(ddi);


        this->reverse_legal_lookup_white[662] = ReverseLegalLookup(Role::White, Piece::Man, 42, 38, NE);
        this->reverse_legal_lookup_white[664] = ReverseLegalLookup(Role::White, Piece::Man, 42, 33, NE);
    }
    // generating for white man 42 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 37, legal: (legal white (move man d 2 c 3))
        ddi->diagonals.emplace_back(37, 663);

        // to position 31, legal: (legal white (move man d 2 b 4))
        ddi->diagonals.emplace_back(31, 665);

        this->diagonal_data[41].push_back(ddi);


        this->reverse_legal_lookup_white[663] = ReverseLegalLookup(Role::White, Piece::Man, 42, 37, NW);
        this->reverse_legal_lookup_white[665] = ReverseLegalLookup(Role::White, Piece::Man, 42, 31, NW);
    }
    // generating for white man 42 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 48, legal: invalid
        ddi->diagonals.emplace_back(48, -1);

        this->diagonal_data[41].push_back(ddi);


    }
    // generating for white man 42 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 47, legal: invalid
        ddi->diagonals.emplace_back(47, -1);

        this->diagonal_data[41].push_back(ddi);


    }
    // generating for white man 43 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 39, legal: (legal white (move man f 2 g 3))
        ddi->diagonals.emplace_back(39, 677);

        // to position 34, legal: (legal white (move man f 2 h 4))
        ddi->diagonals.emplace_back(34, 679);

        this->diagonal_data[42].push_back(ddi);


        this->reverse_legal_lookup_white[677] = ReverseLegalLookup(Role::White, Piece::Man, 43, 39, NE);
        this->reverse_legal_lookup_white[679] = ReverseLegalLookup(Role::White, Piece::Man, 43, 34, NE);
    }
    // generating for white man 43 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 38, legal: (legal white (move man f 2 e 3))
        ddi->diagonals.emplace_back(38, 678);

        // to position 32, legal: (legal white (move man f 2 d 4))
        ddi->diagonals.emplace_back(32, 680);

        this->diagonal_data[42].push_back(ddi);


        this->reverse_legal_lookup_white[678] = ReverseLegalLookup(Role::White, Piece::Man, 43, 38, NW);
        this->reverse_legal_lookup_white[680] = ReverseLegalLookup(Role::White, Piece::Man, 43, 32, NW);
    }
    // generating for white man 43 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 49, legal: invalid
        ddi->diagonals.emplace_back(49, -1);

        this->diagonal_data[42].push_back(ddi);


    }
    // generating for white man 43 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 48, legal: invalid
        ddi->diagonals.emplace_back(48, -1);

        this->diagonal_data[42].push_back(ddi);


    }
    // generating for white man 44 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 40, legal: (legal white (move man h 2 i 3))
        ddi->diagonals.emplace_back(40, 692);

        // to position 35, legal: (legal white (move man h 2 j 4))
        ddi->diagonals.emplace_back(35, 694);

        this->diagonal_data[43].push_back(ddi);


        this->reverse_legal_lookup_white[692] = ReverseLegalLookup(Role::White, Piece::Man, 44, 40, NE);
        this->reverse_legal_lookup_white[694] = ReverseLegalLookup(Role::White, Piece::Man, 44, 35, NE);
    }
    // generating for white man 44 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 39, legal: (legal white (move man h 2 g 3))
        ddi->diagonals.emplace_back(39, 693);

        // to position 33, legal: (legal white (move man h 2 f 4))
        ddi->diagonals.emplace_back(33, 695);

        this->diagonal_data[43].push_back(ddi);


        this->reverse_legal_lookup_white[693] = ReverseLegalLookup(Role::White, Piece::Man, 44, 39, NW);
        this->reverse_legal_lookup_white[695] = ReverseLegalLookup(Role::White, Piece::Man, 44, 33, NW);
    }
    // generating for white man 44 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 50, legal: invalid
        ddi->diagonals.emplace_back(50, -1);

        this->diagonal_data[43].push_back(ddi);


    }
    // generating for white man 44 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 49, legal: invalid
        ddi->diagonals.emplace_back(49, -1);

        this->diagonal_data[43].push_back(ddi);


    }
    // generating for white man 45 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 40, legal: (legal white (move man j 2 i 3))
        ddi->diagonals.emplace_back(40, 707);

        // to position 34, legal: (legal white (move man j 2 h 4))
        ddi->diagonals.emplace_back(34, 708);

        this->diagonal_data[44].push_back(ddi);


        this->reverse_legal_lookup_white[707] = ReverseLegalLookup(Role::White, Piece::Man, 45, 40, NW);
        this->reverse_legal_lookup_white[708] = ReverseLegalLookup(Role::White, Piece::Man, 45, 34, NW);
    }
    // generating for white man 45 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 50, legal: invalid
        ddi->diagonals.emplace_back(50, -1);

        this->diagonal_data[44].push_back(ddi);


    }
    // generating for white man 46 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 41, legal: (legal white (move man a 1 b 2))
        ddi->diagonals.emplace_back(41, 718);

        // to position 37, legal: (legal white (move man a 1 c 3))
        ddi->diagonals.emplace_back(37, 719);

        this->diagonal_data[45].push_back(ddi);


        this->reverse_legal_lookup_white[718] = ReverseLegalLookup(Role::White, Piece::Man, 46, 41, NE);
        this->reverse_legal_lookup_white[719] = ReverseLegalLookup(Role::White, Piece::Man, 46, 37, NE);
    }
    // generating for white man 47 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 42, legal: (legal white (move man c 1 d 2))
        ddi->diagonals.emplace_back(42, 729);

        // to position 38, legal: (legal white (move man c 1 e 3))
        ddi->diagonals.emplace_back(38, 731);

        this->diagonal_data[46].push_back(ddi);


        this->reverse_legal_lookup_white[729] = ReverseLegalLookup(Role::White, Piece::Man, 47, 42, NE);
        this->reverse_legal_lookup_white[731] = ReverseLegalLookup(Role::White, Piece::Man, 47, 38, NE);
    }
    // generating for white man 47 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 41, legal: (legal white (move man c 1 b 2))
        ddi->diagonals.emplace_back(41, 730);

        // to position 36, legal: (legal white (move man c 1 a 3))
        ddi->diagonals.emplace_back(36, 732);

        this->diagonal_data[46].push_back(ddi);


        this->reverse_legal_lookup_white[730] = ReverseLegalLookup(Role::White, Piece::Man, 47, 41, NW);
        this->reverse_legal_lookup_white[732] = ReverseLegalLookup(Role::White, Piece::Man, 47, 36, NW);
    }
    // generating for white man 48 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 43, legal: (legal white (move man e 1 f 2))
        ddi->diagonals.emplace_back(43, 742);

        // to position 39, legal: (legal white (move man e 1 g 3))
        ddi->diagonals.emplace_back(39, 744);

        this->diagonal_data[47].push_back(ddi);


        this->reverse_legal_lookup_white[742] = ReverseLegalLookup(Role::White, Piece::Man, 48, 43, NE);
        this->reverse_legal_lookup_white[744] = ReverseLegalLookup(Role::White, Piece::Man, 48, 39, NE);
    }
    // generating for white man 48 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 42, legal: (legal white (move man e 1 d 2))
        ddi->diagonals.emplace_back(42, 743);

        // to position 37, legal: (legal white (move man e 1 c 3))
        ddi->diagonals.emplace_back(37, 745);

        this->diagonal_data[47].push_back(ddi);


        this->reverse_legal_lookup_white[743] = ReverseLegalLookup(Role::White, Piece::Man, 48, 42, NW);
        this->reverse_legal_lookup_white[745] = ReverseLegalLookup(Role::White, Piece::Man, 48, 37, NW);
    }
    // generating for white man 49 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 44, legal: (legal white (move man g 1 h 2))
        ddi->diagonals.emplace_back(44, 755);

        // to position 40, legal: (legal white (move man g 1 i 3))
        ddi->diagonals.emplace_back(40, 757);

        this->diagonal_data[48].push_back(ddi);


        this->reverse_legal_lookup_white[755] = ReverseLegalLookup(Role::White, Piece::Man, 49, 44, NE);
        this->reverse_legal_lookup_white[757] = ReverseLegalLookup(Role::White, Piece::Man, 49, 40, NE);
    }
    // generating for white man 49 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 43, legal: (legal white (move man g 1 f 2))
        ddi->diagonals.emplace_back(43, 756);

        // to position 38, legal: (legal white (move man g 1 e 3))
        ddi->diagonals.emplace_back(38, 758);

        this->diagonal_data[48].push_back(ddi);


        this->reverse_legal_lookup_white[756] = ReverseLegalLookup(Role::White, Piece::Man, 49, 43, NW);
        this->reverse_legal_lookup_white[758] = ReverseLegalLookup(Role::White, Piece::Man, 49, 38, NW);
    }
    // generating for white man 50 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 45, legal: (legal white (move man i 1 j 2))
        ddi->diagonals.emplace_back(45, 768);

        this->diagonal_data[49].push_back(ddi);


        this->reverse_legal_lookup_white[768] = ReverseLegalLookup(Role::White, Piece::Man, 50, 45, NE);
    }
    // generating for white man 50 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 44, legal: (legal white (move man i 1 h 2))
        ddi->diagonals.emplace_back(44, 769);

        // to position 39, legal: (legal white (move man i 1 g 3))
        ddi->diagonals.emplace_back(39, 770);

        this->diagonal_data[49].push_back(ddi);


        this->reverse_legal_lookup_white[769] = ReverseLegalLookup(Role::White, Piece::Man, 50, 44, NW);
        this->reverse_legal_lookup_white[770] = ReverseLegalLookup(Role::White, Piece::Man, 50, 39, NW);
    }
    // generating for white king 1 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(8);
        // to position 7, legal: (legal white (move king b 10 c 9))
        ddi->diagonals.emplace_back(7, 2);

        // to position 12, legal: (legal white (move king b 10 d 8))
        ddi->diagonals.emplace_back(12, 3);

        // to position 18, legal: (legal white (move king b 10 e 7))
        ddi->diagonals.emplace_back(18, 4);

        // to position 23, legal: (legal white (move king b 10 f 6))
        ddi->diagonals.emplace_back(23, 5);

        // to position 29, legal: (legal white (move king b 10 g 5))
        ddi->diagonals.emplace_back(29, 6);

        // to position 34, legal: (legal white (move king b 10 h 4))
        ddi->diagonals.emplace_back(34, 7);

        // to position 40, legal: (legal white (move king b 10 i 3))
        ddi->diagonals.emplace_back(40, 8);

        // to position 45, legal: (legal white (move king b 10 j 2))
        ddi->diagonals.emplace_back(45, 9);

        this->diagonal_data[50].push_back(ddi);


        this->reverse_legal_lookup_white[2] = ReverseLegalLookup(Role::White, Piece::King, 1, 7, SE);
        this->reverse_legal_lookup_white[3] = ReverseLegalLookup(Role::White, Piece::King, 1, 12, SE);
        this->reverse_legal_lookup_white[4] = ReverseLegalLookup(Role::White, Piece::King, 1, 18, SE);
        this->reverse_legal_lookup_white[5] = ReverseLegalLookup(Role::White, Piece::King, 1, 23, SE);
        this->reverse_legal_lookup_white[6] = ReverseLegalLookup(Role::White, Piece::King, 1, 29, SE);
        this->reverse_legal_lookup_white[7] = ReverseLegalLookup(Role::White, Piece::King, 1, 34, SE);
        this->reverse_legal_lookup_white[8] = ReverseLegalLookup(Role::White, Piece::King, 1, 40, SE);
        this->reverse_legal_lookup_white[9] = ReverseLegalLookup(Role::White, Piece::King, 1, 45, SE);
    }
    // generating for white king 1 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 6, legal: (legal white (move king b 10 a 9))
        ddi->diagonals.emplace_back(6, 10);

        this->diagonal_data[50].push_back(ddi);


        this->reverse_legal_lookup_white[10] = ReverseLegalLookup(Role::White, Piece::King, 1, 6, SW);
    }
    // generating for white king 2 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 8, legal: (legal white (move king d 10 e 9))
        ddi->diagonals.emplace_back(8, 13);

        // to position 13, legal: (legal white (move king d 10 f 8))
        ddi->diagonals.emplace_back(13, 14);

        // to position 19, legal: (legal white (move king d 10 g 7))
        ddi->diagonals.emplace_back(19, 15);

        // to position 24, legal: (legal white (move king d 10 h 6))
        ddi->diagonals.emplace_back(24, 16);

        // to position 30, legal: (legal white (move king d 10 i 5))
        ddi->diagonals.emplace_back(30, 17);

        // to position 35, legal: (legal white (move king d 10 j 4))
        ddi->diagonals.emplace_back(35, 18);

        this->diagonal_data[51].push_back(ddi);


        this->reverse_legal_lookup_white[13] = ReverseLegalLookup(Role::White, Piece::King, 2, 8, SE);
        this->reverse_legal_lookup_white[14] = ReverseLegalLookup(Role::White, Piece::King, 2, 13, SE);
        this->reverse_legal_lookup_white[15] = ReverseLegalLookup(Role::White, Piece::King, 2, 19, SE);
        this->reverse_legal_lookup_white[16] = ReverseLegalLookup(Role::White, Piece::King, 2, 24, SE);
        this->reverse_legal_lookup_white[17] = ReverseLegalLookup(Role::White, Piece::King, 2, 30, SE);
        this->reverse_legal_lookup_white[18] = ReverseLegalLookup(Role::White, Piece::King, 2, 35, SE);
    }
    // generating for white king 2 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 7, legal: (legal white (move king d 10 c 9))
        ddi->diagonals.emplace_back(7, 19);

        // to position 11, legal: (legal white (move king d 10 b 8))
        ddi->diagonals.emplace_back(11, 20);

        // to position 16, legal: (legal white (move king d 10 a 7))
        ddi->diagonals.emplace_back(16, 21);

        this->diagonal_data[51].push_back(ddi);


        this->reverse_legal_lookup_white[19] = ReverseLegalLookup(Role::White, Piece::King, 2, 7, SW);
        this->reverse_legal_lookup_white[20] = ReverseLegalLookup(Role::White, Piece::King, 2, 11, SW);
        this->reverse_legal_lookup_white[21] = ReverseLegalLookup(Role::White, Piece::King, 2, 16, SW);
    }
    // generating for white king 3 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 9, legal: (legal white (move king f 10 g 9))
        ddi->diagonals.emplace_back(9, 24);

        // to position 14, legal: (legal white (move king f 10 h 8))
        ddi->diagonals.emplace_back(14, 25);

        // to position 20, legal: (legal white (move king f 10 i 7))
        ddi->diagonals.emplace_back(20, 26);

        // to position 25, legal: (legal white (move king f 10 j 6))
        ddi->diagonals.emplace_back(25, 27);

        this->diagonal_data[52].push_back(ddi);


        this->reverse_legal_lookup_white[24] = ReverseLegalLookup(Role::White, Piece::King, 3, 9, SE);
        this->reverse_legal_lookup_white[25] = ReverseLegalLookup(Role::White, Piece::King, 3, 14, SE);
        this->reverse_legal_lookup_white[26] = ReverseLegalLookup(Role::White, Piece::King, 3, 20, SE);
        this->reverse_legal_lookup_white[27] = ReverseLegalLookup(Role::White, Piece::King, 3, 25, SE);
    }
    // generating for white king 3 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 8, legal: (legal white (move king f 10 e 9))
        ddi->diagonals.emplace_back(8, 28);

        // to position 12, legal: (legal white (move king f 10 d 8))
        ddi->diagonals.emplace_back(12, 29);

        // to position 17, legal: (legal white (move king f 10 c 7))
        ddi->diagonals.emplace_back(17, 30);

        // to position 21, legal: (legal white (move king f 10 b 6))
        ddi->diagonals.emplace_back(21, 31);

        // to position 26, legal: (legal white (move king f 10 a 5))
        ddi->diagonals.emplace_back(26, 32);

        this->diagonal_data[52].push_back(ddi);


        this->reverse_legal_lookup_white[28] = ReverseLegalLookup(Role::White, Piece::King, 3, 8, SW);
        this->reverse_legal_lookup_white[29] = ReverseLegalLookup(Role::White, Piece::King, 3, 12, SW);
        this->reverse_legal_lookup_white[30] = ReverseLegalLookup(Role::White, Piece::King, 3, 17, SW);
        this->reverse_legal_lookup_white[31] = ReverseLegalLookup(Role::White, Piece::King, 3, 21, SW);
        this->reverse_legal_lookup_white[32] = ReverseLegalLookup(Role::White, Piece::King, 3, 26, SW);
    }
    // generating for white king 4 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal white (move king h 10 i 9))
        ddi->diagonals.emplace_back(10, 35);

        // to position 15, legal: (legal white (move king h 10 j 8))
        ddi->diagonals.emplace_back(15, 36);

        this->diagonal_data[53].push_back(ddi);


        this->reverse_legal_lookup_white[35] = ReverseLegalLookup(Role::White, Piece::King, 4, 10, SE);
        this->reverse_legal_lookup_white[36] = ReverseLegalLookup(Role::White, Piece::King, 4, 15, SE);
    }
    // generating for white king 4 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(7);
        // to position 9, legal: (legal white (move king h 10 g 9))
        ddi->diagonals.emplace_back(9, 37);

        // to position 13, legal: (legal white (move king h 10 f 8))
        ddi->diagonals.emplace_back(13, 38);

        // to position 18, legal: (legal white (move king h 10 e 7))
        ddi->diagonals.emplace_back(18, 39);

        // to position 22, legal: (legal white (move king h 10 d 6))
        ddi->diagonals.emplace_back(22, 40);

        // to position 27, legal: (legal white (move king h 10 c 5))
        ddi->diagonals.emplace_back(27, 41);

        // to position 31, legal: (legal white (move king h 10 b 4))
        ddi->diagonals.emplace_back(31, 42);

        // to position 36, legal: (legal white (move king h 10 a 3))
        ddi->diagonals.emplace_back(36, 43);

        this->diagonal_data[53].push_back(ddi);


        this->reverse_legal_lookup_white[37] = ReverseLegalLookup(Role::White, Piece::King, 4, 9, SW);
        this->reverse_legal_lookup_white[38] = ReverseLegalLookup(Role::White, Piece::King, 4, 13, SW);
        this->reverse_legal_lookup_white[39] = ReverseLegalLookup(Role::White, Piece::King, 4, 18, SW);
        this->reverse_legal_lookup_white[40] = ReverseLegalLookup(Role::White, Piece::King, 4, 22, SW);
        this->reverse_legal_lookup_white[41] = ReverseLegalLookup(Role::White, Piece::King, 4, 27, SW);
        this->reverse_legal_lookup_white[42] = ReverseLegalLookup(Role::White, Piece::King, 4, 31, SW);
        this->reverse_legal_lookup_white[43] = ReverseLegalLookup(Role::White, Piece::King, 4, 36, SW);
    }
    // generating for white king 5 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(9);
        // to position 10, legal: (legal white (move king j 10 i 9))
        ddi->diagonals.emplace_back(10, 45);

        // to position 14, legal: (legal white (move king j 10 h 8))
        ddi->diagonals.emplace_back(14, 46);

        // to position 19, legal: (legal white (move king j 10 g 7))
        ddi->diagonals.emplace_back(19, 47);

        // to position 23, legal: (legal white (move king j 10 f 6))
        ddi->diagonals.emplace_back(23, 48);

        // to position 28, legal: (legal white (move king j 10 e 5))
        ddi->diagonals.emplace_back(28, 49);

        // to position 32, legal: (legal white (move king j 10 d 4))
        ddi->diagonals.emplace_back(32, 50);

        // to position 37, legal: (legal white (move king j 10 c 3))
        ddi->diagonals.emplace_back(37, 51);

        // to position 41, legal: (legal white (move king j 10 b 2))
        ddi->diagonals.emplace_back(41, 52);

        // to position 46, legal: (legal white (move king j 10 a 1))
        ddi->diagonals.emplace_back(46, 53);

        this->diagonal_data[54].push_back(ddi);


        this->reverse_legal_lookup_white[45] = ReverseLegalLookup(Role::White, Piece::King, 5, 10, SW);
        this->reverse_legal_lookup_white[46] = ReverseLegalLookup(Role::White, Piece::King, 5, 14, SW);
        this->reverse_legal_lookup_white[47] = ReverseLegalLookup(Role::White, Piece::King, 5, 19, SW);
        this->reverse_legal_lookup_white[48] = ReverseLegalLookup(Role::White, Piece::King, 5, 23, SW);
        this->reverse_legal_lookup_white[49] = ReverseLegalLookup(Role::White, Piece::King, 5, 28, SW);
        this->reverse_legal_lookup_white[50] = ReverseLegalLookup(Role::White, Piece::King, 5, 32, SW);
        this->reverse_legal_lookup_white[51] = ReverseLegalLookup(Role::White, Piece::King, 5, 37, SW);
        this->reverse_legal_lookup_white[52] = ReverseLegalLookup(Role::White, Piece::King, 5, 41, SW);
        this->reverse_legal_lookup_white[53] = ReverseLegalLookup(Role::White, Piece::King, 5, 46, SW);
    }
    // generating for white king 6 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal white (move king a 9 b 10))
        ddi->diagonals.emplace_back(1, 56);

        this->diagonal_data[55].push_back(ddi);


        this->reverse_legal_lookup_white[56] = ReverseLegalLookup(Role::White, Piece::King, 6, 1, NE);
    }
    // generating for white king 6 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(8);
        // to position 11, legal: (legal white (move king a 9 b 8))
        ddi->diagonals.emplace_back(11, 57);

        // to position 17, legal: (legal white (move king a 9 c 7))
        ddi->diagonals.emplace_back(17, 58);

        // to position 22, legal: (legal white (move king a 9 d 6))
        ddi->diagonals.emplace_back(22, 59);

        // to position 28, legal: (legal white (move king a 9 e 5))
        ddi->diagonals.emplace_back(28, 60);

        // to position 33, legal: (legal white (move king a 9 f 4))
        ddi->diagonals.emplace_back(33, 61);

        // to position 39, legal: (legal white (move king a 9 g 3))
        ddi->diagonals.emplace_back(39, 62);

        // to position 44, legal: (legal white (move king a 9 h 2))
        ddi->diagonals.emplace_back(44, 63);

        // to position 50, legal: (legal white (move king a 9 i 1))
        ddi->diagonals.emplace_back(50, 64);

        this->diagonal_data[55].push_back(ddi);


        this->reverse_legal_lookup_white[57] = ReverseLegalLookup(Role::White, Piece::King, 6, 11, SE);
        this->reverse_legal_lookup_white[58] = ReverseLegalLookup(Role::White, Piece::King, 6, 17, SE);
        this->reverse_legal_lookup_white[59] = ReverseLegalLookup(Role::White, Piece::King, 6, 22, SE);
        this->reverse_legal_lookup_white[60] = ReverseLegalLookup(Role::White, Piece::King, 6, 28, SE);
        this->reverse_legal_lookup_white[61] = ReverseLegalLookup(Role::White, Piece::King, 6, 33, SE);
        this->reverse_legal_lookup_white[62] = ReverseLegalLookup(Role::White, Piece::King, 6, 39, SE);
        this->reverse_legal_lookup_white[63] = ReverseLegalLookup(Role::White, Piece::King, 6, 44, SE);
        this->reverse_legal_lookup_white[64] = ReverseLegalLookup(Role::White, Piece::King, 6, 50, SE);
    }
    // generating for white king 7 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal white (move king c 9 d 10))
        ddi->diagonals.emplace_back(2, 69);

        this->diagonal_data[56].push_back(ddi);


        this->reverse_legal_lookup_white[69] = ReverseLegalLookup(Role::White, Piece::King, 7, 2, NE);
    }
    // generating for white king 7 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal white (move king c 9 b 10))
        ddi->diagonals.emplace_back(1, 70);

        this->diagonal_data[56].push_back(ddi);


        this->reverse_legal_lookup_white[70] = ReverseLegalLookup(Role::White, Piece::King, 7, 1, NW);
    }
    // generating for white king 7 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(7);
        // to position 12, legal: (legal white (move king c 9 d 8))
        ddi->diagonals.emplace_back(12, 71);

        // to position 18, legal: (legal white (move king c 9 e 7))
        ddi->diagonals.emplace_back(18, 72);

        // to position 23, legal: (legal white (move king c 9 f 6))
        ddi->diagonals.emplace_back(23, 73);

        // to position 29, legal: (legal white (move king c 9 g 5))
        ddi->diagonals.emplace_back(29, 74);

        // to position 34, legal: (legal white (move king c 9 h 4))
        ddi->diagonals.emplace_back(34, 75);

        // to position 40, legal: (legal white (move king c 9 i 3))
        ddi->diagonals.emplace_back(40, 76);

        // to position 45, legal: (legal white (move king c 9 j 2))
        ddi->diagonals.emplace_back(45, 77);

        this->diagonal_data[56].push_back(ddi);


        this->reverse_legal_lookup_white[71] = ReverseLegalLookup(Role::White, Piece::King, 7, 12, SE);
        this->reverse_legal_lookup_white[72] = ReverseLegalLookup(Role::White, Piece::King, 7, 18, SE);
        this->reverse_legal_lookup_white[73] = ReverseLegalLookup(Role::White, Piece::King, 7, 23, SE);
        this->reverse_legal_lookup_white[74] = ReverseLegalLookup(Role::White, Piece::King, 7, 29, SE);
        this->reverse_legal_lookup_white[75] = ReverseLegalLookup(Role::White, Piece::King, 7, 34, SE);
        this->reverse_legal_lookup_white[76] = ReverseLegalLookup(Role::White, Piece::King, 7, 40, SE);
        this->reverse_legal_lookup_white[77] = ReverseLegalLookup(Role::White, Piece::King, 7, 45, SE);
    }
    // generating for white king 7 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal white (move king c 9 b 8))
        ddi->diagonals.emplace_back(11, 78);

        // to position 16, legal: (legal white (move king c 9 a 7))
        ddi->diagonals.emplace_back(16, 79);

        this->diagonal_data[56].push_back(ddi);


        this->reverse_legal_lookup_white[78] = ReverseLegalLookup(Role::White, Piece::King, 7, 11, SW);
        this->reverse_legal_lookup_white[79] = ReverseLegalLookup(Role::White, Piece::King, 7, 16, SW);
    }
    // generating for white king 8 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal white (move king e 9 f 10))
        ddi->diagonals.emplace_back(3, 84);

        this->diagonal_data[57].push_back(ddi);


        this->reverse_legal_lookup_white[84] = ReverseLegalLookup(Role::White, Piece::King, 8, 3, NE);
    }
    // generating for white king 8 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal white (move king e 9 d 10))
        ddi->diagonals.emplace_back(2, 85);

        this->diagonal_data[57].push_back(ddi);


        this->reverse_legal_lookup_white[85] = ReverseLegalLookup(Role::White, Piece::King, 8, 2, NW);
    }
    // generating for white king 8 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 13, legal: (legal white (move king e 9 f 8))
        ddi->diagonals.emplace_back(13, 86);

        // to position 19, legal: (legal white (move king e 9 g 7))
        ddi->diagonals.emplace_back(19, 87);

        // to position 24, legal: (legal white (move king e 9 h 6))
        ddi->diagonals.emplace_back(24, 88);

        // to position 30, legal: (legal white (move king e 9 i 5))
        ddi->diagonals.emplace_back(30, 89);

        // to position 35, legal: (legal white (move king e 9 j 4))
        ddi->diagonals.emplace_back(35, 90);

        this->diagonal_data[57].push_back(ddi);


        this->reverse_legal_lookup_white[86] = ReverseLegalLookup(Role::White, Piece::King, 8, 13, SE);
        this->reverse_legal_lookup_white[87] = ReverseLegalLookup(Role::White, Piece::King, 8, 19, SE);
        this->reverse_legal_lookup_white[88] = ReverseLegalLookup(Role::White, Piece::King, 8, 24, SE);
        this->reverse_legal_lookup_white[89] = ReverseLegalLookup(Role::White, Piece::King, 8, 30, SE);
        this->reverse_legal_lookup_white[90] = ReverseLegalLookup(Role::White, Piece::King, 8, 35, SE);
    }
    // generating for white king 8 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 12, legal: (legal white (move king e 9 d 8))
        ddi->diagonals.emplace_back(12, 91);

        // to position 17, legal: (legal white (move king e 9 c 7))
        ddi->diagonals.emplace_back(17, 92);

        // to position 21, legal: (legal white (move king e 9 b 6))
        ddi->diagonals.emplace_back(21, 93);

        // to position 26, legal: (legal white (move king e 9 a 5))
        ddi->diagonals.emplace_back(26, 94);

        this->diagonal_data[57].push_back(ddi);


        this->reverse_legal_lookup_white[91] = ReverseLegalLookup(Role::White, Piece::King, 8, 12, SW);
        this->reverse_legal_lookup_white[92] = ReverseLegalLookup(Role::White, Piece::King, 8, 17, SW);
        this->reverse_legal_lookup_white[93] = ReverseLegalLookup(Role::White, Piece::King, 8, 21, SW);
        this->reverse_legal_lookup_white[94] = ReverseLegalLookup(Role::White, Piece::King, 8, 26, SW);
    }
    // generating for white king 9 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 4, legal: (legal white (move king g 9 h 10))
        ddi->diagonals.emplace_back(4, 99);

        this->diagonal_data[58].push_back(ddi);


        this->reverse_legal_lookup_white[99] = ReverseLegalLookup(Role::White, Piece::King, 9, 4, NE);
    }
    // generating for white king 9 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal white (move king g 9 f 10))
        ddi->diagonals.emplace_back(3, 100);

        this->diagonal_data[58].push_back(ddi);


        this->reverse_legal_lookup_white[100] = ReverseLegalLookup(Role::White, Piece::King, 9, 3, NW);
    }
    // generating for white king 9 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 14, legal: (legal white (move king g 9 h 8))
        ddi->diagonals.emplace_back(14, 101);

        // to position 20, legal: (legal white (move king g 9 i 7))
        ddi->diagonals.emplace_back(20, 102);

        // to position 25, legal: (legal white (move king g 9 j 6))
        ddi->diagonals.emplace_back(25, 103);

        this->diagonal_data[58].push_back(ddi);


        this->reverse_legal_lookup_white[101] = ReverseLegalLookup(Role::White, Piece::King, 9, 14, SE);
        this->reverse_legal_lookup_white[102] = ReverseLegalLookup(Role::White, Piece::King, 9, 20, SE);
        this->reverse_legal_lookup_white[103] = ReverseLegalLookup(Role::White, Piece::King, 9, 25, SE);
    }
    // generating for white king 9 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(6);
        // to position 13, legal: (legal white (move king g 9 f 8))
        ddi->diagonals.emplace_back(13, 104);

        // to position 18, legal: (legal white (move king g 9 e 7))
        ddi->diagonals.emplace_back(18, 105);

        // to position 22, legal: (legal white (move king g 9 d 6))
        ddi->diagonals.emplace_back(22, 106);

        // to position 27, legal: (legal white (move king g 9 c 5))
        ddi->diagonals.emplace_back(27, 107);

        // to position 31, legal: (legal white (move king g 9 b 4))
        ddi->diagonals.emplace_back(31, 108);

        // to position 36, legal: (legal white (move king g 9 a 3))
        ddi->diagonals.emplace_back(36, 109);

        this->diagonal_data[58].push_back(ddi);


        this->reverse_legal_lookup_white[104] = ReverseLegalLookup(Role::White, Piece::King, 9, 13, SW);
        this->reverse_legal_lookup_white[105] = ReverseLegalLookup(Role::White, Piece::King, 9, 18, SW);
        this->reverse_legal_lookup_white[106] = ReverseLegalLookup(Role::White, Piece::King, 9, 22, SW);
        this->reverse_legal_lookup_white[107] = ReverseLegalLookup(Role::White, Piece::King, 9, 27, SW);
        this->reverse_legal_lookup_white[108] = ReverseLegalLookup(Role::White, Piece::King, 9, 31, SW);
        this->reverse_legal_lookup_white[109] = ReverseLegalLookup(Role::White, Piece::King, 9, 36, SW);
    }
    // generating for white king 10 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 5, legal: (legal white (move king i 9 j 10))
        ddi->diagonals.emplace_back(5, 113);

        this->diagonal_data[59].push_back(ddi);


        this->reverse_legal_lookup_white[113] = ReverseLegalLookup(Role::White, Piece::King, 10, 5, NE);
    }
    // generating for white king 10 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 4, legal: (legal white (move king i 9 h 10))
        ddi->diagonals.emplace_back(4, 114);

        this->diagonal_data[59].push_back(ddi);


        this->reverse_legal_lookup_white[114] = ReverseLegalLookup(Role::White, Piece::King, 10, 4, NW);
    }
    // generating for white king 10 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 15, legal: (legal white (move king i 9 j 8))
        ddi->diagonals.emplace_back(15, 115);

        this->diagonal_data[59].push_back(ddi);


        this->reverse_legal_lookup_white[115] = ReverseLegalLookup(Role::White, Piece::King, 10, 15, SE);
    }
    // generating for white king 10 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(8);
        // to position 14, legal: (legal white (move king i 9 h 8))
        ddi->diagonals.emplace_back(14, 116);

        // to position 19, legal: (legal white (move king i 9 g 7))
        ddi->diagonals.emplace_back(19, 117);

        // to position 23, legal: (legal white (move king i 9 f 6))
        ddi->diagonals.emplace_back(23, 118);

        // to position 28, legal: (legal white (move king i 9 e 5))
        ddi->diagonals.emplace_back(28, 119);

        // to position 32, legal: (legal white (move king i 9 d 4))
        ddi->diagonals.emplace_back(32, 120);

        // to position 37, legal: (legal white (move king i 9 c 3))
        ddi->diagonals.emplace_back(37, 121);

        // to position 41, legal: (legal white (move king i 9 b 2))
        ddi->diagonals.emplace_back(41, 122);

        // to position 46, legal: (legal white (move king i 9 a 1))
        ddi->diagonals.emplace_back(46, 123);

        this->diagonal_data[59].push_back(ddi);


        this->reverse_legal_lookup_white[116] = ReverseLegalLookup(Role::White, Piece::King, 10, 14, SW);
        this->reverse_legal_lookup_white[117] = ReverseLegalLookup(Role::White, Piece::King, 10, 19, SW);
        this->reverse_legal_lookup_white[118] = ReverseLegalLookup(Role::White, Piece::King, 10, 23, SW);
        this->reverse_legal_lookup_white[119] = ReverseLegalLookup(Role::White, Piece::King, 10, 28, SW);
        this->reverse_legal_lookup_white[120] = ReverseLegalLookup(Role::White, Piece::King, 10, 32, SW);
        this->reverse_legal_lookup_white[121] = ReverseLegalLookup(Role::White, Piece::King, 10, 37, SW);
        this->reverse_legal_lookup_white[122] = ReverseLegalLookup(Role::White, Piece::King, 10, 41, SW);
        this->reverse_legal_lookup_white[123] = ReverseLegalLookup(Role::White, Piece::King, 10, 46, SW);
    }
    // generating for white king 11 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal white (move king b 8 c 9))
        ddi->diagonals.emplace_back(7, 128);

        // to position 2, legal: (legal white (move king b 8 d 10))
        ddi->diagonals.emplace_back(2, 129);

        this->diagonal_data[60].push_back(ddi);


        this->reverse_legal_lookup_white[128] = ReverseLegalLookup(Role::White, Piece::King, 11, 7, NE);
        this->reverse_legal_lookup_white[129] = ReverseLegalLookup(Role::White, Piece::King, 11, 2, NE);
    }
    // generating for white king 11 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 6, legal: (legal white (move king b 8 a 9))
        ddi->diagonals.emplace_back(6, 130);

        this->diagonal_data[60].push_back(ddi);


        this->reverse_legal_lookup_white[130] = ReverseLegalLookup(Role::White, Piece::King, 11, 6, NW);
    }
    // generating for white king 11 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(7);
        // to position 17, legal: (legal white (move king b 8 c 7))
        ddi->diagonals.emplace_back(17, 131);

        // to position 22, legal: (legal white (move king b 8 d 6))
        ddi->diagonals.emplace_back(22, 132);

        // to position 28, legal: (legal white (move king b 8 e 5))
        ddi->diagonals.emplace_back(28, 133);

        // to position 33, legal: (legal white (move king b 8 f 4))
        ddi->diagonals.emplace_back(33, 134);

        // to position 39, legal: (legal white (move king b 8 g 3))
        ddi->diagonals.emplace_back(39, 135);

        // to position 44, legal: (legal white (move king b 8 h 2))
        ddi->diagonals.emplace_back(44, 136);

        // to position 50, legal: (legal white (move king b 8 i 1))
        ddi->diagonals.emplace_back(50, 137);

        this->diagonal_data[60].push_back(ddi);


        this->reverse_legal_lookup_white[131] = ReverseLegalLookup(Role::White, Piece::King, 11, 17, SE);
        this->reverse_legal_lookup_white[132] = ReverseLegalLookup(Role::White, Piece::King, 11, 22, SE);
        this->reverse_legal_lookup_white[133] = ReverseLegalLookup(Role::White, Piece::King, 11, 28, SE);
        this->reverse_legal_lookup_white[134] = ReverseLegalLookup(Role::White, Piece::King, 11, 33, SE);
        this->reverse_legal_lookup_white[135] = ReverseLegalLookup(Role::White, Piece::King, 11, 39, SE);
        this->reverse_legal_lookup_white[136] = ReverseLegalLookup(Role::White, Piece::King, 11, 44, SE);
        this->reverse_legal_lookup_white[137] = ReverseLegalLookup(Role::White, Piece::King, 11, 50, SE);
    }
    // generating for white king 11 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 16, legal: (legal white (move king b 8 a 7))
        ddi->diagonals.emplace_back(16, 138);

        this->diagonal_data[60].push_back(ddi);


        this->reverse_legal_lookup_white[138] = ReverseLegalLookup(Role::White, Piece::King, 11, 16, SW);
    }
    // generating for white king 12 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal white (move king d 8 e 9))
        ddi->diagonals.emplace_back(8, 145);

        // to position 3, legal: (legal white (move king d 8 f 10))
        ddi->diagonals.emplace_back(3, 146);

        this->diagonal_data[61].push_back(ddi);


        this->reverse_legal_lookup_white[145] = ReverseLegalLookup(Role::White, Piece::King, 12, 8, NE);
        this->reverse_legal_lookup_white[146] = ReverseLegalLookup(Role::White, Piece::King, 12, 3, NE);
    }
    // generating for white king 12 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal white (move king d 8 c 9))
        ddi->diagonals.emplace_back(7, 147);

        // to position 1, legal: (legal white (move king d 8 b 10))
        ddi->diagonals.emplace_back(1, 148);

        this->diagonal_data[61].push_back(ddi);


        this->reverse_legal_lookup_white[147] = ReverseLegalLookup(Role::White, Piece::King, 12, 7, NW);
        this->reverse_legal_lookup_white[148] = ReverseLegalLookup(Role::White, Piece::King, 12, 1, NW);
    }
    // generating for white king 12 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 18, legal: (legal white (move king d 8 e 7))
        ddi->diagonals.emplace_back(18, 149);

        // to position 23, legal: (legal white (move king d 8 f 6))
        ddi->diagonals.emplace_back(23, 150);

        // to position 29, legal: (legal white (move king d 8 g 5))
        ddi->diagonals.emplace_back(29, 151);

        // to position 34, legal: (legal white (move king d 8 h 4))
        ddi->diagonals.emplace_back(34, 152);

        // to position 40, legal: (legal white (move king d 8 i 3))
        ddi->diagonals.emplace_back(40, 153);

        // to position 45, legal: (legal white (move king d 8 j 2))
        ddi->diagonals.emplace_back(45, 154);

        this->diagonal_data[61].push_back(ddi);


        this->reverse_legal_lookup_white[149] = ReverseLegalLookup(Role::White, Piece::King, 12, 18, SE);
        this->reverse_legal_lookup_white[150] = ReverseLegalLookup(Role::White, Piece::King, 12, 23, SE);
        this->reverse_legal_lookup_white[151] = ReverseLegalLookup(Role::White, Piece::King, 12, 29, SE);
        this->reverse_legal_lookup_white[152] = ReverseLegalLookup(Role::White, Piece::King, 12, 34, SE);
        this->reverse_legal_lookup_white[153] = ReverseLegalLookup(Role::White, Piece::King, 12, 40, SE);
        this->reverse_legal_lookup_white[154] = ReverseLegalLookup(Role::White, Piece::King, 12, 45, SE);
    }
    // generating for white king 12 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 17, legal: (legal white (move king d 8 c 7))
        ddi->diagonals.emplace_back(17, 155);

        // to position 21, legal: (legal white (move king d 8 b 6))
        ddi->diagonals.emplace_back(21, 156);

        // to position 26, legal: (legal white (move king d 8 a 5))
        ddi->diagonals.emplace_back(26, 157);

        this->diagonal_data[61].push_back(ddi);


        this->reverse_legal_lookup_white[155] = ReverseLegalLookup(Role::White, Piece::King, 12, 17, SW);
        this->reverse_legal_lookup_white[156] = ReverseLegalLookup(Role::White, Piece::King, 12, 21, SW);
        this->reverse_legal_lookup_white[157] = ReverseLegalLookup(Role::White, Piece::King, 12, 26, SW);
    }
    // generating for white king 13 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal white (move king f 8 g 9))
        ddi->diagonals.emplace_back(9, 164);

        // to position 4, legal: (legal white (move king f 8 h 10))
        ddi->diagonals.emplace_back(4, 165);

        this->diagonal_data[62].push_back(ddi);


        this->reverse_legal_lookup_white[164] = ReverseLegalLookup(Role::White, Piece::King, 13, 9, NE);
        this->reverse_legal_lookup_white[165] = ReverseLegalLookup(Role::White, Piece::King, 13, 4, NE);
    }
    // generating for white king 13 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal white (move king f 8 e 9))
        ddi->diagonals.emplace_back(8, 166);

        // to position 2, legal: (legal white (move king f 8 d 10))
        ddi->diagonals.emplace_back(2, 167);

        this->diagonal_data[62].push_back(ddi);


        this->reverse_legal_lookup_white[166] = ReverseLegalLookup(Role::White, Piece::King, 13, 8, NW);
        this->reverse_legal_lookup_white[167] = ReverseLegalLookup(Role::White, Piece::King, 13, 2, NW);
    }
    // generating for white king 13 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 19, legal: (legal white (move king f 8 g 7))
        ddi->diagonals.emplace_back(19, 168);

        // to position 24, legal: (legal white (move king f 8 h 6))
        ddi->diagonals.emplace_back(24, 169);

        // to position 30, legal: (legal white (move king f 8 i 5))
        ddi->diagonals.emplace_back(30, 170);

        // to position 35, legal: (legal white (move king f 8 j 4))
        ddi->diagonals.emplace_back(35, 171);

        this->diagonal_data[62].push_back(ddi);


        this->reverse_legal_lookup_white[168] = ReverseLegalLookup(Role::White, Piece::King, 13, 19, SE);
        this->reverse_legal_lookup_white[169] = ReverseLegalLookup(Role::White, Piece::King, 13, 24, SE);
        this->reverse_legal_lookup_white[170] = ReverseLegalLookup(Role::White, Piece::King, 13, 30, SE);
        this->reverse_legal_lookup_white[171] = ReverseLegalLookup(Role::White, Piece::King, 13, 35, SE);
    }
    // generating for white king 13 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 18, legal: (legal white (move king f 8 e 7))
        ddi->diagonals.emplace_back(18, 172);

        // to position 22, legal: (legal white (move king f 8 d 6))
        ddi->diagonals.emplace_back(22, 173);

        // to position 27, legal: (legal white (move king f 8 c 5))
        ddi->diagonals.emplace_back(27, 174);

        // to position 31, legal: (legal white (move king f 8 b 4))
        ddi->diagonals.emplace_back(31, 175);

        // to position 36, legal: (legal white (move king f 8 a 3))
        ddi->diagonals.emplace_back(36, 176);

        this->diagonal_data[62].push_back(ddi);


        this->reverse_legal_lookup_white[172] = ReverseLegalLookup(Role::White, Piece::King, 13, 18, SW);
        this->reverse_legal_lookup_white[173] = ReverseLegalLookup(Role::White, Piece::King, 13, 22, SW);
        this->reverse_legal_lookup_white[174] = ReverseLegalLookup(Role::White, Piece::King, 13, 27, SW);
        this->reverse_legal_lookup_white[175] = ReverseLegalLookup(Role::White, Piece::King, 13, 31, SW);
        this->reverse_legal_lookup_white[176] = ReverseLegalLookup(Role::White, Piece::King, 13, 36, SW);
    }
    // generating for white king 14 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal white (move king h 8 i 9))
        ddi->diagonals.emplace_back(10, 183);

        // to position 5, legal: (legal white (move king h 8 j 10))
        ddi->diagonals.emplace_back(5, 184);

        this->diagonal_data[63].push_back(ddi);


        this->reverse_legal_lookup_white[183] = ReverseLegalLookup(Role::White, Piece::King, 14, 10, NE);
        this->reverse_legal_lookup_white[184] = ReverseLegalLookup(Role::White, Piece::King, 14, 5, NE);
    }
    // generating for white king 14 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal white (move king h 8 g 9))
        ddi->diagonals.emplace_back(9, 185);

        // to position 3, legal: (legal white (move king h 8 f 10))
        ddi->diagonals.emplace_back(3, 186);

        this->diagonal_data[63].push_back(ddi);


        this->reverse_legal_lookup_white[185] = ReverseLegalLookup(Role::White, Piece::King, 14, 9, NW);
        this->reverse_legal_lookup_white[186] = ReverseLegalLookup(Role::White, Piece::King, 14, 3, NW);
    }
    // generating for white king 14 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 20, legal: (legal white (move king h 8 i 7))
        ddi->diagonals.emplace_back(20, 187);

        // to position 25, legal: (legal white (move king h 8 j 6))
        ddi->diagonals.emplace_back(25, 188);

        this->diagonal_data[63].push_back(ddi);


        this->reverse_legal_lookup_white[187] = ReverseLegalLookup(Role::White, Piece::King, 14, 20, SE);
        this->reverse_legal_lookup_white[188] = ReverseLegalLookup(Role::White, Piece::King, 14, 25, SE);
    }
    // generating for white king 14 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(7);
        // to position 19, legal: (legal white (move king h 8 g 7))
        ddi->diagonals.emplace_back(19, 189);

        // to position 23, legal: (legal white (move king h 8 f 6))
        ddi->diagonals.emplace_back(23, 190);

        // to position 28, legal: (legal white (move king h 8 e 5))
        ddi->diagonals.emplace_back(28, 191);

        // to position 32, legal: (legal white (move king h 8 d 4))
        ddi->diagonals.emplace_back(32, 192);

        // to position 37, legal: (legal white (move king h 8 c 3))
        ddi->diagonals.emplace_back(37, 193);

        // to position 41, legal: (legal white (move king h 8 b 2))
        ddi->diagonals.emplace_back(41, 194);

        // to position 46, legal: (legal white (move king h 8 a 1))
        ddi->diagonals.emplace_back(46, 195);

        this->diagonal_data[63].push_back(ddi);


        this->reverse_legal_lookup_white[189] = ReverseLegalLookup(Role::White, Piece::King, 14, 19, SW);
        this->reverse_legal_lookup_white[190] = ReverseLegalLookup(Role::White, Piece::King, 14, 23, SW);
        this->reverse_legal_lookup_white[191] = ReverseLegalLookup(Role::White, Piece::King, 14, 28, SW);
        this->reverse_legal_lookup_white[192] = ReverseLegalLookup(Role::White, Piece::King, 14, 32, SW);
        this->reverse_legal_lookup_white[193] = ReverseLegalLookup(Role::White, Piece::King, 14, 37, SW);
        this->reverse_legal_lookup_white[194] = ReverseLegalLookup(Role::White, Piece::King, 14, 41, SW);
        this->reverse_legal_lookup_white[195] = ReverseLegalLookup(Role::White, Piece::King, 14, 46, SW);
    }
    // generating for white king 15 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal white (move king j 8 i 9))
        ddi->diagonals.emplace_back(10, 199);

        // to position 4, legal: (legal white (move king j 8 h 10))
        ddi->diagonals.emplace_back(4, 200);

        this->diagonal_data[64].push_back(ddi);


        this->reverse_legal_lookup_white[199] = ReverseLegalLookup(Role::White, Piece::King, 15, 10, NW);
        this->reverse_legal_lookup_white[200] = ReverseLegalLookup(Role::White, Piece::King, 15, 4, NW);
    }
    // generating for white king 15 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(7);
        // to position 20, legal: (legal white (move king j 8 i 7))
        ddi->diagonals.emplace_back(20, 201);

        // to position 24, legal: (legal white (move king j 8 h 6))
        ddi->diagonals.emplace_back(24, 202);

        // to position 29, legal: (legal white (move king j 8 g 5))
        ddi->diagonals.emplace_back(29, 203);

        // to position 33, legal: (legal white (move king j 8 f 4))
        ddi->diagonals.emplace_back(33, 204);

        // to position 38, legal: (legal white (move king j 8 e 3))
        ddi->diagonals.emplace_back(38, 205);

        // to position 42, legal: (legal white (move king j 8 d 2))
        ddi->diagonals.emplace_back(42, 206);

        // to position 47, legal: (legal white (move king j 8 c 1))
        ddi->diagonals.emplace_back(47, 207);

        this->diagonal_data[64].push_back(ddi);


        this->reverse_legal_lookup_white[201] = ReverseLegalLookup(Role::White, Piece::King, 15, 20, SW);
        this->reverse_legal_lookup_white[202] = ReverseLegalLookup(Role::White, Piece::King, 15, 24, SW);
        this->reverse_legal_lookup_white[203] = ReverseLegalLookup(Role::White, Piece::King, 15, 29, SW);
        this->reverse_legal_lookup_white[204] = ReverseLegalLookup(Role::White, Piece::King, 15, 33, SW);
        this->reverse_legal_lookup_white[205] = ReverseLegalLookup(Role::White, Piece::King, 15, 38, SW);
        this->reverse_legal_lookup_white[206] = ReverseLegalLookup(Role::White, Piece::King, 15, 42, SW);
        this->reverse_legal_lookup_white[207] = ReverseLegalLookup(Role::White, Piece::King, 15, 47, SW);
    }
    // generating for white king 16 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 11, legal: (legal white (move king a 7 b 8))
        ddi->diagonals.emplace_back(11, 211);

        // to position 7, legal: (legal white (move king a 7 c 9))
        ddi->diagonals.emplace_back(7, 212);

        // to position 2, legal: (legal white (move king a 7 d 10))
        ddi->diagonals.emplace_back(2, 213);

        this->diagonal_data[65].push_back(ddi);


        this->reverse_legal_lookup_white[211] = ReverseLegalLookup(Role::White, Piece::King, 16, 11, NE);
        this->reverse_legal_lookup_white[212] = ReverseLegalLookup(Role::White, Piece::King, 16, 7, NE);
        this->reverse_legal_lookup_white[213] = ReverseLegalLookup(Role::White, Piece::King, 16, 2, NE);
    }
    // generating for white king 16 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 21, legal: (legal white (move king a 7 b 6))
        ddi->diagonals.emplace_back(21, 214);

        // to position 27, legal: (legal white (move king a 7 c 5))
        ddi->diagonals.emplace_back(27, 215);

        // to position 32, legal: (legal white (move king a 7 d 4))
        ddi->diagonals.emplace_back(32, 216);

        // to position 38, legal: (legal white (move king a 7 e 3))
        ddi->diagonals.emplace_back(38, 217);

        // to position 43, legal: (legal white (move king a 7 f 2))
        ddi->diagonals.emplace_back(43, 218);

        // to position 49, legal: (legal white (move king a 7 g 1))
        ddi->diagonals.emplace_back(49, 219);

        this->diagonal_data[65].push_back(ddi);


        this->reverse_legal_lookup_white[214] = ReverseLegalLookup(Role::White, Piece::King, 16, 21, SE);
        this->reverse_legal_lookup_white[215] = ReverseLegalLookup(Role::White, Piece::King, 16, 27, SE);
        this->reverse_legal_lookup_white[216] = ReverseLegalLookup(Role::White, Piece::King, 16, 32, SE);
        this->reverse_legal_lookup_white[217] = ReverseLegalLookup(Role::White, Piece::King, 16, 38, SE);
        this->reverse_legal_lookup_white[218] = ReverseLegalLookup(Role::White, Piece::King, 16, 43, SE);
        this->reverse_legal_lookup_white[219] = ReverseLegalLookup(Role::White, Piece::King, 16, 49, SE);
    }
    // generating for white king 17 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 12, legal: (legal white (move king c 7 d 8))
        ddi->diagonals.emplace_back(12, 226);

        // to position 8, legal: (legal white (move king c 7 e 9))
        ddi->diagonals.emplace_back(8, 227);

        // to position 3, legal: (legal white (move king c 7 f 10))
        ddi->diagonals.emplace_back(3, 228);

        this->diagonal_data[66].push_back(ddi);


        this->reverse_legal_lookup_white[226] = ReverseLegalLookup(Role::White, Piece::King, 17, 12, NE);
        this->reverse_legal_lookup_white[227] = ReverseLegalLookup(Role::White, Piece::King, 17, 8, NE);
        this->reverse_legal_lookup_white[228] = ReverseLegalLookup(Role::White, Piece::King, 17, 3, NE);
    }
    // generating for white king 17 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal white (move king c 7 b 8))
        ddi->diagonals.emplace_back(11, 229);

        // to position 6, legal: (legal white (move king c 7 a 9))
        ddi->diagonals.emplace_back(6, 230);

        this->diagonal_data[66].push_back(ddi);


        this->reverse_legal_lookup_white[229] = ReverseLegalLookup(Role::White, Piece::King, 17, 11, NW);
        this->reverse_legal_lookup_white[230] = ReverseLegalLookup(Role::White, Piece::King, 17, 6, NW);
    }
    // generating for white king 17 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 22, legal: (legal white (move king c 7 d 6))
        ddi->diagonals.emplace_back(22, 231);

        // to position 28, legal: (legal white (move king c 7 e 5))
        ddi->diagonals.emplace_back(28, 232);

        // to position 33, legal: (legal white (move king c 7 f 4))
        ddi->diagonals.emplace_back(33, 233);

        // to position 39, legal: (legal white (move king c 7 g 3))
        ddi->diagonals.emplace_back(39, 234);

        // to position 44, legal: (legal white (move king c 7 h 2))
        ddi->diagonals.emplace_back(44, 235);

        // to position 50, legal: (legal white (move king c 7 i 1))
        ddi->diagonals.emplace_back(50, 236);

        this->diagonal_data[66].push_back(ddi);


        this->reverse_legal_lookup_white[231] = ReverseLegalLookup(Role::White, Piece::King, 17, 22, SE);
        this->reverse_legal_lookup_white[232] = ReverseLegalLookup(Role::White, Piece::King, 17, 28, SE);
        this->reverse_legal_lookup_white[233] = ReverseLegalLookup(Role::White, Piece::King, 17, 33, SE);
        this->reverse_legal_lookup_white[234] = ReverseLegalLookup(Role::White, Piece::King, 17, 39, SE);
        this->reverse_legal_lookup_white[235] = ReverseLegalLookup(Role::White, Piece::King, 17, 44, SE);
        this->reverse_legal_lookup_white[236] = ReverseLegalLookup(Role::White, Piece::King, 17, 50, SE);
    }
    // generating for white king 17 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 21, legal: (legal white (move king c 7 b 6))
        ddi->diagonals.emplace_back(21, 237);

        // to position 26, legal: (legal white (move king c 7 a 5))
        ddi->diagonals.emplace_back(26, 238);

        this->diagonal_data[66].push_back(ddi);


        this->reverse_legal_lookup_white[237] = ReverseLegalLookup(Role::White, Piece::King, 17, 21, SW);
        this->reverse_legal_lookup_white[238] = ReverseLegalLookup(Role::White, Piece::King, 17, 26, SW);
    }
    // generating for white king 18 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 13, legal: (legal white (move king e 7 f 8))
        ddi->diagonals.emplace_back(13, 245);

        // to position 9, legal: (legal white (move king e 7 g 9))
        ddi->diagonals.emplace_back(9, 246);

        // to position 4, legal: (legal white (move king e 7 h 10))
        ddi->diagonals.emplace_back(4, 247);

        this->diagonal_data[67].push_back(ddi);


        this->reverse_legal_lookup_white[245] = ReverseLegalLookup(Role::White, Piece::King, 18, 13, NE);
        this->reverse_legal_lookup_white[246] = ReverseLegalLookup(Role::White, Piece::King, 18, 9, NE);
        this->reverse_legal_lookup_white[247] = ReverseLegalLookup(Role::White, Piece::King, 18, 4, NE);
    }
    // generating for white king 18 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 12, legal: (legal white (move king e 7 d 8))
        ddi->diagonals.emplace_back(12, 248);

        // to position 7, legal: (legal white (move king e 7 c 9))
        ddi->diagonals.emplace_back(7, 249);

        // to position 1, legal: (legal white (move king e 7 b 10))
        ddi->diagonals.emplace_back(1, 250);

        this->diagonal_data[67].push_back(ddi);


        this->reverse_legal_lookup_white[248] = ReverseLegalLookup(Role::White, Piece::King, 18, 12, NW);
        this->reverse_legal_lookup_white[249] = ReverseLegalLookup(Role::White, Piece::King, 18, 7, NW);
        this->reverse_legal_lookup_white[250] = ReverseLegalLookup(Role::White, Piece::King, 18, 1, NW);
    }
    // generating for white king 18 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 23, legal: (legal white (move king e 7 f 6))
        ddi->diagonals.emplace_back(23, 251);

        // to position 29, legal: (legal white (move king e 7 g 5))
        ddi->diagonals.emplace_back(29, 252);

        // to position 34, legal: (legal white (move king e 7 h 4))
        ddi->diagonals.emplace_back(34, 253);

        // to position 40, legal: (legal white (move king e 7 i 3))
        ddi->diagonals.emplace_back(40, 254);

        // to position 45, legal: (legal white (move king e 7 j 2))
        ddi->diagonals.emplace_back(45, 255);

        this->diagonal_data[67].push_back(ddi);


        this->reverse_legal_lookup_white[251] = ReverseLegalLookup(Role::White, Piece::King, 18, 23, SE);
        this->reverse_legal_lookup_white[252] = ReverseLegalLookup(Role::White, Piece::King, 18, 29, SE);
        this->reverse_legal_lookup_white[253] = ReverseLegalLookup(Role::White, Piece::King, 18, 34, SE);
        this->reverse_legal_lookup_white[254] = ReverseLegalLookup(Role::White, Piece::King, 18, 40, SE);
        this->reverse_legal_lookup_white[255] = ReverseLegalLookup(Role::White, Piece::King, 18, 45, SE);
    }
    // generating for white king 18 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 22, legal: (legal white (move king e 7 d 6))
        ddi->diagonals.emplace_back(22, 256);

        // to position 27, legal: (legal white (move king e 7 c 5))
        ddi->diagonals.emplace_back(27, 257);

        // to position 31, legal: (legal white (move king e 7 b 4))
        ddi->diagonals.emplace_back(31, 258);

        // to position 36, legal: (legal white (move king e 7 a 3))
        ddi->diagonals.emplace_back(36, 259);

        this->diagonal_data[67].push_back(ddi);


        this->reverse_legal_lookup_white[256] = ReverseLegalLookup(Role::White, Piece::King, 18, 22, SW);
        this->reverse_legal_lookup_white[257] = ReverseLegalLookup(Role::White, Piece::King, 18, 27, SW);
        this->reverse_legal_lookup_white[258] = ReverseLegalLookup(Role::White, Piece::King, 18, 31, SW);
        this->reverse_legal_lookup_white[259] = ReverseLegalLookup(Role::White, Piece::King, 18, 36, SW);
    }
    // generating for white king 19 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 14, legal: (legal white (move king g 7 h 8))
        ddi->diagonals.emplace_back(14, 266);

        // to position 10, legal: (legal white (move king g 7 i 9))
        ddi->diagonals.emplace_back(10, 267);

        // to position 5, legal: (legal white (move king g 7 j 10))
        ddi->diagonals.emplace_back(5, 268);

        this->diagonal_data[68].push_back(ddi);


        this->reverse_legal_lookup_white[266] = ReverseLegalLookup(Role::White, Piece::King, 19, 14, NE);
        this->reverse_legal_lookup_white[267] = ReverseLegalLookup(Role::White, Piece::King, 19, 10, NE);
        this->reverse_legal_lookup_white[268] = ReverseLegalLookup(Role::White, Piece::King, 19, 5, NE);
    }
    // generating for white king 19 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 13, legal: (legal white (move king g 7 f 8))
        ddi->diagonals.emplace_back(13, 269);

        // to position 8, legal: (legal white (move king g 7 e 9))
        ddi->diagonals.emplace_back(8, 270);

        // to position 2, legal: (legal white (move king g 7 d 10))
        ddi->diagonals.emplace_back(2, 271);

        this->diagonal_data[68].push_back(ddi);


        this->reverse_legal_lookup_white[269] = ReverseLegalLookup(Role::White, Piece::King, 19, 13, NW);
        this->reverse_legal_lookup_white[270] = ReverseLegalLookup(Role::White, Piece::King, 19, 8, NW);
        this->reverse_legal_lookup_white[271] = ReverseLegalLookup(Role::White, Piece::King, 19, 2, NW);
    }
    // generating for white king 19 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 24, legal: (legal white (move king g 7 h 6))
        ddi->diagonals.emplace_back(24, 272);

        // to position 30, legal: (legal white (move king g 7 i 5))
        ddi->diagonals.emplace_back(30, 273);

        // to position 35, legal: (legal white (move king g 7 j 4))
        ddi->diagonals.emplace_back(35, 274);

        this->diagonal_data[68].push_back(ddi);


        this->reverse_legal_lookup_white[272] = ReverseLegalLookup(Role::White, Piece::King, 19, 24, SE);
        this->reverse_legal_lookup_white[273] = ReverseLegalLookup(Role::White, Piece::King, 19, 30, SE);
        this->reverse_legal_lookup_white[274] = ReverseLegalLookup(Role::White, Piece::King, 19, 35, SE);
    }
    // generating for white king 19 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(6);
        // to position 23, legal: (legal white (move king g 7 f 6))
        ddi->diagonals.emplace_back(23, 275);

        // to position 28, legal: (legal white (move king g 7 e 5))
        ddi->diagonals.emplace_back(28, 276);

        // to position 32, legal: (legal white (move king g 7 d 4))
        ddi->diagonals.emplace_back(32, 277);

        // to position 37, legal: (legal white (move king g 7 c 3))
        ddi->diagonals.emplace_back(37, 278);

        // to position 41, legal: (legal white (move king g 7 b 2))
        ddi->diagonals.emplace_back(41, 279);

        // to position 46, legal: (legal white (move king g 7 a 1))
        ddi->diagonals.emplace_back(46, 280);

        this->diagonal_data[68].push_back(ddi);


        this->reverse_legal_lookup_white[275] = ReverseLegalLookup(Role::White, Piece::King, 19, 23, SW);
        this->reverse_legal_lookup_white[276] = ReverseLegalLookup(Role::White, Piece::King, 19, 28, SW);
        this->reverse_legal_lookup_white[277] = ReverseLegalLookup(Role::White, Piece::King, 19, 32, SW);
        this->reverse_legal_lookup_white[278] = ReverseLegalLookup(Role::White, Piece::King, 19, 37, SW);
        this->reverse_legal_lookup_white[279] = ReverseLegalLookup(Role::White, Piece::King, 19, 41, SW);
        this->reverse_legal_lookup_white[280] = ReverseLegalLookup(Role::White, Piece::King, 19, 46, SW);
    }
    // generating for white king 20 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 15, legal: (legal white (move king i 7 j 8))
        ddi->diagonals.emplace_back(15, 285);

        this->diagonal_data[69].push_back(ddi);


        this->reverse_legal_lookup_white[285] = ReverseLegalLookup(Role::White, Piece::King, 20, 15, NE);
    }
    // generating for white king 20 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 14, legal: (legal white (move king i 7 h 8))
        ddi->diagonals.emplace_back(14, 286);

        // to position 9, legal: (legal white (move king i 7 g 9))
        ddi->diagonals.emplace_back(9, 287);

        // to position 3, legal: (legal white (move king i 7 f 10))
        ddi->diagonals.emplace_back(3, 288);

        this->diagonal_data[69].push_back(ddi);


        this->reverse_legal_lookup_white[286] = ReverseLegalLookup(Role::White, Piece::King, 20, 14, NW);
        this->reverse_legal_lookup_white[287] = ReverseLegalLookup(Role::White, Piece::King, 20, 9, NW);
        this->reverse_legal_lookup_white[288] = ReverseLegalLookup(Role::White, Piece::King, 20, 3, NW);
    }
    // generating for white king 20 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 25, legal: (legal white (move king i 7 j 6))
        ddi->diagonals.emplace_back(25, 289);

        this->diagonal_data[69].push_back(ddi);


        this->reverse_legal_lookup_white[289] = ReverseLegalLookup(Role::White, Piece::King, 20, 25, SE);
    }
    // generating for white king 20 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(6);
        // to position 24, legal: (legal white (move king i 7 h 6))
        ddi->diagonals.emplace_back(24, 290);

        // to position 29, legal: (legal white (move king i 7 g 5))
        ddi->diagonals.emplace_back(29, 291);

        // to position 33, legal: (legal white (move king i 7 f 4))
        ddi->diagonals.emplace_back(33, 292);

        // to position 38, legal: (legal white (move king i 7 e 3))
        ddi->diagonals.emplace_back(38, 293);

        // to position 42, legal: (legal white (move king i 7 d 2))
        ddi->diagonals.emplace_back(42, 294);

        // to position 47, legal: (legal white (move king i 7 c 1))
        ddi->diagonals.emplace_back(47, 295);

        this->diagonal_data[69].push_back(ddi);


        this->reverse_legal_lookup_white[290] = ReverseLegalLookup(Role::White, Piece::King, 20, 24, SW);
        this->reverse_legal_lookup_white[291] = ReverseLegalLookup(Role::White, Piece::King, 20, 29, SW);
        this->reverse_legal_lookup_white[292] = ReverseLegalLookup(Role::White, Piece::King, 20, 33, SW);
        this->reverse_legal_lookup_white[293] = ReverseLegalLookup(Role::White, Piece::King, 20, 38, SW);
        this->reverse_legal_lookup_white[294] = ReverseLegalLookup(Role::White, Piece::King, 20, 42, SW);
        this->reverse_legal_lookup_white[295] = ReverseLegalLookup(Role::White, Piece::King, 20, 47, SW);
    }
    // generating for white king 21 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 17, legal: (legal white (move king b 6 c 7))
        ddi->diagonals.emplace_back(17, 300);

        // to position 12, legal: (legal white (move king b 6 d 8))
        ddi->diagonals.emplace_back(12, 301);

        // to position 8, legal: (legal white (move king b 6 e 9))
        ddi->diagonals.emplace_back(8, 302);

        // to position 3, legal: (legal white (move king b 6 f 10))
        ddi->diagonals.emplace_back(3, 303);

        this->diagonal_data[70].push_back(ddi);


        this->reverse_legal_lookup_white[300] = ReverseLegalLookup(Role::White, Piece::King, 21, 17, NE);
        this->reverse_legal_lookup_white[301] = ReverseLegalLookup(Role::White, Piece::King, 21, 12, NE);
        this->reverse_legal_lookup_white[302] = ReverseLegalLookup(Role::White, Piece::King, 21, 8, NE);
        this->reverse_legal_lookup_white[303] = ReverseLegalLookup(Role::White, Piece::King, 21, 3, NE);
    }
    // generating for white king 21 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 16, legal: (legal white (move king b 6 a 7))
        ddi->diagonals.emplace_back(16, 304);

        this->diagonal_data[70].push_back(ddi);


        this->reverse_legal_lookup_white[304] = ReverseLegalLookup(Role::White, Piece::King, 21, 16, NW);
    }
    // generating for white king 21 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 27, legal: (legal white (move king b 6 c 5))
        ddi->diagonals.emplace_back(27, 305);

        // to position 32, legal: (legal white (move king b 6 d 4))
        ddi->diagonals.emplace_back(32, 306);

        // to position 38, legal: (legal white (move king b 6 e 3))
        ddi->diagonals.emplace_back(38, 307);

        // to position 43, legal: (legal white (move king b 6 f 2))
        ddi->diagonals.emplace_back(43, 308);

        // to position 49, legal: (legal white (move king b 6 g 1))
        ddi->diagonals.emplace_back(49, 309);

        this->diagonal_data[70].push_back(ddi);


        this->reverse_legal_lookup_white[305] = ReverseLegalLookup(Role::White, Piece::King, 21, 27, SE);
        this->reverse_legal_lookup_white[306] = ReverseLegalLookup(Role::White, Piece::King, 21, 32, SE);
        this->reverse_legal_lookup_white[307] = ReverseLegalLookup(Role::White, Piece::King, 21, 38, SE);
        this->reverse_legal_lookup_white[308] = ReverseLegalLookup(Role::White, Piece::King, 21, 43, SE);
        this->reverse_legal_lookup_white[309] = ReverseLegalLookup(Role::White, Piece::King, 21, 49, SE);
    }
    // generating for white king 21 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 26, legal: (legal white (move king b 6 a 5))
        ddi->diagonals.emplace_back(26, 310);

        this->diagonal_data[70].push_back(ddi);


        this->reverse_legal_lookup_white[310] = ReverseLegalLookup(Role::White, Piece::King, 21, 26, SW);
    }
    // generating for white king 22 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 18, legal: (legal white (move king d 6 e 7))
        ddi->diagonals.emplace_back(18, 317);

        // to position 13, legal: (legal white (move king d 6 f 8))
        ddi->diagonals.emplace_back(13, 318);

        // to position 9, legal: (legal white (move king d 6 g 9))
        ddi->diagonals.emplace_back(9, 319);

        // to position 4, legal: (legal white (move king d 6 h 10))
        ddi->diagonals.emplace_back(4, 320);

        this->diagonal_data[71].push_back(ddi);


        this->reverse_legal_lookup_white[317] = ReverseLegalLookup(Role::White, Piece::King, 22, 18, NE);
        this->reverse_legal_lookup_white[318] = ReverseLegalLookup(Role::White, Piece::King, 22, 13, NE);
        this->reverse_legal_lookup_white[319] = ReverseLegalLookup(Role::White, Piece::King, 22, 9, NE);
        this->reverse_legal_lookup_white[320] = ReverseLegalLookup(Role::White, Piece::King, 22, 4, NE);
    }
    // generating for white king 22 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 17, legal: (legal white (move king d 6 c 7))
        ddi->diagonals.emplace_back(17, 321);

        // to position 11, legal: (legal white (move king d 6 b 8))
        ddi->diagonals.emplace_back(11, 322);

        // to position 6, legal: (legal white (move king d 6 a 9))
        ddi->diagonals.emplace_back(6, 323);

        this->diagonal_data[71].push_back(ddi);


        this->reverse_legal_lookup_white[321] = ReverseLegalLookup(Role::White, Piece::King, 22, 17, NW);
        this->reverse_legal_lookup_white[322] = ReverseLegalLookup(Role::White, Piece::King, 22, 11, NW);
        this->reverse_legal_lookup_white[323] = ReverseLegalLookup(Role::White, Piece::King, 22, 6, NW);
    }
    // generating for white king 22 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 28, legal: (legal white (move king d 6 e 5))
        ddi->diagonals.emplace_back(28, 324);

        // to position 33, legal: (legal white (move king d 6 f 4))
        ddi->diagonals.emplace_back(33, 325);

        // to position 39, legal: (legal white (move king d 6 g 3))
        ddi->diagonals.emplace_back(39, 326);

        // to position 44, legal: (legal white (move king d 6 h 2))
        ddi->diagonals.emplace_back(44, 327);

        // to position 50, legal: (legal white (move king d 6 i 1))
        ddi->diagonals.emplace_back(50, 328);

        this->diagonal_data[71].push_back(ddi);


        this->reverse_legal_lookup_white[324] = ReverseLegalLookup(Role::White, Piece::King, 22, 28, SE);
        this->reverse_legal_lookup_white[325] = ReverseLegalLookup(Role::White, Piece::King, 22, 33, SE);
        this->reverse_legal_lookup_white[326] = ReverseLegalLookup(Role::White, Piece::King, 22, 39, SE);
        this->reverse_legal_lookup_white[327] = ReverseLegalLookup(Role::White, Piece::King, 22, 44, SE);
        this->reverse_legal_lookup_white[328] = ReverseLegalLookup(Role::White, Piece::King, 22, 50, SE);
    }
    // generating for white king 22 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 27, legal: (legal white (move king d 6 c 5))
        ddi->diagonals.emplace_back(27, 329);

        // to position 31, legal: (legal white (move king d 6 b 4))
        ddi->diagonals.emplace_back(31, 330);

        // to position 36, legal: (legal white (move king d 6 a 3))
        ddi->diagonals.emplace_back(36, 331);

        this->diagonal_data[71].push_back(ddi);


        this->reverse_legal_lookup_white[329] = ReverseLegalLookup(Role::White, Piece::King, 22, 27, SW);
        this->reverse_legal_lookup_white[330] = ReverseLegalLookup(Role::White, Piece::King, 22, 31, SW);
        this->reverse_legal_lookup_white[331] = ReverseLegalLookup(Role::White, Piece::King, 22, 36, SW);
    }
    // generating for white king 23 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 19, legal: (legal white (move king f 6 g 7))
        ddi->diagonals.emplace_back(19, 338);

        // to position 14, legal: (legal white (move king f 6 h 8))
        ddi->diagonals.emplace_back(14, 339);

        // to position 10, legal: (legal white (move king f 6 i 9))
        ddi->diagonals.emplace_back(10, 340);

        // to position 5, legal: (legal white (move king f 6 j 10))
        ddi->diagonals.emplace_back(5, 341);

        this->diagonal_data[72].push_back(ddi);


        this->reverse_legal_lookup_white[338] = ReverseLegalLookup(Role::White, Piece::King, 23, 19, NE);
        this->reverse_legal_lookup_white[339] = ReverseLegalLookup(Role::White, Piece::King, 23, 14, NE);
        this->reverse_legal_lookup_white[340] = ReverseLegalLookup(Role::White, Piece::King, 23, 10, NE);
        this->reverse_legal_lookup_white[341] = ReverseLegalLookup(Role::White, Piece::King, 23, 5, NE);
    }
    // generating for white king 23 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 18, legal: (legal white (move king f 6 e 7))
        ddi->diagonals.emplace_back(18, 342);

        // to position 12, legal: (legal white (move king f 6 d 8))
        ddi->diagonals.emplace_back(12, 343);

        // to position 7, legal: (legal white (move king f 6 c 9))
        ddi->diagonals.emplace_back(7, 344);

        // to position 1, legal: (legal white (move king f 6 b 10))
        ddi->diagonals.emplace_back(1, 345);

        this->diagonal_data[72].push_back(ddi);


        this->reverse_legal_lookup_white[342] = ReverseLegalLookup(Role::White, Piece::King, 23, 18, NW);
        this->reverse_legal_lookup_white[343] = ReverseLegalLookup(Role::White, Piece::King, 23, 12, NW);
        this->reverse_legal_lookup_white[344] = ReverseLegalLookup(Role::White, Piece::King, 23, 7, NW);
        this->reverse_legal_lookup_white[345] = ReverseLegalLookup(Role::White, Piece::King, 23, 1, NW);
    }
    // generating for white king 23 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 29, legal: (legal white (move king f 6 g 5))
        ddi->diagonals.emplace_back(29, 346);

        // to position 34, legal: (legal white (move king f 6 h 4))
        ddi->diagonals.emplace_back(34, 347);

        // to position 40, legal: (legal white (move king f 6 i 3))
        ddi->diagonals.emplace_back(40, 348);

        // to position 45, legal: (legal white (move king f 6 j 2))
        ddi->diagonals.emplace_back(45, 349);

        this->diagonal_data[72].push_back(ddi);


        this->reverse_legal_lookup_white[346] = ReverseLegalLookup(Role::White, Piece::King, 23, 29, SE);
        this->reverse_legal_lookup_white[347] = ReverseLegalLookup(Role::White, Piece::King, 23, 34, SE);
        this->reverse_legal_lookup_white[348] = ReverseLegalLookup(Role::White, Piece::King, 23, 40, SE);
        this->reverse_legal_lookup_white[349] = ReverseLegalLookup(Role::White, Piece::King, 23, 45, SE);
    }
    // generating for white king 23 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 28, legal: (legal white (move king f 6 e 5))
        ddi->diagonals.emplace_back(28, 350);

        // to position 32, legal: (legal white (move king f 6 d 4))
        ddi->diagonals.emplace_back(32, 351);

        // to position 37, legal: (legal white (move king f 6 c 3))
        ddi->diagonals.emplace_back(37, 352);

        // to position 41, legal: (legal white (move king f 6 b 2))
        ddi->diagonals.emplace_back(41, 353);

        // to position 46, legal: (legal white (move king f 6 a 1))
        ddi->diagonals.emplace_back(46, 354);

        this->diagonal_data[72].push_back(ddi);


        this->reverse_legal_lookup_white[350] = ReverseLegalLookup(Role::White, Piece::King, 23, 28, SW);
        this->reverse_legal_lookup_white[351] = ReverseLegalLookup(Role::White, Piece::King, 23, 32, SW);
        this->reverse_legal_lookup_white[352] = ReverseLegalLookup(Role::White, Piece::King, 23, 37, SW);
        this->reverse_legal_lookup_white[353] = ReverseLegalLookup(Role::White, Piece::King, 23, 41, SW);
        this->reverse_legal_lookup_white[354] = ReverseLegalLookup(Role::White, Piece::King, 23, 46, SW);
    }
    // generating for white king 24 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 20, legal: (legal white (move king h 6 i 7))
        ddi->diagonals.emplace_back(20, 361);

        // to position 15, legal: (legal white (move king h 6 j 8))
        ddi->diagonals.emplace_back(15, 362);

        this->diagonal_data[73].push_back(ddi);


        this->reverse_legal_lookup_white[361] = ReverseLegalLookup(Role::White, Piece::King, 24, 20, NE);
        this->reverse_legal_lookup_white[362] = ReverseLegalLookup(Role::White, Piece::King, 24, 15, NE);
    }
    // generating for white king 24 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 19, legal: (legal white (move king h 6 g 7))
        ddi->diagonals.emplace_back(19, 363);

        // to position 13, legal: (legal white (move king h 6 f 8))
        ddi->diagonals.emplace_back(13, 364);

        // to position 8, legal: (legal white (move king h 6 e 9))
        ddi->diagonals.emplace_back(8, 365);

        // to position 2, legal: (legal white (move king h 6 d 10))
        ddi->diagonals.emplace_back(2, 366);

        this->diagonal_data[73].push_back(ddi);


        this->reverse_legal_lookup_white[363] = ReverseLegalLookup(Role::White, Piece::King, 24, 19, NW);
        this->reverse_legal_lookup_white[364] = ReverseLegalLookup(Role::White, Piece::King, 24, 13, NW);
        this->reverse_legal_lookup_white[365] = ReverseLegalLookup(Role::White, Piece::King, 24, 8, NW);
        this->reverse_legal_lookup_white[366] = ReverseLegalLookup(Role::White, Piece::King, 24, 2, NW);
    }
    // generating for white king 24 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 30, legal: (legal white (move king h 6 i 5))
        ddi->diagonals.emplace_back(30, 367);

        // to position 35, legal: (legal white (move king h 6 j 4))
        ddi->diagonals.emplace_back(35, 368);

        this->diagonal_data[73].push_back(ddi);


        this->reverse_legal_lookup_white[367] = ReverseLegalLookup(Role::White, Piece::King, 24, 30, SE);
        this->reverse_legal_lookup_white[368] = ReverseLegalLookup(Role::White, Piece::King, 24, 35, SE);
    }
    // generating for white king 24 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 29, legal: (legal white (move king h 6 g 5))
        ddi->diagonals.emplace_back(29, 369);

        // to position 33, legal: (legal white (move king h 6 f 4))
        ddi->diagonals.emplace_back(33, 370);

        // to position 38, legal: (legal white (move king h 6 e 3))
        ddi->diagonals.emplace_back(38, 371);

        // to position 42, legal: (legal white (move king h 6 d 2))
        ddi->diagonals.emplace_back(42, 372);

        // to position 47, legal: (legal white (move king h 6 c 1))
        ddi->diagonals.emplace_back(47, 373);

        this->diagonal_data[73].push_back(ddi);


        this->reverse_legal_lookup_white[369] = ReverseLegalLookup(Role::White, Piece::King, 24, 29, SW);
        this->reverse_legal_lookup_white[370] = ReverseLegalLookup(Role::White, Piece::King, 24, 33, SW);
        this->reverse_legal_lookup_white[371] = ReverseLegalLookup(Role::White, Piece::King, 24, 38, SW);
        this->reverse_legal_lookup_white[372] = ReverseLegalLookup(Role::White, Piece::King, 24, 42, SW);
        this->reverse_legal_lookup_white[373] = ReverseLegalLookup(Role::White, Piece::King, 24, 47, SW);
    }
    // generating for white king 25 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 20, legal: (legal white (move king j 6 i 7))
        ddi->diagonals.emplace_back(20, 377);

        // to position 14, legal: (legal white (move king j 6 h 8))
        ddi->diagonals.emplace_back(14, 378);

        // to position 9, legal: (legal white (move king j 6 g 9))
        ddi->diagonals.emplace_back(9, 379);

        // to position 3, legal: (legal white (move king j 6 f 10))
        ddi->diagonals.emplace_back(3, 380);

        this->diagonal_data[74].push_back(ddi);


        this->reverse_legal_lookup_white[377] = ReverseLegalLookup(Role::White, Piece::King, 25, 20, NW);
        this->reverse_legal_lookup_white[378] = ReverseLegalLookup(Role::White, Piece::King, 25, 14, NW);
        this->reverse_legal_lookup_white[379] = ReverseLegalLookup(Role::White, Piece::King, 25, 9, NW);
        this->reverse_legal_lookup_white[380] = ReverseLegalLookup(Role::White, Piece::King, 25, 3, NW);
    }
    // generating for white king 25 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 30, legal: (legal white (move king j 6 i 5))
        ddi->diagonals.emplace_back(30, 381);

        // to position 34, legal: (legal white (move king j 6 h 4))
        ddi->diagonals.emplace_back(34, 382);

        // to position 39, legal: (legal white (move king j 6 g 3))
        ddi->diagonals.emplace_back(39, 383);

        // to position 43, legal: (legal white (move king j 6 f 2))
        ddi->diagonals.emplace_back(43, 384);

        // to position 48, legal: (legal white (move king j 6 e 1))
        ddi->diagonals.emplace_back(48, 385);

        this->diagonal_data[74].push_back(ddi);


        this->reverse_legal_lookup_white[381] = ReverseLegalLookup(Role::White, Piece::King, 25, 30, SW);
        this->reverse_legal_lookup_white[382] = ReverseLegalLookup(Role::White, Piece::King, 25, 34, SW);
        this->reverse_legal_lookup_white[383] = ReverseLegalLookup(Role::White, Piece::King, 25, 39, SW);
        this->reverse_legal_lookup_white[384] = ReverseLegalLookup(Role::White, Piece::King, 25, 43, SW);
        this->reverse_legal_lookup_white[385] = ReverseLegalLookup(Role::White, Piece::King, 25, 48, SW);
    }
    // generating for white king 26 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 21, legal: (legal white (move king a 5 b 6))
        ddi->diagonals.emplace_back(21, 389);

        // to position 17, legal: (legal white (move king a 5 c 7))
        ddi->diagonals.emplace_back(17, 390);

        // to position 12, legal: (legal white (move king a 5 d 8))
        ddi->diagonals.emplace_back(12, 391);

        // to position 8, legal: (legal white (move king a 5 e 9))
        ddi->diagonals.emplace_back(8, 392);

        // to position 3, legal: (legal white (move king a 5 f 10))
        ddi->diagonals.emplace_back(3, 393);

        this->diagonal_data[75].push_back(ddi);


        this->reverse_legal_lookup_white[389] = ReverseLegalLookup(Role::White, Piece::King, 26, 21, NE);
        this->reverse_legal_lookup_white[390] = ReverseLegalLookup(Role::White, Piece::King, 26, 17, NE);
        this->reverse_legal_lookup_white[391] = ReverseLegalLookup(Role::White, Piece::King, 26, 12, NE);
        this->reverse_legal_lookup_white[392] = ReverseLegalLookup(Role::White, Piece::King, 26, 8, NE);
        this->reverse_legal_lookup_white[393] = ReverseLegalLookup(Role::White, Piece::King, 26, 3, NE);
    }
    // generating for white king 26 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 31, legal: (legal white (move king a 5 b 4))
        ddi->diagonals.emplace_back(31, 394);

        // to position 37, legal: (legal white (move king a 5 c 3))
        ddi->diagonals.emplace_back(37, 395);

        // to position 42, legal: (legal white (move king a 5 d 2))
        ddi->diagonals.emplace_back(42, 396);

        // to position 48, legal: (legal white (move king a 5 e 1))
        ddi->diagonals.emplace_back(48, 397);

        this->diagonal_data[75].push_back(ddi);


        this->reverse_legal_lookup_white[394] = ReverseLegalLookup(Role::White, Piece::King, 26, 31, SE);
        this->reverse_legal_lookup_white[395] = ReverseLegalLookup(Role::White, Piece::King, 26, 37, SE);
        this->reverse_legal_lookup_white[396] = ReverseLegalLookup(Role::White, Piece::King, 26, 42, SE);
        this->reverse_legal_lookup_white[397] = ReverseLegalLookup(Role::White, Piece::King, 26, 48, SE);
    }
    // generating for white king 27 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 22, legal: (legal white (move king c 5 d 6))
        ddi->diagonals.emplace_back(22, 404);

        // to position 18, legal: (legal white (move king c 5 e 7))
        ddi->diagonals.emplace_back(18, 405);

        // to position 13, legal: (legal white (move king c 5 f 8))
        ddi->diagonals.emplace_back(13, 406);

        // to position 9, legal: (legal white (move king c 5 g 9))
        ddi->diagonals.emplace_back(9, 407);

        // to position 4, legal: (legal white (move king c 5 h 10))
        ddi->diagonals.emplace_back(4, 408);

        this->diagonal_data[76].push_back(ddi);


        this->reverse_legal_lookup_white[404] = ReverseLegalLookup(Role::White, Piece::King, 27, 22, NE);
        this->reverse_legal_lookup_white[405] = ReverseLegalLookup(Role::White, Piece::King, 27, 18, NE);
        this->reverse_legal_lookup_white[406] = ReverseLegalLookup(Role::White, Piece::King, 27, 13, NE);
        this->reverse_legal_lookup_white[407] = ReverseLegalLookup(Role::White, Piece::King, 27, 9, NE);
        this->reverse_legal_lookup_white[408] = ReverseLegalLookup(Role::White, Piece::King, 27, 4, NE);
    }
    // generating for white king 27 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 21, legal: (legal white (move king c 5 b 6))
        ddi->diagonals.emplace_back(21, 409);

        // to position 16, legal: (legal white (move king c 5 a 7))
        ddi->diagonals.emplace_back(16, 410);

        this->diagonal_data[76].push_back(ddi);


        this->reverse_legal_lookup_white[409] = ReverseLegalLookup(Role::White, Piece::King, 27, 21, NW);
        this->reverse_legal_lookup_white[410] = ReverseLegalLookup(Role::White, Piece::King, 27, 16, NW);
    }
    // generating for white king 27 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 32, legal: (legal white (move king c 5 d 4))
        ddi->diagonals.emplace_back(32, 411);

        // to position 38, legal: (legal white (move king c 5 e 3))
        ddi->diagonals.emplace_back(38, 412);

        // to position 43, legal: (legal white (move king c 5 f 2))
        ddi->diagonals.emplace_back(43, 413);

        // to position 49, legal: (legal white (move king c 5 g 1))
        ddi->diagonals.emplace_back(49, 414);

        this->diagonal_data[76].push_back(ddi);


        this->reverse_legal_lookup_white[411] = ReverseLegalLookup(Role::White, Piece::King, 27, 32, SE);
        this->reverse_legal_lookup_white[412] = ReverseLegalLookup(Role::White, Piece::King, 27, 38, SE);
        this->reverse_legal_lookup_white[413] = ReverseLegalLookup(Role::White, Piece::King, 27, 43, SE);
        this->reverse_legal_lookup_white[414] = ReverseLegalLookup(Role::White, Piece::King, 27, 49, SE);
    }
    // generating for white king 27 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 31, legal: (legal white (move king c 5 b 4))
        ddi->diagonals.emplace_back(31, 415);

        // to position 36, legal: (legal white (move king c 5 a 3))
        ddi->diagonals.emplace_back(36, 416);

        this->diagonal_data[76].push_back(ddi);


        this->reverse_legal_lookup_white[415] = ReverseLegalLookup(Role::White, Piece::King, 27, 31, SW);
        this->reverse_legal_lookup_white[416] = ReverseLegalLookup(Role::White, Piece::King, 27, 36, SW);
    }
    // generating for white king 28 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 23, legal: (legal white (move king e 5 f 6))
        ddi->diagonals.emplace_back(23, 423);

        // to position 19, legal: (legal white (move king e 5 g 7))
        ddi->diagonals.emplace_back(19, 424);

        // to position 14, legal: (legal white (move king e 5 h 8))
        ddi->diagonals.emplace_back(14, 425);

        // to position 10, legal: (legal white (move king e 5 i 9))
        ddi->diagonals.emplace_back(10, 426);

        // to position 5, legal: (legal white (move king e 5 j 10))
        ddi->diagonals.emplace_back(5, 427);

        this->diagonal_data[77].push_back(ddi);


        this->reverse_legal_lookup_white[423] = ReverseLegalLookup(Role::White, Piece::King, 28, 23, NE);
        this->reverse_legal_lookup_white[424] = ReverseLegalLookup(Role::White, Piece::King, 28, 19, NE);
        this->reverse_legal_lookup_white[425] = ReverseLegalLookup(Role::White, Piece::King, 28, 14, NE);
        this->reverse_legal_lookup_white[426] = ReverseLegalLookup(Role::White, Piece::King, 28, 10, NE);
        this->reverse_legal_lookup_white[427] = ReverseLegalLookup(Role::White, Piece::King, 28, 5, NE);
    }
    // generating for white king 28 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 22, legal: (legal white (move king e 5 d 6))
        ddi->diagonals.emplace_back(22, 428);

        // to position 17, legal: (legal white (move king e 5 c 7))
        ddi->diagonals.emplace_back(17, 429);

        // to position 11, legal: (legal white (move king e 5 b 8))
        ddi->diagonals.emplace_back(11, 430);

        // to position 6, legal: (legal white (move king e 5 a 9))
        ddi->diagonals.emplace_back(6, 431);

        this->diagonal_data[77].push_back(ddi);


        this->reverse_legal_lookup_white[428] = ReverseLegalLookup(Role::White, Piece::King, 28, 22, NW);
        this->reverse_legal_lookup_white[429] = ReverseLegalLookup(Role::White, Piece::King, 28, 17, NW);
        this->reverse_legal_lookup_white[430] = ReverseLegalLookup(Role::White, Piece::King, 28, 11, NW);
        this->reverse_legal_lookup_white[431] = ReverseLegalLookup(Role::White, Piece::King, 28, 6, NW);
    }
    // generating for white king 28 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 33, legal: (legal white (move king e 5 f 4))
        ddi->diagonals.emplace_back(33, 432);

        // to position 39, legal: (legal white (move king e 5 g 3))
        ddi->diagonals.emplace_back(39, 433);

        // to position 44, legal: (legal white (move king e 5 h 2))
        ddi->diagonals.emplace_back(44, 434);

        // to position 50, legal: (legal white (move king e 5 i 1))
        ddi->diagonals.emplace_back(50, 435);

        this->diagonal_data[77].push_back(ddi);


        this->reverse_legal_lookup_white[432] = ReverseLegalLookup(Role::White, Piece::King, 28, 33, SE);
        this->reverse_legal_lookup_white[433] = ReverseLegalLookup(Role::White, Piece::King, 28, 39, SE);
        this->reverse_legal_lookup_white[434] = ReverseLegalLookup(Role::White, Piece::King, 28, 44, SE);
        this->reverse_legal_lookup_white[435] = ReverseLegalLookup(Role::White, Piece::King, 28, 50, SE);
    }
    // generating for white king 28 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 32, legal: (legal white (move king e 5 d 4))
        ddi->diagonals.emplace_back(32, 436);

        // to position 37, legal: (legal white (move king e 5 c 3))
        ddi->diagonals.emplace_back(37, 437);

        // to position 41, legal: (legal white (move king e 5 b 2))
        ddi->diagonals.emplace_back(41, 438);

        // to position 46, legal: (legal white (move king e 5 a 1))
        ddi->diagonals.emplace_back(46, 439);

        this->diagonal_data[77].push_back(ddi);


        this->reverse_legal_lookup_white[436] = ReverseLegalLookup(Role::White, Piece::King, 28, 32, SW);
        this->reverse_legal_lookup_white[437] = ReverseLegalLookup(Role::White, Piece::King, 28, 37, SW);
        this->reverse_legal_lookup_white[438] = ReverseLegalLookup(Role::White, Piece::King, 28, 41, SW);
        this->reverse_legal_lookup_white[439] = ReverseLegalLookup(Role::White, Piece::King, 28, 46, SW);
    }
    // generating for white king 29 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 24, legal: (legal white (move king g 5 h 6))
        ddi->diagonals.emplace_back(24, 446);

        // to position 20, legal: (legal white (move king g 5 i 7))
        ddi->diagonals.emplace_back(20, 447);

        // to position 15, legal: (legal white (move king g 5 j 8))
        ddi->diagonals.emplace_back(15, 448);

        this->diagonal_data[78].push_back(ddi);


        this->reverse_legal_lookup_white[446] = ReverseLegalLookup(Role::White, Piece::King, 29, 24, NE);
        this->reverse_legal_lookup_white[447] = ReverseLegalLookup(Role::White, Piece::King, 29, 20, NE);
        this->reverse_legal_lookup_white[448] = ReverseLegalLookup(Role::White, Piece::King, 29, 15, NE);
    }
    // generating for white king 29 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 23, legal: (legal white (move king g 5 f 6))
        ddi->diagonals.emplace_back(23, 449);

        // to position 18, legal: (legal white (move king g 5 e 7))
        ddi->diagonals.emplace_back(18, 450);

        // to position 12, legal: (legal white (move king g 5 d 8))
        ddi->diagonals.emplace_back(12, 451);

        // to position 7, legal: (legal white (move king g 5 c 9))
        ddi->diagonals.emplace_back(7, 452);

        // to position 1, legal: (legal white (move king g 5 b 10))
        ddi->diagonals.emplace_back(1, 453);

        this->diagonal_data[78].push_back(ddi);


        this->reverse_legal_lookup_white[449] = ReverseLegalLookup(Role::White, Piece::King, 29, 23, NW);
        this->reverse_legal_lookup_white[450] = ReverseLegalLookup(Role::White, Piece::King, 29, 18, NW);
        this->reverse_legal_lookup_white[451] = ReverseLegalLookup(Role::White, Piece::King, 29, 12, NW);
        this->reverse_legal_lookup_white[452] = ReverseLegalLookup(Role::White, Piece::King, 29, 7, NW);
        this->reverse_legal_lookup_white[453] = ReverseLegalLookup(Role::White, Piece::King, 29, 1, NW);
    }
    // generating for white king 29 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 34, legal: (legal white (move king g 5 h 4))
        ddi->diagonals.emplace_back(34, 454);

        // to position 40, legal: (legal white (move king g 5 i 3))
        ddi->diagonals.emplace_back(40, 455);

        // to position 45, legal: (legal white (move king g 5 j 2))
        ddi->diagonals.emplace_back(45, 456);

        this->diagonal_data[78].push_back(ddi);


        this->reverse_legal_lookup_white[454] = ReverseLegalLookup(Role::White, Piece::King, 29, 34, SE);
        this->reverse_legal_lookup_white[455] = ReverseLegalLookup(Role::White, Piece::King, 29, 40, SE);
        this->reverse_legal_lookup_white[456] = ReverseLegalLookup(Role::White, Piece::King, 29, 45, SE);
    }
    // generating for white king 29 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 33, legal: (legal white (move king g 5 f 4))
        ddi->diagonals.emplace_back(33, 457);

        // to position 38, legal: (legal white (move king g 5 e 3))
        ddi->diagonals.emplace_back(38, 458);

        // to position 42, legal: (legal white (move king g 5 d 2))
        ddi->diagonals.emplace_back(42, 459);

        // to position 47, legal: (legal white (move king g 5 c 1))
        ddi->diagonals.emplace_back(47, 460);

        this->diagonal_data[78].push_back(ddi);


        this->reverse_legal_lookup_white[457] = ReverseLegalLookup(Role::White, Piece::King, 29, 33, SW);
        this->reverse_legal_lookup_white[458] = ReverseLegalLookup(Role::White, Piece::King, 29, 38, SW);
        this->reverse_legal_lookup_white[459] = ReverseLegalLookup(Role::White, Piece::King, 29, 42, SW);
        this->reverse_legal_lookup_white[460] = ReverseLegalLookup(Role::White, Piece::King, 29, 47, SW);
    }
    // generating for white king 30 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 25, legal: (legal white (move king i 5 j 6))
        ddi->diagonals.emplace_back(25, 465);

        this->diagonal_data[79].push_back(ddi);


        this->reverse_legal_lookup_white[465] = ReverseLegalLookup(Role::White, Piece::King, 30, 25, NE);
    }
    // generating for white king 30 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 24, legal: (legal white (move king i 5 h 6))
        ddi->diagonals.emplace_back(24, 466);

        // to position 19, legal: (legal white (move king i 5 g 7))
        ddi->diagonals.emplace_back(19, 467);

        // to position 13, legal: (legal white (move king i 5 f 8))
        ddi->diagonals.emplace_back(13, 468);

        // to position 8, legal: (legal white (move king i 5 e 9))
        ddi->diagonals.emplace_back(8, 469);

        // to position 2, legal: (legal white (move king i 5 d 10))
        ddi->diagonals.emplace_back(2, 470);

        this->diagonal_data[79].push_back(ddi);


        this->reverse_legal_lookup_white[466] = ReverseLegalLookup(Role::White, Piece::King, 30, 24, NW);
        this->reverse_legal_lookup_white[467] = ReverseLegalLookup(Role::White, Piece::King, 30, 19, NW);
        this->reverse_legal_lookup_white[468] = ReverseLegalLookup(Role::White, Piece::King, 30, 13, NW);
        this->reverse_legal_lookup_white[469] = ReverseLegalLookup(Role::White, Piece::King, 30, 8, NW);
        this->reverse_legal_lookup_white[470] = ReverseLegalLookup(Role::White, Piece::King, 30, 2, NW);
    }
    // generating for white king 30 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 35, legal: (legal white (move king i 5 j 4))
        ddi->diagonals.emplace_back(35, 471);

        this->diagonal_data[79].push_back(ddi);


        this->reverse_legal_lookup_white[471] = ReverseLegalLookup(Role::White, Piece::King, 30, 35, SE);
    }
    // generating for white king 30 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 34, legal: (legal white (move king i 5 h 4))
        ddi->diagonals.emplace_back(34, 472);

        // to position 39, legal: (legal white (move king i 5 g 3))
        ddi->diagonals.emplace_back(39, 473);

        // to position 43, legal: (legal white (move king i 5 f 2))
        ddi->diagonals.emplace_back(43, 474);

        // to position 48, legal: (legal white (move king i 5 e 1))
        ddi->diagonals.emplace_back(48, 475);

        this->diagonal_data[79].push_back(ddi);


        this->reverse_legal_lookup_white[472] = ReverseLegalLookup(Role::White, Piece::King, 30, 34, SW);
        this->reverse_legal_lookup_white[473] = ReverseLegalLookup(Role::White, Piece::King, 30, 39, SW);
        this->reverse_legal_lookup_white[474] = ReverseLegalLookup(Role::White, Piece::King, 30, 43, SW);
        this->reverse_legal_lookup_white[475] = ReverseLegalLookup(Role::White, Piece::King, 30, 48, SW);
    }
    // generating for white king 31 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(6);
        // to position 27, legal: (legal white (move king b 4 c 5))
        ddi->diagonals.emplace_back(27, 480);

        // to position 22, legal: (legal white (move king b 4 d 6))
        ddi->diagonals.emplace_back(22, 481);

        // to position 18, legal: (legal white (move king b 4 e 7))
        ddi->diagonals.emplace_back(18, 482);

        // to position 13, legal: (legal white (move king b 4 f 8))
        ddi->diagonals.emplace_back(13, 483);

        // to position 9, legal: (legal white (move king b 4 g 9))
        ddi->diagonals.emplace_back(9, 484);

        // to position 4, legal: (legal white (move king b 4 h 10))
        ddi->diagonals.emplace_back(4, 485);

        this->diagonal_data[80].push_back(ddi);


        this->reverse_legal_lookup_white[480] = ReverseLegalLookup(Role::White, Piece::King, 31, 27, NE);
        this->reverse_legal_lookup_white[481] = ReverseLegalLookup(Role::White, Piece::King, 31, 22, NE);
        this->reverse_legal_lookup_white[482] = ReverseLegalLookup(Role::White, Piece::King, 31, 18, NE);
        this->reverse_legal_lookup_white[483] = ReverseLegalLookup(Role::White, Piece::King, 31, 13, NE);
        this->reverse_legal_lookup_white[484] = ReverseLegalLookup(Role::White, Piece::King, 31, 9, NE);
        this->reverse_legal_lookup_white[485] = ReverseLegalLookup(Role::White, Piece::King, 31, 4, NE);
    }
    // generating for white king 31 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 26, legal: (legal white (move king b 4 a 5))
        ddi->diagonals.emplace_back(26, 486);

        this->diagonal_data[80].push_back(ddi);


        this->reverse_legal_lookup_white[486] = ReverseLegalLookup(Role::White, Piece::King, 31, 26, NW);
    }
    // generating for white king 31 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 37, legal: (legal white (move king b 4 c 3))
        ddi->diagonals.emplace_back(37, 487);

        // to position 42, legal: (legal white (move king b 4 d 2))
        ddi->diagonals.emplace_back(42, 488);

        // to position 48, legal: (legal white (move king b 4 e 1))
        ddi->diagonals.emplace_back(48, 489);

        this->diagonal_data[80].push_back(ddi);


        this->reverse_legal_lookup_white[487] = ReverseLegalLookup(Role::White, Piece::King, 31, 37, SE);
        this->reverse_legal_lookup_white[488] = ReverseLegalLookup(Role::White, Piece::King, 31, 42, SE);
        this->reverse_legal_lookup_white[489] = ReverseLegalLookup(Role::White, Piece::King, 31, 48, SE);
    }
    // generating for white king 31 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 36, legal: (legal white (move king b 4 a 3))
        ddi->diagonals.emplace_back(36, 490);

        this->diagonal_data[80].push_back(ddi);


        this->reverse_legal_lookup_white[490] = ReverseLegalLookup(Role::White, Piece::King, 31, 36, SW);
    }
    // generating for white king 32 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(6);
        // to position 28, legal: (legal white (move king d 4 e 5))
        ddi->diagonals.emplace_back(28, 497);

        // to position 23, legal: (legal white (move king d 4 f 6))
        ddi->diagonals.emplace_back(23, 498);

        // to position 19, legal: (legal white (move king d 4 g 7))
        ddi->diagonals.emplace_back(19, 499);

        // to position 14, legal: (legal white (move king d 4 h 8))
        ddi->diagonals.emplace_back(14, 500);

        // to position 10, legal: (legal white (move king d 4 i 9))
        ddi->diagonals.emplace_back(10, 501);

        // to position 5, legal: (legal white (move king d 4 j 10))
        ddi->diagonals.emplace_back(5, 502);

        this->diagonal_data[81].push_back(ddi);


        this->reverse_legal_lookup_white[497] = ReverseLegalLookup(Role::White, Piece::King, 32, 28, NE);
        this->reverse_legal_lookup_white[498] = ReverseLegalLookup(Role::White, Piece::King, 32, 23, NE);
        this->reverse_legal_lookup_white[499] = ReverseLegalLookup(Role::White, Piece::King, 32, 19, NE);
        this->reverse_legal_lookup_white[500] = ReverseLegalLookup(Role::White, Piece::King, 32, 14, NE);
        this->reverse_legal_lookup_white[501] = ReverseLegalLookup(Role::White, Piece::King, 32, 10, NE);
        this->reverse_legal_lookup_white[502] = ReverseLegalLookup(Role::White, Piece::King, 32, 5, NE);
    }
    // generating for white king 32 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 27, legal: (legal white (move king d 4 c 5))
        ddi->diagonals.emplace_back(27, 503);

        // to position 21, legal: (legal white (move king d 4 b 6))
        ddi->diagonals.emplace_back(21, 504);

        // to position 16, legal: (legal white (move king d 4 a 7))
        ddi->diagonals.emplace_back(16, 505);

        this->diagonal_data[81].push_back(ddi);


        this->reverse_legal_lookup_white[503] = ReverseLegalLookup(Role::White, Piece::King, 32, 27, NW);
        this->reverse_legal_lookup_white[504] = ReverseLegalLookup(Role::White, Piece::King, 32, 21, NW);
        this->reverse_legal_lookup_white[505] = ReverseLegalLookup(Role::White, Piece::King, 32, 16, NW);
    }
    // generating for white king 32 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 38, legal: (legal white (move king d 4 e 3))
        ddi->diagonals.emplace_back(38, 506);

        // to position 43, legal: (legal white (move king d 4 f 2))
        ddi->diagonals.emplace_back(43, 507);

        // to position 49, legal: (legal white (move king d 4 g 1))
        ddi->diagonals.emplace_back(49, 508);

        this->diagonal_data[81].push_back(ddi);


        this->reverse_legal_lookup_white[506] = ReverseLegalLookup(Role::White, Piece::King, 32, 38, SE);
        this->reverse_legal_lookup_white[507] = ReverseLegalLookup(Role::White, Piece::King, 32, 43, SE);
        this->reverse_legal_lookup_white[508] = ReverseLegalLookup(Role::White, Piece::King, 32, 49, SE);
    }
    // generating for white king 32 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 37, legal: (legal white (move king d 4 c 3))
        ddi->diagonals.emplace_back(37, 509);

        // to position 41, legal: (legal white (move king d 4 b 2))
        ddi->diagonals.emplace_back(41, 510);

        // to position 46, legal: (legal white (move king d 4 a 1))
        ddi->diagonals.emplace_back(46, 511);

        this->diagonal_data[81].push_back(ddi);


        this->reverse_legal_lookup_white[509] = ReverseLegalLookup(Role::White, Piece::King, 32, 37, SW);
        this->reverse_legal_lookup_white[510] = ReverseLegalLookup(Role::White, Piece::King, 32, 41, SW);
        this->reverse_legal_lookup_white[511] = ReverseLegalLookup(Role::White, Piece::King, 32, 46, SW);
    }
    // generating for white king 33 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 29, legal: (legal white (move king f 4 g 5))
        ddi->diagonals.emplace_back(29, 518);

        // to position 24, legal: (legal white (move king f 4 h 6))
        ddi->diagonals.emplace_back(24, 519);

        // to position 20, legal: (legal white (move king f 4 i 7))
        ddi->diagonals.emplace_back(20, 520);

        // to position 15, legal: (legal white (move king f 4 j 8))
        ddi->diagonals.emplace_back(15, 521);

        this->diagonal_data[82].push_back(ddi);


        this->reverse_legal_lookup_white[518] = ReverseLegalLookup(Role::White, Piece::King, 33, 29, NE);
        this->reverse_legal_lookup_white[519] = ReverseLegalLookup(Role::White, Piece::King, 33, 24, NE);
        this->reverse_legal_lookup_white[520] = ReverseLegalLookup(Role::White, Piece::King, 33, 20, NE);
        this->reverse_legal_lookup_white[521] = ReverseLegalLookup(Role::White, Piece::King, 33, 15, NE);
    }
    // generating for white king 33 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 28, legal: (legal white (move king f 4 e 5))
        ddi->diagonals.emplace_back(28, 522);

        // to position 22, legal: (legal white (move king f 4 d 6))
        ddi->diagonals.emplace_back(22, 523);

        // to position 17, legal: (legal white (move king f 4 c 7))
        ddi->diagonals.emplace_back(17, 524);

        // to position 11, legal: (legal white (move king f 4 b 8))
        ddi->diagonals.emplace_back(11, 525);

        // to position 6, legal: (legal white (move king f 4 a 9))
        ddi->diagonals.emplace_back(6, 526);

        this->diagonal_data[82].push_back(ddi);


        this->reverse_legal_lookup_white[522] = ReverseLegalLookup(Role::White, Piece::King, 33, 28, NW);
        this->reverse_legal_lookup_white[523] = ReverseLegalLookup(Role::White, Piece::King, 33, 22, NW);
        this->reverse_legal_lookup_white[524] = ReverseLegalLookup(Role::White, Piece::King, 33, 17, NW);
        this->reverse_legal_lookup_white[525] = ReverseLegalLookup(Role::White, Piece::King, 33, 11, NW);
        this->reverse_legal_lookup_white[526] = ReverseLegalLookup(Role::White, Piece::King, 33, 6, NW);
    }
    // generating for white king 33 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 39, legal: (legal white (move king f 4 g 3))
        ddi->diagonals.emplace_back(39, 527);

        // to position 44, legal: (legal white (move king f 4 h 2))
        ddi->diagonals.emplace_back(44, 528);

        // to position 50, legal: (legal white (move king f 4 i 1))
        ddi->diagonals.emplace_back(50, 529);

        this->diagonal_data[82].push_back(ddi);


        this->reverse_legal_lookup_white[527] = ReverseLegalLookup(Role::White, Piece::King, 33, 39, SE);
        this->reverse_legal_lookup_white[528] = ReverseLegalLookup(Role::White, Piece::King, 33, 44, SE);
        this->reverse_legal_lookup_white[529] = ReverseLegalLookup(Role::White, Piece::King, 33, 50, SE);
    }
    // generating for white king 33 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 38, legal: (legal white (move king f 4 e 3))
        ddi->diagonals.emplace_back(38, 530);

        // to position 42, legal: (legal white (move king f 4 d 2))
        ddi->diagonals.emplace_back(42, 531);

        // to position 47, legal: (legal white (move king f 4 c 1))
        ddi->diagonals.emplace_back(47, 532);

        this->diagonal_data[82].push_back(ddi);


        this->reverse_legal_lookup_white[530] = ReverseLegalLookup(Role::White, Piece::King, 33, 38, SW);
        this->reverse_legal_lookup_white[531] = ReverseLegalLookup(Role::White, Piece::King, 33, 42, SW);
        this->reverse_legal_lookup_white[532] = ReverseLegalLookup(Role::White, Piece::King, 33, 47, SW);
    }
    // generating for white king 34 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 30, legal: (legal white (move king h 4 i 5))
        ddi->diagonals.emplace_back(30, 539);

        // to position 25, legal: (legal white (move king h 4 j 6))
        ddi->diagonals.emplace_back(25, 540);

        this->diagonal_data[83].push_back(ddi);


        this->reverse_legal_lookup_white[539] = ReverseLegalLookup(Role::White, Piece::King, 34, 30, NE);
        this->reverse_legal_lookup_white[540] = ReverseLegalLookup(Role::White, Piece::King, 34, 25, NE);
    }
    // generating for white king 34 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 29, legal: (legal white (move king h 4 g 5))
        ddi->diagonals.emplace_back(29, 541);

        // to position 23, legal: (legal white (move king h 4 f 6))
        ddi->diagonals.emplace_back(23, 542);

        // to position 18, legal: (legal white (move king h 4 e 7))
        ddi->diagonals.emplace_back(18, 543);

        // to position 12, legal: (legal white (move king h 4 d 8))
        ddi->diagonals.emplace_back(12, 544);

        // to position 7, legal: (legal white (move king h 4 c 9))
        ddi->diagonals.emplace_back(7, 545);

        // to position 1, legal: (legal white (move king h 4 b 10))
        ddi->diagonals.emplace_back(1, 546);

        this->diagonal_data[83].push_back(ddi);


        this->reverse_legal_lookup_white[541] = ReverseLegalLookup(Role::White, Piece::King, 34, 29, NW);
        this->reverse_legal_lookup_white[542] = ReverseLegalLookup(Role::White, Piece::King, 34, 23, NW);
        this->reverse_legal_lookup_white[543] = ReverseLegalLookup(Role::White, Piece::King, 34, 18, NW);
        this->reverse_legal_lookup_white[544] = ReverseLegalLookup(Role::White, Piece::King, 34, 12, NW);
        this->reverse_legal_lookup_white[545] = ReverseLegalLookup(Role::White, Piece::King, 34, 7, NW);
        this->reverse_legal_lookup_white[546] = ReverseLegalLookup(Role::White, Piece::King, 34, 1, NW);
    }
    // generating for white king 34 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 40, legal: (legal white (move king h 4 i 3))
        ddi->diagonals.emplace_back(40, 547);

        // to position 45, legal: (legal white (move king h 4 j 2))
        ddi->diagonals.emplace_back(45, 548);

        this->diagonal_data[83].push_back(ddi);


        this->reverse_legal_lookup_white[547] = ReverseLegalLookup(Role::White, Piece::King, 34, 40, SE);
        this->reverse_legal_lookup_white[548] = ReverseLegalLookup(Role::White, Piece::King, 34, 45, SE);
    }
    // generating for white king 34 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 39, legal: (legal white (move king h 4 g 3))
        ddi->diagonals.emplace_back(39, 549);

        // to position 43, legal: (legal white (move king h 4 f 2))
        ddi->diagonals.emplace_back(43, 550);

        // to position 48, legal: (legal white (move king h 4 e 1))
        ddi->diagonals.emplace_back(48, 551);

        this->diagonal_data[83].push_back(ddi);


        this->reverse_legal_lookup_white[549] = ReverseLegalLookup(Role::White, Piece::King, 34, 39, SW);
        this->reverse_legal_lookup_white[550] = ReverseLegalLookup(Role::White, Piece::King, 34, 43, SW);
        this->reverse_legal_lookup_white[551] = ReverseLegalLookup(Role::White, Piece::King, 34, 48, SW);
    }
    // generating for white king 35 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 30, legal: (legal white (move king j 4 i 5))
        ddi->diagonals.emplace_back(30, 555);

        // to position 24, legal: (legal white (move king j 4 h 6))
        ddi->diagonals.emplace_back(24, 556);

        // to position 19, legal: (legal white (move king j 4 g 7))
        ddi->diagonals.emplace_back(19, 557);

        // to position 13, legal: (legal white (move king j 4 f 8))
        ddi->diagonals.emplace_back(13, 558);

        // to position 8, legal: (legal white (move king j 4 e 9))
        ddi->diagonals.emplace_back(8, 559);

        // to position 2, legal: (legal white (move king j 4 d 10))
        ddi->diagonals.emplace_back(2, 560);

        this->diagonal_data[84].push_back(ddi);


        this->reverse_legal_lookup_white[555] = ReverseLegalLookup(Role::White, Piece::King, 35, 30, NW);
        this->reverse_legal_lookup_white[556] = ReverseLegalLookup(Role::White, Piece::King, 35, 24, NW);
        this->reverse_legal_lookup_white[557] = ReverseLegalLookup(Role::White, Piece::King, 35, 19, NW);
        this->reverse_legal_lookup_white[558] = ReverseLegalLookup(Role::White, Piece::King, 35, 13, NW);
        this->reverse_legal_lookup_white[559] = ReverseLegalLookup(Role::White, Piece::King, 35, 8, NW);
        this->reverse_legal_lookup_white[560] = ReverseLegalLookup(Role::White, Piece::King, 35, 2, NW);
    }
    // generating for white king 35 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 40, legal: (legal white (move king j 4 i 3))
        ddi->diagonals.emplace_back(40, 561);

        // to position 44, legal: (legal white (move king j 4 h 2))
        ddi->diagonals.emplace_back(44, 562);

        // to position 49, legal: (legal white (move king j 4 g 1))
        ddi->diagonals.emplace_back(49, 563);

        this->diagonal_data[84].push_back(ddi);


        this->reverse_legal_lookup_white[561] = ReverseLegalLookup(Role::White, Piece::King, 35, 40, SW);
        this->reverse_legal_lookup_white[562] = ReverseLegalLookup(Role::White, Piece::King, 35, 44, SW);
        this->reverse_legal_lookup_white[563] = ReverseLegalLookup(Role::White, Piece::King, 35, 49, SW);
    }
    // generating for white king 36 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(7);
        // to position 31, legal: (legal white (move king a 3 b 4))
        ddi->diagonals.emplace_back(31, 567);

        // to position 27, legal: (legal white (move king a 3 c 5))
        ddi->diagonals.emplace_back(27, 568);

        // to position 22, legal: (legal white (move king a 3 d 6))
        ddi->diagonals.emplace_back(22, 569);

        // to position 18, legal: (legal white (move king a 3 e 7))
        ddi->diagonals.emplace_back(18, 570);

        // to position 13, legal: (legal white (move king a 3 f 8))
        ddi->diagonals.emplace_back(13, 571);

        // to position 9, legal: (legal white (move king a 3 g 9))
        ddi->diagonals.emplace_back(9, 572);

        // to position 4, legal: (legal white (move king a 3 h 10))
        ddi->diagonals.emplace_back(4, 573);

        this->diagonal_data[85].push_back(ddi);


        this->reverse_legal_lookup_white[567] = ReverseLegalLookup(Role::White, Piece::King, 36, 31, NE);
        this->reverse_legal_lookup_white[568] = ReverseLegalLookup(Role::White, Piece::King, 36, 27, NE);
        this->reverse_legal_lookup_white[569] = ReverseLegalLookup(Role::White, Piece::King, 36, 22, NE);
        this->reverse_legal_lookup_white[570] = ReverseLegalLookup(Role::White, Piece::King, 36, 18, NE);
        this->reverse_legal_lookup_white[571] = ReverseLegalLookup(Role::White, Piece::King, 36, 13, NE);
        this->reverse_legal_lookup_white[572] = ReverseLegalLookup(Role::White, Piece::King, 36, 9, NE);
        this->reverse_legal_lookup_white[573] = ReverseLegalLookup(Role::White, Piece::King, 36, 4, NE);
    }
    // generating for white king 36 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 41, legal: (legal white (move king a 3 b 2))
        ddi->diagonals.emplace_back(41, 574);

        // to position 47, legal: (legal white (move king a 3 c 1))
        ddi->diagonals.emplace_back(47, 575);

        this->diagonal_data[85].push_back(ddi);


        this->reverse_legal_lookup_white[574] = ReverseLegalLookup(Role::White, Piece::King, 36, 41, SE);
        this->reverse_legal_lookup_white[575] = ReverseLegalLookup(Role::White, Piece::King, 36, 47, SE);
    }
    // generating for white king 37 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(7);
        // to position 32, legal: (legal white (move king c 3 d 4))
        ddi->diagonals.emplace_back(32, 582);

        // to position 28, legal: (legal white (move king c 3 e 5))
        ddi->diagonals.emplace_back(28, 583);

        // to position 23, legal: (legal white (move king c 3 f 6))
        ddi->diagonals.emplace_back(23, 584);

        // to position 19, legal: (legal white (move king c 3 g 7))
        ddi->diagonals.emplace_back(19, 585);

        // to position 14, legal: (legal white (move king c 3 h 8))
        ddi->diagonals.emplace_back(14, 586);

        // to position 10, legal: (legal white (move king c 3 i 9))
        ddi->diagonals.emplace_back(10, 587);

        // to position 5, legal: (legal white (move king c 3 j 10))
        ddi->diagonals.emplace_back(5, 588);

        this->diagonal_data[86].push_back(ddi);


        this->reverse_legal_lookup_white[582] = ReverseLegalLookup(Role::White, Piece::King, 37, 32, NE);
        this->reverse_legal_lookup_white[583] = ReverseLegalLookup(Role::White, Piece::King, 37, 28, NE);
        this->reverse_legal_lookup_white[584] = ReverseLegalLookup(Role::White, Piece::King, 37, 23, NE);
        this->reverse_legal_lookup_white[585] = ReverseLegalLookup(Role::White, Piece::King, 37, 19, NE);
        this->reverse_legal_lookup_white[586] = ReverseLegalLookup(Role::White, Piece::King, 37, 14, NE);
        this->reverse_legal_lookup_white[587] = ReverseLegalLookup(Role::White, Piece::King, 37, 10, NE);
        this->reverse_legal_lookup_white[588] = ReverseLegalLookup(Role::White, Piece::King, 37, 5, NE);
    }
    // generating for white king 37 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 31, legal: (legal white (move king c 3 b 4))
        ddi->diagonals.emplace_back(31, 589);

        // to position 26, legal: (legal white (move king c 3 a 5))
        ddi->diagonals.emplace_back(26, 590);

        this->diagonal_data[86].push_back(ddi);


        this->reverse_legal_lookup_white[589] = ReverseLegalLookup(Role::White, Piece::King, 37, 31, NW);
        this->reverse_legal_lookup_white[590] = ReverseLegalLookup(Role::White, Piece::King, 37, 26, NW);
    }
    // generating for white king 37 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 42, legal: (legal white (move king c 3 d 2))
        ddi->diagonals.emplace_back(42, 591);

        // to position 48, legal: (legal white (move king c 3 e 1))
        ddi->diagonals.emplace_back(48, 592);

        this->diagonal_data[86].push_back(ddi);


        this->reverse_legal_lookup_white[591] = ReverseLegalLookup(Role::White, Piece::King, 37, 42, SE);
        this->reverse_legal_lookup_white[592] = ReverseLegalLookup(Role::White, Piece::King, 37, 48, SE);
    }
    // generating for white king 37 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 41, legal: (legal white (move king c 3 b 2))
        ddi->diagonals.emplace_back(41, 593);

        // to position 46, legal: (legal white (move king c 3 a 1))
        ddi->diagonals.emplace_back(46, 594);

        this->diagonal_data[86].push_back(ddi);


        this->reverse_legal_lookup_white[593] = ReverseLegalLookup(Role::White, Piece::King, 37, 41, SW);
        this->reverse_legal_lookup_white[594] = ReverseLegalLookup(Role::White, Piece::King, 37, 46, SW);
    }
    // generating for white king 38 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 33, legal: (legal white (move king e 3 f 4))
        ddi->diagonals.emplace_back(33, 601);

        // to position 29, legal: (legal white (move king e 3 g 5))
        ddi->diagonals.emplace_back(29, 602);

        // to position 24, legal: (legal white (move king e 3 h 6))
        ddi->diagonals.emplace_back(24, 603);

        // to position 20, legal: (legal white (move king e 3 i 7))
        ddi->diagonals.emplace_back(20, 604);

        // to position 15, legal: (legal white (move king e 3 j 8))
        ddi->diagonals.emplace_back(15, 605);

        this->diagonal_data[87].push_back(ddi);


        this->reverse_legal_lookup_white[601] = ReverseLegalLookup(Role::White, Piece::King, 38, 33, NE);
        this->reverse_legal_lookup_white[602] = ReverseLegalLookup(Role::White, Piece::King, 38, 29, NE);
        this->reverse_legal_lookup_white[603] = ReverseLegalLookup(Role::White, Piece::King, 38, 24, NE);
        this->reverse_legal_lookup_white[604] = ReverseLegalLookup(Role::White, Piece::King, 38, 20, NE);
        this->reverse_legal_lookup_white[605] = ReverseLegalLookup(Role::White, Piece::King, 38, 15, NE);
    }
    // generating for white king 38 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 32, legal: (legal white (move king e 3 d 4))
        ddi->diagonals.emplace_back(32, 606);

        // to position 27, legal: (legal white (move king e 3 c 5))
        ddi->diagonals.emplace_back(27, 607);

        // to position 21, legal: (legal white (move king e 3 b 6))
        ddi->diagonals.emplace_back(21, 608);

        // to position 16, legal: (legal white (move king e 3 a 7))
        ddi->diagonals.emplace_back(16, 609);

        this->diagonal_data[87].push_back(ddi);


        this->reverse_legal_lookup_white[606] = ReverseLegalLookup(Role::White, Piece::King, 38, 32, NW);
        this->reverse_legal_lookup_white[607] = ReverseLegalLookup(Role::White, Piece::King, 38, 27, NW);
        this->reverse_legal_lookup_white[608] = ReverseLegalLookup(Role::White, Piece::King, 38, 21, NW);
        this->reverse_legal_lookup_white[609] = ReverseLegalLookup(Role::White, Piece::King, 38, 16, NW);
    }
    // generating for white king 38 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 43, legal: (legal white (move king e 3 f 2))
        ddi->diagonals.emplace_back(43, 610);

        // to position 49, legal: (legal white (move king e 3 g 1))
        ddi->diagonals.emplace_back(49, 611);

        this->diagonal_data[87].push_back(ddi);


        this->reverse_legal_lookup_white[610] = ReverseLegalLookup(Role::White, Piece::King, 38, 43, SE);
        this->reverse_legal_lookup_white[611] = ReverseLegalLookup(Role::White, Piece::King, 38, 49, SE);
    }
    // generating for white king 38 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 42, legal: (legal white (move king e 3 d 2))
        ddi->diagonals.emplace_back(42, 612);

        // to position 47, legal: (legal white (move king e 3 c 1))
        ddi->diagonals.emplace_back(47, 613);

        this->diagonal_data[87].push_back(ddi);


        this->reverse_legal_lookup_white[612] = ReverseLegalLookup(Role::White, Piece::King, 38, 42, SW);
        this->reverse_legal_lookup_white[613] = ReverseLegalLookup(Role::White, Piece::King, 38, 47, SW);
    }
    // generating for white king 39 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 34, legal: (legal white (move king g 3 h 4))
        ddi->diagonals.emplace_back(34, 620);

        // to position 30, legal: (legal white (move king g 3 i 5))
        ddi->diagonals.emplace_back(30, 621);

        // to position 25, legal: (legal white (move king g 3 j 6))
        ddi->diagonals.emplace_back(25, 622);

        this->diagonal_data[88].push_back(ddi);


        this->reverse_legal_lookup_white[620] = ReverseLegalLookup(Role::White, Piece::King, 39, 34, NE);
        this->reverse_legal_lookup_white[621] = ReverseLegalLookup(Role::White, Piece::King, 39, 30, NE);
        this->reverse_legal_lookup_white[622] = ReverseLegalLookup(Role::White, Piece::King, 39, 25, NE);
    }
    // generating for white king 39 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 33, legal: (legal white (move king g 3 f 4))
        ddi->diagonals.emplace_back(33, 623);

        // to position 28, legal: (legal white (move king g 3 e 5))
        ddi->diagonals.emplace_back(28, 624);

        // to position 22, legal: (legal white (move king g 3 d 6))
        ddi->diagonals.emplace_back(22, 625);

        // to position 17, legal: (legal white (move king g 3 c 7))
        ddi->diagonals.emplace_back(17, 626);

        // to position 11, legal: (legal white (move king g 3 b 8))
        ddi->diagonals.emplace_back(11, 627);

        // to position 6, legal: (legal white (move king g 3 a 9))
        ddi->diagonals.emplace_back(6, 628);

        this->diagonal_data[88].push_back(ddi);


        this->reverse_legal_lookup_white[623] = ReverseLegalLookup(Role::White, Piece::King, 39, 33, NW);
        this->reverse_legal_lookup_white[624] = ReverseLegalLookup(Role::White, Piece::King, 39, 28, NW);
        this->reverse_legal_lookup_white[625] = ReverseLegalLookup(Role::White, Piece::King, 39, 22, NW);
        this->reverse_legal_lookup_white[626] = ReverseLegalLookup(Role::White, Piece::King, 39, 17, NW);
        this->reverse_legal_lookup_white[627] = ReverseLegalLookup(Role::White, Piece::King, 39, 11, NW);
        this->reverse_legal_lookup_white[628] = ReverseLegalLookup(Role::White, Piece::King, 39, 6, NW);
    }
    // generating for white king 39 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 44, legal: (legal white (move king g 3 h 2))
        ddi->diagonals.emplace_back(44, 629);

        // to position 50, legal: (legal white (move king g 3 i 1))
        ddi->diagonals.emplace_back(50, 630);

        this->diagonal_data[88].push_back(ddi);


        this->reverse_legal_lookup_white[629] = ReverseLegalLookup(Role::White, Piece::King, 39, 44, SE);
        this->reverse_legal_lookup_white[630] = ReverseLegalLookup(Role::White, Piece::King, 39, 50, SE);
    }
    // generating for white king 39 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 43, legal: (legal white (move king g 3 f 2))
        ddi->diagonals.emplace_back(43, 631);

        // to position 48, legal: (legal white (move king g 3 e 1))
        ddi->diagonals.emplace_back(48, 632);

        this->diagonal_data[88].push_back(ddi);


        this->reverse_legal_lookup_white[631] = ReverseLegalLookup(Role::White, Piece::King, 39, 43, SW);
        this->reverse_legal_lookup_white[632] = ReverseLegalLookup(Role::White, Piece::King, 39, 48, SW);
    }
    // generating for white king 40 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 35, legal: (legal white (move king i 3 j 4))
        ddi->diagonals.emplace_back(35, 637);

        this->diagonal_data[89].push_back(ddi);


        this->reverse_legal_lookup_white[637] = ReverseLegalLookup(Role::White, Piece::King, 40, 35, NE);
    }
    // generating for white king 40 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(7);
        // to position 34, legal: (legal white (move king i 3 h 4))
        ddi->diagonals.emplace_back(34, 638);

        // to position 29, legal: (legal white (move king i 3 g 5))
        ddi->diagonals.emplace_back(29, 639);

        // to position 23, legal: (legal white (move king i 3 f 6))
        ddi->diagonals.emplace_back(23, 640);

        // to position 18, legal: (legal white (move king i 3 e 7))
        ddi->diagonals.emplace_back(18, 641);

        // to position 12, legal: (legal white (move king i 3 d 8))
        ddi->diagonals.emplace_back(12, 642);

        // to position 7, legal: (legal white (move king i 3 c 9))
        ddi->diagonals.emplace_back(7, 643);

        // to position 1, legal: (legal white (move king i 3 b 10))
        ddi->diagonals.emplace_back(1, 644);

        this->diagonal_data[89].push_back(ddi);


        this->reverse_legal_lookup_white[638] = ReverseLegalLookup(Role::White, Piece::King, 40, 34, NW);
        this->reverse_legal_lookup_white[639] = ReverseLegalLookup(Role::White, Piece::King, 40, 29, NW);
        this->reverse_legal_lookup_white[640] = ReverseLegalLookup(Role::White, Piece::King, 40, 23, NW);
        this->reverse_legal_lookup_white[641] = ReverseLegalLookup(Role::White, Piece::King, 40, 18, NW);
        this->reverse_legal_lookup_white[642] = ReverseLegalLookup(Role::White, Piece::King, 40, 12, NW);
        this->reverse_legal_lookup_white[643] = ReverseLegalLookup(Role::White, Piece::King, 40, 7, NW);
        this->reverse_legal_lookup_white[644] = ReverseLegalLookup(Role::White, Piece::King, 40, 1, NW);
    }
    // generating for white king 40 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 45, legal: (legal white (move king i 3 j 2))
        ddi->diagonals.emplace_back(45, 645);

        this->diagonal_data[89].push_back(ddi);


        this->reverse_legal_lookup_white[645] = ReverseLegalLookup(Role::White, Piece::King, 40, 45, SE);
    }
    // generating for white king 40 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 44, legal: (legal white (move king i 3 h 2))
        ddi->diagonals.emplace_back(44, 646);

        // to position 49, legal: (legal white (move king i 3 g 1))
        ddi->diagonals.emplace_back(49, 647);

        this->diagonal_data[89].push_back(ddi);


        this->reverse_legal_lookup_white[646] = ReverseLegalLookup(Role::White, Piece::King, 40, 44, SW);
        this->reverse_legal_lookup_white[647] = ReverseLegalLookup(Role::White, Piece::King, 40, 49, SW);
    }
    // generating for white king 41 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(8);
        // to position 37, legal: (legal white (move king b 2 c 3))
        ddi->diagonals.emplace_back(37, 651);

        // to position 32, legal: (legal white (move king b 2 d 4))
        ddi->diagonals.emplace_back(32, 652);

        // to position 28, legal: (legal white (move king b 2 e 5))
        ddi->diagonals.emplace_back(28, 653);

        // to position 23, legal: (legal white (move king b 2 f 6))
        ddi->diagonals.emplace_back(23, 654);

        // to position 19, legal: (legal white (move king b 2 g 7))
        ddi->diagonals.emplace_back(19, 655);

        // to position 14, legal: (legal white (move king b 2 h 8))
        ddi->diagonals.emplace_back(14, 656);

        // to position 10, legal: (legal white (move king b 2 i 9))
        ddi->diagonals.emplace_back(10, 657);

        // to position 5, legal: (legal white (move king b 2 j 10))
        ddi->diagonals.emplace_back(5, 658);

        this->diagonal_data[90].push_back(ddi);


        this->reverse_legal_lookup_white[651] = ReverseLegalLookup(Role::White, Piece::King, 41, 37, NE);
        this->reverse_legal_lookup_white[652] = ReverseLegalLookup(Role::White, Piece::King, 41, 32, NE);
        this->reverse_legal_lookup_white[653] = ReverseLegalLookup(Role::White, Piece::King, 41, 28, NE);
        this->reverse_legal_lookup_white[654] = ReverseLegalLookup(Role::White, Piece::King, 41, 23, NE);
        this->reverse_legal_lookup_white[655] = ReverseLegalLookup(Role::White, Piece::King, 41, 19, NE);
        this->reverse_legal_lookup_white[656] = ReverseLegalLookup(Role::White, Piece::King, 41, 14, NE);
        this->reverse_legal_lookup_white[657] = ReverseLegalLookup(Role::White, Piece::King, 41, 10, NE);
        this->reverse_legal_lookup_white[658] = ReverseLegalLookup(Role::White, Piece::King, 41, 5, NE);
    }
    // generating for white king 41 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 36, legal: (legal white (move king b 2 a 3))
        ddi->diagonals.emplace_back(36, 659);

        this->diagonal_data[90].push_back(ddi);


        this->reverse_legal_lookup_white[659] = ReverseLegalLookup(Role::White, Piece::King, 41, 36, NW);
    }
    // generating for white king 41 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 47, legal: (legal white (move king b 2 c 1))
        ddi->diagonals.emplace_back(47, 660);

        this->diagonal_data[90].push_back(ddi);


        this->reverse_legal_lookup_white[660] = ReverseLegalLookup(Role::White, Piece::King, 41, 47, SE);
    }
    // generating for white king 41 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 46, legal: (legal white (move king b 2 a 1))
        ddi->diagonals.emplace_back(46, 661);

        this->diagonal_data[90].push_back(ddi);


        this->reverse_legal_lookup_white[661] = ReverseLegalLookup(Role::White, Piece::King, 41, 46, SW);
    }
    // generating for white king 42 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(6);
        // to position 38, legal: (legal white (move king d 2 e 3))
        ddi->diagonals.emplace_back(38, 666);

        // to position 33, legal: (legal white (move king d 2 f 4))
        ddi->diagonals.emplace_back(33, 667);

        // to position 29, legal: (legal white (move king d 2 g 5))
        ddi->diagonals.emplace_back(29, 668);

        // to position 24, legal: (legal white (move king d 2 h 6))
        ddi->diagonals.emplace_back(24, 669);

        // to position 20, legal: (legal white (move king d 2 i 7))
        ddi->diagonals.emplace_back(20, 670);

        // to position 15, legal: (legal white (move king d 2 j 8))
        ddi->diagonals.emplace_back(15, 671);

        this->diagonal_data[91].push_back(ddi);


        this->reverse_legal_lookup_white[666] = ReverseLegalLookup(Role::White, Piece::King, 42, 38, NE);
        this->reverse_legal_lookup_white[667] = ReverseLegalLookup(Role::White, Piece::King, 42, 33, NE);
        this->reverse_legal_lookup_white[668] = ReverseLegalLookup(Role::White, Piece::King, 42, 29, NE);
        this->reverse_legal_lookup_white[669] = ReverseLegalLookup(Role::White, Piece::King, 42, 24, NE);
        this->reverse_legal_lookup_white[670] = ReverseLegalLookup(Role::White, Piece::King, 42, 20, NE);
        this->reverse_legal_lookup_white[671] = ReverseLegalLookup(Role::White, Piece::King, 42, 15, NE);
    }
    // generating for white king 42 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 37, legal: (legal white (move king d 2 c 3))
        ddi->diagonals.emplace_back(37, 672);

        // to position 31, legal: (legal white (move king d 2 b 4))
        ddi->diagonals.emplace_back(31, 673);

        // to position 26, legal: (legal white (move king d 2 a 5))
        ddi->diagonals.emplace_back(26, 674);

        this->diagonal_data[91].push_back(ddi);


        this->reverse_legal_lookup_white[672] = ReverseLegalLookup(Role::White, Piece::King, 42, 37, NW);
        this->reverse_legal_lookup_white[673] = ReverseLegalLookup(Role::White, Piece::King, 42, 31, NW);
        this->reverse_legal_lookup_white[674] = ReverseLegalLookup(Role::White, Piece::King, 42, 26, NW);
    }
    // generating for white king 42 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 48, legal: (legal white (move king d 2 e 1))
        ddi->diagonals.emplace_back(48, 675);

        this->diagonal_data[91].push_back(ddi);


        this->reverse_legal_lookup_white[675] = ReverseLegalLookup(Role::White, Piece::King, 42, 48, SE);
    }
    // generating for white king 42 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 47, legal: (legal white (move king d 2 c 1))
        ddi->diagonals.emplace_back(47, 676);

        this->diagonal_data[91].push_back(ddi);


        this->reverse_legal_lookup_white[676] = ReverseLegalLookup(Role::White, Piece::King, 42, 47, SW);
    }
    // generating for white king 43 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 39, legal: (legal white (move king f 2 g 3))
        ddi->diagonals.emplace_back(39, 681);

        // to position 34, legal: (legal white (move king f 2 h 4))
        ddi->diagonals.emplace_back(34, 682);

        // to position 30, legal: (legal white (move king f 2 i 5))
        ddi->diagonals.emplace_back(30, 683);

        // to position 25, legal: (legal white (move king f 2 j 6))
        ddi->diagonals.emplace_back(25, 684);

        this->diagonal_data[92].push_back(ddi);


        this->reverse_legal_lookup_white[681] = ReverseLegalLookup(Role::White, Piece::King, 43, 39, NE);
        this->reverse_legal_lookup_white[682] = ReverseLegalLookup(Role::White, Piece::King, 43, 34, NE);
        this->reverse_legal_lookup_white[683] = ReverseLegalLookup(Role::White, Piece::King, 43, 30, NE);
        this->reverse_legal_lookup_white[684] = ReverseLegalLookup(Role::White, Piece::King, 43, 25, NE);
    }
    // generating for white king 43 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 38, legal: (legal white (move king f 2 e 3))
        ddi->diagonals.emplace_back(38, 685);

        // to position 32, legal: (legal white (move king f 2 d 4))
        ddi->diagonals.emplace_back(32, 686);

        // to position 27, legal: (legal white (move king f 2 c 5))
        ddi->diagonals.emplace_back(27, 687);

        // to position 21, legal: (legal white (move king f 2 b 6))
        ddi->diagonals.emplace_back(21, 688);

        // to position 16, legal: (legal white (move king f 2 a 7))
        ddi->diagonals.emplace_back(16, 689);

        this->diagonal_data[92].push_back(ddi);


        this->reverse_legal_lookup_white[685] = ReverseLegalLookup(Role::White, Piece::King, 43, 38, NW);
        this->reverse_legal_lookup_white[686] = ReverseLegalLookup(Role::White, Piece::King, 43, 32, NW);
        this->reverse_legal_lookup_white[687] = ReverseLegalLookup(Role::White, Piece::King, 43, 27, NW);
        this->reverse_legal_lookup_white[688] = ReverseLegalLookup(Role::White, Piece::King, 43, 21, NW);
        this->reverse_legal_lookup_white[689] = ReverseLegalLookup(Role::White, Piece::King, 43, 16, NW);
    }
    // generating for white king 43 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 49, legal: (legal white (move king f 2 g 1))
        ddi->diagonals.emplace_back(49, 690);

        this->diagonal_data[92].push_back(ddi);


        this->reverse_legal_lookup_white[690] = ReverseLegalLookup(Role::White, Piece::King, 43, 49, SE);
    }
    // generating for white king 43 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 48, legal: (legal white (move king f 2 e 1))
        ddi->diagonals.emplace_back(48, 691);

        this->diagonal_data[92].push_back(ddi);


        this->reverse_legal_lookup_white[691] = ReverseLegalLookup(Role::White, Piece::King, 43, 48, SW);
    }
    // generating for white king 44 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 40, legal: (legal white (move king h 2 i 3))
        ddi->diagonals.emplace_back(40, 696);

        // to position 35, legal: (legal white (move king h 2 j 4))
        ddi->diagonals.emplace_back(35, 697);

        this->diagonal_data[93].push_back(ddi);


        this->reverse_legal_lookup_white[696] = ReverseLegalLookup(Role::White, Piece::King, 44, 40, NE);
        this->reverse_legal_lookup_white[697] = ReverseLegalLookup(Role::White, Piece::King, 44, 35, NE);
    }
    // generating for white king 44 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(7);
        // to position 39, legal: (legal white (move king h 2 g 3))
        ddi->diagonals.emplace_back(39, 698);

        // to position 33, legal: (legal white (move king h 2 f 4))
        ddi->diagonals.emplace_back(33, 699);

        // to position 28, legal: (legal white (move king h 2 e 5))
        ddi->diagonals.emplace_back(28, 700);

        // to position 22, legal: (legal white (move king h 2 d 6))
        ddi->diagonals.emplace_back(22, 701);

        // to position 17, legal: (legal white (move king h 2 c 7))
        ddi->diagonals.emplace_back(17, 702);

        // to position 11, legal: (legal white (move king h 2 b 8))
        ddi->diagonals.emplace_back(11, 703);

        // to position 6, legal: (legal white (move king h 2 a 9))
        ddi->diagonals.emplace_back(6, 704);

        this->diagonal_data[93].push_back(ddi);


        this->reverse_legal_lookup_white[698] = ReverseLegalLookup(Role::White, Piece::King, 44, 39, NW);
        this->reverse_legal_lookup_white[699] = ReverseLegalLookup(Role::White, Piece::King, 44, 33, NW);
        this->reverse_legal_lookup_white[700] = ReverseLegalLookup(Role::White, Piece::King, 44, 28, NW);
        this->reverse_legal_lookup_white[701] = ReverseLegalLookup(Role::White, Piece::King, 44, 22, NW);
        this->reverse_legal_lookup_white[702] = ReverseLegalLookup(Role::White, Piece::King, 44, 17, NW);
        this->reverse_legal_lookup_white[703] = ReverseLegalLookup(Role::White, Piece::King, 44, 11, NW);
        this->reverse_legal_lookup_white[704] = ReverseLegalLookup(Role::White, Piece::King, 44, 6, NW);
    }
    // generating for white king 44 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 50, legal: (legal white (move king h 2 i 1))
        ddi->diagonals.emplace_back(50, 705);

        this->diagonal_data[93].push_back(ddi);


        this->reverse_legal_lookup_white[705] = ReverseLegalLookup(Role::White, Piece::King, 44, 50, SE);
    }
    // generating for white king 44 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 49, legal: (legal white (move king h 2 g 1))
        ddi->diagonals.emplace_back(49, 706);

        this->diagonal_data[93].push_back(ddi);


        this->reverse_legal_lookup_white[706] = ReverseLegalLookup(Role::White, Piece::King, 44, 49, SW);
    }
    // generating for white king 45 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(8);
        // to position 40, legal: (legal white (move king j 2 i 3))
        ddi->diagonals.emplace_back(40, 709);

        // to position 34, legal: (legal white (move king j 2 h 4))
        ddi->diagonals.emplace_back(34, 710);

        // to position 29, legal: (legal white (move king j 2 g 5))
        ddi->diagonals.emplace_back(29, 711);

        // to position 23, legal: (legal white (move king j 2 f 6))
        ddi->diagonals.emplace_back(23, 712);

        // to position 18, legal: (legal white (move king j 2 e 7))
        ddi->diagonals.emplace_back(18, 713);

        // to position 12, legal: (legal white (move king j 2 d 8))
        ddi->diagonals.emplace_back(12, 714);

        // to position 7, legal: (legal white (move king j 2 c 9))
        ddi->diagonals.emplace_back(7, 715);

        // to position 1, legal: (legal white (move king j 2 b 10))
        ddi->diagonals.emplace_back(1, 716);

        this->diagonal_data[94].push_back(ddi);


        this->reverse_legal_lookup_white[709] = ReverseLegalLookup(Role::White, Piece::King, 45, 40, NW);
        this->reverse_legal_lookup_white[710] = ReverseLegalLookup(Role::White, Piece::King, 45, 34, NW);
        this->reverse_legal_lookup_white[711] = ReverseLegalLookup(Role::White, Piece::King, 45, 29, NW);
        this->reverse_legal_lookup_white[712] = ReverseLegalLookup(Role::White, Piece::King, 45, 23, NW);
        this->reverse_legal_lookup_white[713] = ReverseLegalLookup(Role::White, Piece::King, 45, 18, NW);
        this->reverse_legal_lookup_white[714] = ReverseLegalLookup(Role::White, Piece::King, 45, 12, NW);
        this->reverse_legal_lookup_white[715] = ReverseLegalLookup(Role::White, Piece::King, 45, 7, NW);
        this->reverse_legal_lookup_white[716] = ReverseLegalLookup(Role::White, Piece::King, 45, 1, NW);
    }
    // generating for white king 45 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 50, legal: (legal white (move king j 2 i 1))
        ddi->diagonals.emplace_back(50, 717);

        this->diagonal_data[94].push_back(ddi);


        this->reverse_legal_lookup_white[717] = ReverseLegalLookup(Role::White, Piece::King, 45, 50, SW);
    }
    // generating for white king 46 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(9);
        // to position 41, legal: (legal white (move king a 1 b 2))
        ddi->diagonals.emplace_back(41, 720);

        // to position 37, legal: (legal white (move king a 1 c 3))
        ddi->diagonals.emplace_back(37, 721);

        // to position 32, legal: (legal white (move king a 1 d 4))
        ddi->diagonals.emplace_back(32, 722);

        // to position 28, legal: (legal white (move king a 1 e 5))
        ddi->diagonals.emplace_back(28, 723);

        // to position 23, legal: (legal white (move king a 1 f 6))
        ddi->diagonals.emplace_back(23, 724);

        // to position 19, legal: (legal white (move king a 1 g 7))
        ddi->diagonals.emplace_back(19, 725);

        // to position 14, legal: (legal white (move king a 1 h 8))
        ddi->diagonals.emplace_back(14, 726);

        // to position 10, legal: (legal white (move king a 1 i 9))
        ddi->diagonals.emplace_back(10, 727);

        // to position 5, legal: (legal white (move king a 1 j 10))
        ddi->diagonals.emplace_back(5, 728);

        this->diagonal_data[95].push_back(ddi);


        this->reverse_legal_lookup_white[720] = ReverseLegalLookup(Role::White, Piece::King, 46, 41, NE);
        this->reverse_legal_lookup_white[721] = ReverseLegalLookup(Role::White, Piece::King, 46, 37, NE);
        this->reverse_legal_lookup_white[722] = ReverseLegalLookup(Role::White, Piece::King, 46, 32, NE);
        this->reverse_legal_lookup_white[723] = ReverseLegalLookup(Role::White, Piece::King, 46, 28, NE);
        this->reverse_legal_lookup_white[724] = ReverseLegalLookup(Role::White, Piece::King, 46, 23, NE);
        this->reverse_legal_lookup_white[725] = ReverseLegalLookup(Role::White, Piece::King, 46, 19, NE);
        this->reverse_legal_lookup_white[726] = ReverseLegalLookup(Role::White, Piece::King, 46, 14, NE);
        this->reverse_legal_lookup_white[727] = ReverseLegalLookup(Role::White, Piece::King, 46, 10, NE);
        this->reverse_legal_lookup_white[728] = ReverseLegalLookup(Role::White, Piece::King, 46, 5, NE);
    }
    // generating for white king 47 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(7);
        // to position 42, legal: (legal white (move king c 1 d 2))
        ddi->diagonals.emplace_back(42, 733);

        // to position 38, legal: (legal white (move king c 1 e 3))
        ddi->diagonals.emplace_back(38, 734);

        // to position 33, legal: (legal white (move king c 1 f 4))
        ddi->diagonals.emplace_back(33, 735);

        // to position 29, legal: (legal white (move king c 1 g 5))
        ddi->diagonals.emplace_back(29, 736);

        // to position 24, legal: (legal white (move king c 1 h 6))
        ddi->diagonals.emplace_back(24, 737);

        // to position 20, legal: (legal white (move king c 1 i 7))
        ddi->diagonals.emplace_back(20, 738);

        // to position 15, legal: (legal white (move king c 1 j 8))
        ddi->diagonals.emplace_back(15, 739);

        this->diagonal_data[96].push_back(ddi);


        this->reverse_legal_lookup_white[733] = ReverseLegalLookup(Role::White, Piece::King, 47, 42, NE);
        this->reverse_legal_lookup_white[734] = ReverseLegalLookup(Role::White, Piece::King, 47, 38, NE);
        this->reverse_legal_lookup_white[735] = ReverseLegalLookup(Role::White, Piece::King, 47, 33, NE);
        this->reverse_legal_lookup_white[736] = ReverseLegalLookup(Role::White, Piece::King, 47, 29, NE);
        this->reverse_legal_lookup_white[737] = ReverseLegalLookup(Role::White, Piece::King, 47, 24, NE);
        this->reverse_legal_lookup_white[738] = ReverseLegalLookup(Role::White, Piece::King, 47, 20, NE);
        this->reverse_legal_lookup_white[739] = ReverseLegalLookup(Role::White, Piece::King, 47, 15, NE);
    }
    // generating for white king 47 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 41, legal: (legal white (move king c 1 b 2))
        ddi->diagonals.emplace_back(41, 740);

        // to position 36, legal: (legal white (move king c 1 a 3))
        ddi->diagonals.emplace_back(36, 741);

        this->diagonal_data[96].push_back(ddi);


        this->reverse_legal_lookup_white[740] = ReverseLegalLookup(Role::White, Piece::King, 47, 41, NW);
        this->reverse_legal_lookup_white[741] = ReverseLegalLookup(Role::White, Piece::King, 47, 36, NW);
    }
    // generating for white king 48 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 43, legal: (legal white (move king e 1 f 2))
        ddi->diagonals.emplace_back(43, 746);

        // to position 39, legal: (legal white (move king e 1 g 3))
        ddi->diagonals.emplace_back(39, 747);

        // to position 34, legal: (legal white (move king e 1 h 4))
        ddi->diagonals.emplace_back(34, 748);

        // to position 30, legal: (legal white (move king e 1 i 5))
        ddi->diagonals.emplace_back(30, 749);

        // to position 25, legal: (legal white (move king e 1 j 6))
        ddi->diagonals.emplace_back(25, 750);

        this->diagonal_data[97].push_back(ddi);


        this->reverse_legal_lookup_white[746] = ReverseLegalLookup(Role::White, Piece::King, 48, 43, NE);
        this->reverse_legal_lookup_white[747] = ReverseLegalLookup(Role::White, Piece::King, 48, 39, NE);
        this->reverse_legal_lookup_white[748] = ReverseLegalLookup(Role::White, Piece::King, 48, 34, NE);
        this->reverse_legal_lookup_white[749] = ReverseLegalLookup(Role::White, Piece::King, 48, 30, NE);
        this->reverse_legal_lookup_white[750] = ReverseLegalLookup(Role::White, Piece::King, 48, 25, NE);
    }
    // generating for white king 48 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 42, legal: (legal white (move king e 1 d 2))
        ddi->diagonals.emplace_back(42, 751);

        // to position 37, legal: (legal white (move king e 1 c 3))
        ddi->diagonals.emplace_back(37, 752);

        // to position 31, legal: (legal white (move king e 1 b 4))
        ddi->diagonals.emplace_back(31, 753);

        // to position 26, legal: (legal white (move king e 1 a 5))
        ddi->diagonals.emplace_back(26, 754);

        this->diagonal_data[97].push_back(ddi);


        this->reverse_legal_lookup_white[751] = ReverseLegalLookup(Role::White, Piece::King, 48, 42, NW);
        this->reverse_legal_lookup_white[752] = ReverseLegalLookup(Role::White, Piece::King, 48, 37, NW);
        this->reverse_legal_lookup_white[753] = ReverseLegalLookup(Role::White, Piece::King, 48, 31, NW);
        this->reverse_legal_lookup_white[754] = ReverseLegalLookup(Role::White, Piece::King, 48, 26, NW);
    }
    // generating for white king 49 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 44, legal: (legal white (move king g 1 h 2))
        ddi->diagonals.emplace_back(44, 759);

        // to position 40, legal: (legal white (move king g 1 i 3))
        ddi->diagonals.emplace_back(40, 760);

        // to position 35, legal: (legal white (move king g 1 j 4))
        ddi->diagonals.emplace_back(35, 761);

        this->diagonal_data[98].push_back(ddi);


        this->reverse_legal_lookup_white[759] = ReverseLegalLookup(Role::White, Piece::King, 49, 44, NE);
        this->reverse_legal_lookup_white[760] = ReverseLegalLookup(Role::White, Piece::King, 49, 40, NE);
        this->reverse_legal_lookup_white[761] = ReverseLegalLookup(Role::White, Piece::King, 49, 35, NE);
    }
    // generating for white king 49 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 43, legal: (legal white (move king g 1 f 2))
        ddi->diagonals.emplace_back(43, 762);

        // to position 38, legal: (legal white (move king g 1 e 3))
        ddi->diagonals.emplace_back(38, 763);

        // to position 32, legal: (legal white (move king g 1 d 4))
        ddi->diagonals.emplace_back(32, 764);

        // to position 27, legal: (legal white (move king g 1 c 5))
        ddi->diagonals.emplace_back(27, 765);

        // to position 21, legal: (legal white (move king g 1 b 6))
        ddi->diagonals.emplace_back(21, 766);

        // to position 16, legal: (legal white (move king g 1 a 7))
        ddi->diagonals.emplace_back(16, 767);

        this->diagonal_data[98].push_back(ddi);


        this->reverse_legal_lookup_white[762] = ReverseLegalLookup(Role::White, Piece::King, 49, 43, NW);
        this->reverse_legal_lookup_white[763] = ReverseLegalLookup(Role::White, Piece::King, 49, 38, NW);
        this->reverse_legal_lookup_white[764] = ReverseLegalLookup(Role::White, Piece::King, 49, 32, NW);
        this->reverse_legal_lookup_white[765] = ReverseLegalLookup(Role::White, Piece::King, 49, 27, NW);
        this->reverse_legal_lookup_white[766] = ReverseLegalLookup(Role::White, Piece::King, 49, 21, NW);
        this->reverse_legal_lookup_white[767] = ReverseLegalLookup(Role::White, Piece::King, 49, 16, NW);
    }
    // generating for white king 50 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 45, legal: (legal white (move king i 1 j 2))
        ddi->diagonals.emplace_back(45, 771);

        this->diagonal_data[99].push_back(ddi);


        this->reverse_legal_lookup_white[771] = ReverseLegalLookup(Role::White, Piece::King, 50, 45, NE);
    }
    // generating for white king 50 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(8);
        // to position 44, legal: (legal white (move king i 1 h 2))
        ddi->diagonals.emplace_back(44, 772);

        // to position 39, legal: (legal white (move king i 1 g 3))
        ddi->diagonals.emplace_back(39, 773);

        // to position 33, legal: (legal white (move king i 1 f 4))
        ddi->diagonals.emplace_back(33, 774);

        // to position 28, legal: (legal white (move king i 1 e 5))
        ddi->diagonals.emplace_back(28, 775);

        // to position 22, legal: (legal white (move king i 1 d 6))
        ddi->diagonals.emplace_back(22, 776);

        // to position 17, legal: (legal white (move king i 1 c 7))
        ddi->diagonals.emplace_back(17, 777);

        // to position 11, legal: (legal white (move king i 1 b 8))
        ddi->diagonals.emplace_back(11, 778);

        // to position 6, legal: (legal white (move king i 1 a 9))
        ddi->diagonals.emplace_back(6, 779);

        this->diagonal_data[99].push_back(ddi);


        this->reverse_legal_lookup_white[772] = ReverseLegalLookup(Role::White, Piece::King, 50, 44, NW);
        this->reverse_legal_lookup_white[773] = ReverseLegalLookup(Role::White, Piece::King, 50, 39, NW);
        this->reverse_legal_lookup_white[774] = ReverseLegalLookup(Role::White, Piece::King, 50, 33, NW);
        this->reverse_legal_lookup_white[775] = ReverseLegalLookup(Role::White, Piece::King, 50, 28, NW);
        this->reverse_legal_lookup_white[776] = ReverseLegalLookup(Role::White, Piece::King, 50, 22, NW);
        this->reverse_legal_lookup_white[777] = ReverseLegalLookup(Role::White, Piece::King, 50, 17, NW);
        this->reverse_legal_lookup_white[778] = ReverseLegalLookup(Role::White, Piece::King, 50, 11, NW);
        this->reverse_legal_lookup_white[779] = ReverseLegalLookup(Role::White, Piece::King, 50, 6, NW);
    }

    // generating promotion line for black
    this->black_promotion_line = {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true};


    // generating moves for black
    this->black_legal_moves = {"noop", "(move man b 10 c 9)", "(move man b 10 a 9)", "(move man b 10 d 8)", "(move king b 10 c 9)", "(move king b 10 d 8)", "(move king b 10 e 7)", "(move king b 10 f 6)", "(move king b 10 g 5)", "(move king b 10 h 4)", "(move king b 10 i 3)", "(move king b 10 j 2)", "(move king b 10 a 9)", "(move man d 10 e 9)", "(move man d 10 c 9)", "(move man d 10 f 8)", "(move man d 10 b 8)", "(move king d 10 e 9)", "(move king d 10 f 8)", "(move king d 10 g 7)", "(move king d 10 h 6)", "(move king d 10 i 5)", "(move king d 10 j 4)", "(move king d 10 c 9)", "(move king d 10 b 8)", "(move king d 10 a 7)", "(move man f 10 g 9)", "(move man f 10 e 9)", "(move man f 10 h 8)", "(move man f 10 d 8)", "(move king f 10 g 9)", "(move king f 10 h 8)", "(move king f 10 i 7)", "(move king f 10 j 6)", "(move king f 10 e 9)", "(move king f 10 d 8)", "(move king f 10 c 7)", "(move king f 10 b 6)", "(move king f 10 a 5)", "(move man h 10 i 9)", "(move man h 10 g 9)", "(move man h 10 j 8)", "(move man h 10 f 8)", "(move king h 10 i 9)", "(move king h 10 j 8)", "(move king h 10 g 9)", "(move king h 10 f 8)", "(move king h 10 e 7)", "(move king h 10 d 6)", "(move king h 10 c 5)", "(move king h 10 b 4)", "(move king h 10 a 3)", "(move man j 10 i 9)", "(move man j 10 h 8)", "(move king j 10 i 9)", "(move king j 10 h 8)", "(move king j 10 g 7)", "(move king j 10 f 6)", "(move king j 10 e 5)", "(move king j 10 d 4)", "(move king j 10 c 3)", "(move king j 10 b 2)", "(move king j 10 a 1)", "(move man a 9 b 8)", "(move man a 9 c 7)", "(move king a 9 b 10)", "(move king a 9 b 8)", "(move king a 9 c 7)", "(move king a 9 d 6)", "(move king a 9 e 5)", "(move king a 9 f 4)", "(move king a 9 g 3)", "(move king a 9 h 2)", "(move king a 9 i 1)", "(move man c 9 d 8)", "(move man c 9 b 8)", "(move man c 9 e 7)", "(move man c 9 a 7)", "(move king c 9 d 10)", "(move king c 9 b 10)", "(move king c 9 d 8)", "(move king c 9 e 7)", "(move king c 9 f 6)", "(move king c 9 g 5)", "(move king c 9 h 4)", "(move king c 9 i 3)", "(move king c 9 j 2)", "(move king c 9 b 8)", "(move king c 9 a 7)", "(move man e 9 f 8)", "(move man e 9 d 8)", "(move man e 9 g 7)", "(move man e 9 c 7)", "(move king e 9 f 10)", "(move king e 9 d 10)", "(move king e 9 f 8)", "(move king e 9 g 7)", "(move king e 9 h 6)", "(move king e 9 i 5)", "(move king e 9 j 4)", "(move king e 9 d 8)", "(move king e 9 c 7)", "(move king e 9 b 6)", "(move king e 9 a 5)", "(move man g 9 h 8)", "(move man g 9 f 8)", "(move man g 9 i 7)", "(move man g 9 e 7)", "(move king g 9 h 10)", "(move king g 9 f 10)", "(move king g 9 h 8)", "(move king g 9 i 7)", "(move king g 9 j 6)", "(move king g 9 f 8)", "(move king g 9 e 7)", "(move king g 9 d 6)", "(move king g 9 c 5)", "(move king g 9 b 4)", "(move king g 9 a 3)", "(move man i 9 j 8)", "(move man i 9 h 8)", "(move man i 9 g 7)", "(move king i 9 j 10)", "(move king i 9 h 10)", "(move king i 9 j 8)", "(move king i 9 h 8)", "(move king i 9 g 7)", "(move king i 9 f 6)", "(move king i 9 e 5)", "(move king i 9 d 4)", "(move king i 9 c 3)", "(move king i 9 b 2)", "(move king i 9 a 1)", "(move man b 8 c 7)", "(move man b 8 a 7)", "(move man b 8 d 10)", "(move man b 8 d 6)", "(move king b 8 c 9)", "(move king b 8 d 10)", "(move king b 8 a 9)", "(move king b 8 c 7)", "(move king b 8 d 6)", "(move king b 8 e 5)", "(move king b 8 f 4)", "(move king b 8 g 3)", "(move king b 8 h 2)", "(move king b 8 i 1)", "(move king b 8 a 7)", "(move man d 8 e 7)", "(move man d 8 c 7)", "(move man d 8 f 10)", "(move man d 8 b 10)", "(move man d 8 f 6)", "(move man d 8 b 6)", "(move king d 8 e 9)", "(move king d 8 f 10)", "(move king d 8 c 9)", "(move king d 8 b 10)", "(move king d 8 e 7)", "(move king d 8 f 6)", "(move king d 8 g 5)", "(move king d 8 h 4)", "(move king d 8 i 3)", "(move king d 8 j 2)", "(move king d 8 c 7)", "(move king d 8 b 6)", "(move king d 8 a 5)", "(move man f 8 g 7)", "(move man f 8 e 7)", "(move man f 8 h 10)", "(move man f 8 d 10)", "(move man f 8 h 6)", "(move man f 8 d 6)", "(move king f 8 g 9)", "(move king f 8 h 10)", "(move king f 8 e 9)", "(move king f 8 d 10)", "(move king f 8 g 7)", "(move king f 8 h 6)", "(move king f 8 i 5)", "(move king f 8 j 4)", "(move king f 8 e 7)", "(move king f 8 d 6)", "(move king f 8 c 5)", "(move king f 8 b 4)", "(move king f 8 a 3)", "(move man h 8 i 7)", "(move man h 8 g 7)", "(move man h 8 j 10)", "(move man h 8 f 10)", "(move man h 8 j 6)", "(move man h 8 f 6)", "(move king h 8 i 9)", "(move king h 8 j 10)", "(move king h 8 g 9)", "(move king h 8 f 10)", "(move king h 8 i 7)", "(move king h 8 j 6)", "(move king h 8 g 7)", "(move king h 8 f 6)", "(move king h 8 e 5)", "(move king h 8 d 4)", "(move king h 8 c 3)", "(move king h 8 b 2)", "(move king h 8 a 1)", "(move man j 8 i 7)", "(move man j 8 h 10)", "(move man j 8 h 6)", "(move king j 8 i 9)", "(move king j 8 h 10)", "(move king j 8 i 7)", "(move king j 8 h 6)", "(move king j 8 g 5)", "(move king j 8 f 4)", "(move king j 8 e 3)", "(move king j 8 d 2)", "(move king j 8 c 1)", "(move man a 7 b 6)", "(move man a 7 c 9)", "(move man a 7 c 5)", "(move king a 7 b 8)", "(move king a 7 c 9)", "(move king a 7 d 10)", "(move king a 7 b 6)", "(move king a 7 c 5)", "(move king a 7 d 4)", "(move king a 7 e 3)", "(move king a 7 f 2)", "(move king a 7 g 1)", "(move man c 7 d 6)", "(move man c 7 b 6)", "(move man c 7 e 9)", "(move man c 7 a 9)", "(move man c 7 e 5)", "(move man c 7 a 5)", "(move king c 7 d 8)", "(move king c 7 e 9)", "(move king c 7 f 10)", "(move king c 7 b 8)", "(move king c 7 a 9)", "(move king c 7 d 6)", "(move king c 7 e 5)", "(move king c 7 f 4)", "(move king c 7 g 3)", "(move king c 7 h 2)", "(move king c 7 i 1)", "(move king c 7 b 6)", "(move king c 7 a 5)", "(move man e 7 f 6)", "(move man e 7 d 6)", "(move man e 7 g 9)", "(move man e 7 c 9)", "(move man e 7 g 5)", "(move man e 7 c 5)", "(move king e 7 f 8)", "(move king e 7 g 9)", "(move king e 7 h 10)", "(move king e 7 d 8)", "(move king e 7 c 9)", "(move king e 7 b 10)", "(move king e 7 f 6)", "(move king e 7 g 5)", "(move king e 7 h 4)", "(move king e 7 i 3)", "(move king e 7 j 2)", "(move king e 7 d 6)", "(move king e 7 c 5)", "(move king e 7 b 4)", "(move king e 7 a 3)", "(move man g 7 h 6)", "(move man g 7 f 6)", "(move man g 7 i 9)", "(move man g 7 e 9)", "(move man g 7 i 5)", "(move man g 7 e 5)", "(move king g 7 h 8)", "(move king g 7 i 9)", "(move king g 7 j 10)", "(move king g 7 f 8)", "(move king g 7 e 9)", "(move king g 7 d 10)", "(move king g 7 h 6)", "(move king g 7 i 5)", "(move king g 7 j 4)", "(move king g 7 f 6)", "(move king g 7 e 5)", "(move king g 7 d 4)", "(move king g 7 c 3)", "(move king g 7 b 2)", "(move king g 7 a 1)", "(move man i 7 j 6)", "(move man i 7 h 6)", "(move man i 7 g 9)", "(move man i 7 g 5)", "(move king i 7 j 8)", "(move king i 7 h 8)", "(move king i 7 g 9)", "(move king i 7 f 10)", "(move king i 7 j 6)", "(move king i 7 h 6)", "(move king i 7 g 5)", "(move king i 7 f 4)", "(move king i 7 e 3)", "(move king i 7 d 2)", "(move king i 7 c 1)", "(move man b 6 c 5)", "(move man b 6 a 5)", "(move man b 6 d 8)", "(move man b 6 d 4)", "(move king b 6 c 7)", "(move king b 6 d 8)", "(move king b 6 e 9)", "(move king b 6 f 10)", "(move king b 6 a 7)", "(move king b 6 c 5)", "(move king b 6 d 4)", "(move king b 6 e 3)", "(move king b 6 f 2)", "(move king b 6 g 1)", "(move king b 6 a 5)", "(move man d 6 e 5)", "(move man d 6 c 5)", "(move man d 6 f 8)", "(move man d 6 b 8)", "(move man d 6 f 4)", "(move man d 6 b 4)", "(move king d 6 e 7)", "(move king d 6 f 8)", "(move king d 6 g 9)", "(move king d 6 h 10)", "(move king d 6 c 7)", "(move king d 6 b 8)", "(move king d 6 a 9)", "(move king d 6 e 5)", "(move king d 6 f 4)", "(move king d 6 g 3)", "(move king d 6 h 2)", "(move king d 6 i 1)", "(move king d 6 c 5)", "(move king d 6 b 4)", "(move king d 6 a 3)", "(move man f 6 g 5)", "(move man f 6 e 5)", "(move man f 6 h 8)", "(move man f 6 d 8)", "(move man f 6 h 4)", "(move man f 6 d 4)", "(move king f 6 g 7)", "(move king f 6 h 8)", "(move king f 6 i 9)", "(move king f 6 j 10)", "(move king f 6 e 7)", "(move king f 6 d 8)", "(move king f 6 c 9)", "(move king f 6 b 10)", "(move king f 6 g 5)", "(move king f 6 h 4)", "(move king f 6 i 3)", "(move king f 6 j 2)", "(move king f 6 e 5)", "(move king f 6 d 4)", "(move king f 6 c 3)", "(move king f 6 b 2)", "(move king f 6 a 1)", "(move man h 6 i 5)", "(move man h 6 g 5)", "(move man h 6 j 8)", "(move man h 6 f 8)", "(move man h 6 j 4)", "(move man h 6 f 4)", "(move king h 6 i 7)", "(move king h 6 j 8)", "(move king h 6 g 7)", "(move king h 6 f 8)", "(move king h 6 e 9)", "(move king h 6 d 10)", "(move king h 6 i 5)", "(move king h 6 j 4)", "(move king h 6 g 5)", "(move king h 6 f 4)", "(move king h 6 e 3)", "(move king h 6 d 2)", "(move king h 6 c 1)", "(move man j 6 i 5)", "(move man j 6 h 8)", "(move man j 6 h 4)", "(move king j 6 i 7)", "(move king j 6 h 8)", "(move king j 6 g 9)", "(move king j 6 f 10)", "(move king j 6 i 5)", "(move king j 6 h 4)", "(move king j 6 g 3)", "(move king j 6 f 2)", "(move king j 6 e 1)", "(move man a 5 b 4)", "(move man a 5 c 7)", "(move man a 5 c 3)", "(move king a 5 b 6)", "(move king a 5 c 7)", "(move king a 5 d 8)", "(move king a 5 e 9)", "(move king a 5 f 10)", "(move king a 5 b 4)", "(move king a 5 c 3)", "(move king a 5 d 2)", "(move king a 5 e 1)", "(move man c 5 d 4)", "(move man c 5 b 4)", "(move man c 5 e 7)", "(move man c 5 a 7)", "(move man c 5 e 3)", "(move man c 5 a 3)", "(move king c 5 d 6)", "(move king c 5 e 7)", "(move king c 5 f 8)", "(move king c 5 g 9)", "(move king c 5 h 10)", "(move king c 5 b 6)", "(move king c 5 a 7)", "(move king c 5 d 4)", "(move king c 5 e 3)", "(move king c 5 f 2)", "(move king c 5 g 1)", "(move king c 5 b 4)", "(move king c 5 a 3)", "(move man e 5 f 4)", "(move man e 5 d 4)", "(move man e 5 g 7)", "(move man e 5 c 7)", "(move man e 5 g 3)", "(move man e 5 c 3)", "(move king e 5 f 6)", "(move king e 5 g 7)", "(move king e 5 h 8)", "(move king e 5 i 9)", "(move king e 5 j 10)", "(move king e 5 d 6)", "(move king e 5 c 7)", "(move king e 5 b 8)", "(move king e 5 a 9)", "(move king e 5 f 4)", "(move king e 5 g 3)", "(move king e 5 h 2)", "(move king e 5 i 1)", "(move king e 5 d 4)", "(move king e 5 c 3)", "(move king e 5 b 2)", "(move king e 5 a 1)", "(move man g 5 h 4)", "(move man g 5 f 4)", "(move man g 5 i 7)", "(move man g 5 e 7)", "(move man g 5 i 3)", "(move man g 5 e 3)", "(move king g 5 h 6)", "(move king g 5 i 7)", "(move king g 5 j 8)", "(move king g 5 f 6)", "(move king g 5 e 7)", "(move king g 5 d 8)", "(move king g 5 c 9)", "(move king g 5 b 10)", "(move king g 5 h 4)", "(move king g 5 i 3)", "(move king g 5 j 2)", "(move king g 5 f 4)", "(move king g 5 e 3)", "(move king g 5 d 2)", "(move king g 5 c 1)", "(move man i 5 j 4)", "(move man i 5 h 4)", "(move man i 5 g 7)", "(move man i 5 g 3)", "(move king i 5 j 6)", "(move king i 5 h 6)", "(move king i 5 g 7)", "(move king i 5 f 8)", "(move king i 5 e 9)", "(move king i 5 d 10)", "(move king i 5 j 4)", "(move king i 5 h 4)", "(move king i 5 g 3)", "(move king i 5 f 2)", "(move king i 5 e 1)", "(move man b 4 c 3)", "(move man b 4 a 3)", "(move man b 4 d 6)", "(move man b 4 d 2)", "(move king b 4 c 5)", "(move king b 4 d 6)", "(move king b 4 e 7)", "(move king b 4 f 8)", "(move king b 4 g 9)", "(move king b 4 h 10)", "(move king b 4 a 5)", "(move king b 4 c 3)", "(move king b 4 d 2)", "(move king b 4 e 1)", "(move king b 4 a 3)", "(move man d 4 e 3)", "(move man d 4 c 3)", "(move man d 4 f 6)", "(move man d 4 b 6)", "(move man d 4 f 2)", "(move man d 4 b 2)", "(move king d 4 e 5)", "(move king d 4 f 6)", "(move king d 4 g 7)", "(move king d 4 h 8)", "(move king d 4 i 9)", "(move king d 4 j 10)", "(move king d 4 c 5)", "(move king d 4 b 6)", "(move king d 4 a 7)", "(move king d 4 e 3)", "(move king d 4 f 2)", "(move king d 4 g 1)", "(move king d 4 c 3)", "(move king d 4 b 2)", "(move king d 4 a 1)", "(move man f 4 g 3)", "(move man f 4 e 3)", "(move man f 4 h 6)", "(move man f 4 d 6)", "(move man f 4 h 2)", "(move man f 4 d 2)", "(move king f 4 g 5)", "(move king f 4 h 6)", "(move king f 4 i 7)", "(move king f 4 j 8)", "(move king f 4 e 5)", "(move king f 4 d 6)", "(move king f 4 c 7)", "(move king f 4 b 8)", "(move king f 4 a 9)", "(move king f 4 g 3)", "(move king f 4 h 2)", "(move king f 4 i 1)", "(move king f 4 e 3)", "(move king f 4 d 2)", "(move king f 4 c 1)", "(move man h 4 i 3)", "(move man h 4 g 3)", "(move man h 4 j 6)", "(move man h 4 f 6)", "(move man h 4 j 2)", "(move man h 4 f 2)", "(move king h 4 i 5)", "(move king h 4 j 6)", "(move king h 4 g 5)", "(move king h 4 f 6)", "(move king h 4 e 7)", "(move king h 4 d 8)", "(move king h 4 c 9)", "(move king h 4 b 10)", "(move king h 4 i 3)", "(move king h 4 j 2)", "(move king h 4 g 3)", "(move king h 4 f 2)", "(move king h 4 e 1)", "(move man j 4 i 3)", "(move man j 4 h 6)", "(move man j 4 h 2)", "(move king j 4 i 5)", "(move king j 4 h 6)", "(move king j 4 g 7)", "(move king j 4 f 8)", "(move king j 4 e 9)", "(move king j 4 d 10)", "(move king j 4 i 3)", "(move king j 4 h 2)", "(move king j 4 g 1)", "(move man a 3 b 2)", "(move man a 3 c 5)", "(move man a 3 c 1)", "(move king a 3 b 4)", "(move king a 3 c 5)", "(move king a 3 d 6)", "(move king a 3 e 7)", "(move king a 3 f 8)", "(move king a 3 g 9)", "(move king a 3 h 10)", "(move king a 3 b 2)", "(move king a 3 c 1)", "(move man c 3 d 2)", "(move man c 3 b 2)", "(move man c 3 e 5)", "(move man c 3 a 5)", "(move man c 3 e 1)", "(move man c 3 a 1)", "(move king c 3 d 4)", "(move king c 3 e 5)", "(move king c 3 f 6)", "(move king c 3 g 7)", "(move king c 3 h 8)", "(move king c 3 i 9)", "(move king c 3 j 10)", "(move king c 3 b 4)", "(move king c 3 a 5)", "(move king c 3 d 2)", "(move king c 3 e 1)", "(move king c 3 b 2)", "(move king c 3 a 1)", "(move man e 3 f 2)", "(move man e 3 d 2)", "(move man e 3 g 5)", "(move man e 3 c 5)", "(move man e 3 g 1)", "(move man e 3 c 1)", "(move king e 3 f 4)", "(move king e 3 g 5)", "(move king e 3 h 6)", "(move king e 3 i 7)", "(move king e 3 j 8)", "(move king e 3 d 4)", "(move king e 3 c 5)", "(move king e 3 b 6)", "(move king e 3 a 7)", "(move king e 3 f 2)", "(move king e 3 g 1)", "(move king e 3 d 2)", "(move king e 3 c 1)", "(move man g 3 h 2)", "(move man g 3 f 2)", "(move man g 3 i 5)", "(move man g 3 e 5)", "(move man g 3 i 1)", "(move man g 3 e 1)", "(move king g 3 h 4)", "(move king g 3 i 5)", "(move king g 3 j 6)", "(move king g 3 f 4)", "(move king g 3 e 5)", "(move king g 3 d 6)", "(move king g 3 c 7)", "(move king g 3 b 8)", "(move king g 3 a 9)", "(move king g 3 h 2)", "(move king g 3 i 1)", "(move king g 3 f 2)", "(move king g 3 e 1)", "(move man i 3 j 2)", "(move man i 3 h 2)", "(move man i 3 g 5)", "(move man i 3 g 1)", "(move king i 3 j 4)", "(move king i 3 h 4)", "(move king i 3 g 5)", "(move king i 3 f 6)", "(move king i 3 e 7)", "(move king i 3 d 8)", "(move king i 3 c 9)", "(move king i 3 b 10)", "(move king i 3 j 2)", "(move king i 3 h 2)", "(move king i 3 g 1)", "(move man b 2 c 1)", "(move man b 2 a 1)", "(move man b 2 d 4)", "(move king b 2 c 3)", "(move king b 2 d 4)", "(move king b 2 e 5)", "(move king b 2 f 6)", "(move king b 2 g 7)", "(move king b 2 h 8)", "(move king b 2 i 9)", "(move king b 2 j 10)", "(move king b 2 a 3)", "(move king b 2 c 1)", "(move king b 2 a 1)", "(move man d 2 e 1)", "(move man d 2 c 1)", "(move man d 2 f 4)", "(move man d 2 b 4)", "(move king d 2 e 3)", "(move king d 2 f 4)", "(move king d 2 g 5)", "(move king d 2 h 6)", "(move king d 2 i 7)", "(move king d 2 j 8)", "(move king d 2 c 3)", "(move king d 2 b 4)", "(move king d 2 a 5)", "(move king d 2 e 1)", "(move king d 2 c 1)", "(move man f 2 g 1)", "(move man f 2 e 1)", "(move man f 2 h 4)", "(move man f 2 d 4)", "(move king f 2 g 3)", "(move king f 2 h 4)", "(move king f 2 i 5)", "(move king f 2 j 6)", "(move king f 2 e 3)", "(move king f 2 d 4)", "(move king f 2 c 5)", "(move king f 2 b 6)", "(move king f 2 a 7)", "(move king f 2 g 1)", "(move king f 2 e 1)", "(move man h 2 i 1)", "(move man h 2 g 1)", "(move man h 2 j 4)", "(move man h 2 f 4)", "(move king h 2 i 3)", "(move king h 2 j 4)", "(move king h 2 g 3)", "(move king h 2 f 4)", "(move king h 2 e 5)", "(move king h 2 d 6)", "(move king h 2 c 7)", "(move king h 2 b 8)", "(move king h 2 a 9)", "(move king h 2 i 1)", "(move king h 2 g 1)", "(move man j 2 i 1)", "(move man j 2 h 4)", "(move king j 2 i 3)", "(move king j 2 h 4)", "(move king j 2 g 5)", "(move king j 2 f 6)", "(move king j 2 e 7)", "(move king j 2 d 8)", "(move king j 2 c 9)", "(move king j 2 b 10)", "(move king j 2 i 1)", "(move man a 1 c 3)", "(move king a 1 b 2)", "(move king a 1 c 3)", "(move king a 1 d 4)", "(move king a 1 e 5)", "(move king a 1 f 6)", "(move king a 1 g 7)", "(move king a 1 h 8)", "(move king a 1 i 9)", "(move king a 1 j 10)", "(move man c 1 e 3)", "(move man c 1 a 3)", "(move king c 1 d 2)", "(move king c 1 e 3)", "(move king c 1 f 4)", "(move king c 1 g 5)", "(move king c 1 h 6)", "(move king c 1 i 7)", "(move king c 1 j 8)", "(move king c 1 b 2)", "(move king c 1 a 3)", "(move man e 1 g 3)", "(move man e 1 c 3)", "(move king e 1 f 2)", "(move king e 1 g 3)", "(move king e 1 h 4)", "(move king e 1 i 5)", "(move king e 1 j 6)", "(move king e 1 d 2)", "(move king e 1 c 3)", "(move king e 1 b 4)", "(move king e 1 a 5)", "(move man g 1 i 3)", "(move man g 1 e 3)", "(move king g 1 h 2)", "(move king g 1 i 3)", "(move king g 1 j 4)", "(move king g 1 f 2)", "(move king g 1 e 3)", "(move king g 1 d 4)", "(move king g 1 c 5)", "(move king g 1 b 6)", "(move king g 1 a 7)", "(move man i 1 g 3)", "(move king i 1 j 2)", "(move king i 1 h 2)", "(move king i 1 g 3)", "(move king i 1 f 4)", "(move king i 1 e 5)", "(move king i 1 d 6)", "(move king i 1 c 7)", "(move king i 1 b 8)", "(move king i 1 a 9)"};
    // generating for black man 1 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal black (move man b 10 c 9))
        ddi->diagonals.emplace_back(7, 1);

        // to position 12, legal: (legal black (move man b 10 d 8))
        ddi->diagonals.emplace_back(12, 3);

        this->diagonal_data[100].push_back(ddi);


        this->reverse_legal_lookup_black[1] = ReverseLegalLookup(Role::Black, Piece::Man, 1, 7, SE);
        this->reverse_legal_lookup_black[3] = ReverseLegalLookup(Role::Black, Piece::Man, 1, 12, SE);
    }
    // generating for black man 1 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 6, legal: (legal black (move man b 10 a 9))
        ddi->diagonals.emplace_back(6, 2);

        this->diagonal_data[100].push_back(ddi);


        this->reverse_legal_lookup_black[2] = ReverseLegalLookup(Role::Black, Piece::Man, 1, 6, SW);
    }
    // generating for black man 2 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal black (move man d 10 e 9))
        ddi->diagonals.emplace_back(8, 13);

        // to position 13, legal: (legal black (move man d 10 f 8))
        ddi->diagonals.emplace_back(13, 15);

        this->diagonal_data[101].push_back(ddi);


        this->reverse_legal_lookup_black[13] = ReverseLegalLookup(Role::Black, Piece::Man, 2, 8, SE);
        this->reverse_legal_lookup_black[15] = ReverseLegalLookup(Role::Black, Piece::Man, 2, 13, SE);
    }
    // generating for black man 2 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal black (move man d 10 c 9))
        ddi->diagonals.emplace_back(7, 14);

        // to position 11, legal: (legal black (move man d 10 b 8))
        ddi->diagonals.emplace_back(11, 16);

        this->diagonal_data[101].push_back(ddi);


        this->reverse_legal_lookup_black[14] = ReverseLegalLookup(Role::Black, Piece::Man, 2, 7, SW);
        this->reverse_legal_lookup_black[16] = ReverseLegalLookup(Role::Black, Piece::Man, 2, 11, SW);
    }
    // generating for black man 3 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal black (move man f 10 g 9))
        ddi->diagonals.emplace_back(9, 26);

        // to position 14, legal: (legal black (move man f 10 h 8))
        ddi->diagonals.emplace_back(14, 28);

        this->diagonal_data[102].push_back(ddi);


        this->reverse_legal_lookup_black[26] = ReverseLegalLookup(Role::Black, Piece::Man, 3, 9, SE);
        this->reverse_legal_lookup_black[28] = ReverseLegalLookup(Role::Black, Piece::Man, 3, 14, SE);
    }
    // generating for black man 3 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal black (move man f 10 e 9))
        ddi->diagonals.emplace_back(8, 27);

        // to position 12, legal: (legal black (move man f 10 d 8))
        ddi->diagonals.emplace_back(12, 29);

        this->diagonal_data[102].push_back(ddi);


        this->reverse_legal_lookup_black[27] = ReverseLegalLookup(Role::Black, Piece::Man, 3, 8, SW);
        this->reverse_legal_lookup_black[29] = ReverseLegalLookup(Role::Black, Piece::Man, 3, 12, SW);
    }
    // generating for black man 4 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal black (move man h 10 i 9))
        ddi->diagonals.emplace_back(10, 39);

        // to position 15, legal: (legal black (move man h 10 j 8))
        ddi->diagonals.emplace_back(15, 41);

        this->diagonal_data[103].push_back(ddi);


        this->reverse_legal_lookup_black[39] = ReverseLegalLookup(Role::Black, Piece::Man, 4, 10, SE);
        this->reverse_legal_lookup_black[41] = ReverseLegalLookup(Role::Black, Piece::Man, 4, 15, SE);
    }
    // generating for black man 4 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal black (move man h 10 g 9))
        ddi->diagonals.emplace_back(9, 40);

        // to position 13, legal: (legal black (move man h 10 f 8))
        ddi->diagonals.emplace_back(13, 42);

        this->diagonal_data[103].push_back(ddi);


        this->reverse_legal_lookup_black[40] = ReverseLegalLookup(Role::Black, Piece::Man, 4, 9, SW);
        this->reverse_legal_lookup_black[42] = ReverseLegalLookup(Role::Black, Piece::Man, 4, 13, SW);
    }
    // generating for black man 5 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal black (move man j 10 i 9))
        ddi->diagonals.emplace_back(10, 52);

        // to position 14, legal: (legal black (move man j 10 h 8))
        ddi->diagonals.emplace_back(14, 53);

        this->diagonal_data[104].push_back(ddi);


        this->reverse_legal_lookup_black[52] = ReverseLegalLookup(Role::Black, Piece::Man, 5, 10, SW);
        this->reverse_legal_lookup_black[53] = ReverseLegalLookup(Role::Black, Piece::Man, 5, 14, SW);
    }
    // generating for black man 6 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 1, legal: invalid
        ddi->diagonals.emplace_back(1, -781);

        this->diagonal_data[105].push_back(ddi);


    }
    // generating for black man 6 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal black (move man a 9 b 8))
        ddi->diagonals.emplace_back(11, 63);

        // to position 17, legal: (legal black (move man a 9 c 7))
        ddi->diagonals.emplace_back(17, 64);

        this->diagonal_data[105].push_back(ddi);


        this->reverse_legal_lookup_black[63] = ReverseLegalLookup(Role::Black, Piece::Man, 6, 11, SE);
        this->reverse_legal_lookup_black[64] = ReverseLegalLookup(Role::Black, Piece::Man, 6, 17, SE);
    }
    // generating for black man 7 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 2, legal: invalid
        ddi->diagonals.emplace_back(2, -781);

        this->diagonal_data[106].push_back(ddi);


    }
    // generating for black man 7 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 1, legal: invalid
        ddi->diagonals.emplace_back(1, -781);

        this->diagonal_data[106].push_back(ddi);


    }
    // generating for black man 7 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 12, legal: (legal black (move man c 9 d 8))
        ddi->diagonals.emplace_back(12, 74);

        // to position 18, legal: (legal black (move man c 9 e 7))
        ddi->diagonals.emplace_back(18, 76);

        this->diagonal_data[106].push_back(ddi);


        this->reverse_legal_lookup_black[74] = ReverseLegalLookup(Role::Black, Piece::Man, 7, 12, SE);
        this->reverse_legal_lookup_black[76] = ReverseLegalLookup(Role::Black, Piece::Man, 7, 18, SE);
    }
    // generating for black man 7 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal black (move man c 9 b 8))
        ddi->diagonals.emplace_back(11, 75);

        // to position 16, legal: (legal black (move man c 9 a 7))
        ddi->diagonals.emplace_back(16, 77);

        this->diagonal_data[106].push_back(ddi);


        this->reverse_legal_lookup_black[75] = ReverseLegalLookup(Role::Black, Piece::Man, 7, 11, SW);
        this->reverse_legal_lookup_black[77] = ReverseLegalLookup(Role::Black, Piece::Man, 7, 16, SW);
    }
    // generating for black man 8 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 3, legal: invalid
        ddi->diagonals.emplace_back(3, -781);

        this->diagonal_data[107].push_back(ddi);


    }
    // generating for black man 8 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 2, legal: invalid
        ddi->diagonals.emplace_back(2, -781);

        this->diagonal_data[107].push_back(ddi);


    }
    // generating for black man 8 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 13, legal: (legal black (move man e 9 f 8))
        ddi->diagonals.emplace_back(13, 89);

        // to position 19, legal: (legal black (move man e 9 g 7))
        ddi->diagonals.emplace_back(19, 91);

        this->diagonal_data[107].push_back(ddi);


        this->reverse_legal_lookup_black[89] = ReverseLegalLookup(Role::Black, Piece::Man, 8, 13, SE);
        this->reverse_legal_lookup_black[91] = ReverseLegalLookup(Role::Black, Piece::Man, 8, 19, SE);
    }
    // generating for black man 8 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 12, legal: (legal black (move man e 9 d 8))
        ddi->diagonals.emplace_back(12, 90);

        // to position 17, legal: (legal black (move man e 9 c 7))
        ddi->diagonals.emplace_back(17, 92);

        this->diagonal_data[107].push_back(ddi);


        this->reverse_legal_lookup_black[90] = ReverseLegalLookup(Role::Black, Piece::Man, 8, 12, SW);
        this->reverse_legal_lookup_black[92] = ReverseLegalLookup(Role::Black, Piece::Man, 8, 17, SW);
    }
    // generating for black man 9 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 4, legal: invalid
        ddi->diagonals.emplace_back(4, -781);

        this->diagonal_data[108].push_back(ddi);


    }
    // generating for black man 9 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 3, legal: invalid
        ddi->diagonals.emplace_back(3, -781);

        this->diagonal_data[108].push_back(ddi);


    }
    // generating for black man 9 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 14, legal: (legal black (move man g 9 h 8))
        ddi->diagonals.emplace_back(14, 104);

        // to position 20, legal: (legal black (move man g 9 i 7))
        ddi->diagonals.emplace_back(20, 106);

        this->diagonal_data[108].push_back(ddi);


        this->reverse_legal_lookup_black[104] = ReverseLegalLookup(Role::Black, Piece::Man, 9, 14, SE);
        this->reverse_legal_lookup_black[106] = ReverseLegalLookup(Role::Black, Piece::Man, 9, 20, SE);
    }
    // generating for black man 9 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 13, legal: (legal black (move man g 9 f 8))
        ddi->diagonals.emplace_back(13, 105);

        // to position 18, legal: (legal black (move man g 9 e 7))
        ddi->diagonals.emplace_back(18, 107);

        this->diagonal_data[108].push_back(ddi);


        this->reverse_legal_lookup_black[105] = ReverseLegalLookup(Role::Black, Piece::Man, 9, 13, SW);
        this->reverse_legal_lookup_black[107] = ReverseLegalLookup(Role::Black, Piece::Man, 9, 18, SW);
    }
    // generating for black man 10 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 5, legal: invalid
        ddi->diagonals.emplace_back(5, -781);

        this->diagonal_data[109].push_back(ddi);


    }
    // generating for black man 10 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 4, legal: invalid
        ddi->diagonals.emplace_back(4, -781);

        this->diagonal_data[109].push_back(ddi);


    }
    // generating for black man 10 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 15, legal: (legal black (move man i 9 j 8))
        ddi->diagonals.emplace_back(15, 119);

        this->diagonal_data[109].push_back(ddi);


        this->reverse_legal_lookup_black[119] = ReverseLegalLookup(Role::Black, Piece::Man, 10, 15, SE);
    }
    // generating for black man 10 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 14, legal: (legal black (move man i 9 h 8))
        ddi->diagonals.emplace_back(14, 120);

        // to position 19, legal: (legal black (move man i 9 g 7))
        ddi->diagonals.emplace_back(19, 121);

        this->diagonal_data[109].push_back(ddi);


        this->reverse_legal_lookup_black[120] = ReverseLegalLookup(Role::Black, Piece::Man, 10, 14, SW);
        this->reverse_legal_lookup_black[121] = ReverseLegalLookup(Role::Black, Piece::Man, 10, 19, SW);
    }
    // generating for black man 11 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: invalid
        ddi->diagonals.emplace_back(7, -781);

        // to position 2, legal: (legal black (move man b 8 d 10))
        ddi->diagonals.emplace_back(2, 135);

        this->diagonal_data[110].push_back(ddi);


        this->reverse_legal_lookup_black[135] = ReverseLegalLookup(Role::Black, Piece::Man, 11, 2, NE);
    }
    // generating for black man 11 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 6, legal: invalid
        ddi->diagonals.emplace_back(6, -781);

        this->diagonal_data[110].push_back(ddi);


    }
    // generating for black man 11 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal black (move man b 8 c 7))
        ddi->diagonals.emplace_back(17, 133);

        // to position 22, legal: (legal black (move man b 8 d 6))
        ddi->diagonals.emplace_back(22, 136);

        this->diagonal_data[110].push_back(ddi);


        this->reverse_legal_lookup_black[133] = ReverseLegalLookup(Role::Black, Piece::Man, 11, 17, SE);
        this->reverse_legal_lookup_black[136] = ReverseLegalLookup(Role::Black, Piece::Man, 11, 22, SE);
    }
    // generating for black man 11 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 16, legal: (legal black (move man b 8 a 7))
        ddi->diagonals.emplace_back(16, 134);

        this->diagonal_data[110].push_back(ddi);


        this->reverse_legal_lookup_black[134] = ReverseLegalLookup(Role::Black, Piece::Man, 11, 16, SW);
    }
    // generating for black man 12 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: invalid
        ddi->diagonals.emplace_back(8, -781);

        // to position 3, legal: (legal black (move man d 8 f 10))
        ddi->diagonals.emplace_back(3, 150);

        this->diagonal_data[111].push_back(ddi);


        this->reverse_legal_lookup_black[150] = ReverseLegalLookup(Role::Black, Piece::Man, 12, 3, NE);
    }
    // generating for black man 12 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: invalid
        ddi->diagonals.emplace_back(7, -781);

        // to position 1, legal: (legal black (move man d 8 b 10))
        ddi->diagonals.emplace_back(1, 151);

        this->diagonal_data[111].push_back(ddi);


        this->reverse_legal_lookup_black[151] = ReverseLegalLookup(Role::Black, Piece::Man, 12, 1, NW);
    }
    // generating for black man 12 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 18, legal: (legal black (move man d 8 e 7))
        ddi->diagonals.emplace_back(18, 148);

        // to position 23, legal: (legal black (move man d 8 f 6))
        ddi->diagonals.emplace_back(23, 152);

        this->diagonal_data[111].push_back(ddi);


        this->reverse_legal_lookup_black[148] = ReverseLegalLookup(Role::Black, Piece::Man, 12, 18, SE);
        this->reverse_legal_lookup_black[152] = ReverseLegalLookup(Role::Black, Piece::Man, 12, 23, SE);
    }
    // generating for black man 12 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: (legal black (move man d 8 c 7))
        ddi->diagonals.emplace_back(17, 149);

        // to position 21, legal: (legal black (move man d 8 b 6))
        ddi->diagonals.emplace_back(21, 153);

        this->diagonal_data[111].push_back(ddi);


        this->reverse_legal_lookup_black[149] = ReverseLegalLookup(Role::Black, Piece::Man, 12, 17, SW);
        this->reverse_legal_lookup_black[153] = ReverseLegalLookup(Role::Black, Piece::Man, 12, 21, SW);
    }
    // generating for black man 13 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 9, legal: invalid
        ddi->diagonals.emplace_back(9, -781);

        // to position 4, legal: (legal black (move man f 8 h 10))
        ddi->diagonals.emplace_back(4, 169);

        this->diagonal_data[112].push_back(ddi);


        this->reverse_legal_lookup_black[169] = ReverseLegalLookup(Role::Black, Piece::Man, 13, 4, NE);
    }
    // generating for black man 13 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: invalid
        ddi->diagonals.emplace_back(8, -781);

        // to position 2, legal: (legal black (move man f 8 d 10))
        ddi->diagonals.emplace_back(2, 170);

        this->diagonal_data[112].push_back(ddi);


        this->reverse_legal_lookup_black[170] = ReverseLegalLookup(Role::Black, Piece::Man, 13, 2, NW);
    }
    // generating for black man 13 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 19, legal: (legal black (move man f 8 g 7))
        ddi->diagonals.emplace_back(19, 167);

        // to position 24, legal: (legal black (move man f 8 h 6))
        ddi->diagonals.emplace_back(24, 171);

        this->diagonal_data[112].push_back(ddi);


        this->reverse_legal_lookup_black[167] = ReverseLegalLookup(Role::Black, Piece::Man, 13, 19, SE);
        this->reverse_legal_lookup_black[171] = ReverseLegalLookup(Role::Black, Piece::Man, 13, 24, SE);
    }
    // generating for black man 13 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 18, legal: (legal black (move man f 8 e 7))
        ddi->diagonals.emplace_back(18, 168);

        // to position 22, legal: (legal black (move man f 8 d 6))
        ddi->diagonals.emplace_back(22, 172);

        this->diagonal_data[112].push_back(ddi);


        this->reverse_legal_lookup_black[168] = ReverseLegalLookup(Role::Black, Piece::Man, 13, 18, SW);
        this->reverse_legal_lookup_black[172] = ReverseLegalLookup(Role::Black, Piece::Man, 13, 22, SW);
    }
    // generating for black man 14 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: invalid
        ddi->diagonals.emplace_back(10, -781);

        // to position 5, legal: (legal black (move man h 8 j 10))
        ddi->diagonals.emplace_back(5, 188);

        this->diagonal_data[113].push_back(ddi);


        this->reverse_legal_lookup_black[188] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 5, NE);
    }
    // generating for black man 14 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: invalid
        ddi->diagonals.emplace_back(9, -781);

        // to position 3, legal: (legal black (move man h 8 f 10))
        ddi->diagonals.emplace_back(3, 189);

        this->diagonal_data[113].push_back(ddi);


        this->reverse_legal_lookup_black[189] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 3, NW);
    }
    // generating for black man 14 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 20, legal: (legal black (move man h 8 i 7))
        ddi->diagonals.emplace_back(20, 186);

        // to position 25, legal: (legal black (move man h 8 j 6))
        ddi->diagonals.emplace_back(25, 190);

        this->diagonal_data[113].push_back(ddi);


        this->reverse_legal_lookup_black[186] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 20, SE);
        this->reverse_legal_lookup_black[190] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 25, SE);
    }
    // generating for black man 14 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 19, legal: (legal black (move man h 8 g 7))
        ddi->diagonals.emplace_back(19, 187);

        // to position 23, legal: (legal black (move man h 8 f 6))
        ddi->diagonals.emplace_back(23, 191);

        this->diagonal_data[113].push_back(ddi);


        this->reverse_legal_lookup_black[187] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 19, SW);
        this->reverse_legal_lookup_black[191] = ReverseLegalLookup(Role::Black, Piece::Man, 14, 23, SW);
    }
    // generating for black man 15 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 10, legal: invalid
        ddi->diagonals.emplace_back(10, -781);

        // to position 4, legal: (legal black (move man j 8 h 10))
        ddi->diagonals.emplace_back(4, 206);

        this->diagonal_data[114].push_back(ddi);


        this->reverse_legal_lookup_black[206] = ReverseLegalLookup(Role::Black, Piece::Man, 15, 4, NW);
    }
    // generating for black man 15 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 20, legal: (legal black (move man j 8 i 7))
        ddi->diagonals.emplace_back(20, 205);

        // to position 24, legal: (legal black (move man j 8 h 6))
        ddi->diagonals.emplace_back(24, 207);

        this->diagonal_data[114].push_back(ddi);


        this->reverse_legal_lookup_black[205] = ReverseLegalLookup(Role::Black, Piece::Man, 15, 20, SW);
        this->reverse_legal_lookup_black[207] = ReverseLegalLookup(Role::Black, Piece::Man, 15, 24, SW);
    }
    // generating for black man 16 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 11, legal: invalid
        ddi->diagonals.emplace_back(11, -781);

        // to position 7, legal: (legal black (move man a 7 c 9))
        ddi->diagonals.emplace_back(7, 218);

        this->diagonal_data[115].push_back(ddi);


        this->reverse_legal_lookup_black[218] = ReverseLegalLookup(Role::Black, Piece::Man, 16, 7, NE);
    }
    // generating for black man 16 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 21, legal: (legal black (move man a 7 b 6))
        ddi->diagonals.emplace_back(21, 217);

        // to position 27, legal: (legal black (move man a 7 c 5))
        ddi->diagonals.emplace_back(27, 219);

        this->diagonal_data[115].push_back(ddi);


        this->reverse_legal_lookup_black[217] = ReverseLegalLookup(Role::Black, Piece::Man, 16, 21, SE);
        this->reverse_legal_lookup_black[219] = ReverseLegalLookup(Role::Black, Piece::Man, 16, 27, SE);
    }
    // generating for black man 17 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 12, legal: invalid
        ddi->diagonals.emplace_back(12, -781);

        // to position 8, legal: (legal black (move man c 7 e 9))
        ddi->diagonals.emplace_back(8, 231);

        this->diagonal_data[116].push_back(ddi);


        this->reverse_legal_lookup_black[231] = ReverseLegalLookup(Role::Black, Piece::Man, 17, 8, NE);
    }
    // generating for black man 17 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: invalid
        ddi->diagonals.emplace_back(11, -781);

        // to position 6, legal: (legal black (move man c 7 a 9))
        ddi->diagonals.emplace_back(6, 232);

        this->diagonal_data[116].push_back(ddi);


        this->reverse_legal_lookup_black[232] = ReverseLegalLookup(Role::Black, Piece::Man, 17, 6, NW);
    }
    // generating for black man 17 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 22, legal: (legal black (move man c 7 d 6))
        ddi->diagonals.emplace_back(22, 229);

        // to position 28, legal: (legal black (move man c 7 e 5))
        ddi->diagonals.emplace_back(28, 233);

        this->diagonal_data[116].push_back(ddi);


        this->reverse_legal_lookup_black[229] = ReverseLegalLookup(Role::Black, Piece::Man, 17, 22, SE);
        this->reverse_legal_lookup_black[233] = ReverseLegalLookup(Role::Black, Piece::Man, 17, 28, SE);
    }
    // generating for black man 17 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 21, legal: (legal black (move man c 7 b 6))
        ddi->diagonals.emplace_back(21, 230);

        // to position 26, legal: (legal black (move man c 7 a 5))
        ddi->diagonals.emplace_back(26, 234);

        this->diagonal_data[116].push_back(ddi);


        this->reverse_legal_lookup_black[230] = ReverseLegalLookup(Role::Black, Piece::Man, 17, 21, SW);
        this->reverse_legal_lookup_black[234] = ReverseLegalLookup(Role::Black, Piece::Man, 17, 26, SW);
    }
    // generating for black man 18 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 13, legal: invalid
        ddi->diagonals.emplace_back(13, -781);

        // to position 9, legal: (legal black (move man e 7 g 9))
        ddi->diagonals.emplace_back(9, 250);

        this->diagonal_data[117].push_back(ddi);


        this->reverse_legal_lookup_black[250] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 9, NE);
    }
    // generating for black man 18 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 12, legal: invalid
        ddi->diagonals.emplace_back(12, -781);

        // to position 7, legal: (legal black (move man e 7 c 9))
        ddi->diagonals.emplace_back(7, 251);

        this->diagonal_data[117].push_back(ddi);


        this->reverse_legal_lookup_black[251] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 7, NW);
    }
    // generating for black man 18 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 23, legal: (legal black (move man e 7 f 6))
        ddi->diagonals.emplace_back(23, 248);

        // to position 29, legal: (legal black (move man e 7 g 5))
        ddi->diagonals.emplace_back(29, 252);

        this->diagonal_data[117].push_back(ddi);


        this->reverse_legal_lookup_black[248] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 23, SE);
        this->reverse_legal_lookup_black[252] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 29, SE);
    }
    // generating for black man 18 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 22, legal: (legal black (move man e 7 d 6))
        ddi->diagonals.emplace_back(22, 249);

        // to position 27, legal: (legal black (move man e 7 c 5))
        ddi->diagonals.emplace_back(27, 253);

        this->diagonal_data[117].push_back(ddi);


        this->reverse_legal_lookup_black[249] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 22, SW);
        this->reverse_legal_lookup_black[253] = ReverseLegalLookup(Role::Black, Piece::Man, 18, 27, SW);
    }
    // generating for black man 19 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 14, legal: invalid
        ddi->diagonals.emplace_back(14, -781);

        // to position 10, legal: (legal black (move man g 7 i 9))
        ddi->diagonals.emplace_back(10, 271);

        this->diagonal_data[118].push_back(ddi);


        this->reverse_legal_lookup_black[271] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 10, NE);
    }
    // generating for black man 19 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 13, legal: invalid
        ddi->diagonals.emplace_back(13, -781);

        // to position 8, legal: (legal black (move man g 7 e 9))
        ddi->diagonals.emplace_back(8, 272);

        this->diagonal_data[118].push_back(ddi);


        this->reverse_legal_lookup_black[272] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 8, NW);
    }
    // generating for black man 19 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal black (move man g 7 h 6))
        ddi->diagonals.emplace_back(24, 269);

        // to position 30, legal: (legal black (move man g 7 i 5))
        ddi->diagonals.emplace_back(30, 273);

        this->diagonal_data[118].push_back(ddi);


        this->reverse_legal_lookup_black[269] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 24, SE);
        this->reverse_legal_lookup_black[273] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 30, SE);
    }
    // generating for black man 19 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 23, legal: (legal black (move man g 7 f 6))
        ddi->diagonals.emplace_back(23, 270);

        // to position 28, legal: (legal black (move man g 7 e 5))
        ddi->diagonals.emplace_back(28, 274);

        this->diagonal_data[118].push_back(ddi);


        this->reverse_legal_lookup_black[270] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 23, SW);
        this->reverse_legal_lookup_black[274] = ReverseLegalLookup(Role::Black, Piece::Man, 19, 28, SW);
    }
    // generating for black man 20 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 15, legal: invalid
        ddi->diagonals.emplace_back(15, -781);

        this->diagonal_data[119].push_back(ddi);


    }
    // generating for black man 20 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 14, legal: invalid
        ddi->diagonals.emplace_back(14, -781);

        // to position 9, legal: (legal black (move man i 7 g 9))
        ddi->diagonals.emplace_back(9, 292);

        this->diagonal_data[119].push_back(ddi);


        this->reverse_legal_lookup_black[292] = ReverseLegalLookup(Role::Black, Piece::Man, 20, 9, NW);
    }
    // generating for black man 20 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 25, legal: (legal black (move man i 7 j 6))
        ddi->diagonals.emplace_back(25, 290);

        this->diagonal_data[119].push_back(ddi);


        this->reverse_legal_lookup_black[290] = ReverseLegalLookup(Role::Black, Piece::Man, 20, 25, SE);
    }
    // generating for black man 20 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 24, legal: (legal black (move man i 7 h 6))
        ddi->diagonals.emplace_back(24, 291);

        // to position 29, legal: (legal black (move man i 7 g 5))
        ddi->diagonals.emplace_back(29, 293);

        this->diagonal_data[119].push_back(ddi);


        this->reverse_legal_lookup_black[291] = ReverseLegalLookup(Role::Black, Piece::Man, 20, 24, SW);
        this->reverse_legal_lookup_black[293] = ReverseLegalLookup(Role::Black, Piece::Man, 20, 29, SW);
    }
    // generating for black man 21 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 17, legal: invalid
        ddi->diagonals.emplace_back(17, -781);

        // to position 12, legal: (legal black (move man b 6 d 8))
        ddi->diagonals.emplace_back(12, 307);

        this->diagonal_data[120].push_back(ddi);


        this->reverse_legal_lookup_black[307] = ReverseLegalLookup(Role::Black, Piece::Man, 21, 12, NE);
    }
    // generating for black man 21 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 16, legal: invalid
        ddi->diagonals.emplace_back(16, -781);

        this->diagonal_data[120].push_back(ddi);


    }
    // generating for black man 21 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal black (move man b 6 c 5))
        ddi->diagonals.emplace_back(27, 305);

        // to position 32, legal: (legal black (move man b 6 d 4))
        ddi->diagonals.emplace_back(32, 308);

        this->diagonal_data[120].push_back(ddi);


        this->reverse_legal_lookup_black[305] = ReverseLegalLookup(Role::Black, Piece::Man, 21, 27, SE);
        this->reverse_legal_lookup_black[308] = ReverseLegalLookup(Role::Black, Piece::Man, 21, 32, SE);
    }
    // generating for black man 21 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 26, legal: (legal black (move man b 6 a 5))
        ddi->diagonals.emplace_back(26, 306);

        this->diagonal_data[120].push_back(ddi);


        this->reverse_legal_lookup_black[306] = ReverseLegalLookup(Role::Black, Piece::Man, 21, 26, SW);
    }
    // generating for black man 22 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 18, legal: invalid
        ddi->diagonals.emplace_back(18, -781);

        // to position 13, legal: (legal black (move man d 6 f 8))
        ddi->diagonals.emplace_back(13, 322);

        this->diagonal_data[121].push_back(ddi);


        this->reverse_legal_lookup_black[322] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 13, NE);
    }
    // generating for black man 22 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 17, legal: invalid
        ddi->diagonals.emplace_back(17, -781);

        // to position 11, legal: (legal black (move man d 6 b 8))
        ddi->diagonals.emplace_back(11, 323);

        this->diagonal_data[121].push_back(ddi);


        this->reverse_legal_lookup_black[323] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 11, NW);
    }
    // generating for black man 22 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 28, legal: (legal black (move man d 6 e 5))
        ddi->diagonals.emplace_back(28, 320);

        // to position 33, legal: (legal black (move man d 6 f 4))
        ddi->diagonals.emplace_back(33, 324);

        this->diagonal_data[121].push_back(ddi);


        this->reverse_legal_lookup_black[320] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 28, SE);
        this->reverse_legal_lookup_black[324] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 33, SE);
    }
    // generating for black man 22 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 27, legal: (legal black (move man d 6 c 5))
        ddi->diagonals.emplace_back(27, 321);

        // to position 31, legal: (legal black (move man d 6 b 4))
        ddi->diagonals.emplace_back(31, 325);

        this->diagonal_data[121].push_back(ddi);


        this->reverse_legal_lookup_black[321] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 27, SW);
        this->reverse_legal_lookup_black[325] = ReverseLegalLookup(Role::Black, Piece::Man, 22, 31, SW);
    }
    // generating for black man 23 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 19, legal: invalid
        ddi->diagonals.emplace_back(19, -781);

        // to position 14, legal: (legal black (move man f 6 h 8))
        ddi->diagonals.emplace_back(14, 343);

        this->diagonal_data[122].push_back(ddi);


        this->reverse_legal_lookup_black[343] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 14, NE);
    }
    // generating for black man 23 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 18, legal: invalid
        ddi->diagonals.emplace_back(18, -781);

        // to position 12, legal: (legal black (move man f 6 d 8))
        ddi->diagonals.emplace_back(12, 344);

        this->diagonal_data[122].push_back(ddi);


        this->reverse_legal_lookup_black[344] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 12, NW);
    }
    // generating for black man 23 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 29, legal: (legal black (move man f 6 g 5))
        ddi->diagonals.emplace_back(29, 341);

        // to position 34, legal: (legal black (move man f 6 h 4))
        ddi->diagonals.emplace_back(34, 345);

        this->diagonal_data[122].push_back(ddi);


        this->reverse_legal_lookup_black[341] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 29, SE);
        this->reverse_legal_lookup_black[345] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 34, SE);
    }
    // generating for black man 23 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 28, legal: (legal black (move man f 6 e 5))
        ddi->diagonals.emplace_back(28, 342);

        // to position 32, legal: (legal black (move man f 6 d 4))
        ddi->diagonals.emplace_back(32, 346);

        this->diagonal_data[122].push_back(ddi);


        this->reverse_legal_lookup_black[342] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 28, SW);
        this->reverse_legal_lookup_black[346] = ReverseLegalLookup(Role::Black, Piece::Man, 23, 32, SW);
    }
    // generating for black man 24 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 20, legal: invalid
        ddi->diagonals.emplace_back(20, -781);

        // to position 15, legal: (legal black (move man h 6 j 8))
        ddi->diagonals.emplace_back(15, 366);

        this->diagonal_data[123].push_back(ddi);


        this->reverse_legal_lookup_black[366] = ReverseLegalLookup(Role::Black, Piece::Man, 24, 15, NE);
    }
    // generating for black man 24 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 19, legal: invalid
        ddi->diagonals.emplace_back(19, -781);

        // to position 13, legal: (legal black (move man h 6 f 8))
        ddi->diagonals.emplace_back(13, 367);

        this->diagonal_data[123].push_back(ddi);


        this->reverse_legal_lookup_black[367] = ReverseLegalLookup(Role::Black, Piece::Man, 24, 13, NW);
    }
    // generating for black man 24 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 30, legal: (legal black (move man h 6 i 5))
        ddi->diagonals.emplace_back(30, 364);

        // to position 35, legal: (legal black (move man h 6 j 4))
        ddi->diagonals.emplace_back(35, 368);

        this->diagonal_data[123].push_back(ddi);


        this->reverse_legal_lookup_black[364] = ReverseLegalLookup(Role::Black, Piece::Man, 24, 30, SE);
        this->reverse_legal_lookup_black[368] = ReverseLegalLookup(Role::Black, Piece::Man, 24, 35, SE);
    }
    // generating for black man 24 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 29, legal: (legal black (move man h 6 g 5))
        ddi->diagonals.emplace_back(29, 365);

        // to position 33, legal: (legal black (move man h 6 f 4))
        ddi->diagonals.emplace_back(33, 369);

        this->diagonal_data[123].push_back(ddi);


        this->reverse_legal_lookup_black[365] = ReverseLegalLookup(Role::Black, Piece::Man, 24, 29, SW);
        this->reverse_legal_lookup_black[369] = ReverseLegalLookup(Role::Black, Piece::Man, 24, 33, SW);
    }
    // generating for black man 25 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 20, legal: invalid
        ddi->diagonals.emplace_back(20, -781);

        // to position 14, legal: (legal black (move man j 6 h 8))
        ddi->diagonals.emplace_back(14, 384);

        this->diagonal_data[124].push_back(ddi);


        this->reverse_legal_lookup_black[384] = ReverseLegalLookup(Role::Black, Piece::Man, 25, 14, NW);
    }
    // generating for black man 25 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 30, legal: (legal black (move man j 6 i 5))
        ddi->diagonals.emplace_back(30, 383);

        // to position 34, legal: (legal black (move man j 6 h 4))
        ddi->diagonals.emplace_back(34, 385);

        this->diagonal_data[124].push_back(ddi);


        this->reverse_legal_lookup_black[383] = ReverseLegalLookup(Role::Black, Piece::Man, 25, 30, SW);
        this->reverse_legal_lookup_black[385] = ReverseLegalLookup(Role::Black, Piece::Man, 25, 34, SW);
    }
    // generating for black man 26 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 21, legal: invalid
        ddi->diagonals.emplace_back(21, -781);

        // to position 17, legal: (legal black (move man a 5 c 7))
        ddi->diagonals.emplace_back(17, 396);

        this->diagonal_data[125].push_back(ddi);


        this->reverse_legal_lookup_black[396] = ReverseLegalLookup(Role::Black, Piece::Man, 26, 17, NE);
    }
    // generating for black man 26 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 31, legal: (legal black (move man a 5 b 4))
        ddi->diagonals.emplace_back(31, 395);

        // to position 37, legal: (legal black (move man a 5 c 3))
        ddi->diagonals.emplace_back(37, 397);

        this->diagonal_data[125].push_back(ddi);


        this->reverse_legal_lookup_black[395] = ReverseLegalLookup(Role::Black, Piece::Man, 26, 31, SE);
        this->reverse_legal_lookup_black[397] = ReverseLegalLookup(Role::Black, Piece::Man, 26, 37, SE);
    }
    // generating for black man 27 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 22, legal: invalid
        ddi->diagonals.emplace_back(22, -781);

        // to position 18, legal: (legal black (move man c 5 e 7))
        ddi->diagonals.emplace_back(18, 409);

        this->diagonal_data[126].push_back(ddi);


        this->reverse_legal_lookup_black[409] = ReverseLegalLookup(Role::Black, Piece::Man, 27, 18, NE);
    }
    // generating for black man 27 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 21, legal: invalid
        ddi->diagonals.emplace_back(21, -781);

        // to position 16, legal: (legal black (move man c 5 a 7))
        ddi->diagonals.emplace_back(16, 410);

        this->diagonal_data[126].push_back(ddi);


        this->reverse_legal_lookup_black[410] = ReverseLegalLookup(Role::Black, Piece::Man, 27, 16, NW);
    }
    // generating for black man 27 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 32, legal: (legal black (move man c 5 d 4))
        ddi->diagonals.emplace_back(32, 407);

        // to position 38, legal: (legal black (move man c 5 e 3))
        ddi->diagonals.emplace_back(38, 411);

        this->diagonal_data[126].push_back(ddi);


        this->reverse_legal_lookup_black[407] = ReverseLegalLookup(Role::Black, Piece::Man, 27, 32, SE);
        this->reverse_legal_lookup_black[411] = ReverseLegalLookup(Role::Black, Piece::Man, 27, 38, SE);
    }
    // generating for black man 27 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 31, legal: (legal black (move man c 5 b 4))
        ddi->diagonals.emplace_back(31, 408);

        // to position 36, legal: (legal black (move man c 5 a 3))
        ddi->diagonals.emplace_back(36, 412);

        this->diagonal_data[126].push_back(ddi);


        this->reverse_legal_lookup_black[408] = ReverseLegalLookup(Role::Black, Piece::Man, 27, 31, SW);
        this->reverse_legal_lookup_black[412] = ReverseLegalLookup(Role::Black, Piece::Man, 27, 36, SW);
    }
    // generating for black man 28 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 23, legal: invalid
        ddi->diagonals.emplace_back(23, -781);

        // to position 19, legal: (legal black (move man e 5 g 7))
        ddi->diagonals.emplace_back(19, 428);

        this->diagonal_data[127].push_back(ddi);


        this->reverse_legal_lookup_black[428] = ReverseLegalLookup(Role::Black, Piece::Man, 28, 19, NE);
    }
    // generating for black man 28 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 22, legal: invalid
        ddi->diagonals.emplace_back(22, -781);

        // to position 17, legal: (legal black (move man e 5 c 7))
        ddi->diagonals.emplace_back(17, 429);

        this->diagonal_data[127].push_back(ddi);


        this->reverse_legal_lookup_black[429] = ReverseLegalLookup(Role::Black, Piece::Man, 28, 17, NW);
    }
    // generating for black man 28 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 33, legal: (legal black (move man e 5 f 4))
        ddi->diagonals.emplace_back(33, 426);

        // to position 39, legal: (legal black (move man e 5 g 3))
        ddi->diagonals.emplace_back(39, 430);

        this->diagonal_data[127].push_back(ddi);


        this->reverse_legal_lookup_black[426] = ReverseLegalLookup(Role::Black, Piece::Man, 28, 33, SE);
        this->reverse_legal_lookup_black[430] = ReverseLegalLookup(Role::Black, Piece::Man, 28, 39, SE);
    }
    // generating for black man 28 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 32, legal: (legal black (move man e 5 d 4))
        ddi->diagonals.emplace_back(32, 427);

        // to position 37, legal: (legal black (move man e 5 c 3))
        ddi->diagonals.emplace_back(37, 431);

        this->diagonal_data[127].push_back(ddi);


        this->reverse_legal_lookup_black[427] = ReverseLegalLookup(Role::Black, Piece::Man, 28, 32, SW);
        this->reverse_legal_lookup_black[431] = ReverseLegalLookup(Role::Black, Piece::Man, 28, 37, SW);
    }
    // generating for black man 29 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 24, legal: invalid
        ddi->diagonals.emplace_back(24, -781);

        // to position 20, legal: (legal black (move man g 5 i 7))
        ddi->diagonals.emplace_back(20, 451);

        this->diagonal_data[128].push_back(ddi);


        this->reverse_legal_lookup_black[451] = ReverseLegalLookup(Role::Black, Piece::Man, 29, 20, NE);
    }
    // generating for black man 29 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 23, legal: invalid
        ddi->diagonals.emplace_back(23, -781);

        // to position 18, legal: (legal black (move man g 5 e 7))
        ddi->diagonals.emplace_back(18, 452);

        this->diagonal_data[128].push_back(ddi);


        this->reverse_legal_lookup_black[452] = ReverseLegalLookup(Role::Black, Piece::Man, 29, 18, NW);
    }
    // generating for black man 29 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 34, legal: (legal black (move man g 5 h 4))
        ddi->diagonals.emplace_back(34, 449);

        // to position 40, legal: (legal black (move man g 5 i 3))
        ddi->diagonals.emplace_back(40, 453);

        this->diagonal_data[128].push_back(ddi);


        this->reverse_legal_lookup_black[449] = ReverseLegalLookup(Role::Black, Piece::Man, 29, 34, SE);
        this->reverse_legal_lookup_black[453] = ReverseLegalLookup(Role::Black, Piece::Man, 29, 40, SE);
    }
    // generating for black man 29 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 33, legal: (legal black (move man g 5 f 4))
        ddi->diagonals.emplace_back(33, 450);

        // to position 38, legal: (legal black (move man g 5 e 3))
        ddi->diagonals.emplace_back(38, 454);

        this->diagonal_data[128].push_back(ddi);


        this->reverse_legal_lookup_black[450] = ReverseLegalLookup(Role::Black, Piece::Man, 29, 33, SW);
        this->reverse_legal_lookup_black[454] = ReverseLegalLookup(Role::Black, Piece::Man, 29, 38, SW);
    }
    // generating for black man 30 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 25, legal: invalid
        ddi->diagonals.emplace_back(25, -781);

        this->diagonal_data[129].push_back(ddi);


    }
    // generating for black man 30 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 24, legal: invalid
        ddi->diagonals.emplace_back(24, -781);

        // to position 19, legal: (legal black (move man i 5 g 7))
        ddi->diagonals.emplace_back(19, 472);

        this->diagonal_data[129].push_back(ddi);


        this->reverse_legal_lookup_black[472] = ReverseLegalLookup(Role::Black, Piece::Man, 30, 19, NW);
    }
    // generating for black man 30 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 35, legal: (legal black (move man i 5 j 4))
        ddi->diagonals.emplace_back(35, 470);

        this->diagonal_data[129].push_back(ddi);


        this->reverse_legal_lookup_black[470] = ReverseLegalLookup(Role::Black, Piece::Man, 30, 35, SE);
    }
    // generating for black man 30 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 34, legal: (legal black (move man i 5 h 4))
        ddi->diagonals.emplace_back(34, 471);

        // to position 39, legal: (legal black (move man i 5 g 3))
        ddi->diagonals.emplace_back(39, 473);

        this->diagonal_data[129].push_back(ddi);


        this->reverse_legal_lookup_black[471] = ReverseLegalLookup(Role::Black, Piece::Man, 30, 34, SW);
        this->reverse_legal_lookup_black[473] = ReverseLegalLookup(Role::Black, Piece::Man, 30, 39, SW);
    }
    // generating for black man 31 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 27, legal: invalid
        ddi->diagonals.emplace_back(27, -781);

        // to position 22, legal: (legal black (move man b 4 d 6))
        ddi->diagonals.emplace_back(22, 487);

        this->diagonal_data[130].push_back(ddi);


        this->reverse_legal_lookup_black[487] = ReverseLegalLookup(Role::Black, Piece::Man, 31, 22, NE);
    }
    // generating for black man 31 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 26, legal: invalid
        ddi->diagonals.emplace_back(26, -781);

        this->diagonal_data[130].push_back(ddi);


    }
    // generating for black man 31 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 37, legal: (legal black (move man b 4 c 3))
        ddi->diagonals.emplace_back(37, 485);

        // to position 42, legal: (legal black (move man b 4 d 2))
        ddi->diagonals.emplace_back(42, 488);

        this->diagonal_data[130].push_back(ddi);


        this->reverse_legal_lookup_black[485] = ReverseLegalLookup(Role::Black, Piece::Man, 31, 37, SE);
        this->reverse_legal_lookup_black[488] = ReverseLegalLookup(Role::Black, Piece::Man, 31, 42, SE);
    }
    // generating for black man 31 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 36, legal: (legal black (move man b 4 a 3))
        ddi->diagonals.emplace_back(36, 486);

        this->diagonal_data[130].push_back(ddi);


        this->reverse_legal_lookup_black[486] = ReverseLegalLookup(Role::Black, Piece::Man, 31, 36, SW);
    }
    // generating for black man 32 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 28, legal: invalid
        ddi->diagonals.emplace_back(28, -781);

        // to position 23, legal: (legal black (move man d 4 f 6))
        ddi->diagonals.emplace_back(23, 502);

        this->diagonal_data[131].push_back(ddi);


        this->reverse_legal_lookup_black[502] = ReverseLegalLookup(Role::Black, Piece::Man, 32, 23, NE);
    }
    // generating for black man 32 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 27, legal: invalid
        ddi->diagonals.emplace_back(27, -781);

        // to position 21, legal: (legal black (move man d 4 b 6))
        ddi->diagonals.emplace_back(21, 503);

        this->diagonal_data[131].push_back(ddi);


        this->reverse_legal_lookup_black[503] = ReverseLegalLookup(Role::Black, Piece::Man, 32, 21, NW);
    }
    // generating for black man 32 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 38, legal: (legal black (move man d 4 e 3))
        ddi->diagonals.emplace_back(38, 500);

        // to position 43, legal: (legal black (move man d 4 f 2))
        ddi->diagonals.emplace_back(43, 504);

        this->diagonal_data[131].push_back(ddi);


        this->reverse_legal_lookup_black[500] = ReverseLegalLookup(Role::Black, Piece::Man, 32, 38, SE);
        this->reverse_legal_lookup_black[504] = ReverseLegalLookup(Role::Black, Piece::Man, 32, 43, SE);
    }
    // generating for black man 32 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 37, legal: (legal black (move man d 4 c 3))
        ddi->diagonals.emplace_back(37, 501);

        // to position 41, legal: (legal black (move man d 4 b 2))
        ddi->diagonals.emplace_back(41, 505);

        this->diagonal_data[131].push_back(ddi);


        this->reverse_legal_lookup_black[501] = ReverseLegalLookup(Role::Black, Piece::Man, 32, 37, SW);
        this->reverse_legal_lookup_black[505] = ReverseLegalLookup(Role::Black, Piece::Man, 32, 41, SW);
    }
    // generating for black man 33 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 29, legal: invalid
        ddi->diagonals.emplace_back(29, -781);

        // to position 24, legal: (legal black (move man f 4 h 6))
        ddi->diagonals.emplace_back(24, 523);

        this->diagonal_data[132].push_back(ddi);


        this->reverse_legal_lookup_black[523] = ReverseLegalLookup(Role::Black, Piece::Man, 33, 24, NE);
    }
    // generating for black man 33 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 28, legal: invalid
        ddi->diagonals.emplace_back(28, -781);

        // to position 22, legal: (legal black (move man f 4 d 6))
        ddi->diagonals.emplace_back(22, 524);

        this->diagonal_data[132].push_back(ddi);


        this->reverse_legal_lookup_black[524] = ReverseLegalLookup(Role::Black, Piece::Man, 33, 22, NW);
    }
    // generating for black man 33 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 39, legal: (legal black (move man f 4 g 3))
        ddi->diagonals.emplace_back(39, 521);

        // to position 44, legal: (legal black (move man f 4 h 2))
        ddi->diagonals.emplace_back(44, 525);

        this->diagonal_data[132].push_back(ddi);


        this->reverse_legal_lookup_black[521] = ReverseLegalLookup(Role::Black, Piece::Man, 33, 39, SE);
        this->reverse_legal_lookup_black[525] = ReverseLegalLookup(Role::Black, Piece::Man, 33, 44, SE);
    }
    // generating for black man 33 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 38, legal: (legal black (move man f 4 e 3))
        ddi->diagonals.emplace_back(38, 522);

        // to position 42, legal: (legal black (move man f 4 d 2))
        ddi->diagonals.emplace_back(42, 526);

        this->diagonal_data[132].push_back(ddi);


        this->reverse_legal_lookup_black[522] = ReverseLegalLookup(Role::Black, Piece::Man, 33, 38, SW);
        this->reverse_legal_lookup_black[526] = ReverseLegalLookup(Role::Black, Piece::Man, 33, 42, SW);
    }
    // generating for black man 34 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 30, legal: invalid
        ddi->diagonals.emplace_back(30, -781);

        // to position 25, legal: (legal black (move man h 4 j 6))
        ddi->diagonals.emplace_back(25, 544);

        this->diagonal_data[133].push_back(ddi);


        this->reverse_legal_lookup_black[544] = ReverseLegalLookup(Role::Black, Piece::Man, 34, 25, NE);
    }
    // generating for black man 34 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 29, legal: invalid
        ddi->diagonals.emplace_back(29, -781);

        // to position 23, legal: (legal black (move man h 4 f 6))
        ddi->diagonals.emplace_back(23, 545);

        this->diagonal_data[133].push_back(ddi);


        this->reverse_legal_lookup_black[545] = ReverseLegalLookup(Role::Black, Piece::Man, 34, 23, NW);
    }
    // generating for black man 34 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 40, legal: (legal black (move man h 4 i 3))
        ddi->diagonals.emplace_back(40, 542);

        // to position 45, legal: (legal black (move man h 4 j 2))
        ddi->diagonals.emplace_back(45, 546);

        this->diagonal_data[133].push_back(ddi);


        this->reverse_legal_lookup_black[542] = ReverseLegalLookup(Role::Black, Piece::Man, 34, 40, SE);
        this->reverse_legal_lookup_black[546] = ReverseLegalLookup(Role::Black, Piece::Man, 34, 45, SE);
    }
    // generating for black man 34 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 39, legal: (legal black (move man h 4 g 3))
        ddi->diagonals.emplace_back(39, 543);

        // to position 43, legal: (legal black (move man h 4 f 2))
        ddi->diagonals.emplace_back(43, 547);

        this->diagonal_data[133].push_back(ddi);


        this->reverse_legal_lookup_black[543] = ReverseLegalLookup(Role::Black, Piece::Man, 34, 39, SW);
        this->reverse_legal_lookup_black[547] = ReverseLegalLookup(Role::Black, Piece::Man, 34, 43, SW);
    }
    // generating for black man 35 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 30, legal: invalid
        ddi->diagonals.emplace_back(30, -781);

        // to position 24, legal: (legal black (move man j 4 h 6))
        ddi->diagonals.emplace_back(24, 562);

        this->diagonal_data[134].push_back(ddi);


        this->reverse_legal_lookup_black[562] = ReverseLegalLookup(Role::Black, Piece::Man, 35, 24, NW);
    }
    // generating for black man 35 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 40, legal: (legal black (move man j 4 i 3))
        ddi->diagonals.emplace_back(40, 561);

        // to position 44, legal: (legal black (move man j 4 h 2))
        ddi->diagonals.emplace_back(44, 563);

        this->diagonal_data[134].push_back(ddi);


        this->reverse_legal_lookup_black[561] = ReverseLegalLookup(Role::Black, Piece::Man, 35, 40, SW);
        this->reverse_legal_lookup_black[563] = ReverseLegalLookup(Role::Black, Piece::Man, 35, 44, SW);
    }
    // generating for black man 36 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 31, legal: invalid
        ddi->diagonals.emplace_back(31, -781);

        // to position 27, legal: (legal black (move man a 3 c 5))
        ddi->diagonals.emplace_back(27, 574);

        this->diagonal_data[135].push_back(ddi);


        this->reverse_legal_lookup_black[574] = ReverseLegalLookup(Role::Black, Piece::Man, 36, 27, NE);
    }
    // generating for black man 36 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 41, legal: (legal black (move man a 3 b 2))
        ddi->diagonals.emplace_back(41, 573);

        // to position 47, legal: (legal black (move man a 3 c 1))
        ddi->diagonals.emplace_back(47, 575);

        this->diagonal_data[135].push_back(ddi);


        this->reverse_legal_lookup_black[573] = ReverseLegalLookup(Role::Black, Piece::Man, 36, 41, SE);
        this->reverse_legal_lookup_black[575] = ReverseLegalLookup(Role::Black, Piece::Man, 36, 47, SE);
    }
    // generating for black man 37 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 32, legal: invalid
        ddi->diagonals.emplace_back(32, -781);

        // to position 28, legal: (legal black (move man c 3 e 5))
        ddi->diagonals.emplace_back(28, 587);

        this->diagonal_data[136].push_back(ddi);


        this->reverse_legal_lookup_black[587] = ReverseLegalLookup(Role::Black, Piece::Man, 37, 28, NE);
    }
    // generating for black man 37 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 31, legal: invalid
        ddi->diagonals.emplace_back(31, -781);

        // to position 26, legal: (legal black (move man c 3 a 5))
        ddi->diagonals.emplace_back(26, 588);

        this->diagonal_data[136].push_back(ddi);


        this->reverse_legal_lookup_black[588] = ReverseLegalLookup(Role::Black, Piece::Man, 37, 26, NW);
    }
    // generating for black man 37 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 42, legal: (legal black (move man c 3 d 2))
        ddi->diagonals.emplace_back(42, 585);

        // to position 48, legal: (legal black (move man c 3 e 1))
        ddi->diagonals.emplace_back(48, 589);

        this->diagonal_data[136].push_back(ddi);


        this->reverse_legal_lookup_black[585] = ReverseLegalLookup(Role::Black, Piece::Man, 37, 42, SE);
        this->reverse_legal_lookup_black[589] = ReverseLegalLookup(Role::Black, Piece::Man, 37, 48, SE);
    }
    // generating for black man 37 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 41, legal: (legal black (move man c 3 b 2))
        ddi->diagonals.emplace_back(41, 586);

        // to position 46, legal: (legal black (move man c 3 a 1))
        ddi->diagonals.emplace_back(46, 590);

        this->diagonal_data[136].push_back(ddi);


        this->reverse_legal_lookup_black[586] = ReverseLegalLookup(Role::Black, Piece::Man, 37, 41, SW);
        this->reverse_legal_lookup_black[590] = ReverseLegalLookup(Role::Black, Piece::Man, 37, 46, SW);
    }
    // generating for black man 38 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 33, legal: invalid
        ddi->diagonals.emplace_back(33, -781);

        // to position 29, legal: (legal black (move man e 3 g 5))
        ddi->diagonals.emplace_back(29, 606);

        this->diagonal_data[137].push_back(ddi);


        this->reverse_legal_lookup_black[606] = ReverseLegalLookup(Role::Black, Piece::Man, 38, 29, NE);
    }
    // generating for black man 38 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 32, legal: invalid
        ddi->diagonals.emplace_back(32, -781);

        // to position 27, legal: (legal black (move man e 3 c 5))
        ddi->diagonals.emplace_back(27, 607);

        this->diagonal_data[137].push_back(ddi);


        this->reverse_legal_lookup_black[607] = ReverseLegalLookup(Role::Black, Piece::Man, 38, 27, NW);
    }
    // generating for black man 38 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 43, legal: (legal black (move man e 3 f 2))
        ddi->diagonals.emplace_back(43, 604);

        // to position 49, legal: (legal black (move man e 3 g 1))
        ddi->diagonals.emplace_back(49, 608);

        this->diagonal_data[137].push_back(ddi);


        this->reverse_legal_lookup_black[604] = ReverseLegalLookup(Role::Black, Piece::Man, 38, 43, SE);
        this->reverse_legal_lookup_black[608] = ReverseLegalLookup(Role::Black, Piece::Man, 38, 49, SE);
    }
    // generating for black man 38 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 42, legal: (legal black (move man e 3 d 2))
        ddi->diagonals.emplace_back(42, 605);

        // to position 47, legal: (legal black (move man e 3 c 1))
        ddi->diagonals.emplace_back(47, 609);

        this->diagonal_data[137].push_back(ddi);


        this->reverse_legal_lookup_black[605] = ReverseLegalLookup(Role::Black, Piece::Man, 38, 42, SW);
        this->reverse_legal_lookup_black[609] = ReverseLegalLookup(Role::Black, Piece::Man, 38, 47, SW);
    }
    // generating for black man 39 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 34, legal: invalid
        ddi->diagonals.emplace_back(34, -781);

        // to position 30, legal: (legal black (move man g 3 i 5))
        ddi->diagonals.emplace_back(30, 625);

        this->diagonal_data[138].push_back(ddi);


        this->reverse_legal_lookup_black[625] = ReverseLegalLookup(Role::Black, Piece::Man, 39, 30, NE);
    }
    // generating for black man 39 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 33, legal: invalid
        ddi->diagonals.emplace_back(33, -781);

        // to position 28, legal: (legal black (move man g 3 e 5))
        ddi->diagonals.emplace_back(28, 626);

        this->diagonal_data[138].push_back(ddi);


        this->reverse_legal_lookup_black[626] = ReverseLegalLookup(Role::Black, Piece::Man, 39, 28, NW);
    }
    // generating for black man 39 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 44, legal: (legal black (move man g 3 h 2))
        ddi->diagonals.emplace_back(44, 623);

        // to position 50, legal: (legal black (move man g 3 i 1))
        ddi->diagonals.emplace_back(50, 627);

        this->diagonal_data[138].push_back(ddi);


        this->reverse_legal_lookup_black[623] = ReverseLegalLookup(Role::Black, Piece::Man, 39, 44, SE);
        this->reverse_legal_lookup_black[627] = ReverseLegalLookup(Role::Black, Piece::Man, 39, 50, SE);
    }
    // generating for black man 39 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 43, legal: (legal black (move man g 3 f 2))
        ddi->diagonals.emplace_back(43, 624);

        // to position 48, legal: (legal black (move man g 3 e 1))
        ddi->diagonals.emplace_back(48, 628);

        this->diagonal_data[138].push_back(ddi);


        this->reverse_legal_lookup_black[624] = ReverseLegalLookup(Role::Black, Piece::Man, 39, 43, SW);
        this->reverse_legal_lookup_black[628] = ReverseLegalLookup(Role::Black, Piece::Man, 39, 48, SW);
    }
    // generating for black man 40 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 35, legal: invalid
        ddi->diagonals.emplace_back(35, -781);

        this->diagonal_data[139].push_back(ddi);


    }
    // generating for black man 40 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 34, legal: invalid
        ddi->diagonals.emplace_back(34, -781);

        // to position 29, legal: (legal black (move man i 3 g 5))
        ddi->diagonals.emplace_back(29, 644);

        this->diagonal_data[139].push_back(ddi);


        this->reverse_legal_lookup_black[644] = ReverseLegalLookup(Role::Black, Piece::Man, 40, 29, NW);
    }
    // generating for black man 40 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 45, legal: (legal black (move man i 3 j 2))
        ddi->diagonals.emplace_back(45, 642);

        this->diagonal_data[139].push_back(ddi);


        this->reverse_legal_lookup_black[642] = ReverseLegalLookup(Role::Black, Piece::Man, 40, 45, SE);
    }
    // generating for black man 40 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 44, legal: (legal black (move man i 3 h 2))
        ddi->diagonals.emplace_back(44, 643);

        // to position 49, legal: (legal black (move man i 3 g 1))
        ddi->diagonals.emplace_back(49, 645);

        this->diagonal_data[139].push_back(ddi);


        this->reverse_legal_lookup_black[643] = ReverseLegalLookup(Role::Black, Piece::Man, 40, 44, SW);
        this->reverse_legal_lookup_black[645] = ReverseLegalLookup(Role::Black, Piece::Man, 40, 49, SW);
    }
    // generating for black man 41 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 37, legal: invalid
        ddi->diagonals.emplace_back(37, -781);

        // to position 32, legal: (legal black (move man b 2 d 4))
        ddi->diagonals.emplace_back(32, 659);

        this->diagonal_data[140].push_back(ddi);


        this->reverse_legal_lookup_black[659] = ReverseLegalLookup(Role::Black, Piece::Man, 41, 32, NE);
    }
    // generating for black man 41 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 36, legal: invalid
        ddi->diagonals.emplace_back(36, -781);

        this->diagonal_data[140].push_back(ddi);


    }
    // generating for black man 41 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 47, legal: (legal black (move man b 2 c 1))
        ddi->diagonals.emplace_back(47, 657);

        this->diagonal_data[140].push_back(ddi);


        this->reverse_legal_lookup_black[657] = ReverseLegalLookup(Role::Black, Piece::Man, 41, 47, SE);
    }
    // generating for black man 41 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 46, legal: (legal black (move man b 2 a 1))
        ddi->diagonals.emplace_back(46, 658);

        this->diagonal_data[140].push_back(ddi);


        this->reverse_legal_lookup_black[658] = ReverseLegalLookup(Role::Black, Piece::Man, 41, 46, SW);
    }
    // generating for black man 42 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 38, legal: invalid
        ddi->diagonals.emplace_back(38, -781);

        // to position 33, legal: (legal black (move man d 2 f 4))
        ddi->diagonals.emplace_back(33, 673);

        this->diagonal_data[141].push_back(ddi);


        this->reverse_legal_lookup_black[673] = ReverseLegalLookup(Role::Black, Piece::Man, 42, 33, NE);
    }
    // generating for black man 42 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 37, legal: invalid
        ddi->diagonals.emplace_back(37, -781);

        // to position 31, legal: (legal black (move man d 2 b 4))
        ddi->diagonals.emplace_back(31, 674);

        this->diagonal_data[141].push_back(ddi);


        this->reverse_legal_lookup_black[674] = ReverseLegalLookup(Role::Black, Piece::Man, 42, 31, NW);
    }
    // generating for black man 42 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 48, legal: (legal black (move man d 2 e 1))
        ddi->diagonals.emplace_back(48, 671);

        this->diagonal_data[141].push_back(ddi);


        this->reverse_legal_lookup_black[671] = ReverseLegalLookup(Role::Black, Piece::Man, 42, 48, SE);
    }
    // generating for black man 42 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 47, legal: (legal black (move man d 2 c 1))
        ddi->diagonals.emplace_back(47, 672);

        this->diagonal_data[141].push_back(ddi);


        this->reverse_legal_lookup_black[672] = ReverseLegalLookup(Role::Black, Piece::Man, 42, 47, SW);
    }
    // generating for black man 43 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 39, legal: invalid
        ddi->diagonals.emplace_back(39, -781);

        // to position 34, legal: (legal black (move man f 2 h 4))
        ddi->diagonals.emplace_back(34, 688);

        this->diagonal_data[142].push_back(ddi);


        this->reverse_legal_lookup_black[688] = ReverseLegalLookup(Role::Black, Piece::Man, 43, 34, NE);
    }
    // generating for black man 43 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 38, legal: invalid
        ddi->diagonals.emplace_back(38, -781);

        // to position 32, legal: (legal black (move man f 2 d 4))
        ddi->diagonals.emplace_back(32, 689);

        this->diagonal_data[142].push_back(ddi);


        this->reverse_legal_lookup_black[689] = ReverseLegalLookup(Role::Black, Piece::Man, 43, 32, NW);
    }
    // generating for black man 43 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 49, legal: (legal black (move man f 2 g 1))
        ddi->diagonals.emplace_back(49, 686);

        this->diagonal_data[142].push_back(ddi);


        this->reverse_legal_lookup_black[686] = ReverseLegalLookup(Role::Black, Piece::Man, 43, 49, SE);
    }
    // generating for black man 43 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 48, legal: (legal black (move man f 2 e 1))
        ddi->diagonals.emplace_back(48, 687);

        this->diagonal_data[142].push_back(ddi);


        this->reverse_legal_lookup_black[687] = ReverseLegalLookup(Role::Black, Piece::Man, 43, 48, SW);
    }
    // generating for black man 44 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 40, legal: invalid
        ddi->diagonals.emplace_back(40, -781);

        // to position 35, legal: (legal black (move man h 2 j 4))
        ddi->diagonals.emplace_back(35, 703);

        this->diagonal_data[143].push_back(ddi);


        this->reverse_legal_lookup_black[703] = ReverseLegalLookup(Role::Black, Piece::Man, 44, 35, NE);
    }
    // generating for black man 44 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 39, legal: invalid
        ddi->diagonals.emplace_back(39, -781);

        // to position 33, legal: (legal black (move man h 2 f 4))
        ddi->diagonals.emplace_back(33, 704);

        this->diagonal_data[143].push_back(ddi);


        this->reverse_legal_lookup_black[704] = ReverseLegalLookup(Role::Black, Piece::Man, 44, 33, NW);
    }
    // generating for black man 44 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 50, legal: (legal black (move man h 2 i 1))
        ddi->diagonals.emplace_back(50, 701);

        this->diagonal_data[143].push_back(ddi);


        this->reverse_legal_lookup_black[701] = ReverseLegalLookup(Role::Black, Piece::Man, 44, 50, SE);
    }
    // generating for black man 44 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 49, legal: (legal black (move man h 2 g 1))
        ddi->diagonals.emplace_back(49, 702);

        this->diagonal_data[143].push_back(ddi);


        this->reverse_legal_lookup_black[702] = ReverseLegalLookup(Role::Black, Piece::Man, 44, 49, SW);
    }
    // generating for black man 45 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 40, legal: invalid
        ddi->diagonals.emplace_back(40, -781);

        // to position 34, legal: (legal black (move man j 2 h 4))
        ddi->diagonals.emplace_back(34, 717);

        this->diagonal_data[144].push_back(ddi);


        this->reverse_legal_lookup_black[717] = ReverseLegalLookup(Role::Black, Piece::Man, 45, 34, NW);
    }
    // generating for black man 45 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 50, legal: (legal black (move man j 2 i 1))
        ddi->diagonals.emplace_back(50, 716);

        this->diagonal_data[144].push_back(ddi);


        this->reverse_legal_lookup_black[716] = ReverseLegalLookup(Role::Black, Piece::Man, 45, 50, SW);
    }
    // generating for black man 46 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 41, legal: invalid
        ddi->diagonals.emplace_back(41, -781);

        // to position 37, legal: (legal black (move man a 1 c 3))
        ddi->diagonals.emplace_back(37, 727);

        this->diagonal_data[145].push_back(ddi);


        this->reverse_legal_lookup_black[727] = ReverseLegalLookup(Role::Black, Piece::Man, 46, 37, NE);
    }
    // generating for black man 47 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 42, legal: invalid
        ddi->diagonals.emplace_back(42, -781);

        // to position 38, legal: (legal black (move man c 1 e 3))
        ddi->diagonals.emplace_back(38, 737);

        this->diagonal_data[146].push_back(ddi);


        this->reverse_legal_lookup_black[737] = ReverseLegalLookup(Role::Black, Piece::Man, 47, 38, NE);
    }
    // generating for black man 47 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 41, legal: invalid
        ddi->diagonals.emplace_back(41, -781);

        // to position 36, legal: (legal black (move man c 1 a 3))
        ddi->diagonals.emplace_back(36, 738);

        this->diagonal_data[146].push_back(ddi);


        this->reverse_legal_lookup_black[738] = ReverseLegalLookup(Role::Black, Piece::Man, 47, 36, NW);
    }
    // generating for black man 48 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 43, legal: invalid
        ddi->diagonals.emplace_back(43, -781);

        // to position 39, legal: (legal black (move man e 1 g 3))
        ddi->diagonals.emplace_back(39, 748);

        this->diagonal_data[147].push_back(ddi);


        this->reverse_legal_lookup_black[748] = ReverseLegalLookup(Role::Black, Piece::Man, 48, 39, NE);
    }
    // generating for black man 48 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 42, legal: invalid
        ddi->diagonals.emplace_back(42, -781);

        // to position 37, legal: (legal black (move man e 1 c 3))
        ddi->diagonals.emplace_back(37, 749);

        this->diagonal_data[147].push_back(ddi);


        this->reverse_legal_lookup_black[749] = ReverseLegalLookup(Role::Black, Piece::Man, 48, 37, NW);
    }
    // generating for black man 49 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 44, legal: invalid
        ddi->diagonals.emplace_back(44, -781);

        // to position 40, legal: (legal black (move man g 1 i 3))
        ddi->diagonals.emplace_back(40, 759);

        this->diagonal_data[148].push_back(ddi);


        this->reverse_legal_lookup_black[759] = ReverseLegalLookup(Role::Black, Piece::Man, 49, 40, NE);
    }
    // generating for black man 49 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 43, legal: invalid
        ddi->diagonals.emplace_back(43, -781);

        // to position 38, legal: (legal black (move man g 1 e 3))
        ddi->diagonals.emplace_back(38, 760);

        this->diagonal_data[148].push_back(ddi);


        this->reverse_legal_lookup_black[760] = ReverseLegalLookup(Role::Black, Piece::Man, 49, 38, NW);
    }
    // generating for black man 50 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 45, legal: invalid
        ddi->diagonals.emplace_back(45, -781);

        this->diagonal_data[149].push_back(ddi);


    }
    // generating for black man 50 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 44, legal: invalid
        ddi->diagonals.emplace_back(44, -781);

        // to position 39, legal: (legal black (move man i 1 g 3))
        ddi->diagonals.emplace_back(39, 770);

        this->diagonal_data[149].push_back(ddi);


        this->reverse_legal_lookup_black[770] = ReverseLegalLookup(Role::Black, Piece::Man, 50, 39, NW);
    }
    // generating for black king 1 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(8);
        // to position 7, legal: (legal black (move king b 10 c 9))
        ddi->diagonals.emplace_back(7, 4);

        // to position 12, legal: (legal black (move king b 10 d 8))
        ddi->diagonals.emplace_back(12, 5);

        // to position 18, legal: (legal black (move king b 10 e 7))
        ddi->diagonals.emplace_back(18, 6);

        // to position 23, legal: (legal black (move king b 10 f 6))
        ddi->diagonals.emplace_back(23, 7);

        // to position 29, legal: (legal black (move king b 10 g 5))
        ddi->diagonals.emplace_back(29, 8);

        // to position 34, legal: (legal black (move king b 10 h 4))
        ddi->diagonals.emplace_back(34, 9);

        // to position 40, legal: (legal black (move king b 10 i 3))
        ddi->diagonals.emplace_back(40, 10);

        // to position 45, legal: (legal black (move king b 10 j 2))
        ddi->diagonals.emplace_back(45, 11);

        this->diagonal_data[150].push_back(ddi);


        this->reverse_legal_lookup_black[4] = ReverseLegalLookup(Role::Black, Piece::King, 1, 7, SE);
        this->reverse_legal_lookup_black[5] = ReverseLegalLookup(Role::Black, Piece::King, 1, 12, SE);
        this->reverse_legal_lookup_black[6] = ReverseLegalLookup(Role::Black, Piece::King, 1, 18, SE);
        this->reverse_legal_lookup_black[7] = ReverseLegalLookup(Role::Black, Piece::King, 1, 23, SE);
        this->reverse_legal_lookup_black[8] = ReverseLegalLookup(Role::Black, Piece::King, 1, 29, SE);
        this->reverse_legal_lookup_black[9] = ReverseLegalLookup(Role::Black, Piece::King, 1, 34, SE);
        this->reverse_legal_lookup_black[10] = ReverseLegalLookup(Role::Black, Piece::King, 1, 40, SE);
        this->reverse_legal_lookup_black[11] = ReverseLegalLookup(Role::Black, Piece::King, 1, 45, SE);
    }
    // generating for black king 1 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 6, legal: (legal black (move king b 10 a 9))
        ddi->diagonals.emplace_back(6, 12);

        this->diagonal_data[150].push_back(ddi);


        this->reverse_legal_lookup_black[12] = ReverseLegalLookup(Role::Black, Piece::King, 1, 6, SW);
    }
    // generating for black king 2 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 8, legal: (legal black (move king d 10 e 9))
        ddi->diagonals.emplace_back(8, 17);

        // to position 13, legal: (legal black (move king d 10 f 8))
        ddi->diagonals.emplace_back(13, 18);

        // to position 19, legal: (legal black (move king d 10 g 7))
        ddi->diagonals.emplace_back(19, 19);

        // to position 24, legal: (legal black (move king d 10 h 6))
        ddi->diagonals.emplace_back(24, 20);

        // to position 30, legal: (legal black (move king d 10 i 5))
        ddi->diagonals.emplace_back(30, 21);

        // to position 35, legal: (legal black (move king d 10 j 4))
        ddi->diagonals.emplace_back(35, 22);

        this->diagonal_data[151].push_back(ddi);


        this->reverse_legal_lookup_black[17] = ReverseLegalLookup(Role::Black, Piece::King, 2, 8, SE);
        this->reverse_legal_lookup_black[18] = ReverseLegalLookup(Role::Black, Piece::King, 2, 13, SE);
        this->reverse_legal_lookup_black[19] = ReverseLegalLookup(Role::Black, Piece::King, 2, 19, SE);
        this->reverse_legal_lookup_black[20] = ReverseLegalLookup(Role::Black, Piece::King, 2, 24, SE);
        this->reverse_legal_lookup_black[21] = ReverseLegalLookup(Role::Black, Piece::King, 2, 30, SE);
        this->reverse_legal_lookup_black[22] = ReverseLegalLookup(Role::Black, Piece::King, 2, 35, SE);
    }
    // generating for black king 2 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 7, legal: (legal black (move king d 10 c 9))
        ddi->diagonals.emplace_back(7, 23);

        // to position 11, legal: (legal black (move king d 10 b 8))
        ddi->diagonals.emplace_back(11, 24);

        // to position 16, legal: (legal black (move king d 10 a 7))
        ddi->diagonals.emplace_back(16, 25);

        this->diagonal_data[151].push_back(ddi);


        this->reverse_legal_lookup_black[23] = ReverseLegalLookup(Role::Black, Piece::King, 2, 7, SW);
        this->reverse_legal_lookup_black[24] = ReverseLegalLookup(Role::Black, Piece::King, 2, 11, SW);
        this->reverse_legal_lookup_black[25] = ReverseLegalLookup(Role::Black, Piece::King, 2, 16, SW);
    }
    // generating for black king 3 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 9, legal: (legal black (move king f 10 g 9))
        ddi->diagonals.emplace_back(9, 30);

        // to position 14, legal: (legal black (move king f 10 h 8))
        ddi->diagonals.emplace_back(14, 31);

        // to position 20, legal: (legal black (move king f 10 i 7))
        ddi->diagonals.emplace_back(20, 32);

        // to position 25, legal: (legal black (move king f 10 j 6))
        ddi->diagonals.emplace_back(25, 33);

        this->diagonal_data[152].push_back(ddi);


        this->reverse_legal_lookup_black[30] = ReverseLegalLookup(Role::Black, Piece::King, 3, 9, SE);
        this->reverse_legal_lookup_black[31] = ReverseLegalLookup(Role::Black, Piece::King, 3, 14, SE);
        this->reverse_legal_lookup_black[32] = ReverseLegalLookup(Role::Black, Piece::King, 3, 20, SE);
        this->reverse_legal_lookup_black[33] = ReverseLegalLookup(Role::Black, Piece::King, 3, 25, SE);
    }
    // generating for black king 3 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 8, legal: (legal black (move king f 10 e 9))
        ddi->diagonals.emplace_back(8, 34);

        // to position 12, legal: (legal black (move king f 10 d 8))
        ddi->diagonals.emplace_back(12, 35);

        // to position 17, legal: (legal black (move king f 10 c 7))
        ddi->diagonals.emplace_back(17, 36);

        // to position 21, legal: (legal black (move king f 10 b 6))
        ddi->diagonals.emplace_back(21, 37);

        // to position 26, legal: (legal black (move king f 10 a 5))
        ddi->diagonals.emplace_back(26, 38);

        this->diagonal_data[152].push_back(ddi);


        this->reverse_legal_lookup_black[34] = ReverseLegalLookup(Role::Black, Piece::King, 3, 8, SW);
        this->reverse_legal_lookup_black[35] = ReverseLegalLookup(Role::Black, Piece::King, 3, 12, SW);
        this->reverse_legal_lookup_black[36] = ReverseLegalLookup(Role::Black, Piece::King, 3, 17, SW);
        this->reverse_legal_lookup_black[37] = ReverseLegalLookup(Role::Black, Piece::King, 3, 21, SW);
        this->reverse_legal_lookup_black[38] = ReverseLegalLookup(Role::Black, Piece::King, 3, 26, SW);
    }
    // generating for black king 4 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal black (move king h 10 i 9))
        ddi->diagonals.emplace_back(10, 43);

        // to position 15, legal: (legal black (move king h 10 j 8))
        ddi->diagonals.emplace_back(15, 44);

        this->diagonal_data[153].push_back(ddi);


        this->reverse_legal_lookup_black[43] = ReverseLegalLookup(Role::Black, Piece::King, 4, 10, SE);
        this->reverse_legal_lookup_black[44] = ReverseLegalLookup(Role::Black, Piece::King, 4, 15, SE);
    }
    // generating for black king 4 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(7);
        // to position 9, legal: (legal black (move king h 10 g 9))
        ddi->diagonals.emplace_back(9, 45);

        // to position 13, legal: (legal black (move king h 10 f 8))
        ddi->diagonals.emplace_back(13, 46);

        // to position 18, legal: (legal black (move king h 10 e 7))
        ddi->diagonals.emplace_back(18, 47);

        // to position 22, legal: (legal black (move king h 10 d 6))
        ddi->diagonals.emplace_back(22, 48);

        // to position 27, legal: (legal black (move king h 10 c 5))
        ddi->diagonals.emplace_back(27, 49);

        // to position 31, legal: (legal black (move king h 10 b 4))
        ddi->diagonals.emplace_back(31, 50);

        // to position 36, legal: (legal black (move king h 10 a 3))
        ddi->diagonals.emplace_back(36, 51);

        this->diagonal_data[153].push_back(ddi);


        this->reverse_legal_lookup_black[45] = ReverseLegalLookup(Role::Black, Piece::King, 4, 9, SW);
        this->reverse_legal_lookup_black[46] = ReverseLegalLookup(Role::Black, Piece::King, 4, 13, SW);
        this->reverse_legal_lookup_black[47] = ReverseLegalLookup(Role::Black, Piece::King, 4, 18, SW);
        this->reverse_legal_lookup_black[48] = ReverseLegalLookup(Role::Black, Piece::King, 4, 22, SW);
        this->reverse_legal_lookup_black[49] = ReverseLegalLookup(Role::Black, Piece::King, 4, 27, SW);
        this->reverse_legal_lookup_black[50] = ReverseLegalLookup(Role::Black, Piece::King, 4, 31, SW);
        this->reverse_legal_lookup_black[51] = ReverseLegalLookup(Role::Black, Piece::King, 4, 36, SW);
    }
    // generating for black king 5 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(9);
        // to position 10, legal: (legal black (move king j 10 i 9))
        ddi->diagonals.emplace_back(10, 54);

        // to position 14, legal: (legal black (move king j 10 h 8))
        ddi->diagonals.emplace_back(14, 55);

        // to position 19, legal: (legal black (move king j 10 g 7))
        ddi->diagonals.emplace_back(19, 56);

        // to position 23, legal: (legal black (move king j 10 f 6))
        ddi->diagonals.emplace_back(23, 57);

        // to position 28, legal: (legal black (move king j 10 e 5))
        ddi->diagonals.emplace_back(28, 58);

        // to position 32, legal: (legal black (move king j 10 d 4))
        ddi->diagonals.emplace_back(32, 59);

        // to position 37, legal: (legal black (move king j 10 c 3))
        ddi->diagonals.emplace_back(37, 60);

        // to position 41, legal: (legal black (move king j 10 b 2))
        ddi->diagonals.emplace_back(41, 61);

        // to position 46, legal: (legal black (move king j 10 a 1))
        ddi->diagonals.emplace_back(46, 62);

        this->diagonal_data[154].push_back(ddi);


        this->reverse_legal_lookup_black[54] = ReverseLegalLookup(Role::Black, Piece::King, 5, 10, SW);
        this->reverse_legal_lookup_black[55] = ReverseLegalLookup(Role::Black, Piece::King, 5, 14, SW);
        this->reverse_legal_lookup_black[56] = ReverseLegalLookup(Role::Black, Piece::King, 5, 19, SW);
        this->reverse_legal_lookup_black[57] = ReverseLegalLookup(Role::Black, Piece::King, 5, 23, SW);
        this->reverse_legal_lookup_black[58] = ReverseLegalLookup(Role::Black, Piece::King, 5, 28, SW);
        this->reverse_legal_lookup_black[59] = ReverseLegalLookup(Role::Black, Piece::King, 5, 32, SW);
        this->reverse_legal_lookup_black[60] = ReverseLegalLookup(Role::Black, Piece::King, 5, 37, SW);
        this->reverse_legal_lookup_black[61] = ReverseLegalLookup(Role::Black, Piece::King, 5, 41, SW);
        this->reverse_legal_lookup_black[62] = ReverseLegalLookup(Role::Black, Piece::King, 5, 46, SW);
    }
    // generating for black king 6 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal black (move king a 9 b 10))
        ddi->diagonals.emplace_back(1, 65);

        this->diagonal_data[155].push_back(ddi);


        this->reverse_legal_lookup_black[65] = ReverseLegalLookup(Role::Black, Piece::King, 6, 1, NE);
    }
    // generating for black king 6 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(8);
        // to position 11, legal: (legal black (move king a 9 b 8))
        ddi->diagonals.emplace_back(11, 66);

        // to position 17, legal: (legal black (move king a 9 c 7))
        ddi->diagonals.emplace_back(17, 67);

        // to position 22, legal: (legal black (move king a 9 d 6))
        ddi->diagonals.emplace_back(22, 68);

        // to position 28, legal: (legal black (move king a 9 e 5))
        ddi->diagonals.emplace_back(28, 69);

        // to position 33, legal: (legal black (move king a 9 f 4))
        ddi->diagonals.emplace_back(33, 70);

        // to position 39, legal: (legal black (move king a 9 g 3))
        ddi->diagonals.emplace_back(39, 71);

        // to position 44, legal: (legal black (move king a 9 h 2))
        ddi->diagonals.emplace_back(44, 72);

        // to position 50, legal: (legal black (move king a 9 i 1))
        ddi->diagonals.emplace_back(50, 73);

        this->diagonal_data[155].push_back(ddi);


        this->reverse_legal_lookup_black[66] = ReverseLegalLookup(Role::Black, Piece::King, 6, 11, SE);
        this->reverse_legal_lookup_black[67] = ReverseLegalLookup(Role::Black, Piece::King, 6, 17, SE);
        this->reverse_legal_lookup_black[68] = ReverseLegalLookup(Role::Black, Piece::King, 6, 22, SE);
        this->reverse_legal_lookup_black[69] = ReverseLegalLookup(Role::Black, Piece::King, 6, 28, SE);
        this->reverse_legal_lookup_black[70] = ReverseLegalLookup(Role::Black, Piece::King, 6, 33, SE);
        this->reverse_legal_lookup_black[71] = ReverseLegalLookup(Role::Black, Piece::King, 6, 39, SE);
        this->reverse_legal_lookup_black[72] = ReverseLegalLookup(Role::Black, Piece::King, 6, 44, SE);
        this->reverse_legal_lookup_black[73] = ReverseLegalLookup(Role::Black, Piece::King, 6, 50, SE);
    }
    // generating for black king 7 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal black (move king c 9 d 10))
        ddi->diagonals.emplace_back(2, 78);

        this->diagonal_data[156].push_back(ddi);


        this->reverse_legal_lookup_black[78] = ReverseLegalLookup(Role::Black, Piece::King, 7, 2, NE);
    }
    // generating for black king 7 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 1, legal: (legal black (move king c 9 b 10))
        ddi->diagonals.emplace_back(1, 79);

        this->diagonal_data[156].push_back(ddi);


        this->reverse_legal_lookup_black[79] = ReverseLegalLookup(Role::Black, Piece::King, 7, 1, NW);
    }
    // generating for black king 7 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(7);
        // to position 12, legal: (legal black (move king c 9 d 8))
        ddi->diagonals.emplace_back(12, 80);

        // to position 18, legal: (legal black (move king c 9 e 7))
        ddi->diagonals.emplace_back(18, 81);

        // to position 23, legal: (legal black (move king c 9 f 6))
        ddi->diagonals.emplace_back(23, 82);

        // to position 29, legal: (legal black (move king c 9 g 5))
        ddi->diagonals.emplace_back(29, 83);

        // to position 34, legal: (legal black (move king c 9 h 4))
        ddi->diagonals.emplace_back(34, 84);

        // to position 40, legal: (legal black (move king c 9 i 3))
        ddi->diagonals.emplace_back(40, 85);

        // to position 45, legal: (legal black (move king c 9 j 2))
        ddi->diagonals.emplace_back(45, 86);

        this->diagonal_data[156].push_back(ddi);


        this->reverse_legal_lookup_black[80] = ReverseLegalLookup(Role::Black, Piece::King, 7, 12, SE);
        this->reverse_legal_lookup_black[81] = ReverseLegalLookup(Role::Black, Piece::King, 7, 18, SE);
        this->reverse_legal_lookup_black[82] = ReverseLegalLookup(Role::Black, Piece::King, 7, 23, SE);
        this->reverse_legal_lookup_black[83] = ReverseLegalLookup(Role::Black, Piece::King, 7, 29, SE);
        this->reverse_legal_lookup_black[84] = ReverseLegalLookup(Role::Black, Piece::King, 7, 34, SE);
        this->reverse_legal_lookup_black[85] = ReverseLegalLookup(Role::Black, Piece::King, 7, 40, SE);
        this->reverse_legal_lookup_black[86] = ReverseLegalLookup(Role::Black, Piece::King, 7, 45, SE);
    }
    // generating for black king 7 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal black (move king c 9 b 8))
        ddi->diagonals.emplace_back(11, 87);

        // to position 16, legal: (legal black (move king c 9 a 7))
        ddi->diagonals.emplace_back(16, 88);

        this->diagonal_data[156].push_back(ddi);


        this->reverse_legal_lookup_black[87] = ReverseLegalLookup(Role::Black, Piece::King, 7, 11, SW);
        this->reverse_legal_lookup_black[88] = ReverseLegalLookup(Role::Black, Piece::King, 7, 16, SW);
    }
    // generating for black king 8 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal black (move king e 9 f 10))
        ddi->diagonals.emplace_back(3, 93);

        this->diagonal_data[157].push_back(ddi);


        this->reverse_legal_lookup_black[93] = ReverseLegalLookup(Role::Black, Piece::King, 8, 3, NE);
    }
    // generating for black king 8 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 2, legal: (legal black (move king e 9 d 10))
        ddi->diagonals.emplace_back(2, 94);

        this->diagonal_data[157].push_back(ddi);


        this->reverse_legal_lookup_black[94] = ReverseLegalLookup(Role::Black, Piece::King, 8, 2, NW);
    }
    // generating for black king 8 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 13, legal: (legal black (move king e 9 f 8))
        ddi->diagonals.emplace_back(13, 95);

        // to position 19, legal: (legal black (move king e 9 g 7))
        ddi->diagonals.emplace_back(19, 96);

        // to position 24, legal: (legal black (move king e 9 h 6))
        ddi->diagonals.emplace_back(24, 97);

        // to position 30, legal: (legal black (move king e 9 i 5))
        ddi->diagonals.emplace_back(30, 98);

        // to position 35, legal: (legal black (move king e 9 j 4))
        ddi->diagonals.emplace_back(35, 99);

        this->diagonal_data[157].push_back(ddi);


        this->reverse_legal_lookup_black[95] = ReverseLegalLookup(Role::Black, Piece::King, 8, 13, SE);
        this->reverse_legal_lookup_black[96] = ReverseLegalLookup(Role::Black, Piece::King, 8, 19, SE);
        this->reverse_legal_lookup_black[97] = ReverseLegalLookup(Role::Black, Piece::King, 8, 24, SE);
        this->reverse_legal_lookup_black[98] = ReverseLegalLookup(Role::Black, Piece::King, 8, 30, SE);
        this->reverse_legal_lookup_black[99] = ReverseLegalLookup(Role::Black, Piece::King, 8, 35, SE);
    }
    // generating for black king 8 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 12, legal: (legal black (move king e 9 d 8))
        ddi->diagonals.emplace_back(12, 100);

        // to position 17, legal: (legal black (move king e 9 c 7))
        ddi->diagonals.emplace_back(17, 101);

        // to position 21, legal: (legal black (move king e 9 b 6))
        ddi->diagonals.emplace_back(21, 102);

        // to position 26, legal: (legal black (move king e 9 a 5))
        ddi->diagonals.emplace_back(26, 103);

        this->diagonal_data[157].push_back(ddi);


        this->reverse_legal_lookup_black[100] = ReverseLegalLookup(Role::Black, Piece::King, 8, 12, SW);
        this->reverse_legal_lookup_black[101] = ReverseLegalLookup(Role::Black, Piece::King, 8, 17, SW);
        this->reverse_legal_lookup_black[102] = ReverseLegalLookup(Role::Black, Piece::King, 8, 21, SW);
        this->reverse_legal_lookup_black[103] = ReverseLegalLookup(Role::Black, Piece::King, 8, 26, SW);
    }
    // generating for black king 9 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 4, legal: (legal black (move king g 9 h 10))
        ddi->diagonals.emplace_back(4, 108);

        this->diagonal_data[158].push_back(ddi);


        this->reverse_legal_lookup_black[108] = ReverseLegalLookup(Role::Black, Piece::King, 9, 4, NE);
    }
    // generating for black king 9 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 3, legal: (legal black (move king g 9 f 10))
        ddi->diagonals.emplace_back(3, 109);

        this->diagonal_data[158].push_back(ddi);


        this->reverse_legal_lookup_black[109] = ReverseLegalLookup(Role::Black, Piece::King, 9, 3, NW);
    }
    // generating for black king 9 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 14, legal: (legal black (move king g 9 h 8))
        ddi->diagonals.emplace_back(14, 110);

        // to position 20, legal: (legal black (move king g 9 i 7))
        ddi->diagonals.emplace_back(20, 111);

        // to position 25, legal: (legal black (move king g 9 j 6))
        ddi->diagonals.emplace_back(25, 112);

        this->diagonal_data[158].push_back(ddi);


        this->reverse_legal_lookup_black[110] = ReverseLegalLookup(Role::Black, Piece::King, 9, 14, SE);
        this->reverse_legal_lookup_black[111] = ReverseLegalLookup(Role::Black, Piece::King, 9, 20, SE);
        this->reverse_legal_lookup_black[112] = ReverseLegalLookup(Role::Black, Piece::King, 9, 25, SE);
    }
    // generating for black king 9 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(6);
        // to position 13, legal: (legal black (move king g 9 f 8))
        ddi->diagonals.emplace_back(13, 113);

        // to position 18, legal: (legal black (move king g 9 e 7))
        ddi->diagonals.emplace_back(18, 114);

        // to position 22, legal: (legal black (move king g 9 d 6))
        ddi->diagonals.emplace_back(22, 115);

        // to position 27, legal: (legal black (move king g 9 c 5))
        ddi->diagonals.emplace_back(27, 116);

        // to position 31, legal: (legal black (move king g 9 b 4))
        ddi->diagonals.emplace_back(31, 117);

        // to position 36, legal: (legal black (move king g 9 a 3))
        ddi->diagonals.emplace_back(36, 118);

        this->diagonal_data[158].push_back(ddi);


        this->reverse_legal_lookup_black[113] = ReverseLegalLookup(Role::Black, Piece::King, 9, 13, SW);
        this->reverse_legal_lookup_black[114] = ReverseLegalLookup(Role::Black, Piece::King, 9, 18, SW);
        this->reverse_legal_lookup_black[115] = ReverseLegalLookup(Role::Black, Piece::King, 9, 22, SW);
        this->reverse_legal_lookup_black[116] = ReverseLegalLookup(Role::Black, Piece::King, 9, 27, SW);
        this->reverse_legal_lookup_black[117] = ReverseLegalLookup(Role::Black, Piece::King, 9, 31, SW);
        this->reverse_legal_lookup_black[118] = ReverseLegalLookup(Role::Black, Piece::King, 9, 36, SW);
    }
    // generating for black king 10 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 5, legal: (legal black (move king i 9 j 10))
        ddi->diagonals.emplace_back(5, 122);

        this->diagonal_data[159].push_back(ddi);


        this->reverse_legal_lookup_black[122] = ReverseLegalLookup(Role::Black, Piece::King, 10, 5, NE);
    }
    // generating for black king 10 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 4, legal: (legal black (move king i 9 h 10))
        ddi->diagonals.emplace_back(4, 123);

        this->diagonal_data[159].push_back(ddi);


        this->reverse_legal_lookup_black[123] = ReverseLegalLookup(Role::Black, Piece::King, 10, 4, NW);
    }
    // generating for black king 10 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 15, legal: (legal black (move king i 9 j 8))
        ddi->diagonals.emplace_back(15, 124);

        this->diagonal_data[159].push_back(ddi);


        this->reverse_legal_lookup_black[124] = ReverseLegalLookup(Role::Black, Piece::King, 10, 15, SE);
    }
    // generating for black king 10 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(8);
        // to position 14, legal: (legal black (move king i 9 h 8))
        ddi->diagonals.emplace_back(14, 125);

        // to position 19, legal: (legal black (move king i 9 g 7))
        ddi->diagonals.emplace_back(19, 126);

        // to position 23, legal: (legal black (move king i 9 f 6))
        ddi->diagonals.emplace_back(23, 127);

        // to position 28, legal: (legal black (move king i 9 e 5))
        ddi->diagonals.emplace_back(28, 128);

        // to position 32, legal: (legal black (move king i 9 d 4))
        ddi->diagonals.emplace_back(32, 129);

        // to position 37, legal: (legal black (move king i 9 c 3))
        ddi->diagonals.emplace_back(37, 130);

        // to position 41, legal: (legal black (move king i 9 b 2))
        ddi->diagonals.emplace_back(41, 131);

        // to position 46, legal: (legal black (move king i 9 a 1))
        ddi->diagonals.emplace_back(46, 132);

        this->diagonal_data[159].push_back(ddi);


        this->reverse_legal_lookup_black[125] = ReverseLegalLookup(Role::Black, Piece::King, 10, 14, SW);
        this->reverse_legal_lookup_black[126] = ReverseLegalLookup(Role::Black, Piece::King, 10, 19, SW);
        this->reverse_legal_lookup_black[127] = ReverseLegalLookup(Role::Black, Piece::King, 10, 23, SW);
        this->reverse_legal_lookup_black[128] = ReverseLegalLookup(Role::Black, Piece::King, 10, 28, SW);
        this->reverse_legal_lookup_black[129] = ReverseLegalLookup(Role::Black, Piece::King, 10, 32, SW);
        this->reverse_legal_lookup_black[130] = ReverseLegalLookup(Role::Black, Piece::King, 10, 37, SW);
        this->reverse_legal_lookup_black[131] = ReverseLegalLookup(Role::Black, Piece::King, 10, 41, SW);
        this->reverse_legal_lookup_black[132] = ReverseLegalLookup(Role::Black, Piece::King, 10, 46, SW);
    }
    // generating for black king 11 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal black (move king b 8 c 9))
        ddi->diagonals.emplace_back(7, 137);

        // to position 2, legal: (legal black (move king b 8 d 10))
        ddi->diagonals.emplace_back(2, 138);

        this->diagonal_data[160].push_back(ddi);


        this->reverse_legal_lookup_black[137] = ReverseLegalLookup(Role::Black, Piece::King, 11, 7, NE);
        this->reverse_legal_lookup_black[138] = ReverseLegalLookup(Role::Black, Piece::King, 11, 2, NE);
    }
    // generating for black king 11 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 6, legal: (legal black (move king b 8 a 9))
        ddi->diagonals.emplace_back(6, 139);

        this->diagonal_data[160].push_back(ddi);


        this->reverse_legal_lookup_black[139] = ReverseLegalLookup(Role::Black, Piece::King, 11, 6, NW);
    }
    // generating for black king 11 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(7);
        // to position 17, legal: (legal black (move king b 8 c 7))
        ddi->diagonals.emplace_back(17, 140);

        // to position 22, legal: (legal black (move king b 8 d 6))
        ddi->diagonals.emplace_back(22, 141);

        // to position 28, legal: (legal black (move king b 8 e 5))
        ddi->diagonals.emplace_back(28, 142);

        // to position 33, legal: (legal black (move king b 8 f 4))
        ddi->diagonals.emplace_back(33, 143);

        // to position 39, legal: (legal black (move king b 8 g 3))
        ddi->diagonals.emplace_back(39, 144);

        // to position 44, legal: (legal black (move king b 8 h 2))
        ddi->diagonals.emplace_back(44, 145);

        // to position 50, legal: (legal black (move king b 8 i 1))
        ddi->diagonals.emplace_back(50, 146);

        this->diagonal_data[160].push_back(ddi);


        this->reverse_legal_lookup_black[140] = ReverseLegalLookup(Role::Black, Piece::King, 11, 17, SE);
        this->reverse_legal_lookup_black[141] = ReverseLegalLookup(Role::Black, Piece::King, 11, 22, SE);
        this->reverse_legal_lookup_black[142] = ReverseLegalLookup(Role::Black, Piece::King, 11, 28, SE);
        this->reverse_legal_lookup_black[143] = ReverseLegalLookup(Role::Black, Piece::King, 11, 33, SE);
        this->reverse_legal_lookup_black[144] = ReverseLegalLookup(Role::Black, Piece::King, 11, 39, SE);
        this->reverse_legal_lookup_black[145] = ReverseLegalLookup(Role::Black, Piece::King, 11, 44, SE);
        this->reverse_legal_lookup_black[146] = ReverseLegalLookup(Role::Black, Piece::King, 11, 50, SE);
    }
    // generating for black king 11 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 16, legal: (legal black (move king b 8 a 7))
        ddi->diagonals.emplace_back(16, 147);

        this->diagonal_data[160].push_back(ddi);


        this->reverse_legal_lookup_black[147] = ReverseLegalLookup(Role::Black, Piece::King, 11, 16, SW);
    }
    // generating for black king 12 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal black (move king d 8 e 9))
        ddi->diagonals.emplace_back(8, 154);

        // to position 3, legal: (legal black (move king d 8 f 10))
        ddi->diagonals.emplace_back(3, 155);

        this->diagonal_data[161].push_back(ddi);


        this->reverse_legal_lookup_black[154] = ReverseLegalLookup(Role::Black, Piece::King, 12, 8, NE);
        this->reverse_legal_lookup_black[155] = ReverseLegalLookup(Role::Black, Piece::King, 12, 3, NE);
    }
    // generating for black king 12 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 7, legal: (legal black (move king d 8 c 9))
        ddi->diagonals.emplace_back(7, 156);

        // to position 1, legal: (legal black (move king d 8 b 10))
        ddi->diagonals.emplace_back(1, 157);

        this->diagonal_data[161].push_back(ddi);


        this->reverse_legal_lookup_black[156] = ReverseLegalLookup(Role::Black, Piece::King, 12, 7, NW);
        this->reverse_legal_lookup_black[157] = ReverseLegalLookup(Role::Black, Piece::King, 12, 1, NW);
    }
    // generating for black king 12 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 18, legal: (legal black (move king d 8 e 7))
        ddi->diagonals.emplace_back(18, 158);

        // to position 23, legal: (legal black (move king d 8 f 6))
        ddi->diagonals.emplace_back(23, 159);

        // to position 29, legal: (legal black (move king d 8 g 5))
        ddi->diagonals.emplace_back(29, 160);

        // to position 34, legal: (legal black (move king d 8 h 4))
        ddi->diagonals.emplace_back(34, 161);

        // to position 40, legal: (legal black (move king d 8 i 3))
        ddi->diagonals.emplace_back(40, 162);

        // to position 45, legal: (legal black (move king d 8 j 2))
        ddi->diagonals.emplace_back(45, 163);

        this->diagonal_data[161].push_back(ddi);


        this->reverse_legal_lookup_black[158] = ReverseLegalLookup(Role::Black, Piece::King, 12, 18, SE);
        this->reverse_legal_lookup_black[159] = ReverseLegalLookup(Role::Black, Piece::King, 12, 23, SE);
        this->reverse_legal_lookup_black[160] = ReverseLegalLookup(Role::Black, Piece::King, 12, 29, SE);
        this->reverse_legal_lookup_black[161] = ReverseLegalLookup(Role::Black, Piece::King, 12, 34, SE);
        this->reverse_legal_lookup_black[162] = ReverseLegalLookup(Role::Black, Piece::King, 12, 40, SE);
        this->reverse_legal_lookup_black[163] = ReverseLegalLookup(Role::Black, Piece::King, 12, 45, SE);
    }
    // generating for black king 12 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 17, legal: (legal black (move king d 8 c 7))
        ddi->diagonals.emplace_back(17, 164);

        // to position 21, legal: (legal black (move king d 8 b 6))
        ddi->diagonals.emplace_back(21, 165);

        // to position 26, legal: (legal black (move king d 8 a 5))
        ddi->diagonals.emplace_back(26, 166);

        this->diagonal_data[161].push_back(ddi);


        this->reverse_legal_lookup_black[164] = ReverseLegalLookup(Role::Black, Piece::King, 12, 17, SW);
        this->reverse_legal_lookup_black[165] = ReverseLegalLookup(Role::Black, Piece::King, 12, 21, SW);
        this->reverse_legal_lookup_black[166] = ReverseLegalLookup(Role::Black, Piece::King, 12, 26, SW);
    }
    // generating for black king 13 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal black (move king f 8 g 9))
        ddi->diagonals.emplace_back(9, 173);

        // to position 4, legal: (legal black (move king f 8 h 10))
        ddi->diagonals.emplace_back(4, 174);

        this->diagonal_data[162].push_back(ddi);


        this->reverse_legal_lookup_black[173] = ReverseLegalLookup(Role::Black, Piece::King, 13, 9, NE);
        this->reverse_legal_lookup_black[174] = ReverseLegalLookup(Role::Black, Piece::King, 13, 4, NE);
    }
    // generating for black king 13 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 8, legal: (legal black (move king f 8 e 9))
        ddi->diagonals.emplace_back(8, 175);

        // to position 2, legal: (legal black (move king f 8 d 10))
        ddi->diagonals.emplace_back(2, 176);

        this->diagonal_data[162].push_back(ddi);


        this->reverse_legal_lookup_black[175] = ReverseLegalLookup(Role::Black, Piece::King, 13, 8, NW);
        this->reverse_legal_lookup_black[176] = ReverseLegalLookup(Role::Black, Piece::King, 13, 2, NW);
    }
    // generating for black king 13 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 19, legal: (legal black (move king f 8 g 7))
        ddi->diagonals.emplace_back(19, 177);

        // to position 24, legal: (legal black (move king f 8 h 6))
        ddi->diagonals.emplace_back(24, 178);

        // to position 30, legal: (legal black (move king f 8 i 5))
        ddi->diagonals.emplace_back(30, 179);

        // to position 35, legal: (legal black (move king f 8 j 4))
        ddi->diagonals.emplace_back(35, 180);

        this->diagonal_data[162].push_back(ddi);


        this->reverse_legal_lookup_black[177] = ReverseLegalLookup(Role::Black, Piece::King, 13, 19, SE);
        this->reverse_legal_lookup_black[178] = ReverseLegalLookup(Role::Black, Piece::King, 13, 24, SE);
        this->reverse_legal_lookup_black[179] = ReverseLegalLookup(Role::Black, Piece::King, 13, 30, SE);
        this->reverse_legal_lookup_black[180] = ReverseLegalLookup(Role::Black, Piece::King, 13, 35, SE);
    }
    // generating for black king 13 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 18, legal: (legal black (move king f 8 e 7))
        ddi->diagonals.emplace_back(18, 181);

        // to position 22, legal: (legal black (move king f 8 d 6))
        ddi->diagonals.emplace_back(22, 182);

        // to position 27, legal: (legal black (move king f 8 c 5))
        ddi->diagonals.emplace_back(27, 183);

        // to position 31, legal: (legal black (move king f 8 b 4))
        ddi->diagonals.emplace_back(31, 184);

        // to position 36, legal: (legal black (move king f 8 a 3))
        ddi->diagonals.emplace_back(36, 185);

        this->diagonal_data[162].push_back(ddi);


        this->reverse_legal_lookup_black[181] = ReverseLegalLookup(Role::Black, Piece::King, 13, 18, SW);
        this->reverse_legal_lookup_black[182] = ReverseLegalLookup(Role::Black, Piece::King, 13, 22, SW);
        this->reverse_legal_lookup_black[183] = ReverseLegalLookup(Role::Black, Piece::King, 13, 27, SW);
        this->reverse_legal_lookup_black[184] = ReverseLegalLookup(Role::Black, Piece::King, 13, 31, SW);
        this->reverse_legal_lookup_black[185] = ReverseLegalLookup(Role::Black, Piece::King, 13, 36, SW);
    }
    // generating for black king 14 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal black (move king h 8 i 9))
        ddi->diagonals.emplace_back(10, 192);

        // to position 5, legal: (legal black (move king h 8 j 10))
        ddi->diagonals.emplace_back(5, 193);

        this->diagonal_data[163].push_back(ddi);


        this->reverse_legal_lookup_black[192] = ReverseLegalLookup(Role::Black, Piece::King, 14, 10, NE);
        this->reverse_legal_lookup_black[193] = ReverseLegalLookup(Role::Black, Piece::King, 14, 5, NE);
    }
    // generating for black king 14 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 9, legal: (legal black (move king h 8 g 9))
        ddi->diagonals.emplace_back(9, 194);

        // to position 3, legal: (legal black (move king h 8 f 10))
        ddi->diagonals.emplace_back(3, 195);

        this->diagonal_data[163].push_back(ddi);


        this->reverse_legal_lookup_black[194] = ReverseLegalLookup(Role::Black, Piece::King, 14, 9, NW);
        this->reverse_legal_lookup_black[195] = ReverseLegalLookup(Role::Black, Piece::King, 14, 3, NW);
    }
    // generating for black king 14 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 20, legal: (legal black (move king h 8 i 7))
        ddi->diagonals.emplace_back(20, 196);

        // to position 25, legal: (legal black (move king h 8 j 6))
        ddi->diagonals.emplace_back(25, 197);

        this->diagonal_data[163].push_back(ddi);


        this->reverse_legal_lookup_black[196] = ReverseLegalLookup(Role::Black, Piece::King, 14, 20, SE);
        this->reverse_legal_lookup_black[197] = ReverseLegalLookup(Role::Black, Piece::King, 14, 25, SE);
    }
    // generating for black king 14 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(7);
        // to position 19, legal: (legal black (move king h 8 g 7))
        ddi->diagonals.emplace_back(19, 198);

        // to position 23, legal: (legal black (move king h 8 f 6))
        ddi->diagonals.emplace_back(23, 199);

        // to position 28, legal: (legal black (move king h 8 e 5))
        ddi->diagonals.emplace_back(28, 200);

        // to position 32, legal: (legal black (move king h 8 d 4))
        ddi->diagonals.emplace_back(32, 201);

        // to position 37, legal: (legal black (move king h 8 c 3))
        ddi->diagonals.emplace_back(37, 202);

        // to position 41, legal: (legal black (move king h 8 b 2))
        ddi->diagonals.emplace_back(41, 203);

        // to position 46, legal: (legal black (move king h 8 a 1))
        ddi->diagonals.emplace_back(46, 204);

        this->diagonal_data[163].push_back(ddi);


        this->reverse_legal_lookup_black[198] = ReverseLegalLookup(Role::Black, Piece::King, 14, 19, SW);
        this->reverse_legal_lookup_black[199] = ReverseLegalLookup(Role::Black, Piece::King, 14, 23, SW);
        this->reverse_legal_lookup_black[200] = ReverseLegalLookup(Role::Black, Piece::King, 14, 28, SW);
        this->reverse_legal_lookup_black[201] = ReverseLegalLookup(Role::Black, Piece::King, 14, 32, SW);
        this->reverse_legal_lookup_black[202] = ReverseLegalLookup(Role::Black, Piece::King, 14, 37, SW);
        this->reverse_legal_lookup_black[203] = ReverseLegalLookup(Role::Black, Piece::King, 14, 41, SW);
        this->reverse_legal_lookup_black[204] = ReverseLegalLookup(Role::Black, Piece::King, 14, 46, SW);
    }
    // generating for black king 15 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 10, legal: (legal black (move king j 8 i 9))
        ddi->diagonals.emplace_back(10, 208);

        // to position 4, legal: (legal black (move king j 8 h 10))
        ddi->diagonals.emplace_back(4, 209);

        this->diagonal_data[164].push_back(ddi);


        this->reverse_legal_lookup_black[208] = ReverseLegalLookup(Role::Black, Piece::King, 15, 10, NW);
        this->reverse_legal_lookup_black[209] = ReverseLegalLookup(Role::Black, Piece::King, 15, 4, NW);
    }
    // generating for black king 15 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(7);
        // to position 20, legal: (legal black (move king j 8 i 7))
        ddi->diagonals.emplace_back(20, 210);

        // to position 24, legal: (legal black (move king j 8 h 6))
        ddi->diagonals.emplace_back(24, 211);

        // to position 29, legal: (legal black (move king j 8 g 5))
        ddi->diagonals.emplace_back(29, 212);

        // to position 33, legal: (legal black (move king j 8 f 4))
        ddi->diagonals.emplace_back(33, 213);

        // to position 38, legal: (legal black (move king j 8 e 3))
        ddi->diagonals.emplace_back(38, 214);

        // to position 42, legal: (legal black (move king j 8 d 2))
        ddi->diagonals.emplace_back(42, 215);

        // to position 47, legal: (legal black (move king j 8 c 1))
        ddi->diagonals.emplace_back(47, 216);

        this->diagonal_data[164].push_back(ddi);


        this->reverse_legal_lookup_black[210] = ReverseLegalLookup(Role::Black, Piece::King, 15, 20, SW);
        this->reverse_legal_lookup_black[211] = ReverseLegalLookup(Role::Black, Piece::King, 15, 24, SW);
        this->reverse_legal_lookup_black[212] = ReverseLegalLookup(Role::Black, Piece::King, 15, 29, SW);
        this->reverse_legal_lookup_black[213] = ReverseLegalLookup(Role::Black, Piece::King, 15, 33, SW);
        this->reverse_legal_lookup_black[214] = ReverseLegalLookup(Role::Black, Piece::King, 15, 38, SW);
        this->reverse_legal_lookup_black[215] = ReverseLegalLookup(Role::Black, Piece::King, 15, 42, SW);
        this->reverse_legal_lookup_black[216] = ReverseLegalLookup(Role::Black, Piece::King, 15, 47, SW);
    }
    // generating for black king 16 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 11, legal: (legal black (move king a 7 b 8))
        ddi->diagonals.emplace_back(11, 220);

        // to position 7, legal: (legal black (move king a 7 c 9))
        ddi->diagonals.emplace_back(7, 221);

        // to position 2, legal: (legal black (move king a 7 d 10))
        ddi->diagonals.emplace_back(2, 222);

        this->diagonal_data[165].push_back(ddi);


        this->reverse_legal_lookup_black[220] = ReverseLegalLookup(Role::Black, Piece::King, 16, 11, NE);
        this->reverse_legal_lookup_black[221] = ReverseLegalLookup(Role::Black, Piece::King, 16, 7, NE);
        this->reverse_legal_lookup_black[222] = ReverseLegalLookup(Role::Black, Piece::King, 16, 2, NE);
    }
    // generating for black king 16 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 21, legal: (legal black (move king a 7 b 6))
        ddi->diagonals.emplace_back(21, 223);

        // to position 27, legal: (legal black (move king a 7 c 5))
        ddi->diagonals.emplace_back(27, 224);

        // to position 32, legal: (legal black (move king a 7 d 4))
        ddi->diagonals.emplace_back(32, 225);

        // to position 38, legal: (legal black (move king a 7 e 3))
        ddi->diagonals.emplace_back(38, 226);

        // to position 43, legal: (legal black (move king a 7 f 2))
        ddi->diagonals.emplace_back(43, 227);

        // to position 49, legal: (legal black (move king a 7 g 1))
        ddi->diagonals.emplace_back(49, 228);

        this->diagonal_data[165].push_back(ddi);


        this->reverse_legal_lookup_black[223] = ReverseLegalLookup(Role::Black, Piece::King, 16, 21, SE);
        this->reverse_legal_lookup_black[224] = ReverseLegalLookup(Role::Black, Piece::King, 16, 27, SE);
        this->reverse_legal_lookup_black[225] = ReverseLegalLookup(Role::Black, Piece::King, 16, 32, SE);
        this->reverse_legal_lookup_black[226] = ReverseLegalLookup(Role::Black, Piece::King, 16, 38, SE);
        this->reverse_legal_lookup_black[227] = ReverseLegalLookup(Role::Black, Piece::King, 16, 43, SE);
        this->reverse_legal_lookup_black[228] = ReverseLegalLookup(Role::Black, Piece::King, 16, 49, SE);
    }
    // generating for black king 17 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 12, legal: (legal black (move king c 7 d 8))
        ddi->diagonals.emplace_back(12, 235);

        // to position 8, legal: (legal black (move king c 7 e 9))
        ddi->diagonals.emplace_back(8, 236);

        // to position 3, legal: (legal black (move king c 7 f 10))
        ddi->diagonals.emplace_back(3, 237);

        this->diagonal_data[166].push_back(ddi);


        this->reverse_legal_lookup_black[235] = ReverseLegalLookup(Role::Black, Piece::King, 17, 12, NE);
        this->reverse_legal_lookup_black[236] = ReverseLegalLookup(Role::Black, Piece::King, 17, 8, NE);
        this->reverse_legal_lookup_black[237] = ReverseLegalLookup(Role::Black, Piece::King, 17, 3, NE);
    }
    // generating for black king 17 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 11, legal: (legal black (move king c 7 b 8))
        ddi->diagonals.emplace_back(11, 238);

        // to position 6, legal: (legal black (move king c 7 a 9))
        ddi->diagonals.emplace_back(6, 239);

        this->diagonal_data[166].push_back(ddi);


        this->reverse_legal_lookup_black[238] = ReverseLegalLookup(Role::Black, Piece::King, 17, 11, NW);
        this->reverse_legal_lookup_black[239] = ReverseLegalLookup(Role::Black, Piece::King, 17, 6, NW);
    }
    // generating for black king 17 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(6);
        // to position 22, legal: (legal black (move king c 7 d 6))
        ddi->diagonals.emplace_back(22, 240);

        // to position 28, legal: (legal black (move king c 7 e 5))
        ddi->diagonals.emplace_back(28, 241);

        // to position 33, legal: (legal black (move king c 7 f 4))
        ddi->diagonals.emplace_back(33, 242);

        // to position 39, legal: (legal black (move king c 7 g 3))
        ddi->diagonals.emplace_back(39, 243);

        // to position 44, legal: (legal black (move king c 7 h 2))
        ddi->diagonals.emplace_back(44, 244);

        // to position 50, legal: (legal black (move king c 7 i 1))
        ddi->diagonals.emplace_back(50, 245);

        this->diagonal_data[166].push_back(ddi);


        this->reverse_legal_lookup_black[240] = ReverseLegalLookup(Role::Black, Piece::King, 17, 22, SE);
        this->reverse_legal_lookup_black[241] = ReverseLegalLookup(Role::Black, Piece::King, 17, 28, SE);
        this->reverse_legal_lookup_black[242] = ReverseLegalLookup(Role::Black, Piece::King, 17, 33, SE);
        this->reverse_legal_lookup_black[243] = ReverseLegalLookup(Role::Black, Piece::King, 17, 39, SE);
        this->reverse_legal_lookup_black[244] = ReverseLegalLookup(Role::Black, Piece::King, 17, 44, SE);
        this->reverse_legal_lookup_black[245] = ReverseLegalLookup(Role::Black, Piece::King, 17, 50, SE);
    }
    // generating for black king 17 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 21, legal: (legal black (move king c 7 b 6))
        ddi->diagonals.emplace_back(21, 246);

        // to position 26, legal: (legal black (move king c 7 a 5))
        ddi->diagonals.emplace_back(26, 247);

        this->diagonal_data[166].push_back(ddi);


        this->reverse_legal_lookup_black[246] = ReverseLegalLookup(Role::Black, Piece::King, 17, 21, SW);
        this->reverse_legal_lookup_black[247] = ReverseLegalLookup(Role::Black, Piece::King, 17, 26, SW);
    }
    // generating for black king 18 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 13, legal: (legal black (move king e 7 f 8))
        ddi->diagonals.emplace_back(13, 254);

        // to position 9, legal: (legal black (move king e 7 g 9))
        ddi->diagonals.emplace_back(9, 255);

        // to position 4, legal: (legal black (move king e 7 h 10))
        ddi->diagonals.emplace_back(4, 256);

        this->diagonal_data[167].push_back(ddi);


        this->reverse_legal_lookup_black[254] = ReverseLegalLookup(Role::Black, Piece::King, 18, 13, NE);
        this->reverse_legal_lookup_black[255] = ReverseLegalLookup(Role::Black, Piece::King, 18, 9, NE);
        this->reverse_legal_lookup_black[256] = ReverseLegalLookup(Role::Black, Piece::King, 18, 4, NE);
    }
    // generating for black king 18 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 12, legal: (legal black (move king e 7 d 8))
        ddi->diagonals.emplace_back(12, 257);

        // to position 7, legal: (legal black (move king e 7 c 9))
        ddi->diagonals.emplace_back(7, 258);

        // to position 1, legal: (legal black (move king e 7 b 10))
        ddi->diagonals.emplace_back(1, 259);

        this->diagonal_data[167].push_back(ddi);


        this->reverse_legal_lookup_black[257] = ReverseLegalLookup(Role::Black, Piece::King, 18, 12, NW);
        this->reverse_legal_lookup_black[258] = ReverseLegalLookup(Role::Black, Piece::King, 18, 7, NW);
        this->reverse_legal_lookup_black[259] = ReverseLegalLookup(Role::Black, Piece::King, 18, 1, NW);
    }
    // generating for black king 18 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 23, legal: (legal black (move king e 7 f 6))
        ddi->diagonals.emplace_back(23, 260);

        // to position 29, legal: (legal black (move king e 7 g 5))
        ddi->diagonals.emplace_back(29, 261);

        // to position 34, legal: (legal black (move king e 7 h 4))
        ddi->diagonals.emplace_back(34, 262);

        // to position 40, legal: (legal black (move king e 7 i 3))
        ddi->diagonals.emplace_back(40, 263);

        // to position 45, legal: (legal black (move king e 7 j 2))
        ddi->diagonals.emplace_back(45, 264);

        this->diagonal_data[167].push_back(ddi);


        this->reverse_legal_lookup_black[260] = ReverseLegalLookup(Role::Black, Piece::King, 18, 23, SE);
        this->reverse_legal_lookup_black[261] = ReverseLegalLookup(Role::Black, Piece::King, 18, 29, SE);
        this->reverse_legal_lookup_black[262] = ReverseLegalLookup(Role::Black, Piece::King, 18, 34, SE);
        this->reverse_legal_lookup_black[263] = ReverseLegalLookup(Role::Black, Piece::King, 18, 40, SE);
        this->reverse_legal_lookup_black[264] = ReverseLegalLookup(Role::Black, Piece::King, 18, 45, SE);
    }
    // generating for black king 18 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 22, legal: (legal black (move king e 7 d 6))
        ddi->diagonals.emplace_back(22, 265);

        // to position 27, legal: (legal black (move king e 7 c 5))
        ddi->diagonals.emplace_back(27, 266);

        // to position 31, legal: (legal black (move king e 7 b 4))
        ddi->diagonals.emplace_back(31, 267);

        // to position 36, legal: (legal black (move king e 7 a 3))
        ddi->diagonals.emplace_back(36, 268);

        this->diagonal_data[167].push_back(ddi);


        this->reverse_legal_lookup_black[265] = ReverseLegalLookup(Role::Black, Piece::King, 18, 22, SW);
        this->reverse_legal_lookup_black[266] = ReverseLegalLookup(Role::Black, Piece::King, 18, 27, SW);
        this->reverse_legal_lookup_black[267] = ReverseLegalLookup(Role::Black, Piece::King, 18, 31, SW);
        this->reverse_legal_lookup_black[268] = ReverseLegalLookup(Role::Black, Piece::King, 18, 36, SW);
    }
    // generating for black king 19 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 14, legal: (legal black (move king g 7 h 8))
        ddi->diagonals.emplace_back(14, 275);

        // to position 10, legal: (legal black (move king g 7 i 9))
        ddi->diagonals.emplace_back(10, 276);

        // to position 5, legal: (legal black (move king g 7 j 10))
        ddi->diagonals.emplace_back(5, 277);

        this->diagonal_data[168].push_back(ddi);


        this->reverse_legal_lookup_black[275] = ReverseLegalLookup(Role::Black, Piece::King, 19, 14, NE);
        this->reverse_legal_lookup_black[276] = ReverseLegalLookup(Role::Black, Piece::King, 19, 10, NE);
        this->reverse_legal_lookup_black[277] = ReverseLegalLookup(Role::Black, Piece::King, 19, 5, NE);
    }
    // generating for black king 19 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 13, legal: (legal black (move king g 7 f 8))
        ddi->diagonals.emplace_back(13, 278);

        // to position 8, legal: (legal black (move king g 7 e 9))
        ddi->diagonals.emplace_back(8, 279);

        // to position 2, legal: (legal black (move king g 7 d 10))
        ddi->diagonals.emplace_back(2, 280);

        this->diagonal_data[168].push_back(ddi);


        this->reverse_legal_lookup_black[278] = ReverseLegalLookup(Role::Black, Piece::King, 19, 13, NW);
        this->reverse_legal_lookup_black[279] = ReverseLegalLookup(Role::Black, Piece::King, 19, 8, NW);
        this->reverse_legal_lookup_black[280] = ReverseLegalLookup(Role::Black, Piece::King, 19, 2, NW);
    }
    // generating for black king 19 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 24, legal: (legal black (move king g 7 h 6))
        ddi->diagonals.emplace_back(24, 281);

        // to position 30, legal: (legal black (move king g 7 i 5))
        ddi->diagonals.emplace_back(30, 282);

        // to position 35, legal: (legal black (move king g 7 j 4))
        ddi->diagonals.emplace_back(35, 283);

        this->diagonal_data[168].push_back(ddi);


        this->reverse_legal_lookup_black[281] = ReverseLegalLookup(Role::Black, Piece::King, 19, 24, SE);
        this->reverse_legal_lookup_black[282] = ReverseLegalLookup(Role::Black, Piece::King, 19, 30, SE);
        this->reverse_legal_lookup_black[283] = ReverseLegalLookup(Role::Black, Piece::King, 19, 35, SE);
    }
    // generating for black king 19 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(6);
        // to position 23, legal: (legal black (move king g 7 f 6))
        ddi->diagonals.emplace_back(23, 284);

        // to position 28, legal: (legal black (move king g 7 e 5))
        ddi->diagonals.emplace_back(28, 285);

        // to position 32, legal: (legal black (move king g 7 d 4))
        ddi->diagonals.emplace_back(32, 286);

        // to position 37, legal: (legal black (move king g 7 c 3))
        ddi->diagonals.emplace_back(37, 287);

        // to position 41, legal: (legal black (move king g 7 b 2))
        ddi->diagonals.emplace_back(41, 288);

        // to position 46, legal: (legal black (move king g 7 a 1))
        ddi->diagonals.emplace_back(46, 289);

        this->diagonal_data[168].push_back(ddi);


        this->reverse_legal_lookup_black[284] = ReverseLegalLookup(Role::Black, Piece::King, 19, 23, SW);
        this->reverse_legal_lookup_black[285] = ReverseLegalLookup(Role::Black, Piece::King, 19, 28, SW);
        this->reverse_legal_lookup_black[286] = ReverseLegalLookup(Role::Black, Piece::King, 19, 32, SW);
        this->reverse_legal_lookup_black[287] = ReverseLegalLookup(Role::Black, Piece::King, 19, 37, SW);
        this->reverse_legal_lookup_black[288] = ReverseLegalLookup(Role::Black, Piece::King, 19, 41, SW);
        this->reverse_legal_lookup_black[289] = ReverseLegalLookup(Role::Black, Piece::King, 19, 46, SW);
    }
    // generating for black king 20 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 15, legal: (legal black (move king i 7 j 8))
        ddi->diagonals.emplace_back(15, 294);

        this->diagonal_data[169].push_back(ddi);


        this->reverse_legal_lookup_black[294] = ReverseLegalLookup(Role::Black, Piece::King, 20, 15, NE);
    }
    // generating for black king 20 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 14, legal: (legal black (move king i 7 h 8))
        ddi->diagonals.emplace_back(14, 295);

        // to position 9, legal: (legal black (move king i 7 g 9))
        ddi->diagonals.emplace_back(9, 296);

        // to position 3, legal: (legal black (move king i 7 f 10))
        ddi->diagonals.emplace_back(3, 297);

        this->diagonal_data[169].push_back(ddi);


        this->reverse_legal_lookup_black[295] = ReverseLegalLookup(Role::Black, Piece::King, 20, 14, NW);
        this->reverse_legal_lookup_black[296] = ReverseLegalLookup(Role::Black, Piece::King, 20, 9, NW);
        this->reverse_legal_lookup_black[297] = ReverseLegalLookup(Role::Black, Piece::King, 20, 3, NW);
    }
    // generating for black king 20 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 25, legal: (legal black (move king i 7 j 6))
        ddi->diagonals.emplace_back(25, 298);

        this->diagonal_data[169].push_back(ddi);


        this->reverse_legal_lookup_black[298] = ReverseLegalLookup(Role::Black, Piece::King, 20, 25, SE);
    }
    // generating for black king 20 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(6);
        // to position 24, legal: (legal black (move king i 7 h 6))
        ddi->diagonals.emplace_back(24, 299);

        // to position 29, legal: (legal black (move king i 7 g 5))
        ddi->diagonals.emplace_back(29, 300);

        // to position 33, legal: (legal black (move king i 7 f 4))
        ddi->diagonals.emplace_back(33, 301);

        // to position 38, legal: (legal black (move king i 7 e 3))
        ddi->diagonals.emplace_back(38, 302);

        // to position 42, legal: (legal black (move king i 7 d 2))
        ddi->diagonals.emplace_back(42, 303);

        // to position 47, legal: (legal black (move king i 7 c 1))
        ddi->diagonals.emplace_back(47, 304);

        this->diagonal_data[169].push_back(ddi);


        this->reverse_legal_lookup_black[299] = ReverseLegalLookup(Role::Black, Piece::King, 20, 24, SW);
        this->reverse_legal_lookup_black[300] = ReverseLegalLookup(Role::Black, Piece::King, 20, 29, SW);
        this->reverse_legal_lookup_black[301] = ReverseLegalLookup(Role::Black, Piece::King, 20, 33, SW);
        this->reverse_legal_lookup_black[302] = ReverseLegalLookup(Role::Black, Piece::King, 20, 38, SW);
        this->reverse_legal_lookup_black[303] = ReverseLegalLookup(Role::Black, Piece::King, 20, 42, SW);
        this->reverse_legal_lookup_black[304] = ReverseLegalLookup(Role::Black, Piece::King, 20, 47, SW);
    }
    // generating for black king 21 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 17, legal: (legal black (move king b 6 c 7))
        ddi->diagonals.emplace_back(17, 309);

        // to position 12, legal: (legal black (move king b 6 d 8))
        ddi->diagonals.emplace_back(12, 310);

        // to position 8, legal: (legal black (move king b 6 e 9))
        ddi->diagonals.emplace_back(8, 311);

        // to position 3, legal: (legal black (move king b 6 f 10))
        ddi->diagonals.emplace_back(3, 312);

        this->diagonal_data[170].push_back(ddi);


        this->reverse_legal_lookup_black[309] = ReverseLegalLookup(Role::Black, Piece::King, 21, 17, NE);
        this->reverse_legal_lookup_black[310] = ReverseLegalLookup(Role::Black, Piece::King, 21, 12, NE);
        this->reverse_legal_lookup_black[311] = ReverseLegalLookup(Role::Black, Piece::King, 21, 8, NE);
        this->reverse_legal_lookup_black[312] = ReverseLegalLookup(Role::Black, Piece::King, 21, 3, NE);
    }
    // generating for black king 21 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 16, legal: (legal black (move king b 6 a 7))
        ddi->diagonals.emplace_back(16, 313);

        this->diagonal_data[170].push_back(ddi);


        this->reverse_legal_lookup_black[313] = ReverseLegalLookup(Role::Black, Piece::King, 21, 16, NW);
    }
    // generating for black king 21 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 27, legal: (legal black (move king b 6 c 5))
        ddi->diagonals.emplace_back(27, 314);

        // to position 32, legal: (legal black (move king b 6 d 4))
        ddi->diagonals.emplace_back(32, 315);

        // to position 38, legal: (legal black (move king b 6 e 3))
        ddi->diagonals.emplace_back(38, 316);

        // to position 43, legal: (legal black (move king b 6 f 2))
        ddi->diagonals.emplace_back(43, 317);

        // to position 49, legal: (legal black (move king b 6 g 1))
        ddi->diagonals.emplace_back(49, 318);

        this->diagonal_data[170].push_back(ddi);


        this->reverse_legal_lookup_black[314] = ReverseLegalLookup(Role::Black, Piece::King, 21, 27, SE);
        this->reverse_legal_lookup_black[315] = ReverseLegalLookup(Role::Black, Piece::King, 21, 32, SE);
        this->reverse_legal_lookup_black[316] = ReverseLegalLookup(Role::Black, Piece::King, 21, 38, SE);
        this->reverse_legal_lookup_black[317] = ReverseLegalLookup(Role::Black, Piece::King, 21, 43, SE);
        this->reverse_legal_lookup_black[318] = ReverseLegalLookup(Role::Black, Piece::King, 21, 49, SE);
    }
    // generating for black king 21 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 26, legal: (legal black (move king b 6 a 5))
        ddi->diagonals.emplace_back(26, 319);

        this->diagonal_data[170].push_back(ddi);


        this->reverse_legal_lookup_black[319] = ReverseLegalLookup(Role::Black, Piece::King, 21, 26, SW);
    }
    // generating for black king 22 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 18, legal: (legal black (move king d 6 e 7))
        ddi->diagonals.emplace_back(18, 326);

        // to position 13, legal: (legal black (move king d 6 f 8))
        ddi->diagonals.emplace_back(13, 327);

        // to position 9, legal: (legal black (move king d 6 g 9))
        ddi->diagonals.emplace_back(9, 328);

        // to position 4, legal: (legal black (move king d 6 h 10))
        ddi->diagonals.emplace_back(4, 329);

        this->diagonal_data[171].push_back(ddi);


        this->reverse_legal_lookup_black[326] = ReverseLegalLookup(Role::Black, Piece::King, 22, 18, NE);
        this->reverse_legal_lookup_black[327] = ReverseLegalLookup(Role::Black, Piece::King, 22, 13, NE);
        this->reverse_legal_lookup_black[328] = ReverseLegalLookup(Role::Black, Piece::King, 22, 9, NE);
        this->reverse_legal_lookup_black[329] = ReverseLegalLookup(Role::Black, Piece::King, 22, 4, NE);
    }
    // generating for black king 22 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 17, legal: (legal black (move king d 6 c 7))
        ddi->diagonals.emplace_back(17, 330);

        // to position 11, legal: (legal black (move king d 6 b 8))
        ddi->diagonals.emplace_back(11, 331);

        // to position 6, legal: (legal black (move king d 6 a 9))
        ddi->diagonals.emplace_back(6, 332);

        this->diagonal_data[171].push_back(ddi);


        this->reverse_legal_lookup_black[330] = ReverseLegalLookup(Role::Black, Piece::King, 22, 17, NW);
        this->reverse_legal_lookup_black[331] = ReverseLegalLookup(Role::Black, Piece::King, 22, 11, NW);
        this->reverse_legal_lookup_black[332] = ReverseLegalLookup(Role::Black, Piece::King, 22, 6, NW);
    }
    // generating for black king 22 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(5);
        // to position 28, legal: (legal black (move king d 6 e 5))
        ddi->diagonals.emplace_back(28, 333);

        // to position 33, legal: (legal black (move king d 6 f 4))
        ddi->diagonals.emplace_back(33, 334);

        // to position 39, legal: (legal black (move king d 6 g 3))
        ddi->diagonals.emplace_back(39, 335);

        // to position 44, legal: (legal black (move king d 6 h 2))
        ddi->diagonals.emplace_back(44, 336);

        // to position 50, legal: (legal black (move king d 6 i 1))
        ddi->diagonals.emplace_back(50, 337);

        this->diagonal_data[171].push_back(ddi);


        this->reverse_legal_lookup_black[333] = ReverseLegalLookup(Role::Black, Piece::King, 22, 28, SE);
        this->reverse_legal_lookup_black[334] = ReverseLegalLookup(Role::Black, Piece::King, 22, 33, SE);
        this->reverse_legal_lookup_black[335] = ReverseLegalLookup(Role::Black, Piece::King, 22, 39, SE);
        this->reverse_legal_lookup_black[336] = ReverseLegalLookup(Role::Black, Piece::King, 22, 44, SE);
        this->reverse_legal_lookup_black[337] = ReverseLegalLookup(Role::Black, Piece::King, 22, 50, SE);
    }
    // generating for black king 22 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 27, legal: (legal black (move king d 6 c 5))
        ddi->diagonals.emplace_back(27, 338);

        // to position 31, legal: (legal black (move king d 6 b 4))
        ddi->diagonals.emplace_back(31, 339);

        // to position 36, legal: (legal black (move king d 6 a 3))
        ddi->diagonals.emplace_back(36, 340);

        this->diagonal_data[171].push_back(ddi);


        this->reverse_legal_lookup_black[338] = ReverseLegalLookup(Role::Black, Piece::King, 22, 27, SW);
        this->reverse_legal_lookup_black[339] = ReverseLegalLookup(Role::Black, Piece::King, 22, 31, SW);
        this->reverse_legal_lookup_black[340] = ReverseLegalLookup(Role::Black, Piece::King, 22, 36, SW);
    }
    // generating for black king 23 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 19, legal: (legal black (move king f 6 g 7))
        ddi->diagonals.emplace_back(19, 347);

        // to position 14, legal: (legal black (move king f 6 h 8))
        ddi->diagonals.emplace_back(14, 348);

        // to position 10, legal: (legal black (move king f 6 i 9))
        ddi->diagonals.emplace_back(10, 349);

        // to position 5, legal: (legal black (move king f 6 j 10))
        ddi->diagonals.emplace_back(5, 350);

        this->diagonal_data[172].push_back(ddi);


        this->reverse_legal_lookup_black[347] = ReverseLegalLookup(Role::Black, Piece::King, 23, 19, NE);
        this->reverse_legal_lookup_black[348] = ReverseLegalLookup(Role::Black, Piece::King, 23, 14, NE);
        this->reverse_legal_lookup_black[349] = ReverseLegalLookup(Role::Black, Piece::King, 23, 10, NE);
        this->reverse_legal_lookup_black[350] = ReverseLegalLookup(Role::Black, Piece::King, 23, 5, NE);
    }
    // generating for black king 23 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 18, legal: (legal black (move king f 6 e 7))
        ddi->diagonals.emplace_back(18, 351);

        // to position 12, legal: (legal black (move king f 6 d 8))
        ddi->diagonals.emplace_back(12, 352);

        // to position 7, legal: (legal black (move king f 6 c 9))
        ddi->diagonals.emplace_back(7, 353);

        // to position 1, legal: (legal black (move king f 6 b 10))
        ddi->diagonals.emplace_back(1, 354);

        this->diagonal_data[172].push_back(ddi);


        this->reverse_legal_lookup_black[351] = ReverseLegalLookup(Role::Black, Piece::King, 23, 18, NW);
        this->reverse_legal_lookup_black[352] = ReverseLegalLookup(Role::Black, Piece::King, 23, 12, NW);
        this->reverse_legal_lookup_black[353] = ReverseLegalLookup(Role::Black, Piece::King, 23, 7, NW);
        this->reverse_legal_lookup_black[354] = ReverseLegalLookup(Role::Black, Piece::King, 23, 1, NW);
    }
    // generating for black king 23 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 29, legal: (legal black (move king f 6 g 5))
        ddi->diagonals.emplace_back(29, 355);

        // to position 34, legal: (legal black (move king f 6 h 4))
        ddi->diagonals.emplace_back(34, 356);

        // to position 40, legal: (legal black (move king f 6 i 3))
        ddi->diagonals.emplace_back(40, 357);

        // to position 45, legal: (legal black (move king f 6 j 2))
        ddi->diagonals.emplace_back(45, 358);

        this->diagonal_data[172].push_back(ddi);


        this->reverse_legal_lookup_black[355] = ReverseLegalLookup(Role::Black, Piece::King, 23, 29, SE);
        this->reverse_legal_lookup_black[356] = ReverseLegalLookup(Role::Black, Piece::King, 23, 34, SE);
        this->reverse_legal_lookup_black[357] = ReverseLegalLookup(Role::Black, Piece::King, 23, 40, SE);
        this->reverse_legal_lookup_black[358] = ReverseLegalLookup(Role::Black, Piece::King, 23, 45, SE);
    }
    // generating for black king 23 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 28, legal: (legal black (move king f 6 e 5))
        ddi->diagonals.emplace_back(28, 359);

        // to position 32, legal: (legal black (move king f 6 d 4))
        ddi->diagonals.emplace_back(32, 360);

        // to position 37, legal: (legal black (move king f 6 c 3))
        ddi->diagonals.emplace_back(37, 361);

        // to position 41, legal: (legal black (move king f 6 b 2))
        ddi->diagonals.emplace_back(41, 362);

        // to position 46, legal: (legal black (move king f 6 a 1))
        ddi->diagonals.emplace_back(46, 363);

        this->diagonal_data[172].push_back(ddi);


        this->reverse_legal_lookup_black[359] = ReverseLegalLookup(Role::Black, Piece::King, 23, 28, SW);
        this->reverse_legal_lookup_black[360] = ReverseLegalLookup(Role::Black, Piece::King, 23, 32, SW);
        this->reverse_legal_lookup_black[361] = ReverseLegalLookup(Role::Black, Piece::King, 23, 37, SW);
        this->reverse_legal_lookup_black[362] = ReverseLegalLookup(Role::Black, Piece::King, 23, 41, SW);
        this->reverse_legal_lookup_black[363] = ReverseLegalLookup(Role::Black, Piece::King, 23, 46, SW);
    }
    // generating for black king 24 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 20, legal: (legal black (move king h 6 i 7))
        ddi->diagonals.emplace_back(20, 370);

        // to position 15, legal: (legal black (move king h 6 j 8))
        ddi->diagonals.emplace_back(15, 371);

        this->diagonal_data[173].push_back(ddi);


        this->reverse_legal_lookup_black[370] = ReverseLegalLookup(Role::Black, Piece::King, 24, 20, NE);
        this->reverse_legal_lookup_black[371] = ReverseLegalLookup(Role::Black, Piece::King, 24, 15, NE);
    }
    // generating for black king 24 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 19, legal: (legal black (move king h 6 g 7))
        ddi->diagonals.emplace_back(19, 372);

        // to position 13, legal: (legal black (move king h 6 f 8))
        ddi->diagonals.emplace_back(13, 373);

        // to position 8, legal: (legal black (move king h 6 e 9))
        ddi->diagonals.emplace_back(8, 374);

        // to position 2, legal: (legal black (move king h 6 d 10))
        ddi->diagonals.emplace_back(2, 375);

        this->diagonal_data[173].push_back(ddi);


        this->reverse_legal_lookup_black[372] = ReverseLegalLookup(Role::Black, Piece::King, 24, 19, NW);
        this->reverse_legal_lookup_black[373] = ReverseLegalLookup(Role::Black, Piece::King, 24, 13, NW);
        this->reverse_legal_lookup_black[374] = ReverseLegalLookup(Role::Black, Piece::King, 24, 8, NW);
        this->reverse_legal_lookup_black[375] = ReverseLegalLookup(Role::Black, Piece::King, 24, 2, NW);
    }
    // generating for black king 24 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 30, legal: (legal black (move king h 6 i 5))
        ddi->diagonals.emplace_back(30, 376);

        // to position 35, legal: (legal black (move king h 6 j 4))
        ddi->diagonals.emplace_back(35, 377);

        this->diagonal_data[173].push_back(ddi);


        this->reverse_legal_lookup_black[376] = ReverseLegalLookup(Role::Black, Piece::King, 24, 30, SE);
        this->reverse_legal_lookup_black[377] = ReverseLegalLookup(Role::Black, Piece::King, 24, 35, SE);
    }
    // generating for black king 24 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 29, legal: (legal black (move king h 6 g 5))
        ddi->diagonals.emplace_back(29, 378);

        // to position 33, legal: (legal black (move king h 6 f 4))
        ddi->diagonals.emplace_back(33, 379);

        // to position 38, legal: (legal black (move king h 6 e 3))
        ddi->diagonals.emplace_back(38, 380);

        // to position 42, legal: (legal black (move king h 6 d 2))
        ddi->diagonals.emplace_back(42, 381);

        // to position 47, legal: (legal black (move king h 6 c 1))
        ddi->diagonals.emplace_back(47, 382);

        this->diagonal_data[173].push_back(ddi);


        this->reverse_legal_lookup_black[378] = ReverseLegalLookup(Role::Black, Piece::King, 24, 29, SW);
        this->reverse_legal_lookup_black[379] = ReverseLegalLookup(Role::Black, Piece::King, 24, 33, SW);
        this->reverse_legal_lookup_black[380] = ReverseLegalLookup(Role::Black, Piece::King, 24, 38, SW);
        this->reverse_legal_lookup_black[381] = ReverseLegalLookup(Role::Black, Piece::King, 24, 42, SW);
        this->reverse_legal_lookup_black[382] = ReverseLegalLookup(Role::Black, Piece::King, 24, 47, SW);
    }
    // generating for black king 25 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 20, legal: (legal black (move king j 6 i 7))
        ddi->diagonals.emplace_back(20, 386);

        // to position 14, legal: (legal black (move king j 6 h 8))
        ddi->diagonals.emplace_back(14, 387);

        // to position 9, legal: (legal black (move king j 6 g 9))
        ddi->diagonals.emplace_back(9, 388);

        // to position 3, legal: (legal black (move king j 6 f 10))
        ddi->diagonals.emplace_back(3, 389);

        this->diagonal_data[174].push_back(ddi);


        this->reverse_legal_lookup_black[386] = ReverseLegalLookup(Role::Black, Piece::King, 25, 20, NW);
        this->reverse_legal_lookup_black[387] = ReverseLegalLookup(Role::Black, Piece::King, 25, 14, NW);
        this->reverse_legal_lookup_black[388] = ReverseLegalLookup(Role::Black, Piece::King, 25, 9, NW);
        this->reverse_legal_lookup_black[389] = ReverseLegalLookup(Role::Black, Piece::King, 25, 3, NW);
    }
    // generating for black king 25 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(5);
        // to position 30, legal: (legal black (move king j 6 i 5))
        ddi->diagonals.emplace_back(30, 390);

        // to position 34, legal: (legal black (move king j 6 h 4))
        ddi->diagonals.emplace_back(34, 391);

        // to position 39, legal: (legal black (move king j 6 g 3))
        ddi->diagonals.emplace_back(39, 392);

        // to position 43, legal: (legal black (move king j 6 f 2))
        ddi->diagonals.emplace_back(43, 393);

        // to position 48, legal: (legal black (move king j 6 e 1))
        ddi->diagonals.emplace_back(48, 394);

        this->diagonal_data[174].push_back(ddi);


        this->reverse_legal_lookup_black[390] = ReverseLegalLookup(Role::Black, Piece::King, 25, 30, SW);
        this->reverse_legal_lookup_black[391] = ReverseLegalLookup(Role::Black, Piece::King, 25, 34, SW);
        this->reverse_legal_lookup_black[392] = ReverseLegalLookup(Role::Black, Piece::King, 25, 39, SW);
        this->reverse_legal_lookup_black[393] = ReverseLegalLookup(Role::Black, Piece::King, 25, 43, SW);
        this->reverse_legal_lookup_black[394] = ReverseLegalLookup(Role::Black, Piece::King, 25, 48, SW);
    }
    // generating for black king 26 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 21, legal: (legal black (move king a 5 b 6))
        ddi->diagonals.emplace_back(21, 398);

        // to position 17, legal: (legal black (move king a 5 c 7))
        ddi->diagonals.emplace_back(17, 399);

        // to position 12, legal: (legal black (move king a 5 d 8))
        ddi->diagonals.emplace_back(12, 400);

        // to position 8, legal: (legal black (move king a 5 e 9))
        ddi->diagonals.emplace_back(8, 401);

        // to position 3, legal: (legal black (move king a 5 f 10))
        ddi->diagonals.emplace_back(3, 402);

        this->diagonal_data[175].push_back(ddi);


        this->reverse_legal_lookup_black[398] = ReverseLegalLookup(Role::Black, Piece::King, 26, 21, NE);
        this->reverse_legal_lookup_black[399] = ReverseLegalLookup(Role::Black, Piece::King, 26, 17, NE);
        this->reverse_legal_lookup_black[400] = ReverseLegalLookup(Role::Black, Piece::King, 26, 12, NE);
        this->reverse_legal_lookup_black[401] = ReverseLegalLookup(Role::Black, Piece::King, 26, 8, NE);
        this->reverse_legal_lookup_black[402] = ReverseLegalLookup(Role::Black, Piece::King, 26, 3, NE);
    }
    // generating for black king 26 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 31, legal: (legal black (move king a 5 b 4))
        ddi->diagonals.emplace_back(31, 403);

        // to position 37, legal: (legal black (move king a 5 c 3))
        ddi->diagonals.emplace_back(37, 404);

        // to position 42, legal: (legal black (move king a 5 d 2))
        ddi->diagonals.emplace_back(42, 405);

        // to position 48, legal: (legal black (move king a 5 e 1))
        ddi->diagonals.emplace_back(48, 406);

        this->diagonal_data[175].push_back(ddi);


        this->reverse_legal_lookup_black[403] = ReverseLegalLookup(Role::Black, Piece::King, 26, 31, SE);
        this->reverse_legal_lookup_black[404] = ReverseLegalLookup(Role::Black, Piece::King, 26, 37, SE);
        this->reverse_legal_lookup_black[405] = ReverseLegalLookup(Role::Black, Piece::King, 26, 42, SE);
        this->reverse_legal_lookup_black[406] = ReverseLegalLookup(Role::Black, Piece::King, 26, 48, SE);
    }
    // generating for black king 27 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 22, legal: (legal black (move king c 5 d 6))
        ddi->diagonals.emplace_back(22, 413);

        // to position 18, legal: (legal black (move king c 5 e 7))
        ddi->diagonals.emplace_back(18, 414);

        // to position 13, legal: (legal black (move king c 5 f 8))
        ddi->diagonals.emplace_back(13, 415);

        // to position 9, legal: (legal black (move king c 5 g 9))
        ddi->diagonals.emplace_back(9, 416);

        // to position 4, legal: (legal black (move king c 5 h 10))
        ddi->diagonals.emplace_back(4, 417);

        this->diagonal_data[176].push_back(ddi);


        this->reverse_legal_lookup_black[413] = ReverseLegalLookup(Role::Black, Piece::King, 27, 22, NE);
        this->reverse_legal_lookup_black[414] = ReverseLegalLookup(Role::Black, Piece::King, 27, 18, NE);
        this->reverse_legal_lookup_black[415] = ReverseLegalLookup(Role::Black, Piece::King, 27, 13, NE);
        this->reverse_legal_lookup_black[416] = ReverseLegalLookup(Role::Black, Piece::King, 27, 9, NE);
        this->reverse_legal_lookup_black[417] = ReverseLegalLookup(Role::Black, Piece::King, 27, 4, NE);
    }
    // generating for black king 27 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 21, legal: (legal black (move king c 5 b 6))
        ddi->diagonals.emplace_back(21, 418);

        // to position 16, legal: (legal black (move king c 5 a 7))
        ddi->diagonals.emplace_back(16, 419);

        this->diagonal_data[176].push_back(ddi);


        this->reverse_legal_lookup_black[418] = ReverseLegalLookup(Role::Black, Piece::King, 27, 21, NW);
        this->reverse_legal_lookup_black[419] = ReverseLegalLookup(Role::Black, Piece::King, 27, 16, NW);
    }
    // generating for black king 27 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 32, legal: (legal black (move king c 5 d 4))
        ddi->diagonals.emplace_back(32, 420);

        // to position 38, legal: (legal black (move king c 5 e 3))
        ddi->diagonals.emplace_back(38, 421);

        // to position 43, legal: (legal black (move king c 5 f 2))
        ddi->diagonals.emplace_back(43, 422);

        // to position 49, legal: (legal black (move king c 5 g 1))
        ddi->diagonals.emplace_back(49, 423);

        this->diagonal_data[176].push_back(ddi);


        this->reverse_legal_lookup_black[420] = ReverseLegalLookup(Role::Black, Piece::King, 27, 32, SE);
        this->reverse_legal_lookup_black[421] = ReverseLegalLookup(Role::Black, Piece::King, 27, 38, SE);
        this->reverse_legal_lookup_black[422] = ReverseLegalLookup(Role::Black, Piece::King, 27, 43, SE);
        this->reverse_legal_lookup_black[423] = ReverseLegalLookup(Role::Black, Piece::King, 27, 49, SE);
    }
    // generating for black king 27 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 31, legal: (legal black (move king c 5 b 4))
        ddi->diagonals.emplace_back(31, 424);

        // to position 36, legal: (legal black (move king c 5 a 3))
        ddi->diagonals.emplace_back(36, 425);

        this->diagonal_data[176].push_back(ddi);


        this->reverse_legal_lookup_black[424] = ReverseLegalLookup(Role::Black, Piece::King, 27, 31, SW);
        this->reverse_legal_lookup_black[425] = ReverseLegalLookup(Role::Black, Piece::King, 27, 36, SW);
    }
    // generating for black king 28 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 23, legal: (legal black (move king e 5 f 6))
        ddi->diagonals.emplace_back(23, 432);

        // to position 19, legal: (legal black (move king e 5 g 7))
        ddi->diagonals.emplace_back(19, 433);

        // to position 14, legal: (legal black (move king e 5 h 8))
        ddi->diagonals.emplace_back(14, 434);

        // to position 10, legal: (legal black (move king e 5 i 9))
        ddi->diagonals.emplace_back(10, 435);

        // to position 5, legal: (legal black (move king e 5 j 10))
        ddi->diagonals.emplace_back(5, 436);

        this->diagonal_data[177].push_back(ddi);


        this->reverse_legal_lookup_black[432] = ReverseLegalLookup(Role::Black, Piece::King, 28, 23, NE);
        this->reverse_legal_lookup_black[433] = ReverseLegalLookup(Role::Black, Piece::King, 28, 19, NE);
        this->reverse_legal_lookup_black[434] = ReverseLegalLookup(Role::Black, Piece::King, 28, 14, NE);
        this->reverse_legal_lookup_black[435] = ReverseLegalLookup(Role::Black, Piece::King, 28, 10, NE);
        this->reverse_legal_lookup_black[436] = ReverseLegalLookup(Role::Black, Piece::King, 28, 5, NE);
    }
    // generating for black king 28 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 22, legal: (legal black (move king e 5 d 6))
        ddi->diagonals.emplace_back(22, 437);

        // to position 17, legal: (legal black (move king e 5 c 7))
        ddi->diagonals.emplace_back(17, 438);

        // to position 11, legal: (legal black (move king e 5 b 8))
        ddi->diagonals.emplace_back(11, 439);

        // to position 6, legal: (legal black (move king e 5 a 9))
        ddi->diagonals.emplace_back(6, 440);

        this->diagonal_data[177].push_back(ddi);


        this->reverse_legal_lookup_black[437] = ReverseLegalLookup(Role::Black, Piece::King, 28, 22, NW);
        this->reverse_legal_lookup_black[438] = ReverseLegalLookup(Role::Black, Piece::King, 28, 17, NW);
        this->reverse_legal_lookup_black[439] = ReverseLegalLookup(Role::Black, Piece::King, 28, 11, NW);
        this->reverse_legal_lookup_black[440] = ReverseLegalLookup(Role::Black, Piece::King, 28, 6, NW);
    }
    // generating for black king 28 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(4);
        // to position 33, legal: (legal black (move king e 5 f 4))
        ddi->diagonals.emplace_back(33, 441);

        // to position 39, legal: (legal black (move king e 5 g 3))
        ddi->diagonals.emplace_back(39, 442);

        // to position 44, legal: (legal black (move king e 5 h 2))
        ddi->diagonals.emplace_back(44, 443);

        // to position 50, legal: (legal black (move king e 5 i 1))
        ddi->diagonals.emplace_back(50, 444);

        this->diagonal_data[177].push_back(ddi);


        this->reverse_legal_lookup_black[441] = ReverseLegalLookup(Role::Black, Piece::King, 28, 33, SE);
        this->reverse_legal_lookup_black[442] = ReverseLegalLookup(Role::Black, Piece::King, 28, 39, SE);
        this->reverse_legal_lookup_black[443] = ReverseLegalLookup(Role::Black, Piece::King, 28, 44, SE);
        this->reverse_legal_lookup_black[444] = ReverseLegalLookup(Role::Black, Piece::King, 28, 50, SE);
    }
    // generating for black king 28 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 32, legal: (legal black (move king e 5 d 4))
        ddi->diagonals.emplace_back(32, 445);

        // to position 37, legal: (legal black (move king e 5 c 3))
        ddi->diagonals.emplace_back(37, 446);

        // to position 41, legal: (legal black (move king e 5 b 2))
        ddi->diagonals.emplace_back(41, 447);

        // to position 46, legal: (legal black (move king e 5 a 1))
        ddi->diagonals.emplace_back(46, 448);

        this->diagonal_data[177].push_back(ddi);


        this->reverse_legal_lookup_black[445] = ReverseLegalLookup(Role::Black, Piece::King, 28, 32, SW);
        this->reverse_legal_lookup_black[446] = ReverseLegalLookup(Role::Black, Piece::King, 28, 37, SW);
        this->reverse_legal_lookup_black[447] = ReverseLegalLookup(Role::Black, Piece::King, 28, 41, SW);
        this->reverse_legal_lookup_black[448] = ReverseLegalLookup(Role::Black, Piece::King, 28, 46, SW);
    }
    // generating for black king 29 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 24, legal: (legal black (move king g 5 h 6))
        ddi->diagonals.emplace_back(24, 455);

        // to position 20, legal: (legal black (move king g 5 i 7))
        ddi->diagonals.emplace_back(20, 456);

        // to position 15, legal: (legal black (move king g 5 j 8))
        ddi->diagonals.emplace_back(15, 457);

        this->diagonal_data[178].push_back(ddi);


        this->reverse_legal_lookup_black[455] = ReverseLegalLookup(Role::Black, Piece::King, 29, 24, NE);
        this->reverse_legal_lookup_black[456] = ReverseLegalLookup(Role::Black, Piece::King, 29, 20, NE);
        this->reverse_legal_lookup_black[457] = ReverseLegalLookup(Role::Black, Piece::King, 29, 15, NE);
    }
    // generating for black king 29 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 23, legal: (legal black (move king g 5 f 6))
        ddi->diagonals.emplace_back(23, 458);

        // to position 18, legal: (legal black (move king g 5 e 7))
        ddi->diagonals.emplace_back(18, 459);

        // to position 12, legal: (legal black (move king g 5 d 8))
        ddi->diagonals.emplace_back(12, 460);

        // to position 7, legal: (legal black (move king g 5 c 9))
        ddi->diagonals.emplace_back(7, 461);

        // to position 1, legal: (legal black (move king g 5 b 10))
        ddi->diagonals.emplace_back(1, 462);

        this->diagonal_data[178].push_back(ddi);


        this->reverse_legal_lookup_black[458] = ReverseLegalLookup(Role::Black, Piece::King, 29, 23, NW);
        this->reverse_legal_lookup_black[459] = ReverseLegalLookup(Role::Black, Piece::King, 29, 18, NW);
        this->reverse_legal_lookup_black[460] = ReverseLegalLookup(Role::Black, Piece::King, 29, 12, NW);
        this->reverse_legal_lookup_black[461] = ReverseLegalLookup(Role::Black, Piece::King, 29, 7, NW);
        this->reverse_legal_lookup_black[462] = ReverseLegalLookup(Role::Black, Piece::King, 29, 1, NW);
    }
    // generating for black king 29 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 34, legal: (legal black (move king g 5 h 4))
        ddi->diagonals.emplace_back(34, 463);

        // to position 40, legal: (legal black (move king g 5 i 3))
        ddi->diagonals.emplace_back(40, 464);

        // to position 45, legal: (legal black (move king g 5 j 2))
        ddi->diagonals.emplace_back(45, 465);

        this->diagonal_data[178].push_back(ddi);


        this->reverse_legal_lookup_black[463] = ReverseLegalLookup(Role::Black, Piece::King, 29, 34, SE);
        this->reverse_legal_lookup_black[464] = ReverseLegalLookup(Role::Black, Piece::King, 29, 40, SE);
        this->reverse_legal_lookup_black[465] = ReverseLegalLookup(Role::Black, Piece::King, 29, 45, SE);
    }
    // generating for black king 29 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 33, legal: (legal black (move king g 5 f 4))
        ddi->diagonals.emplace_back(33, 466);

        // to position 38, legal: (legal black (move king g 5 e 3))
        ddi->diagonals.emplace_back(38, 467);

        // to position 42, legal: (legal black (move king g 5 d 2))
        ddi->diagonals.emplace_back(42, 468);

        // to position 47, legal: (legal black (move king g 5 c 1))
        ddi->diagonals.emplace_back(47, 469);

        this->diagonal_data[178].push_back(ddi);


        this->reverse_legal_lookup_black[466] = ReverseLegalLookup(Role::Black, Piece::King, 29, 33, SW);
        this->reverse_legal_lookup_black[467] = ReverseLegalLookup(Role::Black, Piece::King, 29, 38, SW);
        this->reverse_legal_lookup_black[468] = ReverseLegalLookup(Role::Black, Piece::King, 29, 42, SW);
        this->reverse_legal_lookup_black[469] = ReverseLegalLookup(Role::Black, Piece::King, 29, 47, SW);
    }
    // generating for black king 30 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 25, legal: (legal black (move king i 5 j 6))
        ddi->diagonals.emplace_back(25, 474);

        this->diagonal_data[179].push_back(ddi);


        this->reverse_legal_lookup_black[474] = ReverseLegalLookup(Role::Black, Piece::King, 30, 25, NE);
    }
    // generating for black king 30 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 24, legal: (legal black (move king i 5 h 6))
        ddi->diagonals.emplace_back(24, 475);

        // to position 19, legal: (legal black (move king i 5 g 7))
        ddi->diagonals.emplace_back(19, 476);

        // to position 13, legal: (legal black (move king i 5 f 8))
        ddi->diagonals.emplace_back(13, 477);

        // to position 8, legal: (legal black (move king i 5 e 9))
        ddi->diagonals.emplace_back(8, 478);

        // to position 2, legal: (legal black (move king i 5 d 10))
        ddi->diagonals.emplace_back(2, 479);

        this->diagonal_data[179].push_back(ddi);


        this->reverse_legal_lookup_black[475] = ReverseLegalLookup(Role::Black, Piece::King, 30, 24, NW);
        this->reverse_legal_lookup_black[476] = ReverseLegalLookup(Role::Black, Piece::King, 30, 19, NW);
        this->reverse_legal_lookup_black[477] = ReverseLegalLookup(Role::Black, Piece::King, 30, 13, NW);
        this->reverse_legal_lookup_black[478] = ReverseLegalLookup(Role::Black, Piece::King, 30, 8, NW);
        this->reverse_legal_lookup_black[479] = ReverseLegalLookup(Role::Black, Piece::King, 30, 2, NW);
    }
    // generating for black king 30 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 35, legal: (legal black (move king i 5 j 4))
        ddi->diagonals.emplace_back(35, 480);

        this->diagonal_data[179].push_back(ddi);


        this->reverse_legal_lookup_black[480] = ReverseLegalLookup(Role::Black, Piece::King, 30, 35, SE);
    }
    // generating for black king 30 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(4);
        // to position 34, legal: (legal black (move king i 5 h 4))
        ddi->diagonals.emplace_back(34, 481);

        // to position 39, legal: (legal black (move king i 5 g 3))
        ddi->diagonals.emplace_back(39, 482);

        // to position 43, legal: (legal black (move king i 5 f 2))
        ddi->diagonals.emplace_back(43, 483);

        // to position 48, legal: (legal black (move king i 5 e 1))
        ddi->diagonals.emplace_back(48, 484);

        this->diagonal_data[179].push_back(ddi);


        this->reverse_legal_lookup_black[481] = ReverseLegalLookup(Role::Black, Piece::King, 30, 34, SW);
        this->reverse_legal_lookup_black[482] = ReverseLegalLookup(Role::Black, Piece::King, 30, 39, SW);
        this->reverse_legal_lookup_black[483] = ReverseLegalLookup(Role::Black, Piece::King, 30, 43, SW);
        this->reverse_legal_lookup_black[484] = ReverseLegalLookup(Role::Black, Piece::King, 30, 48, SW);
    }
    // generating for black king 31 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(6);
        // to position 27, legal: (legal black (move king b 4 c 5))
        ddi->diagonals.emplace_back(27, 489);

        // to position 22, legal: (legal black (move king b 4 d 6))
        ddi->diagonals.emplace_back(22, 490);

        // to position 18, legal: (legal black (move king b 4 e 7))
        ddi->diagonals.emplace_back(18, 491);

        // to position 13, legal: (legal black (move king b 4 f 8))
        ddi->diagonals.emplace_back(13, 492);

        // to position 9, legal: (legal black (move king b 4 g 9))
        ddi->diagonals.emplace_back(9, 493);

        // to position 4, legal: (legal black (move king b 4 h 10))
        ddi->diagonals.emplace_back(4, 494);

        this->diagonal_data[180].push_back(ddi);


        this->reverse_legal_lookup_black[489] = ReverseLegalLookup(Role::Black, Piece::King, 31, 27, NE);
        this->reverse_legal_lookup_black[490] = ReverseLegalLookup(Role::Black, Piece::King, 31, 22, NE);
        this->reverse_legal_lookup_black[491] = ReverseLegalLookup(Role::Black, Piece::King, 31, 18, NE);
        this->reverse_legal_lookup_black[492] = ReverseLegalLookup(Role::Black, Piece::King, 31, 13, NE);
        this->reverse_legal_lookup_black[493] = ReverseLegalLookup(Role::Black, Piece::King, 31, 9, NE);
        this->reverse_legal_lookup_black[494] = ReverseLegalLookup(Role::Black, Piece::King, 31, 4, NE);
    }
    // generating for black king 31 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 26, legal: (legal black (move king b 4 a 5))
        ddi->diagonals.emplace_back(26, 495);

        this->diagonal_data[180].push_back(ddi);


        this->reverse_legal_lookup_black[495] = ReverseLegalLookup(Role::Black, Piece::King, 31, 26, NW);
    }
    // generating for black king 31 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 37, legal: (legal black (move king b 4 c 3))
        ddi->diagonals.emplace_back(37, 496);

        // to position 42, legal: (legal black (move king b 4 d 2))
        ddi->diagonals.emplace_back(42, 497);

        // to position 48, legal: (legal black (move king b 4 e 1))
        ddi->diagonals.emplace_back(48, 498);

        this->diagonal_data[180].push_back(ddi);


        this->reverse_legal_lookup_black[496] = ReverseLegalLookup(Role::Black, Piece::King, 31, 37, SE);
        this->reverse_legal_lookup_black[497] = ReverseLegalLookup(Role::Black, Piece::King, 31, 42, SE);
        this->reverse_legal_lookup_black[498] = ReverseLegalLookup(Role::Black, Piece::King, 31, 48, SE);
    }
    // generating for black king 31 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 36, legal: (legal black (move king b 4 a 3))
        ddi->diagonals.emplace_back(36, 499);

        this->diagonal_data[180].push_back(ddi);


        this->reverse_legal_lookup_black[499] = ReverseLegalLookup(Role::Black, Piece::King, 31, 36, SW);
    }
    // generating for black king 32 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(6);
        // to position 28, legal: (legal black (move king d 4 e 5))
        ddi->diagonals.emplace_back(28, 506);

        // to position 23, legal: (legal black (move king d 4 f 6))
        ddi->diagonals.emplace_back(23, 507);

        // to position 19, legal: (legal black (move king d 4 g 7))
        ddi->diagonals.emplace_back(19, 508);

        // to position 14, legal: (legal black (move king d 4 h 8))
        ddi->diagonals.emplace_back(14, 509);

        // to position 10, legal: (legal black (move king d 4 i 9))
        ddi->diagonals.emplace_back(10, 510);

        // to position 5, legal: (legal black (move king d 4 j 10))
        ddi->diagonals.emplace_back(5, 511);

        this->diagonal_data[181].push_back(ddi);


        this->reverse_legal_lookup_black[506] = ReverseLegalLookup(Role::Black, Piece::King, 32, 28, NE);
        this->reverse_legal_lookup_black[507] = ReverseLegalLookup(Role::Black, Piece::King, 32, 23, NE);
        this->reverse_legal_lookup_black[508] = ReverseLegalLookup(Role::Black, Piece::King, 32, 19, NE);
        this->reverse_legal_lookup_black[509] = ReverseLegalLookup(Role::Black, Piece::King, 32, 14, NE);
        this->reverse_legal_lookup_black[510] = ReverseLegalLookup(Role::Black, Piece::King, 32, 10, NE);
        this->reverse_legal_lookup_black[511] = ReverseLegalLookup(Role::Black, Piece::King, 32, 5, NE);
    }
    // generating for black king 32 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 27, legal: (legal black (move king d 4 c 5))
        ddi->diagonals.emplace_back(27, 512);

        // to position 21, legal: (legal black (move king d 4 b 6))
        ddi->diagonals.emplace_back(21, 513);

        // to position 16, legal: (legal black (move king d 4 a 7))
        ddi->diagonals.emplace_back(16, 514);

        this->diagonal_data[181].push_back(ddi);


        this->reverse_legal_lookup_black[512] = ReverseLegalLookup(Role::Black, Piece::King, 32, 27, NW);
        this->reverse_legal_lookup_black[513] = ReverseLegalLookup(Role::Black, Piece::King, 32, 21, NW);
        this->reverse_legal_lookup_black[514] = ReverseLegalLookup(Role::Black, Piece::King, 32, 16, NW);
    }
    // generating for black king 32 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 38, legal: (legal black (move king d 4 e 3))
        ddi->diagonals.emplace_back(38, 515);

        // to position 43, legal: (legal black (move king d 4 f 2))
        ddi->diagonals.emplace_back(43, 516);

        // to position 49, legal: (legal black (move king d 4 g 1))
        ddi->diagonals.emplace_back(49, 517);

        this->diagonal_data[181].push_back(ddi);


        this->reverse_legal_lookup_black[515] = ReverseLegalLookup(Role::Black, Piece::King, 32, 38, SE);
        this->reverse_legal_lookup_black[516] = ReverseLegalLookup(Role::Black, Piece::King, 32, 43, SE);
        this->reverse_legal_lookup_black[517] = ReverseLegalLookup(Role::Black, Piece::King, 32, 49, SE);
    }
    // generating for black king 32 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 37, legal: (legal black (move king d 4 c 3))
        ddi->diagonals.emplace_back(37, 518);

        // to position 41, legal: (legal black (move king d 4 b 2))
        ddi->diagonals.emplace_back(41, 519);

        // to position 46, legal: (legal black (move king d 4 a 1))
        ddi->diagonals.emplace_back(46, 520);

        this->diagonal_data[181].push_back(ddi);


        this->reverse_legal_lookup_black[518] = ReverseLegalLookup(Role::Black, Piece::King, 32, 37, SW);
        this->reverse_legal_lookup_black[519] = ReverseLegalLookup(Role::Black, Piece::King, 32, 41, SW);
        this->reverse_legal_lookup_black[520] = ReverseLegalLookup(Role::Black, Piece::King, 32, 46, SW);
    }
    // generating for black king 33 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 29, legal: (legal black (move king f 4 g 5))
        ddi->diagonals.emplace_back(29, 527);

        // to position 24, legal: (legal black (move king f 4 h 6))
        ddi->diagonals.emplace_back(24, 528);

        // to position 20, legal: (legal black (move king f 4 i 7))
        ddi->diagonals.emplace_back(20, 529);

        // to position 15, legal: (legal black (move king f 4 j 8))
        ddi->diagonals.emplace_back(15, 530);

        this->diagonal_data[182].push_back(ddi);


        this->reverse_legal_lookup_black[527] = ReverseLegalLookup(Role::Black, Piece::King, 33, 29, NE);
        this->reverse_legal_lookup_black[528] = ReverseLegalLookup(Role::Black, Piece::King, 33, 24, NE);
        this->reverse_legal_lookup_black[529] = ReverseLegalLookup(Role::Black, Piece::King, 33, 20, NE);
        this->reverse_legal_lookup_black[530] = ReverseLegalLookup(Role::Black, Piece::King, 33, 15, NE);
    }
    // generating for black king 33 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 28, legal: (legal black (move king f 4 e 5))
        ddi->diagonals.emplace_back(28, 531);

        // to position 22, legal: (legal black (move king f 4 d 6))
        ddi->diagonals.emplace_back(22, 532);

        // to position 17, legal: (legal black (move king f 4 c 7))
        ddi->diagonals.emplace_back(17, 533);

        // to position 11, legal: (legal black (move king f 4 b 8))
        ddi->diagonals.emplace_back(11, 534);

        // to position 6, legal: (legal black (move king f 4 a 9))
        ddi->diagonals.emplace_back(6, 535);

        this->diagonal_data[182].push_back(ddi);


        this->reverse_legal_lookup_black[531] = ReverseLegalLookup(Role::Black, Piece::King, 33, 28, NW);
        this->reverse_legal_lookup_black[532] = ReverseLegalLookup(Role::Black, Piece::King, 33, 22, NW);
        this->reverse_legal_lookup_black[533] = ReverseLegalLookup(Role::Black, Piece::King, 33, 17, NW);
        this->reverse_legal_lookup_black[534] = ReverseLegalLookup(Role::Black, Piece::King, 33, 11, NW);
        this->reverse_legal_lookup_black[535] = ReverseLegalLookup(Role::Black, Piece::King, 33, 6, NW);
    }
    // generating for black king 33 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(3);
        // to position 39, legal: (legal black (move king f 4 g 3))
        ddi->diagonals.emplace_back(39, 536);

        // to position 44, legal: (legal black (move king f 4 h 2))
        ddi->diagonals.emplace_back(44, 537);

        // to position 50, legal: (legal black (move king f 4 i 1))
        ddi->diagonals.emplace_back(50, 538);

        this->diagonal_data[182].push_back(ddi);


        this->reverse_legal_lookup_black[536] = ReverseLegalLookup(Role::Black, Piece::King, 33, 39, SE);
        this->reverse_legal_lookup_black[537] = ReverseLegalLookup(Role::Black, Piece::King, 33, 44, SE);
        this->reverse_legal_lookup_black[538] = ReverseLegalLookup(Role::Black, Piece::King, 33, 50, SE);
    }
    // generating for black king 33 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 38, legal: (legal black (move king f 4 e 3))
        ddi->diagonals.emplace_back(38, 539);

        // to position 42, legal: (legal black (move king f 4 d 2))
        ddi->diagonals.emplace_back(42, 540);

        // to position 47, legal: (legal black (move king f 4 c 1))
        ddi->diagonals.emplace_back(47, 541);

        this->diagonal_data[182].push_back(ddi);


        this->reverse_legal_lookup_black[539] = ReverseLegalLookup(Role::Black, Piece::King, 33, 38, SW);
        this->reverse_legal_lookup_black[540] = ReverseLegalLookup(Role::Black, Piece::King, 33, 42, SW);
        this->reverse_legal_lookup_black[541] = ReverseLegalLookup(Role::Black, Piece::King, 33, 47, SW);
    }
    // generating for black king 34 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 30, legal: (legal black (move king h 4 i 5))
        ddi->diagonals.emplace_back(30, 548);

        // to position 25, legal: (legal black (move king h 4 j 6))
        ddi->diagonals.emplace_back(25, 549);

        this->diagonal_data[183].push_back(ddi);


        this->reverse_legal_lookup_black[548] = ReverseLegalLookup(Role::Black, Piece::King, 34, 30, NE);
        this->reverse_legal_lookup_black[549] = ReverseLegalLookup(Role::Black, Piece::King, 34, 25, NE);
    }
    // generating for black king 34 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 29, legal: (legal black (move king h 4 g 5))
        ddi->diagonals.emplace_back(29, 550);

        // to position 23, legal: (legal black (move king h 4 f 6))
        ddi->diagonals.emplace_back(23, 551);

        // to position 18, legal: (legal black (move king h 4 e 7))
        ddi->diagonals.emplace_back(18, 552);

        // to position 12, legal: (legal black (move king h 4 d 8))
        ddi->diagonals.emplace_back(12, 553);

        // to position 7, legal: (legal black (move king h 4 c 9))
        ddi->diagonals.emplace_back(7, 554);

        // to position 1, legal: (legal black (move king h 4 b 10))
        ddi->diagonals.emplace_back(1, 555);

        this->diagonal_data[183].push_back(ddi);


        this->reverse_legal_lookup_black[550] = ReverseLegalLookup(Role::Black, Piece::King, 34, 29, NW);
        this->reverse_legal_lookup_black[551] = ReverseLegalLookup(Role::Black, Piece::King, 34, 23, NW);
        this->reverse_legal_lookup_black[552] = ReverseLegalLookup(Role::Black, Piece::King, 34, 18, NW);
        this->reverse_legal_lookup_black[553] = ReverseLegalLookup(Role::Black, Piece::King, 34, 12, NW);
        this->reverse_legal_lookup_black[554] = ReverseLegalLookup(Role::Black, Piece::King, 34, 7, NW);
        this->reverse_legal_lookup_black[555] = ReverseLegalLookup(Role::Black, Piece::King, 34, 1, NW);
    }
    // generating for black king 34 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 40, legal: (legal black (move king h 4 i 3))
        ddi->diagonals.emplace_back(40, 556);

        // to position 45, legal: (legal black (move king h 4 j 2))
        ddi->diagonals.emplace_back(45, 557);

        this->diagonal_data[183].push_back(ddi);


        this->reverse_legal_lookup_black[556] = ReverseLegalLookup(Role::Black, Piece::King, 34, 40, SE);
        this->reverse_legal_lookup_black[557] = ReverseLegalLookup(Role::Black, Piece::King, 34, 45, SE);
    }
    // generating for black king 34 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 39, legal: (legal black (move king h 4 g 3))
        ddi->diagonals.emplace_back(39, 558);

        // to position 43, legal: (legal black (move king h 4 f 2))
        ddi->diagonals.emplace_back(43, 559);

        // to position 48, legal: (legal black (move king h 4 e 1))
        ddi->diagonals.emplace_back(48, 560);

        this->diagonal_data[183].push_back(ddi);


        this->reverse_legal_lookup_black[558] = ReverseLegalLookup(Role::Black, Piece::King, 34, 39, SW);
        this->reverse_legal_lookup_black[559] = ReverseLegalLookup(Role::Black, Piece::King, 34, 43, SW);
        this->reverse_legal_lookup_black[560] = ReverseLegalLookup(Role::Black, Piece::King, 34, 48, SW);
    }
    // generating for black king 35 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 30, legal: (legal black (move king j 4 i 5))
        ddi->diagonals.emplace_back(30, 564);

        // to position 24, legal: (legal black (move king j 4 h 6))
        ddi->diagonals.emplace_back(24, 565);

        // to position 19, legal: (legal black (move king j 4 g 7))
        ddi->diagonals.emplace_back(19, 566);

        // to position 13, legal: (legal black (move king j 4 f 8))
        ddi->diagonals.emplace_back(13, 567);

        // to position 8, legal: (legal black (move king j 4 e 9))
        ddi->diagonals.emplace_back(8, 568);

        // to position 2, legal: (legal black (move king j 4 d 10))
        ddi->diagonals.emplace_back(2, 569);

        this->diagonal_data[184].push_back(ddi);


        this->reverse_legal_lookup_black[564] = ReverseLegalLookup(Role::Black, Piece::King, 35, 30, NW);
        this->reverse_legal_lookup_black[565] = ReverseLegalLookup(Role::Black, Piece::King, 35, 24, NW);
        this->reverse_legal_lookup_black[566] = ReverseLegalLookup(Role::Black, Piece::King, 35, 19, NW);
        this->reverse_legal_lookup_black[567] = ReverseLegalLookup(Role::Black, Piece::King, 35, 13, NW);
        this->reverse_legal_lookup_black[568] = ReverseLegalLookup(Role::Black, Piece::King, 35, 8, NW);
        this->reverse_legal_lookup_black[569] = ReverseLegalLookup(Role::Black, Piece::King, 35, 2, NW);
    }
    // generating for black king 35 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(3);
        // to position 40, legal: (legal black (move king j 4 i 3))
        ddi->diagonals.emplace_back(40, 570);

        // to position 44, legal: (legal black (move king j 4 h 2))
        ddi->diagonals.emplace_back(44, 571);

        // to position 49, legal: (legal black (move king j 4 g 1))
        ddi->diagonals.emplace_back(49, 572);

        this->diagonal_data[184].push_back(ddi);


        this->reverse_legal_lookup_black[570] = ReverseLegalLookup(Role::Black, Piece::King, 35, 40, SW);
        this->reverse_legal_lookup_black[571] = ReverseLegalLookup(Role::Black, Piece::King, 35, 44, SW);
        this->reverse_legal_lookup_black[572] = ReverseLegalLookup(Role::Black, Piece::King, 35, 49, SW);
    }
    // generating for black king 36 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(7);
        // to position 31, legal: (legal black (move king a 3 b 4))
        ddi->diagonals.emplace_back(31, 576);

        // to position 27, legal: (legal black (move king a 3 c 5))
        ddi->diagonals.emplace_back(27, 577);

        // to position 22, legal: (legal black (move king a 3 d 6))
        ddi->diagonals.emplace_back(22, 578);

        // to position 18, legal: (legal black (move king a 3 e 7))
        ddi->diagonals.emplace_back(18, 579);

        // to position 13, legal: (legal black (move king a 3 f 8))
        ddi->diagonals.emplace_back(13, 580);

        // to position 9, legal: (legal black (move king a 3 g 9))
        ddi->diagonals.emplace_back(9, 581);

        // to position 4, legal: (legal black (move king a 3 h 10))
        ddi->diagonals.emplace_back(4, 582);

        this->diagonal_data[185].push_back(ddi);


        this->reverse_legal_lookup_black[576] = ReverseLegalLookup(Role::Black, Piece::King, 36, 31, NE);
        this->reverse_legal_lookup_black[577] = ReverseLegalLookup(Role::Black, Piece::King, 36, 27, NE);
        this->reverse_legal_lookup_black[578] = ReverseLegalLookup(Role::Black, Piece::King, 36, 22, NE);
        this->reverse_legal_lookup_black[579] = ReverseLegalLookup(Role::Black, Piece::King, 36, 18, NE);
        this->reverse_legal_lookup_black[580] = ReverseLegalLookup(Role::Black, Piece::King, 36, 13, NE);
        this->reverse_legal_lookup_black[581] = ReverseLegalLookup(Role::Black, Piece::King, 36, 9, NE);
        this->reverse_legal_lookup_black[582] = ReverseLegalLookup(Role::Black, Piece::King, 36, 4, NE);
    }
    // generating for black king 36 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 41, legal: (legal black (move king a 3 b 2))
        ddi->diagonals.emplace_back(41, 583);

        // to position 47, legal: (legal black (move king a 3 c 1))
        ddi->diagonals.emplace_back(47, 584);

        this->diagonal_data[185].push_back(ddi);


        this->reverse_legal_lookup_black[583] = ReverseLegalLookup(Role::Black, Piece::King, 36, 41, SE);
        this->reverse_legal_lookup_black[584] = ReverseLegalLookup(Role::Black, Piece::King, 36, 47, SE);
    }
    // generating for black king 37 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(7);
        // to position 32, legal: (legal black (move king c 3 d 4))
        ddi->diagonals.emplace_back(32, 591);

        // to position 28, legal: (legal black (move king c 3 e 5))
        ddi->diagonals.emplace_back(28, 592);

        // to position 23, legal: (legal black (move king c 3 f 6))
        ddi->diagonals.emplace_back(23, 593);

        // to position 19, legal: (legal black (move king c 3 g 7))
        ddi->diagonals.emplace_back(19, 594);

        // to position 14, legal: (legal black (move king c 3 h 8))
        ddi->diagonals.emplace_back(14, 595);

        // to position 10, legal: (legal black (move king c 3 i 9))
        ddi->diagonals.emplace_back(10, 596);

        // to position 5, legal: (legal black (move king c 3 j 10))
        ddi->diagonals.emplace_back(5, 597);

        this->diagonal_data[186].push_back(ddi);


        this->reverse_legal_lookup_black[591] = ReverseLegalLookup(Role::Black, Piece::King, 37, 32, NE);
        this->reverse_legal_lookup_black[592] = ReverseLegalLookup(Role::Black, Piece::King, 37, 28, NE);
        this->reverse_legal_lookup_black[593] = ReverseLegalLookup(Role::Black, Piece::King, 37, 23, NE);
        this->reverse_legal_lookup_black[594] = ReverseLegalLookup(Role::Black, Piece::King, 37, 19, NE);
        this->reverse_legal_lookup_black[595] = ReverseLegalLookup(Role::Black, Piece::King, 37, 14, NE);
        this->reverse_legal_lookup_black[596] = ReverseLegalLookup(Role::Black, Piece::King, 37, 10, NE);
        this->reverse_legal_lookup_black[597] = ReverseLegalLookup(Role::Black, Piece::King, 37, 5, NE);
    }
    // generating for black king 37 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 31, legal: (legal black (move king c 3 b 4))
        ddi->diagonals.emplace_back(31, 598);

        // to position 26, legal: (legal black (move king c 3 a 5))
        ddi->diagonals.emplace_back(26, 599);

        this->diagonal_data[186].push_back(ddi);


        this->reverse_legal_lookup_black[598] = ReverseLegalLookup(Role::Black, Piece::King, 37, 31, NW);
        this->reverse_legal_lookup_black[599] = ReverseLegalLookup(Role::Black, Piece::King, 37, 26, NW);
    }
    // generating for black king 37 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 42, legal: (legal black (move king c 3 d 2))
        ddi->diagonals.emplace_back(42, 600);

        // to position 48, legal: (legal black (move king c 3 e 1))
        ddi->diagonals.emplace_back(48, 601);

        this->diagonal_data[186].push_back(ddi);


        this->reverse_legal_lookup_black[600] = ReverseLegalLookup(Role::Black, Piece::King, 37, 42, SE);
        this->reverse_legal_lookup_black[601] = ReverseLegalLookup(Role::Black, Piece::King, 37, 48, SE);
    }
    // generating for black king 37 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 41, legal: (legal black (move king c 3 b 2))
        ddi->diagonals.emplace_back(41, 602);

        // to position 46, legal: (legal black (move king c 3 a 1))
        ddi->diagonals.emplace_back(46, 603);

        this->diagonal_data[186].push_back(ddi);


        this->reverse_legal_lookup_black[602] = ReverseLegalLookup(Role::Black, Piece::King, 37, 41, SW);
        this->reverse_legal_lookup_black[603] = ReverseLegalLookup(Role::Black, Piece::King, 37, 46, SW);
    }
    // generating for black king 38 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 33, legal: (legal black (move king e 3 f 4))
        ddi->diagonals.emplace_back(33, 610);

        // to position 29, legal: (legal black (move king e 3 g 5))
        ddi->diagonals.emplace_back(29, 611);

        // to position 24, legal: (legal black (move king e 3 h 6))
        ddi->diagonals.emplace_back(24, 612);

        // to position 20, legal: (legal black (move king e 3 i 7))
        ddi->diagonals.emplace_back(20, 613);

        // to position 15, legal: (legal black (move king e 3 j 8))
        ddi->diagonals.emplace_back(15, 614);

        this->diagonal_data[187].push_back(ddi);


        this->reverse_legal_lookup_black[610] = ReverseLegalLookup(Role::Black, Piece::King, 38, 33, NE);
        this->reverse_legal_lookup_black[611] = ReverseLegalLookup(Role::Black, Piece::King, 38, 29, NE);
        this->reverse_legal_lookup_black[612] = ReverseLegalLookup(Role::Black, Piece::King, 38, 24, NE);
        this->reverse_legal_lookup_black[613] = ReverseLegalLookup(Role::Black, Piece::King, 38, 20, NE);
        this->reverse_legal_lookup_black[614] = ReverseLegalLookup(Role::Black, Piece::King, 38, 15, NE);
    }
    // generating for black king 38 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 32, legal: (legal black (move king e 3 d 4))
        ddi->diagonals.emplace_back(32, 615);

        // to position 27, legal: (legal black (move king e 3 c 5))
        ddi->diagonals.emplace_back(27, 616);

        // to position 21, legal: (legal black (move king e 3 b 6))
        ddi->diagonals.emplace_back(21, 617);

        // to position 16, legal: (legal black (move king e 3 a 7))
        ddi->diagonals.emplace_back(16, 618);

        this->diagonal_data[187].push_back(ddi);


        this->reverse_legal_lookup_black[615] = ReverseLegalLookup(Role::Black, Piece::King, 38, 32, NW);
        this->reverse_legal_lookup_black[616] = ReverseLegalLookup(Role::Black, Piece::King, 38, 27, NW);
        this->reverse_legal_lookup_black[617] = ReverseLegalLookup(Role::Black, Piece::King, 38, 21, NW);
        this->reverse_legal_lookup_black[618] = ReverseLegalLookup(Role::Black, Piece::King, 38, 16, NW);
    }
    // generating for black king 38 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 43, legal: (legal black (move king e 3 f 2))
        ddi->diagonals.emplace_back(43, 619);

        // to position 49, legal: (legal black (move king e 3 g 1))
        ddi->diagonals.emplace_back(49, 620);

        this->diagonal_data[187].push_back(ddi);


        this->reverse_legal_lookup_black[619] = ReverseLegalLookup(Role::Black, Piece::King, 38, 43, SE);
        this->reverse_legal_lookup_black[620] = ReverseLegalLookup(Role::Black, Piece::King, 38, 49, SE);
    }
    // generating for black king 38 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 42, legal: (legal black (move king e 3 d 2))
        ddi->diagonals.emplace_back(42, 621);

        // to position 47, legal: (legal black (move king e 3 c 1))
        ddi->diagonals.emplace_back(47, 622);

        this->diagonal_data[187].push_back(ddi);


        this->reverse_legal_lookup_black[621] = ReverseLegalLookup(Role::Black, Piece::King, 38, 42, SW);
        this->reverse_legal_lookup_black[622] = ReverseLegalLookup(Role::Black, Piece::King, 38, 47, SW);
    }
    // generating for black king 39 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 34, legal: (legal black (move king g 3 h 4))
        ddi->diagonals.emplace_back(34, 629);

        // to position 30, legal: (legal black (move king g 3 i 5))
        ddi->diagonals.emplace_back(30, 630);

        // to position 25, legal: (legal black (move king g 3 j 6))
        ddi->diagonals.emplace_back(25, 631);

        this->diagonal_data[188].push_back(ddi);


        this->reverse_legal_lookup_black[629] = ReverseLegalLookup(Role::Black, Piece::King, 39, 34, NE);
        this->reverse_legal_lookup_black[630] = ReverseLegalLookup(Role::Black, Piece::King, 39, 30, NE);
        this->reverse_legal_lookup_black[631] = ReverseLegalLookup(Role::Black, Piece::King, 39, 25, NE);
    }
    // generating for black king 39 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 33, legal: (legal black (move king g 3 f 4))
        ddi->diagonals.emplace_back(33, 632);

        // to position 28, legal: (legal black (move king g 3 e 5))
        ddi->diagonals.emplace_back(28, 633);

        // to position 22, legal: (legal black (move king g 3 d 6))
        ddi->diagonals.emplace_back(22, 634);

        // to position 17, legal: (legal black (move king g 3 c 7))
        ddi->diagonals.emplace_back(17, 635);

        // to position 11, legal: (legal black (move king g 3 b 8))
        ddi->diagonals.emplace_back(11, 636);

        // to position 6, legal: (legal black (move king g 3 a 9))
        ddi->diagonals.emplace_back(6, 637);

        this->diagonal_data[188].push_back(ddi);


        this->reverse_legal_lookup_black[632] = ReverseLegalLookup(Role::Black, Piece::King, 39, 33, NW);
        this->reverse_legal_lookup_black[633] = ReverseLegalLookup(Role::Black, Piece::King, 39, 28, NW);
        this->reverse_legal_lookup_black[634] = ReverseLegalLookup(Role::Black, Piece::King, 39, 22, NW);
        this->reverse_legal_lookup_black[635] = ReverseLegalLookup(Role::Black, Piece::King, 39, 17, NW);
        this->reverse_legal_lookup_black[636] = ReverseLegalLookup(Role::Black, Piece::King, 39, 11, NW);
        this->reverse_legal_lookup_black[637] = ReverseLegalLookup(Role::Black, Piece::King, 39, 6, NW);
    }
    // generating for black king 39 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(2);
        // to position 44, legal: (legal black (move king g 3 h 2))
        ddi->diagonals.emplace_back(44, 638);

        // to position 50, legal: (legal black (move king g 3 i 1))
        ddi->diagonals.emplace_back(50, 639);

        this->diagonal_data[188].push_back(ddi);


        this->reverse_legal_lookup_black[638] = ReverseLegalLookup(Role::Black, Piece::King, 39, 44, SE);
        this->reverse_legal_lookup_black[639] = ReverseLegalLookup(Role::Black, Piece::King, 39, 50, SE);
    }
    // generating for black king 39 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 43, legal: (legal black (move king g 3 f 2))
        ddi->diagonals.emplace_back(43, 640);

        // to position 48, legal: (legal black (move king g 3 e 1))
        ddi->diagonals.emplace_back(48, 641);

        this->diagonal_data[188].push_back(ddi);


        this->reverse_legal_lookup_black[640] = ReverseLegalLookup(Role::Black, Piece::King, 39, 43, SW);
        this->reverse_legal_lookup_black[641] = ReverseLegalLookup(Role::Black, Piece::King, 39, 48, SW);
    }
    // generating for black king 40 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 35, legal: (legal black (move king i 3 j 4))
        ddi->diagonals.emplace_back(35, 646);

        this->diagonal_data[189].push_back(ddi);


        this->reverse_legal_lookup_black[646] = ReverseLegalLookup(Role::Black, Piece::King, 40, 35, NE);
    }
    // generating for black king 40 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(7);
        // to position 34, legal: (legal black (move king i 3 h 4))
        ddi->diagonals.emplace_back(34, 647);

        // to position 29, legal: (legal black (move king i 3 g 5))
        ddi->diagonals.emplace_back(29, 648);

        // to position 23, legal: (legal black (move king i 3 f 6))
        ddi->diagonals.emplace_back(23, 649);

        // to position 18, legal: (legal black (move king i 3 e 7))
        ddi->diagonals.emplace_back(18, 650);

        // to position 12, legal: (legal black (move king i 3 d 8))
        ddi->diagonals.emplace_back(12, 651);

        // to position 7, legal: (legal black (move king i 3 c 9))
        ddi->diagonals.emplace_back(7, 652);

        // to position 1, legal: (legal black (move king i 3 b 10))
        ddi->diagonals.emplace_back(1, 653);

        this->diagonal_data[189].push_back(ddi);


        this->reverse_legal_lookup_black[647] = ReverseLegalLookup(Role::Black, Piece::King, 40, 34, NW);
        this->reverse_legal_lookup_black[648] = ReverseLegalLookup(Role::Black, Piece::King, 40, 29, NW);
        this->reverse_legal_lookup_black[649] = ReverseLegalLookup(Role::Black, Piece::King, 40, 23, NW);
        this->reverse_legal_lookup_black[650] = ReverseLegalLookup(Role::Black, Piece::King, 40, 18, NW);
        this->reverse_legal_lookup_black[651] = ReverseLegalLookup(Role::Black, Piece::King, 40, 12, NW);
        this->reverse_legal_lookup_black[652] = ReverseLegalLookup(Role::Black, Piece::King, 40, 7, NW);
        this->reverse_legal_lookup_black[653] = ReverseLegalLookup(Role::Black, Piece::King, 40, 1, NW);
    }
    // generating for black king 40 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 45, legal: (legal black (move king i 3 j 2))
        ddi->diagonals.emplace_back(45, 654);

        this->diagonal_data[189].push_back(ddi);


        this->reverse_legal_lookup_black[654] = ReverseLegalLookup(Role::Black, Piece::King, 40, 45, SE);
    }
    // generating for black king 40 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(2);
        // to position 44, legal: (legal black (move king i 3 h 2))
        ddi->diagonals.emplace_back(44, 655);

        // to position 49, legal: (legal black (move king i 3 g 1))
        ddi->diagonals.emplace_back(49, 656);

        this->diagonal_data[189].push_back(ddi);


        this->reverse_legal_lookup_black[655] = ReverseLegalLookup(Role::Black, Piece::King, 40, 44, SW);
        this->reverse_legal_lookup_black[656] = ReverseLegalLookup(Role::Black, Piece::King, 40, 49, SW);
    }
    // generating for black king 41 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(8);
        // to position 37, legal: (legal black (move king b 2 c 3))
        ddi->diagonals.emplace_back(37, 660);

        // to position 32, legal: (legal black (move king b 2 d 4))
        ddi->diagonals.emplace_back(32, 661);

        // to position 28, legal: (legal black (move king b 2 e 5))
        ddi->diagonals.emplace_back(28, 662);

        // to position 23, legal: (legal black (move king b 2 f 6))
        ddi->diagonals.emplace_back(23, 663);

        // to position 19, legal: (legal black (move king b 2 g 7))
        ddi->diagonals.emplace_back(19, 664);

        // to position 14, legal: (legal black (move king b 2 h 8))
        ddi->diagonals.emplace_back(14, 665);

        // to position 10, legal: (legal black (move king b 2 i 9))
        ddi->diagonals.emplace_back(10, 666);

        // to position 5, legal: (legal black (move king b 2 j 10))
        ddi->diagonals.emplace_back(5, 667);

        this->diagonal_data[190].push_back(ddi);


        this->reverse_legal_lookup_black[660] = ReverseLegalLookup(Role::Black, Piece::King, 41, 37, NE);
        this->reverse_legal_lookup_black[661] = ReverseLegalLookup(Role::Black, Piece::King, 41, 32, NE);
        this->reverse_legal_lookup_black[662] = ReverseLegalLookup(Role::Black, Piece::King, 41, 28, NE);
        this->reverse_legal_lookup_black[663] = ReverseLegalLookup(Role::Black, Piece::King, 41, 23, NE);
        this->reverse_legal_lookup_black[664] = ReverseLegalLookup(Role::Black, Piece::King, 41, 19, NE);
        this->reverse_legal_lookup_black[665] = ReverseLegalLookup(Role::Black, Piece::King, 41, 14, NE);
        this->reverse_legal_lookup_black[666] = ReverseLegalLookup(Role::Black, Piece::King, 41, 10, NE);
        this->reverse_legal_lookup_black[667] = ReverseLegalLookup(Role::Black, Piece::King, 41, 5, NE);
    }
    // generating for black king 41 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(1);
        // to position 36, legal: (legal black (move king b 2 a 3))
        ddi->diagonals.emplace_back(36, 668);

        this->diagonal_data[190].push_back(ddi);


        this->reverse_legal_lookup_black[668] = ReverseLegalLookup(Role::Black, Piece::King, 41, 36, NW);
    }
    // generating for black king 41 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 47, legal: (legal black (move king b 2 c 1))
        ddi->diagonals.emplace_back(47, 669);

        this->diagonal_data[190].push_back(ddi);


        this->reverse_legal_lookup_black[669] = ReverseLegalLookup(Role::Black, Piece::King, 41, 47, SE);
    }
    // generating for black king 41 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 46, legal: (legal black (move king b 2 a 1))
        ddi->diagonals.emplace_back(46, 670);

        this->diagonal_data[190].push_back(ddi);


        this->reverse_legal_lookup_black[670] = ReverseLegalLookup(Role::Black, Piece::King, 41, 46, SW);
    }
    // generating for black king 42 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(6);
        // to position 38, legal: (legal black (move king d 2 e 3))
        ddi->diagonals.emplace_back(38, 675);

        // to position 33, legal: (legal black (move king d 2 f 4))
        ddi->diagonals.emplace_back(33, 676);

        // to position 29, legal: (legal black (move king d 2 g 5))
        ddi->diagonals.emplace_back(29, 677);

        // to position 24, legal: (legal black (move king d 2 h 6))
        ddi->diagonals.emplace_back(24, 678);

        // to position 20, legal: (legal black (move king d 2 i 7))
        ddi->diagonals.emplace_back(20, 679);

        // to position 15, legal: (legal black (move king d 2 j 8))
        ddi->diagonals.emplace_back(15, 680);

        this->diagonal_data[191].push_back(ddi);


        this->reverse_legal_lookup_black[675] = ReverseLegalLookup(Role::Black, Piece::King, 42, 38, NE);
        this->reverse_legal_lookup_black[676] = ReverseLegalLookup(Role::Black, Piece::King, 42, 33, NE);
        this->reverse_legal_lookup_black[677] = ReverseLegalLookup(Role::Black, Piece::King, 42, 29, NE);
        this->reverse_legal_lookup_black[678] = ReverseLegalLookup(Role::Black, Piece::King, 42, 24, NE);
        this->reverse_legal_lookup_black[679] = ReverseLegalLookup(Role::Black, Piece::King, 42, 20, NE);
        this->reverse_legal_lookup_black[680] = ReverseLegalLookup(Role::Black, Piece::King, 42, 15, NE);
    }
    // generating for black king 42 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(3);
        // to position 37, legal: (legal black (move king d 2 c 3))
        ddi->diagonals.emplace_back(37, 681);

        // to position 31, legal: (legal black (move king d 2 b 4))
        ddi->diagonals.emplace_back(31, 682);

        // to position 26, legal: (legal black (move king d 2 a 5))
        ddi->diagonals.emplace_back(26, 683);

        this->diagonal_data[191].push_back(ddi);


        this->reverse_legal_lookup_black[681] = ReverseLegalLookup(Role::Black, Piece::King, 42, 37, NW);
        this->reverse_legal_lookup_black[682] = ReverseLegalLookup(Role::Black, Piece::King, 42, 31, NW);
        this->reverse_legal_lookup_black[683] = ReverseLegalLookup(Role::Black, Piece::King, 42, 26, NW);
    }
    // generating for black king 42 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 48, legal: (legal black (move king d 2 e 1))
        ddi->diagonals.emplace_back(48, 684);

        this->diagonal_data[191].push_back(ddi);


        this->reverse_legal_lookup_black[684] = ReverseLegalLookup(Role::Black, Piece::King, 42, 48, SE);
    }
    // generating for black king 42 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 47, legal: (legal black (move king d 2 c 1))
        ddi->diagonals.emplace_back(47, 685);

        this->diagonal_data[191].push_back(ddi);


        this->reverse_legal_lookup_black[685] = ReverseLegalLookup(Role::Black, Piece::King, 42, 47, SW);
    }
    // generating for black king 43 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(4);
        // to position 39, legal: (legal black (move king f 2 g 3))
        ddi->diagonals.emplace_back(39, 690);

        // to position 34, legal: (legal black (move king f 2 h 4))
        ddi->diagonals.emplace_back(34, 691);

        // to position 30, legal: (legal black (move king f 2 i 5))
        ddi->diagonals.emplace_back(30, 692);

        // to position 25, legal: (legal black (move king f 2 j 6))
        ddi->diagonals.emplace_back(25, 693);

        this->diagonal_data[192].push_back(ddi);


        this->reverse_legal_lookup_black[690] = ReverseLegalLookup(Role::Black, Piece::King, 43, 39, NE);
        this->reverse_legal_lookup_black[691] = ReverseLegalLookup(Role::Black, Piece::King, 43, 34, NE);
        this->reverse_legal_lookup_black[692] = ReverseLegalLookup(Role::Black, Piece::King, 43, 30, NE);
        this->reverse_legal_lookup_black[693] = ReverseLegalLookup(Role::Black, Piece::King, 43, 25, NE);
    }
    // generating for black king 43 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(5);
        // to position 38, legal: (legal black (move king f 2 e 3))
        ddi->diagonals.emplace_back(38, 694);

        // to position 32, legal: (legal black (move king f 2 d 4))
        ddi->diagonals.emplace_back(32, 695);

        // to position 27, legal: (legal black (move king f 2 c 5))
        ddi->diagonals.emplace_back(27, 696);

        // to position 21, legal: (legal black (move king f 2 b 6))
        ddi->diagonals.emplace_back(21, 697);

        // to position 16, legal: (legal black (move king f 2 a 7))
        ddi->diagonals.emplace_back(16, 698);

        this->diagonal_data[192].push_back(ddi);


        this->reverse_legal_lookup_black[694] = ReverseLegalLookup(Role::Black, Piece::King, 43, 38, NW);
        this->reverse_legal_lookup_black[695] = ReverseLegalLookup(Role::Black, Piece::King, 43, 32, NW);
        this->reverse_legal_lookup_black[696] = ReverseLegalLookup(Role::Black, Piece::King, 43, 27, NW);
        this->reverse_legal_lookup_black[697] = ReverseLegalLookup(Role::Black, Piece::King, 43, 21, NW);
        this->reverse_legal_lookup_black[698] = ReverseLegalLookup(Role::Black, Piece::King, 43, 16, NW);
    }
    // generating for black king 43 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 49, legal: (legal black (move king f 2 g 1))
        ddi->diagonals.emplace_back(49, 699);

        this->diagonal_data[192].push_back(ddi);


        this->reverse_legal_lookup_black[699] = ReverseLegalLookup(Role::Black, Piece::King, 43, 49, SE);
    }
    // generating for black king 43 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 48, legal: (legal black (move king f 2 e 1))
        ddi->diagonals.emplace_back(48, 700);

        this->diagonal_data[192].push_back(ddi);


        this->reverse_legal_lookup_black[700] = ReverseLegalLookup(Role::Black, Piece::King, 43, 48, SW);
    }
    // generating for black king 44 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(2);
        // to position 40, legal: (legal black (move king h 2 i 3))
        ddi->diagonals.emplace_back(40, 705);

        // to position 35, legal: (legal black (move king h 2 j 4))
        ddi->diagonals.emplace_back(35, 706);

        this->diagonal_data[193].push_back(ddi);


        this->reverse_legal_lookup_black[705] = ReverseLegalLookup(Role::Black, Piece::King, 44, 40, NE);
        this->reverse_legal_lookup_black[706] = ReverseLegalLookup(Role::Black, Piece::King, 44, 35, NE);
    }
    // generating for black king 44 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(7);
        // to position 39, legal: (legal black (move king h 2 g 3))
        ddi->diagonals.emplace_back(39, 707);

        // to position 33, legal: (legal black (move king h 2 f 4))
        ddi->diagonals.emplace_back(33, 708);

        // to position 28, legal: (legal black (move king h 2 e 5))
        ddi->diagonals.emplace_back(28, 709);

        // to position 22, legal: (legal black (move king h 2 d 6))
        ddi->diagonals.emplace_back(22, 710);

        // to position 17, legal: (legal black (move king h 2 c 7))
        ddi->diagonals.emplace_back(17, 711);

        // to position 11, legal: (legal black (move king h 2 b 8))
        ddi->diagonals.emplace_back(11, 712);

        // to position 6, legal: (legal black (move king h 2 a 9))
        ddi->diagonals.emplace_back(6, 713);

        this->diagonal_data[193].push_back(ddi);


        this->reverse_legal_lookup_black[707] = ReverseLegalLookup(Role::Black, Piece::King, 44, 39, NW);
        this->reverse_legal_lookup_black[708] = ReverseLegalLookup(Role::Black, Piece::King, 44, 33, NW);
        this->reverse_legal_lookup_black[709] = ReverseLegalLookup(Role::Black, Piece::King, 44, 28, NW);
        this->reverse_legal_lookup_black[710] = ReverseLegalLookup(Role::Black, Piece::King, 44, 22, NW);
        this->reverse_legal_lookup_black[711] = ReverseLegalLookup(Role::Black, Piece::King, 44, 17, NW);
        this->reverse_legal_lookup_black[712] = ReverseLegalLookup(Role::Black, Piece::King, 44, 11, NW);
        this->reverse_legal_lookup_black[713] = ReverseLegalLookup(Role::Black, Piece::King, 44, 6, NW);
    }
    // generating for black king 44 SE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SE;
        ddi->diagonals.reserve(1);
        // to position 50, legal: (legal black (move king h 2 i 1))
        ddi->diagonals.emplace_back(50, 714);

        this->diagonal_data[193].push_back(ddi);


        this->reverse_legal_lookup_black[714] = ReverseLegalLookup(Role::Black, Piece::King, 44, 50, SE);
    }
    // generating for black king 44 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 49, legal: (legal black (move king h 2 g 1))
        ddi->diagonals.emplace_back(49, 715);

        this->diagonal_data[193].push_back(ddi);


        this->reverse_legal_lookup_black[715] = ReverseLegalLookup(Role::Black, Piece::King, 44, 49, SW);
    }
    // generating for black king 45 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(8);
        // to position 40, legal: (legal black (move king j 2 i 3))
        ddi->diagonals.emplace_back(40, 718);

        // to position 34, legal: (legal black (move king j 2 h 4))
        ddi->diagonals.emplace_back(34, 719);

        // to position 29, legal: (legal black (move king j 2 g 5))
        ddi->diagonals.emplace_back(29, 720);

        // to position 23, legal: (legal black (move king j 2 f 6))
        ddi->diagonals.emplace_back(23, 721);

        // to position 18, legal: (legal black (move king j 2 e 7))
        ddi->diagonals.emplace_back(18, 722);

        // to position 12, legal: (legal black (move king j 2 d 8))
        ddi->diagonals.emplace_back(12, 723);

        // to position 7, legal: (legal black (move king j 2 c 9))
        ddi->diagonals.emplace_back(7, 724);

        // to position 1, legal: (legal black (move king j 2 b 10))
        ddi->diagonals.emplace_back(1, 725);

        this->diagonal_data[194].push_back(ddi);


        this->reverse_legal_lookup_black[718] = ReverseLegalLookup(Role::Black, Piece::King, 45, 40, NW);
        this->reverse_legal_lookup_black[719] = ReverseLegalLookup(Role::Black, Piece::King, 45, 34, NW);
        this->reverse_legal_lookup_black[720] = ReverseLegalLookup(Role::Black, Piece::King, 45, 29, NW);
        this->reverse_legal_lookup_black[721] = ReverseLegalLookup(Role::Black, Piece::King, 45, 23, NW);
        this->reverse_legal_lookup_black[722] = ReverseLegalLookup(Role::Black, Piece::King, 45, 18, NW);
        this->reverse_legal_lookup_black[723] = ReverseLegalLookup(Role::Black, Piece::King, 45, 12, NW);
        this->reverse_legal_lookup_black[724] = ReverseLegalLookup(Role::Black, Piece::King, 45, 7, NW);
        this->reverse_legal_lookup_black[725] = ReverseLegalLookup(Role::Black, Piece::King, 45, 1, NW);
    }
    // generating for black king 45 SW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = SW;
        ddi->diagonals.reserve(1);
        // to position 50, legal: (legal black (move king j 2 i 1))
        ddi->diagonals.emplace_back(50, 726);

        this->diagonal_data[194].push_back(ddi);


        this->reverse_legal_lookup_black[726] = ReverseLegalLookup(Role::Black, Piece::King, 45, 50, SW);
    }
    // generating for black king 46 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(9);
        // to position 41, legal: (legal black (move king a 1 b 2))
        ddi->diagonals.emplace_back(41, 728);

        // to position 37, legal: (legal black (move king a 1 c 3))
        ddi->diagonals.emplace_back(37, 729);

        // to position 32, legal: (legal black (move king a 1 d 4))
        ddi->diagonals.emplace_back(32, 730);

        // to position 28, legal: (legal black (move king a 1 e 5))
        ddi->diagonals.emplace_back(28, 731);

        // to position 23, legal: (legal black (move king a 1 f 6))
        ddi->diagonals.emplace_back(23, 732);

        // to position 19, legal: (legal black (move king a 1 g 7))
        ddi->diagonals.emplace_back(19, 733);

        // to position 14, legal: (legal black (move king a 1 h 8))
        ddi->diagonals.emplace_back(14, 734);

        // to position 10, legal: (legal black (move king a 1 i 9))
        ddi->diagonals.emplace_back(10, 735);

        // to position 5, legal: (legal black (move king a 1 j 10))
        ddi->diagonals.emplace_back(5, 736);

        this->diagonal_data[195].push_back(ddi);


        this->reverse_legal_lookup_black[728] = ReverseLegalLookup(Role::Black, Piece::King, 46, 41, NE);
        this->reverse_legal_lookup_black[729] = ReverseLegalLookup(Role::Black, Piece::King, 46, 37, NE);
        this->reverse_legal_lookup_black[730] = ReverseLegalLookup(Role::Black, Piece::King, 46, 32, NE);
        this->reverse_legal_lookup_black[731] = ReverseLegalLookup(Role::Black, Piece::King, 46, 28, NE);
        this->reverse_legal_lookup_black[732] = ReverseLegalLookup(Role::Black, Piece::King, 46, 23, NE);
        this->reverse_legal_lookup_black[733] = ReverseLegalLookup(Role::Black, Piece::King, 46, 19, NE);
        this->reverse_legal_lookup_black[734] = ReverseLegalLookup(Role::Black, Piece::King, 46, 14, NE);
        this->reverse_legal_lookup_black[735] = ReverseLegalLookup(Role::Black, Piece::King, 46, 10, NE);
        this->reverse_legal_lookup_black[736] = ReverseLegalLookup(Role::Black, Piece::King, 46, 5, NE);
    }
    // generating for black king 47 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(7);
        // to position 42, legal: (legal black (move king c 1 d 2))
        ddi->diagonals.emplace_back(42, 739);

        // to position 38, legal: (legal black (move king c 1 e 3))
        ddi->diagonals.emplace_back(38, 740);

        // to position 33, legal: (legal black (move king c 1 f 4))
        ddi->diagonals.emplace_back(33, 741);

        // to position 29, legal: (legal black (move king c 1 g 5))
        ddi->diagonals.emplace_back(29, 742);

        // to position 24, legal: (legal black (move king c 1 h 6))
        ddi->diagonals.emplace_back(24, 743);

        // to position 20, legal: (legal black (move king c 1 i 7))
        ddi->diagonals.emplace_back(20, 744);

        // to position 15, legal: (legal black (move king c 1 j 8))
        ddi->diagonals.emplace_back(15, 745);

        this->diagonal_data[196].push_back(ddi);


        this->reverse_legal_lookup_black[739] = ReverseLegalLookup(Role::Black, Piece::King, 47, 42, NE);
        this->reverse_legal_lookup_black[740] = ReverseLegalLookup(Role::Black, Piece::King, 47, 38, NE);
        this->reverse_legal_lookup_black[741] = ReverseLegalLookup(Role::Black, Piece::King, 47, 33, NE);
        this->reverse_legal_lookup_black[742] = ReverseLegalLookup(Role::Black, Piece::King, 47, 29, NE);
        this->reverse_legal_lookup_black[743] = ReverseLegalLookup(Role::Black, Piece::King, 47, 24, NE);
        this->reverse_legal_lookup_black[744] = ReverseLegalLookup(Role::Black, Piece::King, 47, 20, NE);
        this->reverse_legal_lookup_black[745] = ReverseLegalLookup(Role::Black, Piece::King, 47, 15, NE);
    }
    // generating for black king 47 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(2);
        // to position 41, legal: (legal black (move king c 1 b 2))
        ddi->diagonals.emplace_back(41, 746);

        // to position 36, legal: (legal black (move king c 1 a 3))
        ddi->diagonals.emplace_back(36, 747);

        this->diagonal_data[196].push_back(ddi);


        this->reverse_legal_lookup_black[746] = ReverseLegalLookup(Role::Black, Piece::King, 47, 41, NW);
        this->reverse_legal_lookup_black[747] = ReverseLegalLookup(Role::Black, Piece::King, 47, 36, NW);
    }
    // generating for black king 48 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(5);
        // to position 43, legal: (legal black (move king e 1 f 2))
        ddi->diagonals.emplace_back(43, 750);

        // to position 39, legal: (legal black (move king e 1 g 3))
        ddi->diagonals.emplace_back(39, 751);

        // to position 34, legal: (legal black (move king e 1 h 4))
        ddi->diagonals.emplace_back(34, 752);

        // to position 30, legal: (legal black (move king e 1 i 5))
        ddi->diagonals.emplace_back(30, 753);

        // to position 25, legal: (legal black (move king e 1 j 6))
        ddi->diagonals.emplace_back(25, 754);

        this->diagonal_data[197].push_back(ddi);


        this->reverse_legal_lookup_black[750] = ReverseLegalLookup(Role::Black, Piece::King, 48, 43, NE);
        this->reverse_legal_lookup_black[751] = ReverseLegalLookup(Role::Black, Piece::King, 48, 39, NE);
        this->reverse_legal_lookup_black[752] = ReverseLegalLookup(Role::Black, Piece::King, 48, 34, NE);
        this->reverse_legal_lookup_black[753] = ReverseLegalLookup(Role::Black, Piece::King, 48, 30, NE);
        this->reverse_legal_lookup_black[754] = ReverseLegalLookup(Role::Black, Piece::King, 48, 25, NE);
    }
    // generating for black king 48 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(4);
        // to position 42, legal: (legal black (move king e 1 d 2))
        ddi->diagonals.emplace_back(42, 755);

        // to position 37, legal: (legal black (move king e 1 c 3))
        ddi->diagonals.emplace_back(37, 756);

        // to position 31, legal: (legal black (move king e 1 b 4))
        ddi->diagonals.emplace_back(31, 757);

        // to position 26, legal: (legal black (move king e 1 a 5))
        ddi->diagonals.emplace_back(26, 758);

        this->diagonal_data[197].push_back(ddi);


        this->reverse_legal_lookup_black[755] = ReverseLegalLookup(Role::Black, Piece::King, 48, 42, NW);
        this->reverse_legal_lookup_black[756] = ReverseLegalLookup(Role::Black, Piece::King, 48, 37, NW);
        this->reverse_legal_lookup_black[757] = ReverseLegalLookup(Role::Black, Piece::King, 48, 31, NW);
        this->reverse_legal_lookup_black[758] = ReverseLegalLookup(Role::Black, Piece::King, 48, 26, NW);
    }
    // generating for black king 49 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(3);
        // to position 44, legal: (legal black (move king g 1 h 2))
        ddi->diagonals.emplace_back(44, 761);

        // to position 40, legal: (legal black (move king g 1 i 3))
        ddi->diagonals.emplace_back(40, 762);

        // to position 35, legal: (legal black (move king g 1 j 4))
        ddi->diagonals.emplace_back(35, 763);

        this->diagonal_data[198].push_back(ddi);


        this->reverse_legal_lookup_black[761] = ReverseLegalLookup(Role::Black, Piece::King, 49, 44, NE);
        this->reverse_legal_lookup_black[762] = ReverseLegalLookup(Role::Black, Piece::King, 49, 40, NE);
        this->reverse_legal_lookup_black[763] = ReverseLegalLookup(Role::Black, Piece::King, 49, 35, NE);
    }
    // generating for black king 49 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(6);
        // to position 43, legal: (legal black (move king g 1 f 2))
        ddi->diagonals.emplace_back(43, 764);

        // to position 38, legal: (legal black (move king g 1 e 3))
        ddi->diagonals.emplace_back(38, 765);

        // to position 32, legal: (legal black (move king g 1 d 4))
        ddi->diagonals.emplace_back(32, 766);

        // to position 27, legal: (legal black (move king g 1 c 5))
        ddi->diagonals.emplace_back(27, 767);

        // to position 21, legal: (legal black (move king g 1 b 6))
        ddi->diagonals.emplace_back(21, 768);

        // to position 16, legal: (legal black (move king g 1 a 7))
        ddi->diagonals.emplace_back(16, 769);

        this->diagonal_data[198].push_back(ddi);


        this->reverse_legal_lookup_black[764] = ReverseLegalLookup(Role::Black, Piece::King, 49, 43, NW);
        this->reverse_legal_lookup_black[765] = ReverseLegalLookup(Role::Black, Piece::King, 49, 38, NW);
        this->reverse_legal_lookup_black[766] = ReverseLegalLookup(Role::Black, Piece::King, 49, 32, NW);
        this->reverse_legal_lookup_black[767] = ReverseLegalLookup(Role::Black, Piece::King, 49, 27, NW);
        this->reverse_legal_lookup_black[768] = ReverseLegalLookup(Role::Black, Piece::King, 49, 21, NW);
        this->reverse_legal_lookup_black[769] = ReverseLegalLookup(Role::Black, Piece::King, 49, 16, NW);
    }
    // generating for black king 50 NE
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NE;
        ddi->diagonals.reserve(1);
        // to position 45, legal: (legal black (move king i 1 j 2))
        ddi->diagonals.emplace_back(45, 771);

        this->diagonal_data[199].push_back(ddi);


        this->reverse_legal_lookup_black[771] = ReverseLegalLookup(Role::Black, Piece::King, 50, 45, NE);
    }
    // generating for black king 50 NW
    {
        DiagonalDirectionInfo* ddi = new DiagonalDirectionInfo;
        ddi->direction = NW;
        ddi->diagonals.reserve(8);
        // to position 44, legal: (legal black (move king i 1 h 2))
        ddi->diagonals.emplace_back(44, 772);

        // to position 39, legal: (legal black (move king i 1 g 3))
        ddi->diagonals.emplace_back(39, 773);

        // to position 33, legal: (legal black (move king i 1 f 4))
        ddi->diagonals.emplace_back(33, 774);

        // to position 28, legal: (legal black (move king i 1 e 5))
        ddi->diagonals.emplace_back(28, 775);

        // to position 22, legal: (legal black (move king i 1 d 6))
        ddi->diagonals.emplace_back(22, 776);

        // to position 17, legal: (legal black (move king i 1 c 7))
        ddi->diagonals.emplace_back(17, 777);

        // to position 11, legal: (legal black (move king i 1 b 8))
        ddi->diagonals.emplace_back(11, 778);

        // to position 6, legal: (legal black (move king i 1 a 9))
        ddi->diagonals.emplace_back(6, 779);

        this->diagonal_data[199].push_back(ddi);


        this->reverse_legal_lookup_black[772] = ReverseLegalLookup(Role::Black, Piece::King, 50, 44, NW);
        this->reverse_legal_lookup_black[773] = ReverseLegalLookup(Role::Black, Piece::King, 50, 39, NW);
        this->reverse_legal_lookup_black[774] = ReverseLegalLookup(Role::Black, Piece::King, 50, 33, NW);
        this->reverse_legal_lookup_black[775] = ReverseLegalLookup(Role::Black, Piece::King, 50, 28, NW);
        this->reverse_legal_lookup_black[776] = ReverseLegalLookup(Role::Black, Piece::King, 50, 22, NW);
        this->reverse_legal_lookup_black[777] = ReverseLegalLookup(Role::Black, Piece::King, 50, 17, NW);
        this->reverse_legal_lookup_black[778] = ReverseLegalLookup(Role::Black, Piece::King, 50, 11, NW);
        this->reverse_legal_lookup_black[779] = ReverseLegalLookup(Role::Black, Piece::King, 50, 6, NW);
    }


} // end of BoardDescription::initBoard_10x10
// end of file

