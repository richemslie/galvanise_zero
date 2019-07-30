import random
from pprint import pprint
from functools import partial

from ggplib.db import lookup
from ggpzero.util import symmetry as sym
from ggpzero.nn.manager import get_manager


def test_reflect_vertical_odd():
    cords_x = '12345'
    cords_y = 'abcde'

    assert sym.reflect_vertical('1', 'b', cords_x, cords_y) == ('5', 'b')
    assert sym.reflect_vertical('2', 'c', cords_x, cords_y) == ('4', 'c')
    assert sym.reflect_vertical('3', 'a', cords_x, cords_y) == ('3', 'a')


def test_reflect_vertical_even():
    cords_x = '123456'
    cords_y = 'abcdef'

    assert sym.reflect_vertical('1', 'b', cords_x, cords_y) == ('6', 'b')
    assert sym.reflect_vertical('2', 'c', cords_x, cords_y) == ('5', 'c')
    assert sym.reflect_vertical('3', 'a', cords_x, cords_y) == ('4', 'a')
    assert sym.reflect_vertical('4', 'f', cords_x, cords_y) == ('3', 'f')


def test_rotate_90_odd():
    '''
    e
    d
    c
    b X
    a
      1 2 3 4 5
    '''

    cords_x = '12345'
    cords_y = 'abcde'

    assert sym.rotate_90('1', 'b', cords_x, cords_y) == ('2', 'e')
    assert sym.rotate_90('2', 'e', cords_x, cords_y) == ('5', 'd')
    assert sym.rotate_90('5', 'd', cords_x, cords_y) == ('4', 'a')
    assert sym.rotate_90('4', 'a', cords_x, cords_y) == ('1', 'b')


def test_rotate_90_even():
    '''
    f
    e
    d
    c
    b X
    a
      1 2 3 4 5 6
    '''

    cords_x = '123456'
    cords_y = 'abcdef'

    assert sym.rotate_90('1', 'b', cords_x, cords_y) == ('2', 'f')
    assert sym.rotate_90('2', 'f', cords_x, cords_y) == ('6', 'e')
    assert sym.rotate_90('6', 'e', cords_x, cords_y) == ('5', 'a')
    assert sym.rotate_90('5', 'a', cords_x, cords_y) == ('1', 'b')


def test_translator_reversi1():
    info = lookup.by_name("reversi")
    sm = info.get_sm()
    b = sm.get_initial_state().to_list()

    t = sym.Translator(info, '12345678', '12345678')
    t.add_basetype('cell', 1, 2)
    pprint(t.base_root_term_indexes)
    pprint(t.base_root_term_to_mapping)

    t.add_skip_base('control')

    new_basestate = t.translate_basestate(b, True, 0)
    print info.model.basestate_to_str(b)
    print info.model.basestate_to_str(new_basestate)

    t.add_skip_action('noop')

    t.add_action_type('move', 1, 2)
    print "action_root_term_indexes:"
    pprint(t.action_root_term_indexes)

    assert t.translate_action(0, 0, False, 1) == 0
    assert t.translate_action(1, 1, False, 1) == 8


def advance_state(sm, basestate):
    sm.update_bases(basestate)

    # leaks, but who cares, it is a test
    joint_move = sm.get_joint_move()
    base_state = sm.new_base_state()

    for role_index in range(len(sm.get_roles())):
        ls = sm.get_legal_state(role_index)
        choice = ls.get_legal(random.randrange(0, ls.get_count()))
        joint_move.set(role_index, choice)

    # play move, the base_state will be new state
    sm.next_state(joint_move, base_state)
    return base_state


def all_moves(sm, basestate):
    sm.update_bases(basestate)

    def f(ri, ls, i):
        return sm.legal_to_move(ri, ls.get_legal(i))

    result = []
    for ri, role in enumerate(sm.get_roles()):
        ls = sm.get_legal_state(ri)
        moves = [f(ri, ls, ii) for ii in range(ls.get_count())]
        result.append((role, set(moves)))

    return result


def translate_moves(sm, basestate, translator, do_reflection, rot_count):
    sm.update_bases(basestate)
    moves = []
    for ri, role in enumerate(sm.get_roles()):
        ls = sm.get_legal_state(ri)
        translated_moves = []
        for indx in range(ls.get_count()):
            legal = translator.translate_action(ri, ls.get_legal(indx), do_reflection, rot_count)

            translated_moves.append(sm.legal_to_move(ri, legal))

        moves.append((role, set(translated_moves)))

    return moves


def test_translator_reversi2():
    from ggpzero.battle.reversi import pretty_board

    info = lookup.by_name("reversi")

    t = sym.Translator(info, '12345678', '12345678')
    t.add_basetype('cell', 1, 2)
    t.add_action_type('move', 1, 2)
    t.add_skip_base('control')
    t.add_skip_action('noop')

    sm = info.get_sm()
    sm.reset()

    pretty_board(8, sm)
    basestate = sm.get_initial_state()
    for i in range(30):
        basestate = advance_state(sm, basestate)
        sm.update_bases(basestate)

    # print board & moves
    pretty_board(8, sm)
    for role, moves in all_moves(sm, basestate):
        print role, moves

    basestate_list = t.translate_basestate(basestate.to_list(), True, 0)
    basestate.from_list(basestate_list)
    sm.update_bases(basestate)

    # print board & moves
    pretty_board(8, sm)
    for role, moves in all_moves(sm, basestate):
        print role, moves

    for i in range(4):
        basestate_list = t.translate_basestate(basestate.to_list(), 0, 1)
        basestate.from_list(basestate_list)
        sm.update_bases(basestate)

        # print board & moves
        pretty_board(8, sm)
        for role, moves in all_moves(sm, basestate):
            print role, moves


def game_test(game, pretty_board, advance_state_count):
    # game stuff
    info = lookup.by_name(game)
    transformer = get_manager().get_transformer(game)

    # the translator
    t = sym.create_translator(info, transformer.game_desc, transformer.get_symmetries_desc())

    # start with a statemachine - and advance 5 moves
    sm = info.get_sm()
    sm.reset()

    basestate = sm.get_initial_state()
    for i in range(advance_state_count):
        basestate = advance_state(sm, basestate)
        sm.update_bases(basestate)

    # print board & moves
    print "original board:"
    pretty_board(sm)

    prescription = sym.Prescription(transformer.get_symmetries_desc())
    translated_basestate = sm.new_base_state()

    # do all reflections / rotations in prescription
    for do_reflection, rot_count in prescription:
        print "reflection", do_reflection, "rotations", rot_count

        # translate state/moves
        basestate_list = t.translate_basestate(basestate.to_list(), do_reflection, rot_count)
        basestate2_list = t.translate_basestate_faster(basestate.to_list(), do_reflection, rot_count)

        assert basestate_list == basestate2_list

        translated_moves = translate_moves(sm, basestate, t, do_reflection, rot_count)

        translated_basestate.from_list(basestate_list)
        assert all_moves(sm, translated_basestate) == translated_moves
        sm.update_bases(translated_basestate)

        # print board & moves
        pretty_board(sm)
        for role, moves in all_moves(sm, translated_basestate):
            print role, moves


def test_game_reversi():
    from ggpzero.battle.reversi import pretty_board
    game_test("reversi", partial(pretty_board, 8), 5)
    game_test("reversi", partial(pretty_board, 8), 15)


def test_game_bts():
    from ggpzero.battle.bt import pretty_board
    game_test("breakthroughSmall", partial(pretty_board, 6), 2)
    game_test("breakthroughSmall", partial(pretty_board, 6), 8)


def test_game_c6():
    from ggpzero.battle.connect6 import MatchInfo
    match_info = MatchInfo()
    game_test("connect6", match_info.print_board, 3)
    game_test("connect6", match_info.print_board, 10)


def test_game_hex11():
    from ggpzero.battle.hex import MatchInfo
    match_info = MatchInfo(11)

    game_test("hexLG11", match_info.print_board, 3)
    game_test("hexLG11", match_info.print_board, 10)
    game_test("hexLG11", match_info.print_board, 16)


def test_game_hex13():
    from ggpzero.battle.hex import MatchInfo
    match_info = MatchInfo(13)

    game_test("hexLG13", match_info.print_board, 3)
    game_test("hexLG13", match_info.print_board, 10)
    game_test("hexLG13", match_info.print_board, 16)


def test_game_amazons():
    from ggpzero.battle.amazons import MatchInfo
    match_info = MatchInfo()

    game_test("amazons_10x10", match_info.print_board, 3)
    game_test("amazons_10x10", match_info.print_board, 10)
    game_test("amazons_10x10", match_info.print_board, 16)

