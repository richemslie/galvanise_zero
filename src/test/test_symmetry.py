import random
from pprint import pprint

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


def dump_moves(sm, basestate):
    sm.update_bases(basestate)

    def f(ri, ls, i):
        return sm.legal_to_move(ri, ls.get_legal(i))

    for ri, role in enumerate(sm.get_roles()):
        print "role %s:" % role
        ls = sm.get_legal_state(ri)
        moves = [f(ri, ls, ii) for ii in range(ls.get_count())]
        pprint(moves)


def test_translator_reversi2():
    ''' 1. print board
        2. randomly advance board
        3. print board with various translations
    '''
    from gzero_games.battle.reversi import pretty_board

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
    dump_moves(sm, basestate)

    basestate_list = t.translate_basestate(basestate.to_list(), True, 0)
    basestate.from_list(basestate_list)
    sm.update_bases(basestate)

    # print board & moves
    pretty_board(8, sm)
    dump_moves(sm, basestate)

    for i in range(4):
        basestate_list = t.translate_basestate(basestate.to_list(), 0, 1)
        basestate.from_list(basestate_list)
        sm.update_bases(basestate)

        # print board & moves
        pretty_board(8, sm)
        dump_moves(sm, basestate)


def test_translator_reversi3():
    ''' 1. print board
        2. randomly advance board
        3. print board with various translations
    '''
    from gzero_games.battle.reversi import pretty_board

    info = lookup.by_name("reversi")
    transformer = get_manager().get_transformer("reversi")

    t = sym.create_translator(info, transformer.game_desc, transformer.get_symmetries_desc())

    sm = info.get_sm()
    sm.reset()

    pretty_board(8, sm)
    basestate = sm.get_initial_state()
    for i in range(30):
        basestate = advance_state(sm, basestate)
        sm.update_bases(basestate)

    # print board & moves
    pretty_board(8, sm)
    dump_moves(sm, basestate)

    prescription = sym.Prescription(transformer.get_symmetries_desc())
    translated_basestate = sm.new_base_state()

    for do_reflection, rot_count in prescription:
        print do_reflection, rot_count

        basestate_list = t.translate_basestate(basestate.to_list(), do_reflection, rot_count)
        translated_basestate.from_list(basestate_list)
        sm.update_bases(translated_basestate)

        # print board & moves
        pretty_board(8, sm)
        dump_moves(sm, translated_basestate)


def test_translator_bt():
    ''' 1. print board
        2. randomly advance board
        3. print board with various translations
    '''
    from gzero_games.battle.bt import pretty_board

    info = lookup.by_name("breakthroughSmall")
    transformer = get_manager().get_transformer("breakthroughSmall")

    t = sym.create_translator(info, transformer.game_desc, transformer.get_symmetries_desc())

    sm = info.get_sm()
    sm.reset()

    pretty_board(6, sm)
    basestate = sm.get_initial_state()
    for i in range(14):
        basestate = advance_state(sm, basestate)
        sm.update_bases(basestate)

    # print board & moves
    pretty_board(6, sm)
    dump_moves(sm, basestate)

    prescription = sym.Prescription(transformer.get_symmetries_desc())
    translated_basestate = sm.new_base_state()

    for do_reflection, rot_count in prescription:
        print do_reflection, rot_count

        basestate_list = t.translate_basestate(basestate.to_list(), do_reflection, rot_count)
        translated_basestate.from_list(basestate_list)
        sm.update_bases(translated_basestate)

        # print board & moves
        pretty_board(6, sm)
        dump_moves(sm, translated_basestate)


def test_translator_c6():
    ''' 1. print board
        2. randomly advance board
        3. print board with various translations
    '''
    # for printing
    from gzero_games.battle.connect6 import MatchInfo
    match_info = MatchInfo()
    pretty_board = match_info.print_board

    info = lookup.by_name("connect6")
    transformer = get_manager().get_transformer("connect6")

    t = sym.create_translator(info, transformer.game_desc, transformer.get_symmetries_desc())

    sm = info.get_sm()
    sm.reset()

    pretty_board(sm)
    basestate = sm.get_initial_state()
    for i in range(14):
        basestate = advance_state(sm, basestate)
        sm.update_bases(basestate)

    # print board & moves
    pretty_board(sm)

    prescription = sym.Prescription(transformer.get_symmetries_desc())
    translated_basestate = sm.new_base_state()

    for do_reflection, rot_count in prescription:
        print do_reflection, rot_count

        basestate_list = t.translate_basestate(basestate.to_list(), do_reflection, rot_count)
        translated_basestate.from_list(basestate_list)
        sm.update_bases(translated_basestate)

        # print board & moves
        pretty_board(sm)


