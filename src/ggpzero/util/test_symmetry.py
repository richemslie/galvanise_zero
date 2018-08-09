from ggplib.db import lookup
from ggpzero.util import symmetry as sym


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


def test_translator():
    info = lookup.by_name("reversi")
    sm = info.get_sm()
    b = sm.get_initial_state().to_list()

    t = sym.Translater(info, '12345678', '12345678')
    t.add_basetype('cell', 1, 2)
    t.add_skip_base('control')

    new_basestate = t.translate_basestate(b, True, 0)
    print info.model.basestate_to_str(b)
    print info.model.basestate_to_str(new_basestate)

    print t.action_list
    t.add_skip_action('noop')
    t.add_action_type('move', 1, 2)

    assert t.translate_action(0, 0, False, 1) == 0
    assert t.translate_action(1, 1, False, 1) == 8
