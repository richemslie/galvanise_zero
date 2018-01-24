import attr

from ggpzero.util.attrutil import register_attrs


@register_attrs
class ControlTerm(object):
    # pieces
    base_term = attr.ib(3)

    # list of extra terms - must match absolutely
    extra_terms = attr.ib(default=attr.Factory(list))

    # we set the channel to this value
    value = attr.ib(1)


@register_attrs
class PieceTerm(object):
    piece_term_idx = attr.ib(3)

    # allow pieces
    pieces = ['white', 'black', 'arrow']


@register_attrs
class BoardTerm(object):
    base_term = attr.ib("cell")
    x_term_idx = attr.ib(1)
    y_term_idx = attr.ib(2)

    # list of ExtraTerm (if any - each one will be for index 2,3,4...)
    # will create a cross product
    pieces = attr.ib(default=attr.Factory(list))


@register_attrs
class GameDesc(object):
    game = "checkers"
    x_cords = "a b c d e f g h".split()
    y_cords = "1 2 3 4 5 6 7 8".split()

    # list of BoardTerm (length kind of needs to be >= 1, or not much using convs)
    board_terms = attr.ib(default=attr.Factory(list))

    # list of list of BoardTerm (length kind of needs to be >= 1, or not much using convs)
    control_terms = attr.ib(default=attr.Factory(list))


class GameDefines(object):

    def __init__(self):
        pass

    def breakthrough(self):
        control_terms = [ControlTerm("control", "black", 0),
                         ControlTerm("control", "white", 1)]

        board_term = BoardTerm("cellHolds", 1, 2, 3,
                               PieceTerm(3, ['white', 'black']))

        return GameDesc("breakthrough",
                        "1 2 3 4 5 6 7 8".split(),
                        "1 2 3 4 5 6 7 8".split(),
                        [board_term],
                        [control_terms])
