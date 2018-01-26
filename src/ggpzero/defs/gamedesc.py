from ggpzero.util.attrutil import register_attrs, attribute, attr_factory


@register_attrs
class ControlTerm(object):
    # pieces
    base_term = attribute(3)

    # list of extra terms - must match absolutely
    extra_terms = attribute(default=attr_factory(list))

    # we set the channel to this value
    value = attribute(1)


@register_attrs
class PieceTerm(object):
    piece_term_idx = attribute(3)

    # allow pieces
    pieces = ['white', 'black', 'arrow']


@register_attrs
class BoardTerm(object):
    base_term = attribute("cell")
    x_term_idx = attribute(1)
    y_term_idx = attribute(2)

    # list of ExtraTerm (if any - each one will be for index 2,3,4...)
    # will create a cross product
    pieces = attribute(default=attr_factory(list))


@register_attrs
class GameDesc(object):
    game = "checkers"
    x_cords = "a b c d e f g h".split()
    y_cords = "1 2 3 4 5 6 7 8".split()

    # list of BoardTerm (length kind of needs to be >= 1, or not much using convs)
    board_terms = attribute(default=attr_factory(list))

    # list of list of BoardTerm (length kind of needs to be >= 1, or not much using convs)
    control_terms = attribute(default=attr_factory(list))


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
