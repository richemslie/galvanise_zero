from ggpzero.util.attrutil import register_attrs, attribute, attr_factory


@register_attrs
class ControlBase(object):
    ''' a control base is basically a mapping from a gdl base to a value '''

    # list of argument terms - which must match exactly
    arg_terms = attribute(attr_factory(list))

    # we set the channel to this value
    value = attribute(1)


@register_attrs
class ControlChannel(object):
    ''' Creates a single channel.  The control bases need to be mutually exclusive (ie only one set
        at a time).  If none are set the value of the channel will be zero.  If a channel is set,
        it is the value defined in the ControlBase '''
    # a list of control bases.
    control_bases = attribute(attr_factory(list))


@register_attrs
class BoardTerm(object):
    ''' For nxn boards, we identify which terms we use and index into the base. '''

    term_idx = attribute(3)

    # terms = ["white", "black", "arrow"]
    terms = attribute(attr_factory(list))


@register_attrs
class BoardChannels(object):
    ''' board channels are defined by
        (a) the base term
        (b) a cross product of the board terms

    The value set on the channel itself will be if a matching base is set on the x/y cordinates.
    '''

    base_term = attribute("cell")

    # these are index to the term identifying the coordinates
    x_term_idx = attribute(1)
    y_term_idx = attribute(2)

    # list of BoardTerm (if any) - will result in taking a cross product if len() > 1
    board_terms = attribute(attr_factory(list))


@register_attrs
class GameDesc(object):
    game = attribute("checkers")

    # x_cords = "a b c d e f g h".split()
    x_cords = attribute(attr_factory(list))

    # y_cords = "1 2 3 4 5 6 7 8".split()
    y_cords = attribute(attr_factory(list))

    # list of BoardChannels (length kind of needs to be >= 1, or not much using convs)
    board_channels = attribute(attr_factory(list))

    # list of list of ControlChannels
    control_channels = attribute(attr_factory(list))


###############################################################################
# helpers, to cut down on verbosity

def simple_control(*terms):
    return ControlChannel([ControlBase(terms, 1)])


def binary_control(base_term, a_term, b_term):
    return ControlChannel([ControlBase([base_term, a_term], 0),
                           ControlBase([base_term, b_term], 1)])


def step_control(base_term, start, end):
    divisor = float(end - start + 1)
    step_control = ControlChannel([ControlBase([base_term, str(ii)], ii / divisor)
                                   for ii in range(start, end + 1)])
    return step_control


def simple_board_channels(base, pieces):
    ''' common usage '''
    return BoardChannels(base, 1, 2, [BoardTerm(3, pieces)])


###############################################################################

class Games(object):
    ''' class is only here to create a namespace '''

    def breakthrough(self):
        # one channel, sharing black/white
        control = binary_control("control", "black", "white")
        cell_holds = simple_board_channels("cellHolds", ["white", "black"])

        return GameDesc("breakthrough",
                        "1 2 3 4 5 6 7 8".split(),
                        "1 2 3 4 5 6 7 8".split(),
                        [cell_holds], [control])

    def reversi(self):
        # one control channel
        control = binary_control("control", "black", "red")
        cell = simple_board_channels("cell", ["black", "red"])

        return GameDesc("reversi",
                        "1 2 3 4 5 6 7 8".split(),
                        "1 2 3 4 5 6 7 8".split(),
                        [cell], [control])

    def reversi_10x10(self):
        # one control channel
        control = binary_control("control", "black", "white")
        cell = simple_board_channels("cell", ["black", "white"])

        return GameDesc("reversi",
                        "1 2 3 4 5 6 7 8 9 10".split(),
                        "1 2 3 4 5 6 7 8 9 10".split(),
                        [cell], [control])

    def breakthroughSmall(self):
        control = binary_control("control", "white", "black")
        cell = simple_board_channels("cell", ["white", "black"])

        return GameDesc("breakthroughSmall",
                        "1 2 3 4 5 6".split(),
                        "1 2 3 4 5 6".split(),
                        [cell], [control])

    def cittaceot(self):
        # one channel, sharing roles
        control = binary_control("control", "xplayer", "yplayer")

        # note dropping b - for blank (not needed)
        cell = simple_board_channels("cell", ["x", "o"])

        return GameDesc("cittaceot",
                        "1 2 3 4 5".split(),
                        "1 2 3 4 5".split(),
                        [cell], [control])

    def checkers(self):
        control = binary_control("control", "red", "black")

        step = step_control("step", 1, 201)

        # drop b.  Love how it is red and black, then uses bp wp. :)
        cell = simple_board_channels("cell", "bp bk wk wp".split())

        return GameDesc("checkers",
                        "a b c d e f g h".split(),
                        "1 2 3 4 5 6 7 8".split(),
                        [cell], [control, step])

    def _amazons_like(self, game):
        # 4 control channels
        controls = [simple_control("turn", "black", "move"),
                    simple_control("turn", "black", "fire"),
                    simple_control("turn", "white", "move"),
                    simple_control("turn", "white", "fire")]

        just_moved = BoardChannels("justMoved", 1, 2)
        cell = simple_board_channels("cell", ["white", "black", "arrow"])

        return GameDesc(game,
                        "1 2 3 4 5 6 7 8 9 10".split(),
                        "1 2 3 4 5 6 7 8 9 10".split(),
                        [just_moved, cell], controls)

    def amazons_10x10(self):
        return self._amazons_like("amazons_10x10")

    def amazonsLGcross(self):
        return self._amazons_like("amazonsLGcross")

    def amazonsSuicide_10x10(self):
        return self._amazons_like("amazonsSuicide_10x10")

    def atariGo_7x7(self):
        # one channel, sharing roles
        control = binary_control("control", "white", "black")

        cell = simple_board_channels("cell", ["white", "black"])

        return GameDesc("atariGo_7x7",
                        "1 2 3 4 5 6 7".split(),
                        "1 2 3 4 5 6 7".split(),
                        [cell], [control])

    def escortLatch(self):
        # one channel, sharing roles
        control = binary_control("control", "white", "black")

        step = step_control("step", 1, 61)

        cell = simple_board_channels("cell", "bp wp wk bk".split())

        return GameDesc("escortLatch",
                        "a b c d e f g h".split(),
                        "1 2 3 4 5 6 7 8".split(),
                        [cell], [control, step])

    def escortLatch2(self):
        # one channel, sharing roles
        control = binary_control("control", "white", "black")

        step = step_control("step", 1, 101)

        cell = simple_board_channels("cell", "bp wp wk bk".split())

        return GameDesc("escortLatch2",
                        "a b c d e f g h".split(),
                        "1 2 3 4 5 6 7 8".split(),
                        [cell], [control, step])

    def tron_10x10(self):
        # no controls
        cell = simple_board_channels("cell", ["v"])
        position = BoardChannels("position", 2, 3, [BoardTerm(1, ["blue", "red"])])

        return GameDesc("tron_10x10",
                        "0 1 2 3 4 5 6 7 8 9 10 11".split(),
                        "0 1 2 3 4 5 6 7 8 9 10 11".split(),
                        [cell, position], [])

    def hex(self):
        control = binary_control("control", "red", "blue")
        cell = simple_board_channels("cell", "red blue".split())

        # bases: owner, connected and step - are optimisation tricks for propnet?

        return GameDesc("hex",
                        "a b c d e f g h i".split(),
                        "1 2 3 4 5 6 7 8 9".split(),
                        [cell], [control])

    def hexLG11(self):
        control = binary_control("control", "black", "white")
        cell = simple_board_channels("cell", "black white".split())

        # bases: owner, connected and step - are optimisation tricks for propnet?

        return GameDesc("hex",
                        "a b c d e f g h i j k".split(),
                        "1 2 3 4 5 6 7 8 9 10 11".split(),
                        [cell], [control])

    def hexLG13(self):
        control = binary_control("control", "black", "white")
        cell = simple_board_channels("cell", "black white".split())

        # bases: owner, connected and step - are optimisation tricks for propnet?

        return GameDesc("hex",
                        "a b c d e f g h i j k l m".split(),
                        "1 2 3 4 5 6 7 8 9 10 11 12 13".split(),
                        [cell], [control])

    def connectFour(self):
        control = binary_control("control", "red", "black")
        cell = simple_board_channels("cell", ["red", "black"])

        # turned the coord upside down ... well just because it looks pretty!
        return GameDesc("connectFour",
                        "1 2 3 4 5 6 7 8".split(),
                        "6 5 4 3 2 1".split(),
                        [cell], [control])

    def _chess_like(self, game, steps):
        control = binary_control("control", "white", "black")
        step = step_control("step", 1, steps)

        has_moveds = []
        for s in "kingHasMoved hRookHasMoved aRookHasMoved".split():
            has_moveds.append(simple_control(s, "black"))
            has_moveds.append(simple_control(s, "white"))

        # cross product
        cell = BoardChannels("cell", 1, 2, [BoardTerm(3, "white black".split()),
                                            BoardTerm(4, "king rook pawn knight queen bishop".split())])

        return GameDesc(game,
                        "a b c d e f g h".split(),
                        "1 2 3 4 5 6 7 8".split(),
                        [cell], [control, step] + has_moveds)

    def speedChess(self):
        assert False, "check steps"
        return self._chess_like("speedChess", 150)

    def chess_150(self):
        return self._chess_like("chess_150", 151)

    def chess_200(self):
        assert False, "check steps"
        return self._chess_like("chess_200", 200)

    def skirmishNew(self):
        assert False, "check steps"
        return self._chess_like("skirmishNew", 100)

    def skirmishZeroSum(self):
        assert False, "check steps"
        return self._chess_like("skirmishZeroSum", 100)

    def skirmishSTK(self):
        assert False, "check steps"
        return self._chess_like("skirmishSTK", 100)

    def englishDraughts(self):
        last_to_move = binary_control("lastToMove", "black", "white")
        step = step_control("step", 1, 20)

        cell = BoardChannels("cell", 1, 2, [BoardTerm(3, "white black".split()),
                                            BoardTerm(4, "pawn king".split())])
        capturing_piece = BoardChannels("capturingPiece", 1, 2)

        return GameDesc("englishDraughts",
                        "a b c d e f g h".split(),
                        "1 2 3 4 5 6 7 8".split(),
                        [cell, capturing_piece], [step, last_to_move])

    def gomoku_11x11(self):
        # one control channel
        control = binary_control("control", "white", "black")
        cell = simple_board_channels("cell", ["white", "black"])

        return GameDesc("gomoku_11x11",
                        "1 2 3 4 5 6 7 8 9 10 11".split(),
                        "1 2 3 4 5 6 7 8 9 10 11".split(),
                        [cell], [control])

    def draughts_bt_8x8(self):
        interim_status = simple_control("interim_status")
        control = binary_control("control", "white", "black")

        cell = BoardChannels("cell", 2, 3, [BoardTerm(1, "white black".split()),
                                            BoardTerm(4, "man king".split())])

        capturing_piece = BoardChannels("capturing_piece", 1, 2)
        last_piece = BoardChannels("last_at", 1, 2)

        return GameDesc("draughts_bt_8x8",
                        "a b c d e f g h".split(),
                        "1 2 3 4 5 6 7 8".split(),
                        [cell, capturing_piece, last_piece], [interim_status, control])
