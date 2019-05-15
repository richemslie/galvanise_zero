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

@register_attrs
class ApplySymmetry(object):
    base_term = attribute("cell")

    # these are index to the term identifying the coordinates
    x_terms_idx = attribute(attr_factory(list))
    y_terms_idx = attribute(attr_factory(list))


@register_attrs
class Symmetries(object):
    ''' defines the bases which symmetries can be applied  '''

    # list of ApplySymmetry
    apply_bases = attribute(attr_factory(list))

    # list of terms
    skip_bases = attribute(attr_factory(list))

    # list of ApplySymmetry
    apply_actions = attribute(attr_factory(list))

    # list of terms
    skip_actions = attribute(attr_factory(list))

    # do horizontal reflection
    do_reflection = attribute(False)

    # rotate x4
    do_rotations_90 = attribute(False)

    # rotate x2
    do_rotations_180 = attribute(False)


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

    def _chess_like(self, game, steps=None):
        control = binary_control("control", "white", "black")
        if steps is not None:
            step = step_control("step", 1, steps)

        has_moveds = []
        for s in "kingHasMoved hRookHasMoved aRookHasMoved".split():
            has_moveds.append(simple_control(s, "black"))
            has_moveds.append(simple_control(s, "white"))

        # cross product
        cell = BoardChannels("cell", 1, 2, [BoardTerm(3, "white black".split()),
                                            BoardTerm(4, "king rook pawn knight queen bishop".split())])

        controls = [control] + has_moveds
        if steps is not None:
            controls.append(step)

        return GameDesc(game,
                        "a b c d e f g h".split(),
                        "1 2 3 4 5 6 7 8".split(),
                        [cell], controls)

    def speedChess(self):
        assert False, "check steps"
        return self._chess_like("speedChess", 150)

    def chess_200(self):
        return self._chess_like("chess_200")

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


    def _draughts_helper(self):
        interim_status = simple_control("interim_status")
        control = binary_control("control", "white", "black")

        cell = BoardChannels("cell", 2, 3, [BoardTerm(1, "white black".split()),
                                            BoardTerm(4, "man king".split())])

        capturing_piece = BoardChannels("capturing_piece", 1, 2)
        last_piece = BoardChannels("last_at", 1, 2)
        return [cell, capturing_piece, last_piece], [interim_status, control]

    def draughts_bt_8x8(self):
        board_channels, controls = self._draughts_helper()

        return GameDesc("draughts_bt_8x8",
                        "a b c d e f g h".split(),
                        "1 2 3 4 5 6 7 8".split(),
                        board_channels, controls)

    def draughts_10x10(self):
        board_channels, controls = self._draughts_helper()

        return GameDesc("draughts_10x10",
                        "a b c d e f g h i j".split(),
                        "1 2 3 4 5 6 7 8 9 10".split(),
                        board_channels, controls)

    def connect6(self):
        # 4 control channels (black/white place twice per turn)
        controls = [simple_control("control", "black_turn0"),
                    simple_control("control", "black_turn1"),
                    simple_control("control", "white_turn0"),
                    simple_control("control", "white_turn1")]
        cell = simple_board_channels("cell", ["black", "white"])
        return GameDesc("connect6",
                        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19".split(),
                        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19".split(),
                        [cell], controls)

    def baduk_9x9(self):
        control = binary_control("control", "white", "black")
        simple_controls = [simple_control("passed", "black"),
                           simple_control("passed", "white"),
                           simple_control("ko_set")]

        cell = BoardChannels("cell", 2, 3, [BoardTerm(1, "white black".split())])
        ko_point = BoardChannels("ko_point", 1, 2)
        ko_captured = BoardChannels("ko_captured", 1, 2)
        return GameDesc("baduk_9x9",
                        "a b c d e f g h i".split(),
                        "1 2 3 4 5 6 7 8 9".split(),
                        [cell, ko_point, ko_captured], simple_controls + [control])

    def baduk_19x19(self):
        control = binary_control("control", "white", "black")
        simple_controls = [simple_control("passed", "black"),
                           simple_control("passed", "white"),
                           simple_control("ko_set")]

        cell = BoardChannels("cell", 2, 3, [BoardTerm(1, "white black".split())])
        ko_point = BoardChannels("ko_point", 1, 2)
        ko_captured = BoardChannels("ko_captured", 1, 2)
        return GameDesc("baduk_19x19",
                        "a b c d e f g h i  j  k  l  m  n  o  p  q  r  s".split(),
                        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19".split(),
                        [cell, ko_point, ko_captured], simple_controls + [control])


class GameSymmetries(object):
    ''' class is only here to create a namespace '''

    def breakthroughSmall(self):
        return Symmetries(skip_bases=["control"],
                          apply_bases=[ApplySymmetry("cell", 1, 2)],
                          skip_actions=["noop"],
                          apply_actions=[ApplySymmetry("move", [1, 3], [2, 4])],
                          do_reflection=True)

    def breakthrough(self):
        return Symmetries(skip_bases=["control"],
                          apply_bases=[ApplySymmetry("cellHolds", 1, 2)],
                          skip_actions=["noop"],
                          apply_actions=[ApplySymmetry("move", [1, 3], [2, 4])],
                          do_reflection=True)

    def reversi(self):
        return Symmetries(skip_bases=["control"],
                          apply_bases=[ApplySymmetry("cell", 1, 2)],
                          skip_actions=["noop"],
                          apply_actions=[ApplySymmetry("move", 1, 2)],
                          do_rotations_90=True,
                          do_reflection=True)

    def reversi_10x10(self):
        return Symmetries(skip_bases=["control"],
                          apply_bases=[ApplySymmetry("cell", 1, 2)],
                          skip_actions=["noop"],
                          apply_actions=[ApplySymmetry("move", 1, 2)],
                          do_rotations_90=True,
                          do_reflection=True)

    def connect6(self):
        return Symmetries(skip_bases=["control"],
                          apply_bases=[ApplySymmetry("cell", 1, 2)],
                          skip_actions=["noop"],
                          apply_actions=[ApplySymmetry("place", 1, 2)],
                          do_rotations_90=True,
                          do_reflection=True)

    def _hex(self):
        ''' Thanks to Niall Cardin (who in turn extended his thanks to Jeff Klingner) for setting
            me right on my hex symmetries. '''
        return Symmetries(skip_bases=["control", "step", "owner", "canSwap"],
                          apply_bases=[ApplySymmetry("cell", 1, 2),
                                       ApplySymmetry("connected", 2, 3)],
                          skip_actions=["noop", "swap"],
                          apply_actions=[ApplySymmetry("place", 1, 2)],
                          do_rotations_90=False,
                          do_rotations_180=True,
                          do_reflection=False)
    hexLG11 = _hex
    hexLG13 = _hex

    def baduk(self):
        return Symmetries(skip_bases=["control", "passed", "ko_set"],
                          apply_bases=[ApplySymmetry("cell", 2, 3),
                                       ApplySymmetry("ko_point", 1, 2),
                                       ApplySymmetry("ko_captured", 1, 2)],
                          skip_actions=["noop", "pass"],
                          apply_actions=[ApplySymmetry("place", 1, 2)],
                          do_rotations_90=True,
                          do_reflection=True)

    baduk_9x9 = baduk
    baduk_19x19 = baduk
