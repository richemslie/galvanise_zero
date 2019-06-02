from builtins import super

from ggplib.db import lookup

from ggpzero.battle.common import MatchGameInfo


def pretty_board(board_size, sm):
    assert board_size == 8 or board_size == 10
    game_info = lookup.by_name("reversi") if board_size == 8 else lookup.by_name("reversi_10x10")

    from ggplib.util.symbols import SymbolFactory
    as_str = game_info.model.basestate_to_str(sm.get_current_state())
    sf = SymbolFactory()
    print list(sf.to_symbols(as_str))

    mapping = {}
    control = None
    for s in sf.to_symbols(as_str):
        if s[1][0] == "control":
            control = s[1][1]
        else:
            assert s[1][0] == "cell"

            key = int(s[1][1]), int(s[1][2])
            mapping[key] = s[1][3]

    lines = []
    line_len = board_size * 4 + 1
    lines.append("     +" + "-" * (line_len - 2) + "+")
    for i in reversed(range(1, board_size + 1)):
        ll = [" %2s  |" % i]
        for j in reversed(range(1, board_size + 1)):
            key = j, i
            if key in mapping:
                if mapping[key] == "black":
                    ll.append(" %s |" % u"\u2659")
                else:
                    assert mapping[key] in ("red", "white")
                    ll.append(" %s |" % u"\u265F")
            else:
                ll.append("   |")

        lines.append("".join(ll))
        if i > 1:
            lines.append("     " + "-" * line_len)

    lines.append("     +" + "-" * (line_len - 2) + "+")
    if board_size == 8:
        lines.append("      " + ' '.join(' %s ' % c for c in 'abcdefgh'))
    else:
        lines.append("      " + ' '.join(' %s ' % c for c in 'abcdef'))

    print
    print
    print "\n".join(lines)
    print "Control:", control


class MatchInfo8(MatchGameInfo):
    def __init__(self):
        game = "reversi"
        game_info = lookup.by_name(game)
        super().__init__(game_info)

    def convert_move_to_gdl(self, move):
        if move == "pass":
            yield "noop"
            return

        assert len(move) == 2

        def cord_x(c):
            return "hgfedcba".index(c) + 1

        def cord_y(c):
            return "abcdefgh".index(c) + 1

        yield "(move %s %s)" % (cord_x(move[0]), cord_y(move[1]))

    def gdl_to_lg(self, move):
        if move == "noop":
            return "pass"

        move = move.replace("(move", "").replace(")", "")

        def cord_x(c):
            return "hgfedcba"[int(c) - 1]

        def cord_y(c):
            return "abcdefgh"[int(c) - 1]

        x, y = move.split()
        return "%s%s" % (cord_x(x), cord_y(y))

    def print_board(self, sm):
        pretty_board(8, sm)


class MatchInfo10(MatchGameInfo):
    def __init__(self):
        game = "reversi_10x10"
        game_info = lookup.by_name(game)
        super().__init__(game_info)

    def convert_move_to_gdl(self, move):
        if move == "pass":
            yield "noop"
            return

        assert len(move) == 2

        def cord(c):
            return "abcdefghij".index(c) + 1

        yield "(move %s %s)" % tuple(cord(c) for c in move)

    def gdl_to_lg(self, move):
        if move == "noop":
            return "pass"

        move = move.replace("(move", "").replace(")", "")

        def cord(c):
            return "abcdefghij"[int(c) - 1]

        return "%s%s" % tuple(cord(c) for c in move.split())

    def print_board(self, sm):
        pretty_board(10, sm)
