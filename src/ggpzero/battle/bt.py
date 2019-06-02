
from builtins import super

from ggplib.db import lookup

from ggpzero.battle.common import MatchGameInfo


def get_game_info(board_size):
    # add players
    if board_size == 8:
        game = "breakthrough"
    else:
        assert board_size == 6
        game = "breakthroughSmall"

    return lookup.by_name(game)


def pretty_board(board_size, sm):
    ' pretty print board current state of match '

    from ggplib.util.symbols import SymbolFactory
    as_str = get_game_info(board_size).model.basestate_to_str(sm.get_current_state())
    sf = SymbolFactory()
    states = sf.to_symbols(as_str)
    mapping = {}
    control = None
    for s in list(states):
        if s[1][0] == "control":
            control = s[1][1]
        else:
            if board_size == 8:
                assert s[1][0] == "cellHolds"
            else:
                assert s[1][0] == "cell"

            key = int(s[1][1]), int(s[1][2])
            mapping[key] = s[1][3]

    lines = []
    line_len = board_size * 4 + 1
    lines.append("    +" + "-" * (line_len - 2) + "+")
    for i in reversed(range(1, board_size + 1)):
        ll = [" %s  |" % i]
        for j in reversed(range(1, board_size + 1)):
            key = j, i
            if key in mapping:
                if mapping[key] == "black":
                    ll.append(" %s |" % u"\u2659")
                else:
                    assert mapping[key] == "white"
                    ll.append(" %s |" % u"\u265F")
            else:
                ll.append("   |")

        lines.append("".join(ll))
        if i > 1:
            lines.append("    " + "-" * line_len)

    lines.append("    +" + "-" * (line_len - 2) + "+")
    if board_size == 8:
        lines.append("     " + ' '.join(' %s ' % c for c in 'abcdefgh'))
    else:
        lines.append("     " + ' '.join(' %s ' % c for c in 'abcdef'))

    print
    print
    print "\n".join(lines)
    print "Control:", control


def parse_sgf(txt):
    ''' not actually sgf, but whatever '''
    moves = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("1"):
            expect = 1
            moves = []
            for count, token in enumerate(line.split()):
                if token == "*":
                    break
                if count % 3 == 0:
                    expect_str = "%s." % expect
                    assert token == expect_str, "expected '%s', got '%s'" % (expect_str, token)
                    expect += 1
                else:
                    moves.append(token)
    return moves


class MatchInfo(MatchGameInfo):
    def __init__(self, board_size):
        self.board_size = board_size

        # add players
        if board_size == 8:
            game = "breakthrough"
        else:
            assert board_size == 6
            game = "breakthroughSmall"

        game_info = lookup.by_name(game)
        super().__init__(game_info)

    def print_board(self, sm):
        return pretty_board(self.board_size, sm)

    def convert_move_to_gdl(self, move):
        def to_cords(s):
            if self.board_size == 8:
                mapping_x_cord = {x0 : x1 for x0, x1 in zip('abcdefgh', '87654321')}
            else:
                mapping_x_cord = {x0 : x1 for x0, x1 in zip('abcdef', '654321')}
            return mapping_x_cord[s[0]], s[1]

        move = move.lower()
        split_chr = '-' if "-" in move else 'x'
        from_, to_ = map(to_cords, move.split(split_chr))
        yield "(move %s %s %s %s)" % (from_[0], from_[1], to_[0], to_[1])

    def gdl_to_sgf(self, move):
        # XXX captures How?
        # XXX move = move[lead_role_index]
        move = move.replace("(move", "").replace(")", "")
        a, b, c, d = move.split()
        if self.board_size == 8:
            mapping_x_cord = {x0 : x1 for x0, x1 in zip('87654321', 'abcdefgh')}
        else:
            mapping_x_cord = {x0 : x1 for x0, x1 in zip('654321', 'abcdef')}

        return "%s%s-%s%s" % (mapping_x_cord[a], b, mapping_x_cord[c], d)

    def gdl_to_lg(self, move):
        move = move.replace("(move", "").replace(")", "")
        a, b, c, d = move.split()
        a = 8 - int(a)
        b = int(b) - 1
        c = 8 - int(c)
        d = int(d) - 1
        return "%s%s%s%s" % (a, b, c, d)

    def parse_sgf(self, sgf):
        return parse_sgf(sgf)


    def print_board(self, sm):
        pretty_board(self.board_size, sm)
