from builtins import super
from pprint import pprint

from ggplib.db import lookup

from ggpzero.battle.common import MatchGameInfo


def print_board(game_info, sm):
    basestate = sm.get_current_state()

    def valid_bases():
        for ii in range(basestate.len()):
            if basestate.get(ii):
                base = game_info.model.bases[ii]
                base = base.replace("(true (", "").replace(")", "")
                yield base.split()

    # pprint(list(valid_bases()))

    board_map = {}

    to_unicode = {}
    to_unicode["black", "king"] = u"\u2654"
    to_unicode["black", "queen"] = u"\u2655"
    to_unicode["black", "rook"] = u"\u2656"
    to_unicode["black", "bishop"] = u"\u2657"
    to_unicode["black", "knight"] = u"\u2658"
    to_unicode["black", "pawn"] = u"\u2659"

    to_unicode["white", "king"] = u"\u265A"
    to_unicode["white", "queen"] = u"\u265B"
    to_unicode["white", "rook"] = u"\u265C"
    to_unicode["white", "bishop"] = u"\u265D"
    to_unicode["white", "knight"] = u"\u265E"
    to_unicode["white", "pawn"] = u"\u265F"

    for base in valid_bases():
        if base[0] == "cell":
            key = "_abcdefgh".index(base[1]), int(base[2])
            board_map[key] = base[3], base[4]
        elif base[0] in ("step", "kingHasMoved", "aRookHasMoved", "control",
                         "hRookHasMoved", "canEnPassantCapture"):
            print "control", base
            continue
        else:
            assert False, "what is this?: %s" % base

    # pprint(board_map)

    board_size = 8
    lines = []
    line_len = board_size * 2 + 1
    lines.append("     +" + "-" * line_len + "+")

    for i in range(board_size):
        y = board_size - i
        ll = [" %2d  |" % y]
        for j in range(board_size):
            x = j + 1
            key = x, y
            if key in board_map:
                what = board_map[key]
                c = to_unicode[what]
                ll.append(" %s" % c)

            else:
                ll.append(" .")

        lines.append("".join(ll) + " |")

    lines.append("     +" + "-" * line_len + "+")
    lines.append("       " + ' '.join('%s' % c for c in 'abcdefgh'))

    print
    print
    print "\n".join(lines)
    print


class MatchInfo(MatchGameInfo):
    def __init__(self, short_50=False):
        if short_50:
            game = "chess_15d"
        else:
            game = "chess_50d"

        self.game_info = lookup.by_name(game)
        super().__init__(self.game_info)

    def print_board(self, sm):
        print_board(self.game_info, sm)
