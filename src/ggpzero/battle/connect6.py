from builtins import super

import re

from ggplib.db import lookup

from ggpzero.battle.common import MatchGameInfo


class MatchInfo(MatchGameInfo):
    def __init__(self, match_cb=None):
        game_info = lookup.by_name("connect6")
        super().__init__(game_info)

        self.pattern = re.compile('[a-s]\d+')
        self.match_cb = match_cb

    def play_cb(self, players, match_depth):
        if self.match_cb:
            self.match_cb(players, match_depth)

    def convert_move_to_gdl(self, move):
        move = move.lower()
        if move == "j10":
            return

        def lg_to_ggp(k):
            return ("abcdefghijklmnopqrs".index(k[0]) + 1), int(k[1:])

        # always 2 moves
        a, b = self.pattern.findall(move)

        yield "(place %s %s)" % lg_to_ggp(a)
        yield "(place %s %s)" % lg_to_ggp(b)

    def gdl_to_lg(self, move):
        move_a, move_b = move
        move_a = move_a.replace("(place", "").replace(")", "").split()
        move_b = move_b.replace("(place", "").replace(")", "").split()

        def to_cord(x):
            return "_abcdefghijklmnopqrs"[int(x)]

        return "%s%s%s%s" % tuple(to_cord(x) for x in move_a + move_b)

    def print_board(self, sm):
        from ggplib.util.symbols import SymbolFactory

        as_str = self.game_info.model.basestate_to_str(sm.get_current_state())

        sf = SymbolFactory()
        states = sf.to_symbols(as_str)

        control = None
        board_map = {}

        for s in list(states):
            base = s[1]
            if base[0] == "control":
                control = base[1]
            elif base[0] == "cell":
                key = int(base[1]), int(base[2])
                board_map[key] = base[3]

        board_size = 19
        lines = []
        line_len = board_size * 2 + 1
        lines.append("     +" + "-" * line_len + "+")

        for y in range(board_size, 0, -1):
            ll = [" %2d  |" % y]
            for j in range(board_size):
                x = j + 1
                key = x, y
                if key in board_map:
                    if board_map[key] == "black":
                        ll.append(" %s" % u"\u26C0")
                    else:
                        assert board_map[key] == "white"
                        ll.append(" %s" % u"\u26C2")
                else:
                    ll.append(" .")

            lines.append("".join(ll) + " |")

        lines.append("     +" + "-" * line_len + "+")
        lines.append("       " + ' '.join('%s' % c for c in 'abcdefghijklmnopqrs'))

        print
        print
        print "\n".join(lines)
        print "Control:", control
