from builtins import super

import time
import hashlib

import colorama

from ggplib.db import lookup

from ggpzero.battle.common import MatchGameInfo


class MatchInfo(MatchGameInfo):
    def __init__(self, size=None):
        game = "hex_lg_%s" % size
        self.size = size

        game_info = lookup.by_name(game)
        super().__init__(game_info)

    def convert_move_to_gdl(self, move):
        if move == "resign":
            return

        if move == "swap":
            yield move
        else:
            gdl_role_move = "(place %s %s)" % (move[0],
                                               "abcdefghijklmnopqrstuvwxyz".index(move[1]) + 1)
            yield gdl_role_move

    def gdl_to_lg(self, move):
        if move != "swap":
            move = move.replace("(place", "").replace(")", "")
            parts = move.split()
            move = parts[0] + "_abcdefghijklmnopqrstuvwxyz"[int(parts[1])]
        return move

    def print_board(self, sm):
        from ggplib.util.symbols import SymbolFactory

        as_str = self.game_info.model.basestate_to_str(sm.get_current_state())

        sf = SymbolFactory()
        states = sf.to_symbols(as_str)

        control = None
        board_map = {}
        board_map_colours = {}

        for s in list(states):
            base = s[1]
            print base
            if base[0] == "control":
                control = base[1]
            elif base[0] == "cell":
                key = "abcdefghijklmnoprstuvwxyz".index(base[2]) + 1, int(base[3]),
                if base[1] == "white" or base[1] == "black":
                    board_map[key] = base[1]
                else:
                    board_map_colours[key] = base

        board_size = self.size
        lines = []
        line_len = board_size * 2 + 1

        def indent(y):
            return y * ' '

        lines.append("    %s +" % indent(0) + "-" * line_len + "+")
        for i in range(board_size):
            y = i + 1
            ll = [" %2d %s \\" % (y, indent(y))]

            for j in range(board_size):
                x = j + 1
                key = x, y
                if key in board_map:
                    if key in board_map_colours:
                        ll.append(colorama.Fore.GREEN)
                    if board_map[key] == "black":
                        ll.append(" %s" % u"\u26C0")
                    else:
                        assert board_map[key] == "white"
                        ll.append(" %s" % u"\u26C2")

                    if key in board_map_colours:
                        ll.append(colorama.Style.RESET_ALL)

                else:
                    ll.append(" .")

            lines.append("".join(ll) + " \\")

        lines.append("   %s  +" % indent(board_size + 1) + "-" * line_len + "+")
        lines.append("   %s   " % indent(board_size + 1) + ' '.join('%s' % c for c in 'abcdefghijklmnopqrs'[:board_size]))

        print
        print
        print "\n".join(lines)
        print "Control:", control
