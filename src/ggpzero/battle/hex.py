from builtins import super

import time
import hashlib

from ggplib.db import lookup

from ggpzero.battle.common import MatchGameInfo


def write_hexgui_sgf(black_name, white_name, moves, game_size):
    game_id = hashlib.md5(hashlib.sha1("%.5f" % time.time()).hexdigest()).hexdigest()[:6]

    with open("hex%s_%s_%s_%s.sgf" % (game_size,
                                      black_name,
                                      white_name,
                                      game_id), "w") as f:
        f.write("(;FF[4]EV[null]PB[%s]PW[%s]SZ[%s]GC[game#%s];" % (black_name,
                                                                   white_name,
                                                                   game_size,
                                                                   game_id))

        # Note: piece colours are swapped for hexgui from LG
        for ri, m in moves:
            f.write("%s[%s];" % ("B" if ri == 0 else "W", m))
        f.write(")\n")


def dump_trmph_url(moves, game_size):
    s = "http://www.trmph.com/hex/board#%d," % game_size
    for _, m in moves:
        if m == "swap":
            continue
        s += m

    print s


def convert_trmph(ex):
    alpha = "_abcdefghijklmnop"
    moves = ex.split(",")[-1]
    first = None
    while moves:
        c0 = moves[0]
        assert c0 in alpha
        try:
            ii = int(moves[1] + moves[2])
            c1 = alpha[ii]
            move = c0 + c1
            moves = moves[3:]
        except:
            ii = int(moves[1])
            c1 = alpha[ii]
            move = c0 + c1
            moves = moves[2:]

        if first is None:
            first = move
        else:
            if first == move:
                move = "swap"
            first = -42

        yield move

class MatchInfo(MatchGameInfo):
    def __init__(self, size=None):
        game = "hexLG%s" % size
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
                                               "abcdefghijklmnop".index(move[1]) + 1)
            yield gdl_role_move

    def gdl_to_lg(self, move):
        if move != "swap":
            move = move.replace("(place", "").replace(")", "")
            parts = move.split()
            move = parts[0] + "_abcdefghijklm"[int(parts[1])]
        return move

    def export(self, players, result):
        sgf_moves = []

        def remove_gdl(m):
            return m.replace("(place ", "").replace(")", "").strip().replace(' ', '')

        def swapaxis(s):
            mapping_x = {x1 : x0 for x0, x1 in zip('abcdefghi', '123456789')}
            mapping_y = {x0 : x1 for x0, x1 in zip('abcdefghi', '123456789')}
            for x0, x1 in zip(('j', 'k', 'l', 'm'), ('10', '11', '12', '13')):
                mapping_x[x1] = x0
                mapping_y[x0] = x1

            return "%s%s" % (mapping_x[s[1]], mapping_y[s[0]])

        def add_sgf_move(ri, m):
            sgf_moves.append((ri, m))
            if m == "swap":
                assert ri == 1
                assert len(sgf_moves) == 2

                # hexgui does do swap like LG.   This is a (double) hack.
                moved_move = swapaxis(sgf_moves[0][1])
                sgf_moves[0] = (0, moved_move)
                sgf_moves.append((1, moved_move))

        for match_depth, move, move_info in result:
            ri = 1 if move[0] == "noop" else 0
            str_move = remove_gdl(move[ri])
            add_sgf_move(ri, str_move)

        player0, player1 = players
        dump_trmph_url(sgf_moves, self.size)
        write_hexgui_sgf(player0.get_name(), player1.get_name(), sgf_moves, self.size)

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
                key = "abcdefghijklmnop".index(base[1]) + 1, int(base[2]),
                board_map[key] = base[3]

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
                    if board_map[key] == "black":
                        ll.append(" %s" % u"\u26C0")
                    else:
                        assert board_map[key] == "white"
                        ll.append(" %s" % u"\u26C2")
                else:
                    ll.append(" .")

            lines.append("".join(ll) + " \\")

        lines.append("   %s  +" % indent(board_size + 1) + "-" * line_len + "+")
        lines.append("   %s   " % indent(board_size + 1) + ' '.join('%s' % c for c in 'abcdefghijklmnopqrs'[:board_size]))

        print
        print
        print "\n".join(lines)
        print "Control:", control
