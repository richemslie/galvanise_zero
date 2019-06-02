from builtins import super

from ggplib.util.symbols import SymbolFactory
from ggplib.db import lookup

from ggpzero.battle.common import MatchGameInfo


class MatchInfo(MatchGameInfo):
    def __init__(self, cross=False, match_cb=None):
        game = "amazonsLGcross" if cross else "amazons_10x10"
        self.cross = cross
        self.match_cb = match_cb

        game_info = lookup.by_name(game)
        super().__init__(game_info)

    def play_cb(self, players, match_depth):
        if self.match_cb:
            self.match_cb(players, match_depth)

    def convert_move_to_gdl(self, move):
        def lg_to_ggp(k):
            return ("jihgfedcba".index(k[0]) + 1), int(k[1:])

        # actually 2 moves
        amazon, fire = move.split("/")
        from_pos, to_pos = amazon.split("-")
        from_pos, to_pos, fire = map(lg_to_ggp, (from_pos, to_pos, fire))

        yield "(move %s %s %s %s)" % (from_pos + to_pos)
        yield "(fire %s %s)" % fire

    def ggp_to_sgf(self, move):
        amazon_move, fire_move = move
        amazon_move = amazon_move.replace("(move", "").replace(")", "")
        fire_move = fire_move.replace("(fire", "").replace(")", "")
        cords = map(int, amazon_move.split()) + map(int, fire_move.split())

        def ggp_to_cord(x, y):
            return "%s%s" % ("abcdefghij"[10 - x], y)

        move = "%s-%s/%s" % (ggp_to_cord(cords[0], cords[1]),
                             ggp_to_cord(cords[2], cords[3]),
                             ggp_to_cord(cords[4], cords[5]))

        return move

    def gdl_to_lg(self, move):
        amazon_move, fire_move = move
        amazon_move = amazon_move.replace("(move", "").replace(")", "")
        fire_move = fire_move.replace("(fire", "").replace(")", "")
        cords = map(int, amazon_move.split()) + map(int, fire_move.split())

        def ggp_to_lg(x, y):
            return "%s%s" % (10 - x, y - 1)

        move = "%s%s%s" % (ggp_to_lg(cords[0], cords[1]),
                           ggp_to_lg(cords[2], cords[3]),
                           ggp_to_lg(cords[4], cords[5]))

        return move

    def print_board(self, sm):
        as_str = self.game_info.model.basestate_to_str(sm.get_current_state())
        print as_str

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

        def row(i):
            yield ' '
            for j in range(10, 0, -1):
                key = j, i
                if key in board_map:
                    if board_map[key] == "arrow":
                        yield " %s " % u"\u25C8"

                    elif board_map[key] == "black":
                        yield " B "

                    else:
                        assert board_map[key] == "white"
                        yield " W "

                else:
                    yield ' . '

        def lines():
            for i in range(10, 0, -1):
                yield "".join(row(i))

        print
        print
        print "\n".join(lines())
        print "Control:", control
