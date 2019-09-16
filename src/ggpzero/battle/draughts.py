from builtins import super

from ggplib.db import lookup

from ggpzero.battle.common import MatchGameInfo
from ggplib.non_gdl_games.draughts import desc


class Draughts_MatchInfo(MatchGameInfo):
    def __init__(self, killer=False):
        if killer:
            game_info = lookup.by_name("draughts_killer_10x10")
        else:
            game_info = lookup.by_name("draughts_10x10")

        super().__init__(game_info)

        self.board_desc = desc.BoardDesc(10)

    def print_board(self, sm):
        self.board_desc.print_board_sm(sm)
