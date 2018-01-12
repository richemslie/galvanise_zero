from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game

from ggpzero.defs import templates
from ggpzero.player.cpuctplayer import CppPUCTPlayer


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

def test_play():
    conf = templates.puct_config_template("none", "test")

    gm = GameMaster(get_gdl_for_game("breakthroughSmall"))

    gm.add_player(get.get_player("random"), "white")
    gm.add_player(CppPUCTPlayer(conf=conf), "black")

    gm.start(meta_time=30, move_time=15)
    gm.play_to_end()

