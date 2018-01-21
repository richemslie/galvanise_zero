from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game

from ggpzero.defs import templates
from ggpzero.nn.manager import get_manager

from ggpzero.player.cpuctplayer import CppPUCTPlayer


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

def test_play():
    game = "connectFour"
    gen = "test_play"

    # ensure we have a network
    man = get_manager()
    nn = man.create_new_network(game, "smaller")
    man.save_network(nn, game, gen)

    conf = templates.puct_config_template(gen, "test")

    gm = GameMaster(get_gdl_for_game("connectFour"))

    gm.add_player(CppPUCTPlayer(conf=conf), "red")
    gm.add_player(get.get_player("random"), "black")

    gm.start(meta_time=30, move_time=15)
    gm.play_to_end()

