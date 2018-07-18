from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game

from gurgeh.player import GurgehPlayer


def setup():
    from ggplib.util.init import setup_once
    setup_once()


def test_breakthrough():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    # add two players
    pymcs = get.get_player("pymcs")
    pymcs.max_run_time = 0.25

    gurgeh = GurgehPlayer("Gurgeh")
    gurgeh.max_tree_search_time = 1
    gurgeh.max_tree_search_time = 1
    gurgeh.thread_workers = 2

    gm.add_player(pymcs, "white")
    gm.add_player(gurgeh, "black")

    gm.start(meta_time=30, move_time=15)
    gm.play_to_end()

    print "white_score", pymcs.name, gm.get_score("white")
    print "black_score", gurgeh.name, gm.get_score("black")
