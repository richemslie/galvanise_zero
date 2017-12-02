from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game
from ggplearn.player.simple import NNPlayerOneShot

def test_breakthrough():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    # add two players
    white = get.get_player("pymcs")
    white.max_run_time = 0.5
    black = NNPlayerOneShot()

    gm.add_player(black, "white")
    gm.add_player(white, "black")

    gm.start(meta_time=30, move_time=15)
    gm.play_to_end()

def test_reversi():
    gm = GameMaster(get_gdl_for_game("reversi"))

    # add two players
    white = get.get_player("pymcs")
    white.max_run_time = 0.5
    black = NNPlayerOneShot()

    gm.add_player(black, "black")
    gm.add_player(white, "red")

    gm.start(meta_time=30, move_time=15)
    gm.play_to_end()
