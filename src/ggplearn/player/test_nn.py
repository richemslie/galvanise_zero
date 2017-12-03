from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game
from ggplearn.player.simple import NNPlayerOneShot


def setup():
    from ggplib.util.init import setup_once
    setup_once()



# def test_breakthrough():
#     gm = GameMaster(get_gdl_for_game("breakthrough"))

#     # add two players
#     white = get.get_player("pymcs")
#     white.max_run_time = 0.5
#     black = NNPlayerOneShot()

#     gm.add_player(black, "white")
#     gm.add_player(white, "black")

#     gm.start(meta_time=30, move_time=15)
#     gm.play_to_end()

def test_reversi_tournament():
    gm = GameMaster(get_gdl_for_game("reversi"))

    # add two players
    pymcs = get.get_player("pymcs")
    pymcs.max_run_time = 0.25

    random = get.get_player("pyrandom")
    nn = NNPlayerOneShot()

    gm.add_player(pymcs, "black")
    gm.add_player(nn, "red")

    acc_black_score = 0
    acc_red_score = 0
    for i in range(10):
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["red"]

    print "black_score", gm.players_map["black"], acc_black_score
    print "red_score", gm.players_map["red"], acc_red_score
