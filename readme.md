gzero/galvanise_zero
====================
gzero provides a framework for neural networks to learn solely based on self play.  This is largely based off Deepmind's papers on AlphaGo, AlphaGo Zero and AlphaZero, as well as the excellent
Expert Iteration [paper](https://arxiv.org/abs/1705.08439). A number of Alpha*Zero open source projects were also inspirational.

The name gzero stems from the fact that this project was initially a spin off my galvanise player
in [GGP](https://en.wikipedia.org/wiki/General_game_playing).

Status
------
All games are written in [GDL](https://en.wikipedia.org/wiki/Game_Description_Language) unless otherwise stated.  There is *no* game specific code other than 
a single python file describing mappings for policy and state (see [here](https://github.com/richemslie/galvanise_zero/issues/1) for more information).

Games with significant training:

* [breakthrough](https://github.com/richemslie/gzero_data/tree/master/breakthrough)
* reversi (8 and 10 board sizes)
* hex (11 and 13 board sizes)
* amazons
* connect6
* chess (with no 50 rule)
* go(baduk) 9x9 (no super ko, statemachine in c++)
* international draughts (statemachine in c++)

LG Champion in last attempt @ Amazons, Breakthrough and Hex 13x13 (joint).

Amazons and Breakthrough models were strong enough to win gold medals at ICGA 2018 Computer Olympiad. :clap: :clap:

Reversi is also strong relative to humans on LG, yet performs a bit worse than top AB programs (about ntest level 20 the last time I tested).

Hex/Connect6 play around somewhere top human level on LG.

Chess and Baduk 9x9 are reasonably strong for the little time they were trained.  Baduk 9x9 had a rating ~2900 elo on CGOS after 2-3 week of training.  Chess was harder to test due to not having 50 rule, but somewhere about 2200-2600 elo would be a decent guess.

Also, Chess and Connect6 "cheated" as experimented with adding data from historical games
as well as the self play data.

All the models can (eventually) be found [here](https://github.com/richemslie/gzero_data).

--------------------

The code is in fairly good shape, but could do with some refactoring and
documentation (especially a how to guide on how to train a game).  It would definitely be good to
have an extra pair of eyes on it.  I'll welcome and support anyone willing to try training a game
for themselves.  Some notes:

1. python is 2.7
2. requires a GPU/tensorflow
3. good starting point is https://github.com/richemslie/ggp-zero/blob/dev/src/ggpzero/defs
4. the self play method is very different from A0, and not documented anywhere.  the code is here:
    https://github.com/richemslie/ggp-zero/blob/dev/src/cpp/selfplay.cpp
5. cpp puct/puct2 really needs to be combined.


Little Golem
------------
Most trained games are available to play on Little Golem website.  Send an invite to play
[gzero_bot](http://littlegolem.net/jsp/info/player.jsp?plid=58835).


project goal(s)
---------------
The initial goal of this project was to be able to train any game in
[GGP](https://en.wikipedia.org/wiki/General_game_playing) ecosystem, to play at a higher level than
the state of the art GGP players, given some (relatively) small time frame to train the game (a few
days to a week, on commodity hardware - and not months/years worth of training time on hundreds of
GPUs).

Some game types which would be interesting to try:

* non-zero sum games (such as the non zero sum variant of Skirmish)
* multiplayer games (games with > 2 players)
* games that are not easily represented as a 2D array of channels
* simultaneous games


Related repos (will be merged eventually here)
----------------------------------------------
* ggpzero is extension of [ggplib](https://github.com/ggplib/ggplib)
* Custom games, game specific code (*) can be found [here](https://github.com/richemslie/gzero_games)


(*)  Most game specific game is for testing purposes, printing the board to console, or connecting
to platforms/programs, such as GTP in go and UCI in chess.  State machines for go(Baduk) and
International Draughts are written in C++.


