ggpzero/gzero/galvanise
=======================
gzero provides a framework for neural networks to learn to play solely based on self-play.  This
is loosely based on the papers of AlphaZero and exIT, and a number of zero open source were greatly
inspirational.

The name gzero stems from the fact that this project was initially a spin off my galvanise player
in GGP.

Status September 2018
-------------------
Games trained for > 5 days (from most recent)

* go(baduk) 9x9 (no super ko, statemachine in c++)
* chess (with no 50 rule)
* connect 6
* amazons
* hex (11 and 13 board sizes)
* reversi (8 and 10 board sizes)
* breakthrough

See https://github.com/richemslie/ggp-zero/blob/dev/src/ggpzero/defs/gamedesc.py for full list.

Amazons and Breakthrough models were strong enough to win gold medals at ICGA 2018 Computer Olympiad.  Reversi performs a litte under AB players (about ntest level 20) and Hex/Connect6 play around somewhere around top 10 human level on LG (at a guess, really don't know for sure).  Chess and Go are reasonably strong, but no where near the level one could achieve if spending 100s of GPU days training.  Go 9x9 has a rating 2560 elo on CGOS after about a week of training.  Chess was harder to test due to not having 50 rule, but somewhere about 2200-2600 elo would be my best guess.

Also, Chess and Connect6 "cheated" as experimented with adding some data from historical games
during the self play.

--------------------

The project is currently on hiatus.

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


Little Gollem
-------------
Search for user
[gzero_bot](http://littlegolem.net/jsp/info/player.jsp?plid=58835) on Little Golem if you would
like to challenge it to a game.


project goal(s)
---------------
The initial goal of this project was to be able to train any game in
[GGP](https://en.wikipedia.org/wiki/General_game_playing) ecosystem, to play at a higher level than
the state of the art GGP players, given some (relatively) small time frame to train the game (a few
days to a week, on commodity hardware - and not months/years worth of training time on hundreds of
GPUs).

Some game types which would be interesting to try:

* non-zero board games (such as non zero sum variant of Skirmish)
* multiplayer games
* games that are not easily represented as a 2D array of channels
* simultaneous games
* single player games (puzzles)


Related repos
-------------
* ggpzero is extension of [ggplib](https://github.com/ggplib/ggplib)
* Models can be found [here](https://github.com/richemslie/gzero_data)
* Custom games, game specific code (*) can be found [here](https://github.com/richemslie/gzero_games)


(*)  Most game specific game is for testing purposes, printing the board to console, or connecting
to platforms/programs, such as GTP in go and UCI in chess.  State machines for go(Baduk) and
Internation Draughts are written in C++.


