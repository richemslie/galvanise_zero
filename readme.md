gzero/galvanise_zero
====================
gzero provides a framework for neural networks to learn solely based on self play.  This is largely based off Deepmind's papers on AlphaGo, AlphaGo Zero and AlphaZero, as well as the excellent
Expert Iteration [paper](https://arxiv.org/abs/1705.08439). A number of Alpha*Zero open source projects were also inspirational.

The name gzero stems from the fact that this project was initially a spin off my galvanise player
in [GGP](https://en.wikipedia.org/wiki/General_game_playing).

* this is an extension of [ggplib](https://github.com/ggplib/ggplib)

Status
------
All games are written in [GDL](https://en.wikipedia.org/wiki/Game_Description_Language) unless otherwise stated.  There is *no* game specific code other than 
a single python file describing mappings for policy and state (see [here](https://github.com/richemslie/galvanise_zero/issues/1) for more information).

Games with significant training, links to elo graphs and models:

* [chess](https://github.com/richemslie/gzero_data/tree/master/data/chess)
* [connect6](https://github.com/richemslie/gzero_data/tree/master/data/connect6)
* [hex13](https://github.com/richemslie/gzero_data/tree/master/data/hexLG13)
* [reversi10](https://github.com/richemslie/gzero_data/tree/master/data/reversi_10x10)
* [reversi8](https://github.com/richemslie/gzero_data/tree/master/data/reversi_8x8)
* [amazons](https://github.com/richemslie/gzero_data/tree/master/data/amazons_10x10)
* [breakthrough](https://github.com/richemslie/gzero_data/tree/master/data/breakthrough)


Amazons and Breakthrough won gold medals at ICGA 2018 Computer Olympiad. :clap: :clap:

LG Champion in last attempt @ Amazons, Breakthrough and Hex 13 (joint).

Hex 13 / Connect6 are currently rated 2nd on active users on Little Golem.

Reversi is also strong relative to humans on LG, yet performs a bit worse than top AB programs (about ntest level 20 the last time I tested).

Also trained Baduk 9x9, it had a rating ~2900 elo on CGOS after 2-3 week of training.

--------------------

The code is in fairly good shape, but could do with some refactoring and
documentation (especially a how to guide on how to train a game).  It would definitely be good to
have an extra pair of eyes on it.  I'll welcome and support anyone willing to try training a game
for themselves.  Some notes:

1. python is 2.7
2. requires a GPU/tensorflow
3. good starting point is https://github.com/richemslie/ggp-zero/blob/dev/src/ggpzero/defs
4. cpp puct/puct2 really needs to be combined.

How to run and install instruction coming soon!


Little Golem
------------
Some games are have had success on the Little Golem website
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



