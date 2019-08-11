What?
=====
galvanise is a [General Game Player](https://en.wikipedia.org/wiki/General_game_playing), where
games are written in [GDL](https://en.wikipedia.org/wiki/Game_Description_Language).  The original
galvanise code was converted to a library [ggplib](https://github.com/richemslie/ggplib) and
galvanise_zero adds AlphaZero style learning.  Much inspiration was from Deepmind's related papers,
and the excellent Expert Iteration [paper](https://arxiv.org/abs/1705.08439). A number of
Alpha*Zero open source projects were also inspirational: LeelaZero and KataGo (XXX add links).

Features
--------
* there is *no* game specific code other than the GDL description of the games, a high level python
configuration file describing GDL symbols to state mapping and symmetries (see
[here](https://github.com/richemslie/galvanise_zero/issues/1) for more information).
* fully automated, put in oven and strong model is baked
* network replaced during training games
* training is very fast using proper coroutines at the C level.  1000s of concurrent games are trained
  using large batch sizes on GPU (for small networks).  Im generally takes 3-5 days in many of the
  trained game types below to become super human strength.
* used same setting for training all games types (cpuct 0.85, fpu 0.25).
* uses smaller number of evaluations (200) than A0, oscillating sampling during training (75% of
  moves are skipped, using much less evals to do so).
* policy squashing and extra noise to prevent overfitting
* models use dropout, global average pooling and squeeze_excite blocks (optional)

See [gzero_bot](http://littlegolem.net/jsp/info/player.jsp?plid=58835) for how to play on Little Golem.


Status
------
Games with significant training, links to elo graphs and models:

* [chess](https://github.com/richemslie/gzero_data/tree/master/data/chess)
* [connect6](https://github.com/richemslie/gzero_data/tree/master/data/connect6)
* [hex13](https://github.com/richemslie/gzero_data/tree/master/data/hexLG13)
* [reversi10](https://github.com/richemslie/gzero_data/tree/master/data/reversi_10x10)
* [reversi8](https://github.com/richemslie/gzero_data/tree/master/data/reversi_8x8)
* [amazons](https://github.com/richemslie/gzero_data/tree/master/data/amazons_10x10)
* [breakthrough](https://github.com/richemslie/gzero_data/tree/master/data/breakthrough)
* [hex11](https://github.com/richemslie/gzero_data/tree/master/data/hexLG11)

Little Golem Champion in last attempts @ Connect6, Hex13, Amazons and Breakthrough, winning all matches.
Retired from further Championships.  Connect6 and Hex 13 are currently rated 1st and 2nd
respectively on active users.

Amazons and Breakthrough won gold medals at ICGA 2018 Computer Olympiad. :clap: :clap:

Reversi is also strong relative to humans on LG, yet performs a bit worse than top AB programs
(about ntest level 20 the last time I tested).

Also trained Baduk 9x9, it had a rating ~2900 elo on CGOS after 2-3 week of training.

--------------------

The code is in fairly good shape, but could do with some refactoring and documentation (especially
a how to guide on how to train a game).  It would definitely be good to have an extra pair of eyes
on it.  I'll welcome and support anyone willing to try training a game for themselves.  Some notes:

1. python is 2.7
2. requires a GPU/tensorflow
3. good starting point is https://github.com/richemslie/ggp-zero/blob/dev/src/ggpzero/defs

How to run and install instruction coming soon!

