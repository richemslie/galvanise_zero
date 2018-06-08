gzero
=====
gzero provides a framework for neural networks to learn to play solely based on self-play.  This
is loosely based on the papers of AlphaZero and exIT, and a number of zero open source were greatly
inspirational.

The name gzero stems from the fact that this project was initially a spin off my galvanise player
in GGP.


project goal(s)
---------------
The initial goal of this project was to be able to train any game in
[GGP](https://en.wikipedia.org/wiki/General_game_playing) ecosystem, to play at a higher level than
the state of the art GGP players, given some (relatively) small time frame to train the game (a few
days to a week, on commodity hardware - and not months/years worth of training time on hundreds of
GPUs).

I feel this goal has been somewhat achieved for zero sum games, where the game state can easily be
represented as a 2D array of channels.

Some games have been very friendly to such an approach, such as Breakthrough which has achieved
super human and super bot level.

Although many games have been trainined, there is a multitude of games left to try.  There are some
game types which are completely unsupported right now, for starters:

* non-zero board games (such as non zero sum variant of Skirmish)
* multiplayer games
* games that are not easily represented as a 2D array of channels
* simultaneous games
* single player games (puzzles)


Little Gollem
-------------
Search for user
[gzero_bot](http://littlegolem.net/jsp/info/player.jsp?plid=58835) on Little Golem if you would
like to challenge it to a game of

 * Breakthrough
 * Reversi 10x10
 * Reversi 8x8
 * Hex 13x13
 * Hex 11x11
 * Amazons 10x10

Breakthrough and Reversi have retired from any further competitions on LG.  It is still possible to
play these games via an invite.
