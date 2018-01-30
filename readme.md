ggp-zero
========

recent
------

[reversi training](https://github.com/ggplib/ggp-zero/blob/dev/doc/reversi-record.md).

[Watch live! *Be back soon*!](http://simulated.tech:8800/index.html/)

Post to https://github.com/mokemokechicken/reversi-alpha-zero/issues/40

I have started a recent run.  This is using (ggp-zero)[https://github.com/ggplib/ggp-zero] (
reversi-alpha-zero implementation was inspiration!).  ggp-zero is a generic implementation of a
'zero' method, and can train many different games.  ie By zero I mean start with a random network
and train via self play using (PUCT or variant) MCTS.  However, at this point the implementation
(and goals) are very divirgent from AlphaZero (I also drew inspiration from 'Thinking Fast and
Slow').  For this run, I am running with multiple policies and multiple value heads, with no turn
flipping of the network and no symmetry/rotation of the network.

A previous run achieved approximately ntest level 7, however there were no records.

This time going to keep detailed record [here](https://github.com/ggplib/ggp-zero/blob/dev/doc/reversi-record.md), currently
approximately ntest level 2.



about
------

[General Game Playing](https://en.wikipedia.org/wiki/General_game_playing) with
reinforcement learning, starting from zero.

Current games trained:

 * reversi (strong)
 * breakthrough (strong)
 * cittaceot (solved, network reports 99% probability of first move win)
 * hex
 * connect four
 * checkers-variant
 * escort latched breakthrough (variant) and speedChess (variant)

The trained model for reversi is very strong, and plays at ntest level 7.

Based on [GGPLib](https://github.com/ggplib/ggplib).


roadmap
-------
 * flesh out "zero battleground", visualisation and pretty elo graphs

 * train new variant of escort latched breakthrough - which is super hard for mcts players (and
   since method of learning is similar to mcts, can it learn to play?)

 * add previous states & multiple policy heads

 * lots of refactoring

 * update install instructions.  finish refactoring & quick polish up.  write a little about how it works.  post first working version.


other
-----
* old / early [results](https://github.com/ggplib/ggp-zero/blob/dev/doc/old_results.md).
* old install [instructions](https://github.com/ggplib/ggp-zero/blob/dev/doc/install.md).
