ggp-zero
========

recent
------

[reversi training v10](https://github.com/ggplib/ggp-zero/blob/dev/doc/reversi_record.md).
[reversi training v9](https://github.com/ggplib/ggp-zero/blob/dev/doc/reversi_record_v9.md).

[Watch live! *Be back soon*!](http://simulated.tech:8800/index.html/)

Post to https://github.com/mokemokechicken/reversi-alpha-zero/issues/40

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
 * create PUCT+ evaluator, bunch of the galvanise ideas carried over to PUCT MCTS

 * fix bases transformer to use a game description, and not harded coded classes.  add enough
   support to transform chess & skirmish & tron.

 * flesh out "zero battleground", visualisation and pretty elo graphs

 * train new variant of escort latched breakthrough - which is super hard for mcts players (and
   since method of learning is similar to MCTS, can it learn to play?  Without guidance MCTS will
   score 50/50.)

 * train a non-zero sum game (skirmish), and a simultaneous game (tron)

 * train chess!

 * docs.  update install instructions.  finish refactoring & quick polish up.  write a little about
   how it works.  post first working version.


other
-----
* old / early [results](https://github.com/ggplib/ggp-zero/blob/dev/doc/old_results.md).
* old install [instructions](https://github.com/ggplib/ggp-zero/blob/dev/doc/install.md).
