ggp-zero
========

After adding the new game description / bases transformer (which allows for many more games without
missing vital information), will be slowing down any new developments for a while (mostly for my
own sanity).  The idea is to go back and write some docs, fix config options to actually reflect
what they say, minor refactors and cleanups.  Not exactly aiming for 'release', but more of a
sanitize at this point.


roadmap
-------
 * docs.  update install instructions.  finish refactoring & quick polish up.  write a little about
   how it works.

 * train new variant of escort latched breakthrough - which is super hard for mcts players (and
   since method of learning is similar to MCTS, can it learn to play?  Without guidance MCTS will
   score 50/50.)

 * test a simultaneous game (tron) (this needs new PUCT+ evaluator)

 * create PUCT+ evaluator, shared tree between self plays and try a new self play idea.

 * chess and chess variant games give ggpzero a very hard time.  I currently run out of memory
   after 200k samples (very large policies) and saving the data as json can cause pauses of up to 5
   minutes.  chess itself looks like as if it will be extremely challenging to get any traction, I
   think it will need 2-3 million samples with a random network to save going into local minima of
   worse than random self play (which will end up as draw).  Need performance fixes for the above
   before retrying.

 * flesh out "zero battleground", visualisation and pretty elo graphs


recent
------
[reversi training](https://github.com/ggplib/ggp-zero/blob/dev/doc/reversi_record.md).
Post to https://github.com/mokemokechicken/reversi-alpha-zero/issues/40

[Watch live! *Be back sometime hopefully*!](http://simulated.tech:8800/index.html/)


about
------
[General Game Playing](https://en.wikipedia.org/wiki/General_game_playing) with
reinforcement learning, starting from zero.

Current games trained:

 * reversi
 * breakthrough
 * breakthroughSmall
 * cittaceot
 * hex
 * connect four
 * checkers-variant
 * escort latched breakthrough (variant) and speedChess (variant)

Based on [GGPLib](https://github.com/ggplib/ggplib).


other
-----
* old / early [results](https://github.com/ggplib/ggp-zero/blob/dev/doc/old_results.md).
* old install [instructions](https://github.com/ggplib/ggp-zero/blob/dev/doc/install.md).
