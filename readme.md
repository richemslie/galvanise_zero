ggp-zero
========

[General Game Playing](https://en.wikipedia.org/wiki/General_game_playing) and
reinforcement learning with
[AlphaZero](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
and [Thinking Fast And Slow](https://arxiv.org/abs/1705.08439v4) methods.

Based on [GGPLib](https://github.com/ggplib/ggplib).

Current games in training:

 * connect four
 * breakthrough
 * reversi
 * hex

connect four / breakthrough and reversi are all a good bit stronger than baseline MCTS player - which does 3-4 orders of magnitude in
playouts that ggp-zero.

hex is in early stages of training.

roadmap
-------
 * add evaluator stage 'game master', with visualisation and pretty elo graphs

 * train c4/bt/hex/reversi in parallel (cycle a generation each)

 * need some other players to judge progress (other alpha-zero projects?)

 * update install instructions.  finish refactoring & quick polish up.  write a little about how it works.  post first working version.

 * add a non-zero sum game to the mix (skirmish variant)

 * experiment with adding previous states

 * experiment with multiple policy heads

 * reuse network from skirmish variant for chess (just to try something different)


other
-----
* previous results [here](https://github.com/ggplib/ggp-zero/blob/dev/doc/old_results.md).
* old install instructions [here](https://github.com/ggplib/ggp-zero/blob/dev/doc/install.md).
