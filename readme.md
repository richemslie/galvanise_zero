ggp-zero
========

[General Game Playing](https://en.wikipedia.org/wiki/General_game_playing) and
reinforcement learning with
[AlphaZero](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
and [Thinking Fast And Slow](https://arxiv.org/abs/1705.08439v4) methods.

Based on [GGPLib](https://github.com/ggplib/ggplib).


roadmap
-------

 * test new fast self play which uses thread/coroutines/c++ to attempt to saturate GPU.

 * network training redux.  add self modifying weighting for value/reward head, better previous
   network, better callbacks and tensorboard.  Still with keras.

 * support more ggp games

   * automate the creation of configs
   * save the configs in json, rather in python classes

 * create a generation object with each network - has options

    * previous states to the neural network (option)
    * add multiple policy heads to neural network (option).
    * the data_format used for convolutions


 * test a non-zero sum game (skirmish variant)

other
-----
* previous results [here](https://github.com/ggplib/ggp-zero/blob/dev/doc/old_results.md).
* old install instructions [here](https://github.com/ggplib/ggp-zero/blob/dev/doc/install.md).
