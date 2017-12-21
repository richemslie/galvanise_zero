GGP - AlphaZero like learning
==============================
[General Game Playing](https://en.wikipedia.org/wiki/General_game_playing) experiments with
reinforcement learning related to
[AlphaZero](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
and [Thinking Fast And Slow](https://arxiv.org/abs/1705.08439v4).

Everything here is a WIP.

Based on [GGPLib](https://github.com/richemslie/ggplib).


Breakthrough games
------------------
This is the third attempt, where I started completely over from fresh.

The first run was riddled with bugs and misunderstandings, the second was completely unstable due
to bad selection of moves for training, so its 3rd time lucky I hope.  This time the entire process
is 95% automated, and distributed - with little to none hand holding.  After 24 hours of training
using 1 gpu and 6 cores, 25 generations was complete.  That is 155k samples to train the network
on.  I pitched the trained network against Gurgeh - which is a very fast MCTS player (which is
doing 1.2 million tree playouts per move).  The matches gave the players 15 seonds thinking time
per move. The network based PUCT player (which was set at 800 iterations) is only using a 1-3
second of.  It wins quite comfortable:

 * [game 1](http://www.ggp.org/view/all/matches/82745815a8ab7ea9a80be4c03626c04d7608eebb/)
 * [game 2](http://www.ggp.org/view/all/matches/3097bd5b1a64df66d611e612357f7ddf0a802988/)

Note the game is notoriously bad for MCTS players.  Most players in GGP community generally add
heuristics on top of MCTS play and play significantly stronger than the above.  Gurgeh doesn't
employ any special heuristics, so this is a raw MCTS player.  Next up is against galvanise.


Self play
---------
These are some self play games, only using the policy part of the network (greedily taking the most
probable move) - of gen 20 versus gen 25.  Each generation here is basically 5k new games, where
each generation takes approximately 1 hour to generate and train.

 * [gen25 v gen20](http://www.ggp.org/view/all/matches/91d2cf9cefc7075b33152e0127b1f3e7b12aeef1/)
 * [gen20 v gen25](http://www.ggp.org/view/all/matches/dc77c121f3958d2cbefcc75f8430dad8f2b52312/)


gen25 is very aggressive in exchanging pieces, which is an interesting tactic.


Current status
--------------
The trained data is initially sampled from mostly random rollouts.  This is very fast - and used to
just let the network learn the rules.  For each random rollout (may have some extra bells such as
greedily taking wins) - there is no search.  One random sample is taking from each rollout.
Subsequently the real training happens.  The network is trained in generations.  A single sample is
taken from a self play game, where the self play game only uses the policy part of the network with
a random choice of move (moves with higher probability have more chance of being played than moves
with lower probability).  Upon selection of a game state, 800 iterations of a PUCT player is
performed and the distribution of visits on the root node defines a policy distribution to train
on.  Finally the game is then played to the end starting from the selected state - via the policy
part of the network only taking the most probably move at each step.  By generating the data this
way, and by increasing to larger networks as the number of samples increase (ie starting with a
tiny neural network) - and a bunch of other optimisations - all in all sped up training by over a
magnitude over the first couple of attempts.  With the extra bonus that the network does seem to
improve this time around.
