GGP - AlphaZero like learning
==============================

[General Game Playing](https://en.wikipedia.org/wiki/General_game_playing) experiments with
reinforcement learning related to
[AlphaZero](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
and [Thinking Fast And Slow](https://arxiv.org/abs/1705.08439v4).

Just started - WIP.  Lots of hand holding going on.

Based on [GGPLib](https://github.com/richemslie/ggplib).


Breakthrough games
------------------
The first attempt was to try learning the game Breakthrough.

After 10 generations, with bugs and misunderstandings in the implementation along the way, it can
now beat a vanilla MCTS player which is doing ~250k playouts per turn.  The trained network based
player in the same time frame does only ~400 'iterations' (running on CPU).  That is roughly 3
orders of magnitude difference in playouts.

It does however play similar to the MCTS players (and MCS player it was initially trained on) -
which is very kamikaze like, throwing away pawns for no reason.  I am hoping after a bunch more
training generations, it finally figures out that this is not the greatest idea.

However, what is interesting in these games is - at the end of the game and throwing away 60% of its pawns,
it seems to always have a nice base to protect itself whereas the MCTS player is wide open.

Here are the last four games (with gen8, currupted gen9 network with bad PUCT values)

 * [game 1](http://www.ggp.org/view/all/matches/a8468283b34055be6e315951499d57d7af21fa67/)
 * [game 2](http://www.ggp.org/view/all/matches/7fb8f051dd46d51bd491bcb28f66d7629344e1fd/)
 * [game 3](http://www.ggp.org/view/all/matches/db0f4ce99613445ddcf89b12b160d1e58686975e/)
 * [game 4](http://www.ggp.org/view/all/matches/a67ea10203cf74b367f7f4e16dfaa3c7923a21af/)


Note the game is notoriously bad for MCTS players.  Most players in GGP community generally add
heuristics on top of MCTS play and play significantly stronger than the above.


Self play
---------
These are some self play games, with only 42 iterations.  In other words it plays according to what
the policy network has learned.  Admittedly not brilliant, but it has improved.

 * [gen8 v gen2](http://www.ggp.org/view/all/matches/a0d60b298799ea9be497ae54b719acbc0316a365/)
 * [gen2 v gen8](http://www.ggp.org/view/all/matches/b761a1029b667935fe13c988246a0bf5bc9d19c6/)


Current status
--------------
Neural network is general to support many games.  There is a combination of
[GDL](http://alloyggp.blogspot.co.uk/2012/12/the-game-description-language.html) 'base' states as
inputs to the network.

The coordinate looking base-states are turned into planes and fed into a residual part of the
network.  The non-coordinate base-states go through a single fully connected layer.  GDL does
specifiy whether they are coordinates or not, and for now these are hard coded.  It shouldn't be
too hard to infer them.

Both part os the network are combined and fed to two outputs heads: a single list of moves and the
final score of the game (similar to policy and score in AlphaZero).

The trained data is initially sampled from self play of a dumb Monte Carlo search player.  Each
turn takes 0.25 seconds and it was enough to prime the network and learn the rules of the game.
This differs from AlphaZero which skips right to the chase and starts from a completely random
network.  One thing that seems apparent is that after 10 generations it still plays similar to this
dumb player, and feels like it is caught in a local optima right from the get go.  Next time around
I plan to start from a random neural network.

Subsequently the network is trained in generations.  A single sample is taken from a self play game
using Monte Carlo playouts with the policy and score values, using PUCT and Dirichlet noise for
exploration.  This is played approximately, using different configurations of a Monte Carlo base
player to arrive at a single state to evaluate thoroughly (800 iterations) for the policy part.
Then it plays to the end without noise to come up with a final score.  This cuts down in computation
significantly and is a clever hack (credit ThinkingFastSlow).

A generation is made up of ~10k of these samples and then the neural network retrains from scratch.


Comments
--------
The hyperparameters are very sensitive and getting these right for playing and training seems quite
hard.

I've no idea what to do with duplicate states.  Should later generations be allowed to update with
a better quality policies and scores - and by replace can just append to the buffer.  What about
duplicate states in the same generation.  Also keeping these in sync across machines is a pain.

PUCT constant is the new UCT constant.  Can never find that quite the right value for all
circumstances.  Hence why I made it auto tuning for each node in galvanise.  And I just found the
players don't play well at all without dirichlet noise.  Which means my training player has been
producing basically rubbish.  Initialising each edge to be zero, exploration will never happen, and
dirichlet noise will at least release some of those nodes.

