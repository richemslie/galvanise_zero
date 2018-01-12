
run 6 - Reversi
---------------
Easy wins versus gurgeh, after 2 days of training.  Need better competition as Gurgeh was beat too
easily.


run 5 - part 2
--------------
After disposing of galvanise 40-0

After 5 more generations (25k samples later), I was checking in and noticed how the policy network
looked rather strong - without any search or using the value part of the network.  So once again up
against Gurgeh, using only the policy part of the network - well the results speak for them self.
Not perfect play - there a few kamikaze moves, but some interesting tactical play that the network
has learned.

Gurgeh is running 3 threads and for 10 seconds per turn.  That is approximately 1.5 million MCTS
playouts/simulations per second.

http://www.ggp.org/view/all/matches/46c69cd7a742cb2c868f76215c8d9321e985eaf5/
http://www.ggp.org/view/all/matches/f138fa208e916e98b87d6daf1b571a2a7c32a7b7/


run 5
-----
Starting training from scratch again on Breakthrough.  This time it was **zero** - ie no prior data
from other any other means was given.  That is start with random weights on the neural network and
let it evolve. The results weren't overly bad.

However, very early on there was a huge bias towards the black player and the scores are about 80%
in favour when they should be 50%.  I tried doing more simulations per sample and increasing the
temperature/randomness of move selections but it didn't want to pull back despite doing another
100k samples.  There was a total of 155k samples for the entire run. It takes 2 seconds per sample
doing on average 2300 network predictions for each sample.

This is actually very slow - benchmarks show that the GPU can handle 20k predictions per second and
I have a newer card I got for Christmas which would be twice as fast.  Indeed monitoring the GPU
card showed it getting 6% saturation and 2300/20000 is about 6%.  This was all despite my best
efforts to optimise the python code and run 128 concurrent games and batch the predictions on the
GPU (it did speed things about 2-4 fold).


run 4
-----
This time I tried Reversi (Othello).  Unfortunately it just barely learned the legal moves and
didn't conclusively beat a random player.  Was very, very slow to train.


Breakthrough games - run 3
--------------------------
The first run was riddled with bugs and misunderstandings, the second was completely unstable due
to bad selection of moves for training, so its 3rd time lucky I hope.  This time the entire process
is 95% automated, and distributed - with little to none hand holding.  After 24 hours of training
using 1 gpu and 6 cores, 25 generations was complete.  That is 155k samples (50k were provided from
random rollouts - see below) to train the network on.  I pitting the trained network against
Gurgeh - which is a very fast MCTS player (which is doing 1.2 million tree playouts per move).  The
matches gave the players 15 seconds thinking time per move. The network based PUCT player (which
was set at 800 iterations) is only using a 1-3 second of.  It wins quite comfortable:

 * [game 1](http://www.ggp.org/view/all/matches/82745815a8ab7ea9a80be4c03626c04d7608eebb/)
 * [game 2](http://www.ggp.org/view/all/matches/3097bd5b1a64df66d611e612357f7ddf0a802988/)

Note the game is notoriously bad for MCTS players.  Most players in GGP community generally add
heuristics on top of MCTS play and play significantly stronger than the above.  Gurgeh doesn't
employ any special heuristics, so this is a raw MCTS player.  Next up is against galvanise.


Self play - run 3
-----------------
These are some self play games, only using the policy part of the network (greedily taking the most
probable move) - of gen 20 versus gen 25.  Each generation here is basically 5k new games, where
each generation takes approximately 1 hour to generate and train.

 * [gen25 v gen20](http://www.ggp.org/view/all/matches/91d2cf9cefc7075b33152e0127b1f3e7b12aeef1/)
 * [gen20 v gen25](http://www.ggp.org/view/all/matches/dc77c121f3958d2cbefcc75f8430dad8f2b52312/)


gen25 is very aggressive in exchanging pieces, which is an interesting tactic.


Run 3 comments
--------------
The trained data is initially sampled from mostly random rollouts (50k).  This is very fast - and
used to just let the network learn the rules.  For each random rollout (may have some extra bells
such as greedily taking wins) - there is no search.  One random sample is taking from each rollout.

Subsequently the real training happens.  The network is fitted in generations.  A single sample is
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
