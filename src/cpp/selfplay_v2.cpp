/*

Idea for self play 2.

First off - switch to puctplus.  This gives us a shared MCTS tree, with transpositions, and
traversals on children (need traversals if using transpositions - currently puct evaluator uses
visits on nodes).

roll: At the start of each roll will clear the tree.

Every self play does 100 iterations per move, up to n moves.  (n=10 in reversi?)

Then go back to the root and select a starting point for this by walking the tree (using select
parameters).  And return the first node it finds where:

 * node depth > n OR has less "evaluations/iterations" than 800
 (or whatever it is configured to).

Then evaluate as per normal and create samples, all the way to the end of game (or until it
resigns).  Perform backprop as per normal - no need to backprop all the way up the tree.  The
selection process next time around will take advantage of any nicely fleshed out downwind nodes,
and update itself accordingly.

At roll time, go through all nodes in tree from depth from 0-n, and emit a sample for any nodes
over 800 (or what ever sample_iterations is set to).  For the reward, return the mcts score (will
be most accurate).  Alternatively could clamp the value.

Then we clear the tree for the next nueral network (goto roll).  If feeling brave, we could
possibly not clear the tree between every generation - however, would run into issues where the
tree wouldn't reflect the current network's evaluations.

----

Conceptually, the idea is that we avoid massive amounts of dupes for the first n moves.  The
selection process should effectively choose an interesting starting point, whereas the current way
is pretty much nonsense.

Most importantly, it eliminates bad rewards due to random selection for a sample.  Since samples
will be for the most part taking top visits, then this will give the most accurate score.

Currently, the number of players per self play manager is 1024, this would give 10k iterations on
the first move.  Since could do more than one self play per generation, this could grow well beyond
10k.

*/
