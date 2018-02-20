reversi training - x4
=====================

results
-------
* gzero_x_y - where x is generation, y is number of model predictions per move * 100.
* gzero_t_x_y - gzero tiered.  This a variation of gzero that applies a tiered multiplier to the
  number of evaluation as the game approaches the end (*2 at move 38, *4 at move 45).

* ntest_x - where x is nboard depth/level
* results - win/loss/draw


See below for some extra notes on results.


| player       | opponent   | overall   | black   | white   |
|:-------------|:-----------|:----------|:--------|:--------|
| gzero_t_80_8 | ntest_13   | 9/7/4     | 1/6/3   | 8/1/1   |
| gzero_t_80_8 | ntest_11   | 5/3/2     | 1/2/2   | 4/1/0   |
| gzero_t_78_8 | ntest_11   | 8/1/1     | 4/0/1   | 4/1/0   |
| gzero_77_8   | ntest_9    | 4/2/0     | 3/0/0   | 1/2/0   |
| gzero_77_8   | ntest_9    | 4/2/0     | 3/0/0   | 1/2/0   |
| gzero_75_8   | ntest_7    | 4/0/2     | 2/0/1   | 2/0/1   |
| gzero_70_8   | ntest_11   | 2/5/1     | 1/3/0   | 1/2/1   |
| gzero_70_8   | ntest_9    | 4/4/0     | 4/0/0   | 0/4/0   |
| gzero_70_8   | ntest_7    | 3/5/0     | 2/2/0   | 1/3/0   |
| gzero_70_8   | ntest_5    | 8/0/0     | 4/0/0   | 4/0/0   |
| gzero_62_8   | ntest_11   | 3/5/0     | 0/4/0   | 3/1/0   |
| gzero_62_8   | ntest_9    | 5/5/0     | 0/5/0   | 5/0/0   |
| gzero_62_8   | ntest_7    | 6/4/0     | 3/2/0   | 3/2/0   |
| gzero_61_8   | ntest_7    | 9/11/0    | 1/9/0   | 8/2/0   |
| gzero_60_8   | ntest_7    | 3/6/1     | 3/2/0   | 0/4/1   |
| gzero_54_8   | ntest_7    | 3/5/0     | 0/4/0   | 3/1/0   |
| gzero_53_8   | ntest_7    | 0/7/1     | 0/4/0   | 0/3/1   |
| gzero_53_8   | ntest_6    | 2/5/1     | 1/2/1   | 1/3/0   |
| gzero_52_8   | ntest_6    | 1/7/0     | 1/3/0   | 0/4/0   |
| gzero_52_8   | ntest_5    | 6/1/1     | 3/1/0   | 3/0/1   |
| gzero_51_8   | ntest_5    | 1/6/1     | 1/2/1   | 0/4/0   |
| gzero_51_8   | ntest_4    | 3/5/0     | 2/2/0   | 1/3/0   |
| gzero_50_8   | ntest_3    | 7/3/0     | 4/1/0   | 3/2/0   |
| gzero_35_8   | ntest_1    | 10/0/0    | 5/0/0   | 5/0/0   |
| gzero_20_8   | ntest_1    | 6/4/0     | 2/3/0   | 4/1/0   |


----


| player       | opponent   | overall   | black   | white   |
|:-------------|:-----------|:----------|:--------|:--------|
| gzero_80_1   | ntest_7    | 5/5/0     | 4/1/0   | 1/4/0   |
| gzero_80_1   | ntest_5    | 5/5/0     | 1/4/0   | 4/1/0   |
| gzero_80_1   | ntest_4    | 5/5/0     | 0/5/0   | 5/0/0   |
| gzero_80_1   | ntest_3    | 5/3/2     | 0/3/2   | 5/0/0   |
| gzero_69_1   | ntest_7    | 0/7/1     | 0/4/0   | 0/3/1   |
| gzero_69_1   | ntest_5    | 3/5/0     | 2/2/0   | 1/3/0   |
| gzero_69_1   | ntest_4    | 1/6/1     | 1/3/0   | 0/3/1   |
| gzero_69_1   | ntest_3    | 5/1/2     | 2/0/2   | 3/1/0   |
| gzero_57_1   | ntest_3    | 3/8/1     | 1/5/0   | 2/3/1   |
| gzero_57_1   | ntest_2    | 6/1/3     | 2/0/3   | 4/1/0   |
| gzero_53_1   | ntest_2    | 5/3/0     | 3/1/0   | 2/2/0   |
| gzero_52_1   | ntest_2    | 4/4/0     | 2/2/0   | 2/2/0   |
| gzero_52_1   | ntest_1    | 8/0/0     | 4/0/0   | 4/0/0   |
| gzero_20_1   | simplemcts | 10/0/0    | 5/0/0   | 5/0/0   |
| gzero_20_1   | pymcs      | 10/0/0    | 5/0/0   | 5/0/0   |


notes
-----

1. In the most recent results, there are number of exactly the same matches played.  There isn't
   much I can do to prevent this currently.  The way it decides upon the move to be played, is by
   randomly selecting from the top 50% of moves in the probability distribution over the visits
   after n MCTS evaluations.  There is also a temperature of 2 applied, after the first few moves.
   A large number of moves in the game have to top move with > 50% of the distribution, so it ends
   up similar to taking the top move.

   The temperature is 4 for gzero_1_xxx, as not enough playouts to form a stable distribution.

   XXX I should change it to at least select the first move with equal probability - so that can test
   the symmetry of the learned model (this is AlphaZero where no the data is not augmented with
   symmetry/rotation).


2. The tiered version doesn't add much overall time difference to the entire match.  Approximately
   around move 45 we start hitting terminal nodes - that don't need to evaluated.  The c++ code is
   extremely fast in this case, and can perform 1-2 million tree search per second (versus about
   100 per second with model evaluations).

   Currently this number of playouts is limited in we start hitting the same terminal node
   repeated - it is something in the region of 16 * max number evaluations.


3. Alpha*Zero evaluation uses 10k of evaluations per move.  It would interesting to try this, but
   doing 1 evaluation per playout is very slow due to the latency of keras/tensorflow/pcie/gpu.
   Increasing the batch size is one approach, but needs some tweaking of the network scheduler c++
   code.


4. As of gen 81, the number of samples in the epoch is close to 3 million.  And getting pretty
   close to running out of RAM.  Will need to use train_on_batch() from keras and read the data
   from disk.  Although I also thinking about switching to pytorch for extra control, and using
   caffe2 for evaluations in c++ (or at least have the option to do so - the model library is
   already quite well abstracted out).


previous runs
-------------
The names are arbitrary.

* [x2](https://github.com/ggplib/ggp-zero/blob/dev/doc/reversi_record_x2.md).
* [v9](https://github.com/ggplib/ggp-zero/blob/dev/doc/reversi_record_v9.md).
