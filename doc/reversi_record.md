reversi generation 0
====================

restarted again!


Results
-------
results @ 30th January

| player     | opponent   | black_result   | white_result   |
|:-----------|:-----------|:---------------|:---------------|
| gzero_5_1  | random     | 3/2/0          | 3/2/0          |
| gzero_10_1 | random     | 5/0/0          | 3/2/0          |
| gzero_10_1 | pymcs      | 0/2/0          | 0/2/0          |
| gzero_15_2 | pymcs      | 2/3/0          | 1/2/1          |
| gzero_20_1 | random     | 5/0/0          | 5/0/0          |
| gzero_20_2 | pymcs      | 3/1/1          | 2/2/0          |
| gzero_20_4 | simplecmts | 2/0/0          | 1/1/0          |
| gzero_30_1 | pymcs      | 5/0/0          | 4/1/0          |
| gzero_30_4 | ntest_1    | 5/0/0          | 4/1/0          |
| gzero_30_4 | ntest_2    | 0/6/4          | 1/9/0          |
| gzero_30_8 | ntest_2    | 1/4/0          | 1/4/0          |
| gzero_35_1 | pymcs      | 5/0/0          | 4/1/0          |
| gzero_35_1 | simplecmts | 4/1/0          | 3/2/0          |
| gzero_35_2 | simplecmts | 5/0/0          | 4/1/0          |
| gzero_35_4 | ntest_1    | 5/0/0          | 4/1/0          |
| gzero_35_4 | ntest_2    | 0/5/0          | 1/4/0          |
| gzero_35_8 | ntest_2    | 2/3/0          | 3/2/0          |
| gzero_40_4 | ntest_2    | 1/4/0          | 3/2/0          |



Plan
----
20 generations with 2 samples per game - 10k each, 200k total

15 generations with 4 samples per game - 40k each, 600k total
15 generations with 6 samples per game - 40k each. 600k total

15 generations with 8 samples per game - 60k each. 800k total
15 generations with 10 samples per game - 60k each. 800k total


major features testing
----------------------
 * multiple policies
 * 3 previous states
 * no drop out on policy heads
 * resampling from generations each epoch (bucket/percent based)


methodology
-----------
Start with a small network, and progresively increase the size (manually).
Also manually reduce resign_false_positive_retry_percentage.

At the end of every 5th/10th generation, perform an evaluation against opponents.


evaluation configuration
------------------------
No dirichlet noise, small amount of random choice of action.

 * gzero-1 : 100 sims per turn.
 * gzero-2 : 200 sims per turn.
 * gzero-4 : 400 sims per turn.
 * gzero-8 : 800 sims per turn.


opponents
---------
* random
* pymcs - a monte carlo player, with no tree.  number iterations 4k per move
* simplecmts - a vanilla mcts player.  number iterations 4k per move
* gurgeh - a multithreaded fast mcts player, with some extra bells&whistles.
  number iterations 40k per move
* ntest lvl 1-20


problems anticipated
--------------------
Without dropout on policy heads, I imagine it will overfit really fast.  As training the network is
using early stopping with accuracy of the policy, we may not be training as long as we used to.


configuration
=============

initial
-------

```json

{
        "base_generation_description": {
            "channel_last": false,
            "game": "reversi",
            "multiple_policy_heads": true,
            "name": "v9_0",
            "num_previous_states": 3,
        },
        "base_network_model": {
            "cnn_filter_size": 64,
            "cnn_kernel_size": 3,
            "dropout_rate_policy": -1,
            "dropout_rate_value": 0.5,
            "input_channels": 10,
            "input_columns": 8,
            "input_rows": 8,
            "l2_regularisation": false,
            "leaky_relu": false,
            "multiple_policies": true,
            "policy_dist_count": [
                65,
                65
            ],
            "residual_layers": 8,
            "role_count": 2,
            "value_hidden_size": 64
        },
        "base_training_config": {
            "batch_size": 256,
            "compile_strategy": "adam",
            "drop_dupes_count": 5,
            "epochs": 20,
            "game": "reversi",
            "generation_prefix": "v9",
            "learning_rate": null,
            "max_epoch_samples_count": 250000,
            "max_sample_count": 1000000,
            "next_step": 20,
            "overwrite_existing": false,
            "starting_step": 0,
            "use_previous": true,
            "validation_split": 0.9
        },
        "checkpoint_interval": 120.0,
        "current_step": 0,
        "game": "reversi",
        "generation_prefix": "v9",
        "max_samples_growth": 0.8,
        "num_samples_to_train": 10000,
        "port": 9000,
        "self_play_config": {
            "max_number_of_samples": 2,
            "resign_false_positive_retry_percentage": 0.99,
            "resign_score_probability": 0.02,
            "sample_iterations": 800,
            "sample_puct_config": {
                "choose": "choose_temperature",
                "depth_temperature_increment": 0.2,
                "depth_temperature_max": 5.0,
                "depth_temperature_start": 6,
                "depth_temperature_stop": 100,
                "dirichlet_noise_alpha": 0.15,
                "dirichlet_noise_pct": 0.25,
                "puct_before_expansions": 3,
                "puct_before_root_expansions": 5,
                "puct_constant_after": 0.75,
                "puct_constant_before": 3.0,
                "random_scale": 0.6,
                "root_expansions_preset_visits": 1,
                "temperature": 1.0,
            },
            "score_iterations": 100,
            "score_puct_config": {
                "choose": "choose_top_visits",
                "dirichlet_noise_alpha": 0.15,
                "dirichlet_noise_pct": 0.25,
                "puct_before_expansions": 3,
                "puct_before_root_expansions": 5,
                "puct_constant_after": 0.75,
                "puct_constant_before": 3.0,
                "root_expansions_preset_visits": 1,
            },
            "select_iterations": 0,
            "select_puct_config": {
                "choose": "choose_temperature",
                "depth_temperature_increment": 0.1,
                "depth_temperature_max": 3.0,
                "depth_temperature_start": 30,
                "depth_temperature_stop": 100,
                "puct_before_expansions": 3,
                "puct_before_root_expansions": 5,
                "puct_constant_after": 0.75,
                "puct_constant_before": 3.0,
                "random_scale": 0.75,
                "temperature": 1.0,
            },
        },
}
```

gen 15 changes
--------------

actually resign a bit

* self_play_config.resign_false_positive_retry_percentage = 0.8
* self_play_config.resign_score_probability = 0.05

up the size of the network (to small)

* conf.cnn_filter_size = 96
* conf.value_hidden_size = 128

ignore the first 3 steps (mostly noise)

* base_training_config.starting_step = 3


gen 18 changes
--------------
introduced buckets for training

* remove base_training_config.max_epoch_samples_count
* base_training_config.resample_buckets = [[5, 1.0], [10, 0.8], [15, 0.6], [20, 0.4], [0, 0.2]]
* base_training_config.drop_dupes_count = 7


gen 20 changes
--------------
4 samples per game

* num_samples_to_train = 40000
* self_play_config.max_number_of_samples = 4

less false positive resigning tests

* self_play_config.resign_false_positive_retry_percentage = 0.6


gen 24 changes
--------------
increase network (medium-small), lower false positive retry

* conf.cnn_filter_size = 112
* self_play_config.resign_false_positive_retry_percentage = 0.5


gen 35 changes
--------------
increase network (medium), resign earlier, more samples per game

* conf.cnn_filter_size = 128
* conf.value_hidden_size = 192
* self_play_config.resign_score_probability = 0.075
* self_play_config.max_number_of_samples = 6

with the resampling, drop more at the tail
* base_training_config.resample_buckets = [[5, 1.0], [10, 0.8], [15, 0.6], [20, 0.4], [25, 0.2], [0, 0.1]]
* base_training_config.starting_step = 5


gen 38 issues/changes
---------------------
We are barely running any epochs like we prior to multiple policies.  I don't want to change the
early stopping logic mid run.  And I really didn't want to make any massive changes to the network
structure either.  However, playing around with some configurations - decided to add in leaky relus
and increase the batch_size during training to 1024.  Just going to go with that.  If it becomes
unstable, then we abandon the run.

* base_network_model.config.leaky_relu = True
* base_training_config.batch_size = 1024
* base_training_config.starting_step = 5
* base_training_config.resample_buckets =  [[5, 1.0], [15, 0.8], [25, 0.6], [35, 0.4], [45, 0.2], [0, 0.1]]


raw results json
----------------

```json
{ "results" : [
              [ ["gzero", 5, 1], ["random", -1, -1], [3, 2, 0], [3, 2, 0] ],

              [ ["gzero", 10, 1], ["random", -1, -1], [5, 0, 0], [3, 2, 0] ],
              [ ["gzero", 10, 1], ["pymcs", -1, -1], [0, 2, 0], [0, 2, 0] ],

              [ ["gzero", 15, 2], ["pymcs", -1, -1], [2, 3, 0], [1, 2, 1] ],

              [ ["gzero", 20, 1], ["random", -1, -1], [5, 0, 0], [5, 0, 0] ],
              [ ["gzero", 20, 2], ["pymcs", -1, -1], [3, 1, 1], [2, 2, 0] ],
              [ ["gzero", 20, 4], ["simplecmts", -1, -1], [2, 0, 0], [1, 1, 0] ],

              [ ["gzero", 30, 1], ["pymcs", -1, -1], [5, 0, 0], [4, 1, 0] ],
              [ ["gzero", 30, 4], ["ntest", 1, -1], [5, 0, 0], [4, 1, 0] ],
              [ ["gzero", 30, 4], ["ntest", 2, -1], [0, 6, 4], [1, 9, 0] ],
              [ ["gzero", 30, 8], ["ntest", 2, -1], [1, 4, 0], [1, 4, 0] ],

              [ ["gzero", 35, 1], ["pymcs", -1, -1], [5, 0, 0], [4, 1, 0] ],
              [ ["gzero", 35, 1], ["simplecmts", -1, -1], [4, 1, 0], [3, 2, 0] ],
              [ ["gzero", 35, 2], ["simplecmts", -1, -1], [5, 0, 0], [4, 1, 0] ],

              [ ["gzero", 35, 4], ["ntest", 1, -1], [5, 0, 0], [4, 1, 0] ],
              [ ["gzero", 35, 4], ["ntest", 2, -1], [0, 5, 0], [1, 4, 0] ],
              [ ["gzero", 35, 8], ["ntest", 2, -1], [2, 3, 0], [3, 2, 0] ]

              [ ["gzero", 40, 4], ["ntest", 2, -1], [1, 4, 0], [3, 2, 0] ]
              ] }

```
