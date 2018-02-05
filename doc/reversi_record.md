reversi training
================

Started on 2nd February.


finished
--------
5th February.  Ran 400k more samples (4 more gens) after gen 50, and it started
stagnating/regressing, and don't really have the time to figure out why.
So that's it folks for now.  Not too bad result.


results
-------
* gzero_x_y - where x is generation, y is number of playouts per move * 100.

* ntest_x - where x is nboard depth/level
* results - win/loss/draw


| player     | opponent   | black_result   | white_result   |
|:-----------|:-----------|:---------------|:---------------|
| gzero_50_8 | ntest_10   | 3/2/0          | 0/5/0          |
| gzero_50_8 | ntest_9    | 0/5/0          | 0/5/0          |
| gzero_50_8 | ntest_8    | 1/0/4          | 5/0/0          |
| gzero_50_8 | ntest_7    | 4/5/1          | 6/4/0          |
| gzero_50_8 | ntest_5    | 1/3/1          | 5/0/0          |
| gzero_50_4 | ntest_7    | 1/4/0          | 0/5/0          |
| gzero_50_4 | ntest_5    | 1/3/1          | 1/4/1          |
| gzero_50_4 | ntest_3    | 4/0/1          | 4/0/1          |
| gzero_50_2 | ntest_5    | 0/5/0          | 0/4/1          |
| gzero_50_2 | ntest_4    | 4/1/0          | 2/2/1          |
| gzero_50_2 | ntest_3    | 4/1/0          | 3/2/0          |
| gzero_50_2 | ntest_2    | 2/3/0          | 2/3/0          |
| gzero_50_1 | ntest_3    | 1/4/0          | 3/2/0          |
| gzero_50_1 | ntest_2    | 2/3/0          | 1/3/1          |
| gzero_50_1 | ntest_1    | 5/0/0          | 4/1/0          |
| gzero_36_8 | ntest_7    | 0/4/0          | 0/4/0          |
| gzero_36_8 | ntest_5    | 3/1/0          | 1/3/0          |
| gzero_36_8 | ntest_4    | 1/3/0          | 1/3/0          |
| gzero_36_8 | ntest_3    | 4/0/0          | 2/1/1          |
| gzero_36_4 | ntest_4    | 0/4/0          | 1/3/0          |
| gzero_36_4 | ntest_3    | 1/3/0          | 1/3/0          |
| gzero_36_4 | ntest_2    | 4/0/0          | 4/0/0          |
| gzero_36_2 | ntest_2    | 4/0/0          | 4/0/0          |
| gzero_36_2 | ntest_1    | 8/2/0          | 7/3/0          |
| gzero_36_1 | ntest_1    | 3/1/0          | 2/2/0          |
| gzero_30_8 | ntest_2    | 1/4/0          | 3/2/0          |
| gzero_30_4 | ntest_2    | 0/5/0          | 1/4/0          |
| gzero_30_4 | ntest_1    | 4/1/0          | 4/1/0          |
| gzero_30_1 | simplemcts | 5/0/0          | 5/0/0          |
| gzero_15_2 | simplemcts | 4/1/0          | 5/0/0          |
| gzero_12_2 | pymcs      | 4/1/0          | 5/0/0          |
| gzero_5_2  | pymcs      | 3/2/0          | 2/3/0          |


gzero evaluation similar configuration to sample_puct_config, with no dirichlet noise.

updated 4th February



opponents
---------

* random - completely random player
* pymcs - a monte carlo player, with no tree.  number iterations 4k per move
* simplecmts - a vanilla MCTS player.  number iterations 4k per move
* gurgeh - a multithreaded fast MCTS player, with some extra bells&whistles.
  number iterations 40k per move
* ntest lvl 1-20 - [ntest](https://github.com/weltyc/ntest)


plan
----
(very rough)

* 20 generations with 4 samples per game - 20k each, 400k total
* 20 generations with 8 samples per game - 40k each, 800k total
* 20 generations with 20 samples per game - 60k each, 800k total


environment
-----------
* Linux Ubuntu.
* Self play & training mostly on single 1080ti with some help from a 1080.  CPU i7-6900K.


methodology
-----------
Start small, and go back to configuration that (somewhat) worked.  Major differences

 * 1 previous state
 * add drop out on policy heads


configuration
=============

initial
-------

```json

{
  "base_generation_description": {
    "channel_last": false,
    "date_created": "2018\/02\/02 10:20",
    "game": "reversi",
    "multiple_policy_heads": true,
    "name": "x2_0",
    "num_previous_states": 1,
  },
  "base_network_model": {
    "cnn_filter_size": 64,
    "cnn_kernel_size": 3,
    "dropout_rate_policy": 0.25,
    "dropout_rate_value": 0.5,
    "input_channels": 6,
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
    "batch_size": 128,
    "compile_strategy": "adam",
    "drop_dupes_count": 3,
    "epochs": 20,
    "game": "reversi",
    "generation_prefix": "x2",
    "learning_rate": null,
    "max_sample_count": 300000,
    "next_step": 0,
    "overwrite_existing": false,
    "resample_buckets": [
      [
        100,
        1.0
      ]
    ],
    "starting_step": 0,
    "use_previous": true,
    "validation_split": 0.9
  },
  "checkpoint_interval": 120.0,
  "current_step": 0,
  "game": "reversi",
  "generation_prefix": "x2",
  "max_samples_growth": 0.8,
  "num_samples_to_train": 20000,
  "port": 9000,
  "self_play_config": {
    "max_number_of_samples": 4,
    "resign0_false_positive_retry_percentage": 0.95,
    "resign0_score_probability": 0.1,
    "resign1_false_positive_retry_percentage": 0.95,
    "resign1_score_probability": 0.02,
    "sample_iterations": 400,
    "sample_puct_config": {
      "choose": "choose_temperature",
      "depth_temperature_increment": 1.0,
      "depth_temperature_max": 6.0,
      "depth_temperature_start": 0,
      "depth_temperature_stop": 20,
      "dirichlet_noise_alpha": 0.1,
      "dirichlet_noise_pct": 0.25,
      "max_dump_depth": 2,
      "puct_before_expansions": 4,
      "puct_before_root_expansions": 5,
      "puct_constant_after": 0.75,
      "puct_constant_before": 3.0,
      "random_scale": 0.95,
      "root_expansions_preset_visits": 1,
      "temperature": 1.0,
      "verbose": false
    },
    "score_iterations": 100,
    "score_puct_config": {
      "choose": "choose_top_visits",
      "depth_temperature_increment": 0.5,
      "depth_temperature_max": 5.0,
      "depth_temperature_start": 5,
      "depth_temperature_stop": 10,
      "dirichlet_noise_alpha": 0.1,
      "dirichlet_noise_pct": 0.25,
      "max_dump_depth": 2,
      "puct_before_expansions": 4,
      "puct_before_root_expansions": 5,
      "puct_constant_after": 0.75,
      "puct_constant_before": 3.0,
      "random_scale": 0.5,
      "root_expansions_preset_visits": 1,
      "temperature": 1.0,
      "verbose": false
    },
    "select_iterations": 0,
    "select_puct_config": {
      "choose": "choose_temperature",
      "depth_temperature_increment": 0.25,
      "depth_temperature_max": 5.0,
      "depth_temperature_start": 2,
      "depth_temperature_stop": 40,
      "dirichlet_noise_alpha": -1,
      "dirichlet_noise_pct": 0.25,
      "max_dump_depth": 2,
      "puct_before_expansions": 4,
      "puct_before_root_expansions": 5,
      "puct_constant_after": 0.75,
      "puct_constant_before": 3.0,
      "random_scale": 0.95,
      "root_expansions_preset_visits": -1,
      "temperature": 1.0,
      "verbose": false
    }
  }
}
```

gen 5 changes
--------------
* base_network_model.dropout_rate_policy = 0.2
* self_play_config.sample_iterations = 800
* base_training_config.starting_step = 2
* resign0_false_positive_retry_percentage = 0.9
* resign1_false_positive_retry_percentage = 0.75


gen 10 changes
--------------
* base_training_config.batch_size = 256
* resign0_false_positive_retry_percentage = 0.85
* resign1_false_positive_retry_percentage = 0.3
* base_training_config.resample_buckets = [[10, 1.0], [0, 0.8]]
* base_training_config.starting_step = 3


gen 20 changes
--------------
* base_network_model.cnn_filter_size = 112
* base_network_model.value_hidden_size = 128
* num_samples_to_train = 40000
* self_play_config.max_number_of_samples = 8
* base_training_config.starting_step = 5
* base_training_config.resample_buckets = [[10, 1.0], [20, 0.9], [30, 0.7], [0, 0.5]]
* base_training_config.max_number_of_samples = 1000000
* base_training_config.drop_dupes_count = 7


gen 50 changes
--------------
Lost track... trying hard not to change things, but there has been lots.
