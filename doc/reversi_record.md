reversi generation 0 - v10
==========================

Restarted again!  Started on 2nd February.


results
-------
results @ 2nd February

* gzero_x_y - where x is generation, y is number of playouts per move * 100.

* ntest_x - where x is nboard depth/level
* results - win/loss/draw

| player    | opponent   | black_result   | white_result   |
|:----------|:-----------|:---------------|:---------------|
| gzero_5_1 | pymcs      | 0/0/0          | 0/0/0          |
| gzero_2_1 | random     | 5/0/0          | 5/0/0          |


gzero run with no dirichlet noise, small amount of random choice of action.


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
(rough)

* 20 generations with 2 samples per game - 10k each, 200k total
* 15 generations with 4 samples per game - 40k each, 600k total
* 15 generations with 6 samples per game - 40k each. 600k total
* 15 generations with 8 samples per game - 60k each. 800k total
* 15 generations with 10 samples per game - 60k each. 800k total


major features testing
----------------------
 * multiple policies
 * 5 previous states
 * no drop out on policy heads
 * resampling from generations each epoch (bucket/percent based)
 * experimental prover puct mcts (see if helps convergence at end of game)


methodology
-----------
Start with a similar configuration when I stopped previous run.  I adopted different dirichlet and
cpuct values - (from mokemokechicken and gooooloo runs).

Will tweak a lot less this run.  Will have to reduce resigns manaully at some point.

At the end of every 5th/10th generation, perform an evaluation against opponents.


problems anticipated
--------------------
Lots of changes.  Complete lack of testing.


configuration
=============

initial
-------

```json

{
  "base_training_config": {
    "validation_split": 0.9,
    "use_previous": true,
    "starting_step": 0,
    "resample_buckets": [
      [
        5,
        1.0
      ],
      [
        15,
        0.8
      ],
      [
        25,
        0.6
      ],
      [
        35,
        0.4
      ],
      [
        45,
        0.2
      ],
      [
        0,
        0.1
      ]
    ],
    "overwrite_existing": false,
    "next_step": 20,
    "max_sample_count": 1000000,
    "learning_rate": null,
    "generation_prefix": "v10",
    "game": "reversi",
    "epochs": 20,
    "drop_dupes_count": -1,
    "compile_strategy": "adam",
    "batch_size": 1024
  },
  "base_generation_description": {
    "num_previous_states": 5,
    "name": "v10_0",
    "multiple_policy_heads": true,
    "game": "reversi",
    "date_created": "2018-02-01 0:00",
    "channel_last": false
  },
  "base_network_model": {
    "cnn_filter_size": 128,
    "cnn_kernel_size": 3,
    "dropout_rate_policy": -1,
    "dropout_rate_value": 0.5,
    "input_channels": 14,
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
    "value_hidden_size": 192
  },
  "checkpoint_interval": 120.0,
  "current_step": 0,
  "game": "reversi",
  "generation_prefix": "v10",
  "max_samples_growth": 0.8,
  "num_samples_to_train": 10000,
  "port": 9000,
  "self_play_config": {
    "max_number_of_samples": 2,
    "resign0_false_positive_retry_percentage": 0.99,
    "resign0_score_probability": 0.08,
    "resign1_false_positive_retry_percentage": 0.99,
    "resign1_score_probability": 0.02,
    "sample_iterations": 800,
    "sample_puct_config": {
      "choose": "choose_temperature",
      "depth_temperature_increment": 1.0,
      "depth_temperature_max": 6.0,
      "depth_temperature_start": 0,
      "depth_temperature_stop": 20,
      "dirichlet_noise_alpha": 0.3,
      "dirichlet_noise_pct": 0.25,
      "puct_before_expansions": 3,
      "puct_before_root_expansions": 5,
      "puct_constant_after": 1.0,
      "puct_constant_before": 5.0,
      "random_scale": 0.9,
      "root_expansions_preset_visits": 1,
      "temperature": 1.0
    },
    "score_iterations": 100,
    "score_puct_config": {
      "choose": "choose_top_visits",
      "dirichlet_noise_alpha": 0.3,
      "dirichlet_noise_pct": 0.25,
      "puct_before_expansions": 3,
      "puct_before_root_expansions": 5,
      "puct_constant_after": 1.0,
      "puct_constant_before": 5.0,
      "root_expansions_preset_visits": 1
    },
    "select_iterations": 0,
    "select_puct_config": {
      "choose": "choose_temperature",
      "depth_temperature_increment": 0.1,
      "depth_temperature_max": 3.0,
      "depth_temperature_start": 30,
      "depth_temperature_stop": 100,
      "temperature": 1.0
    }
  }
}
```

raw results json
----------------

```json
{ "results" : [
              [ ["gzero", 2, 1], ["random", -1, -1], [5, 0, 0], [5, 0, 0] ],
              [ ["gzero", 5, 1], ["pymcs", -1, -1], [0, 0, 0], [0, 0, 0] ]
}

```
