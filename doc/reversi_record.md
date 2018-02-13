reversi training - x3
=====================

Previous runs (names are arbitrary):

* [x2](https://github.com/ggplib/ggp-zero/blob/dev/doc/reversi_record_x2.md).
* [v9](https://github.com/ggplib/ggp-zero/blob/dev/doc/reversi_record_v9.md).

Started on 11am 12th February.

Fixed a subtle bug with PUCT constant.  I also disabled the a horrible hack (was minimaxing during
backprop, which was something I added likely because of the PUCT constant bug).

For this run, will keep number of epochs at approximately 1 for each generation.  And always use
previous network, unless changing network configuration.  The batch size is smaller (128) and
ignoring duplicates across generations (no de-duping).

And I am using 5 previous states (why not?).


results
-------
* gzero_x_y - where x is generation, y is number of playouts per move * 100.

* ntest_x - where x is nboard depth/level
* results - win/loss/draw

| player     | opponent   | black_result   | white_result   |
|:-----------|:-----------|:---------------|:---------------|
| gzero_34_8 | ntest_5    | 0/4/1          | 0/5/0          |
| gzero_34_8 | ntest_4    | 0/5/0          | 3/2/0          |
| gzero_34_8 | ntest_3    | 3/2/0          | 1/4/0          |
| gzero_34_8 | ntest_2    | 3/2/0          | 1/4/0          |
| gzero_34_4 | ntest_2    | 3/2/0          | 1/4/0          |
| gzero_34_2 | ntest_1    | 3/2/0          | 5/0/0          |
| gzero_30_1 | simplemcts | 5/0/0          | 5/0/0          |
| gzero_20_2 | simplemcts | 4/1/0          | 4/1/0          |
| gzero_20_1 | simplemcts | 3/2/0          | 4/1/0          |
| gzero_12_2 | simplemcts | 4/1/0          | 4/1/0          |
| gzero_10_2 | pymcs      | 5/0/0          | 5/0/0          |


configuration
=============

initial
-------

```json

{
  "obj": {
    "self_play_config": {
      "select_puct_config": {
        "verbose": false,
        "temperature": 1.0,
        "root_expansions_preset_visits": -1,
        "random_scale": 1.0,
        "puct_constant_before": 3.0,
        "puct_constant_after": 0.75,
        "puct_before_root_expansions": 4,
        "puct_before_expansions": 3,
        "max_dump_depth": 2,
        "dirichlet_noise_pct": 0.25,
        "dirichlet_noise_alpha": -1,
        "depth_temperature_stop": 70,
        "depth_temperature_start": 4,
        "depth_temperature_max": 2.5,
        "depth_temperature_increment": 0.2,
        "choose": "choose_temperature"
      },
      "select_iterations": 1,
      "score_puct_config": {
        "verbose": false,
        "temperature": 1.0,
        "root_expansions_preset_visits": -1,
        "random_scale": 0.5,
        "puct_constant_before": 3.0,
        "puct_constant_after": 0.75,
        "puct_before_root_expansions": 4,
        "puct_before_expansions": 3,
        "max_dump_depth": 2,
        "dirichlet_noise_pct": 0.25,
        "dirichlet_noise_alpha": 0.1,
        "depth_temperature_stop": 10,
        "depth_temperature_start": 5,
        "depth_temperature_max": 5.0,
        "depth_temperature_increment": 0.5,
        "choose": "choose_top_visits"
      },
      "score_iterations": 100,
      "sample_puct_config": {
        "verbose": false,
        "temperature": 1.0,
        "root_expansions_preset_visits": -1,
        "random_scale": 0.85,
        "puct_constant_before": 3.0,
        "puct_constant_after": 0.75,
        "puct_before_root_expansions": 4,
        "puct_before_expansions": 3,
        "max_dump_depth": 2,
        "dirichlet_noise_pct": 0.25,
        "dirichlet_noise_alpha": 0.1,
        "depth_temperature_stop": 16,
        "depth_temperature_start": 4,
        "depth_temperature_max": 3.0,
        "depth_temperature_increment": 0.25,
        "choose": "choose_temperature"
      },
      "sample_iterations": 800,
      "resign1_score_probability": 0.02,
      "resign1_false_positive_retry_percentage": 0.99,
      "resign0_score_probability": 0.1,
      "resign0_false_positive_retry_percentage": 0.99,
      "max_number_of_samples": 4
    },
    "port": 9000,
    "num_samples_to_train": 20000,
    "max_samples_growth": 0.8,
    "generation_prefix": "x3",
    "game": "reversi",
    "current_step": 2,
    "checkpoint_interval": 120.0,
    "base_training_config": {
      "validation_split": 0.9,
      "use_previous": true,
      "starting_step": 0,
      "resample_buckets": [
        [
          20,
          1.0
        ],
        [
          30,
          0.8
        ],
        [
          0,
          0.5
        ]
      ],
      "overwrite_existing": false,
      "next_step": 2,
      "max_sample_count": 1000000,
      "learning_rate": null,
      "generation_prefix": "x3",
      "game": "reversi",
      "epochs": 2,
      "drop_dupes_count": -1,
      "compile_strategy": "adam",
      "batch_size": 128
    },
    "base_network_model": {
      "value_hidden_size": 64,
      "role_count": 2,
      "residual_layers": 8,
      "policy_dist_count": [
        65,
        65
      ],
      "multiple_policies": true,
      "leaky_relu": false,
      "l2_regularisation": false,
      "input_rows": 8,
      "input_columns": 8,
      "input_channels": 13,
      "dropout_rate_value": 0.5,
      "dropout_rate_policy": 0.1,
      "cnn_kernel_size": 3,
      "cnn_filter_size": 64
    },
    "base_generation_description": {
      "transformer_description": null,
      "trained_value_accuracy": "not set",
      "trained_validation_losses": "not set",
      "trained_policy_accuracy": "not set",
      "trained_losses": "not set",
      "num_previous_states": 5,
      "name": "x3_0",
      "multiple_policy_heads": true,
      "game": "reversi",
      "date_created": "2018\/02\/02 10:20",
      "channel_last": false
    }
  }
}
```

gen 7 changes
--------------
Up network size.

* base_network_model.cnn_filter_size = 96,
* base_network_model.value_hidden_size = 96

Set resign.

* resign0_score_probability = 0.05
* resign0_false_positive_retry_percentage = 0.9
* resign1_score_probability = 0.02
* resign1_false_positive_retry_percentage = 0.8


gen 20 changes
--------------
Up network size.

* base_network_model.value_hidden_size = 128

Increase number of samples, more resign, some resampling.

* base_training_config.resample_buckets = [ 20, 1.0], [25, 0.50], [0, 0.25]
* self_play_config.max_number_of_samples = 6
* num_samples_to_train = 30000
* resign1_false_positive_retry_percentage = 0.75

gen 35 changes
--------------
Up network size.

* base_network_model.cnn_filter_size = 128
* base_network_model.value_hidden_size = 192

* resign0_false_positive_retry_percentage = 0.5
* resign1_false_positive_retry_percentage = 0.25


