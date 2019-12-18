from datetime import datetime

from ggpzero.defs import confs, datadesc


def default_generation_desc(game, name="default", **kwds):
    desc = datadesc.GenerationDescription(game)

    desc.name = name
    desc.date_created = datetime.now().strftime("%Y/%m/%d %H:%M")

    desc.channel_last = False
    desc.multiple_policy_heads = True
    desc.num_previous_states = 0
    for k, v in kwds.items():
        setattr(desc, k, v)

    return desc


def nn_model_config_template(game, network_size_hint, transformer, features=False):
    ' helper for creating NNModelConfig templates '

    conf = confs.NNModelConfig()

    # from transformer
    conf.role_count = transformer.role_count

    conf.input_rows = transformer.num_rows
    conf.input_columns = transformer.num_cols
    conf.input_channels = transformer.num_channels

    # policy distribution head
    conf.policy_dist_count = transformer.policy_dist_count
    assert isinstance(conf.policy_dist_count, list) and len(conf.policy_dist_count) > 0

    # normal defaults
    conf.cnn_kernel_size = 3
    conf.dropout_rate_policy = 0.25
    conf.dropout_rate_value = 0.5

    if network_size_hint == "small":
        conf.cnn_filter_size = 64
        conf.residual_layers = 5
        conf.value_hidden_size = 256

    elif network_size_hint == "medium":
        conf.cnn_filter_size = 96
        conf.residual_layers = 5
        conf.value_hidden_size = 512

    elif network_size_hint == "large":
        conf.cnn_filter_size = 96
        conf.residual_layers = 10
        conf.value_hidden_size = 512

    else:
        assert False, "network_size_hint %s, not recognised" % network_size_hint

    conf.leaky_relu = False
    if features:
        conf.resnet_v2 = True
        conf.squeeze_excite_layers = True
        conf.global_pooling_value = True
    else:
        conf.resnet_v2 = False
        conf.squeeze_excite_layers = False
        conf.global_pooling_value = False

    return conf


def base_puct_config(**kwds):
    config = confs.PUCTEvaluatorConfig(verbose=False,
                                       backup_finalised=False,
                                       batch_size=1,

                                       dirichlet_noise_pct=-1,

                                       puct_constant=0.85,
                                       puct_constant_root=0.85,

                                       fpu_prior_discount=0.25,
                                       fpu_prior_discount_root=0.25,

                                       choose="choose_temperature",
                                       temperature=1.0,
                                       depth_temperature_max=5.0,
                                       depth_temperature_start=2,
                                       depth_temperature_increment=0.2,
                                       depth_temperature_stop=6,
                                       random_scale=0.95,

                                       think_time=-1,
                                       max_dump_depth=0,
                                       top_visits_best_guess_converge_ratio=0.85,
                                       converged_visits=1,
                                       evaluation_multiplier_to_convergence=2.0)
    # ZZZ everyting else?

    for k, v in kwds.items():
        setattr(config, k, v)

    return config


def selfplay_config_template():
    conf = confs.SelfPlayConfig()
    conf.oscillate_sampling_pct = 0.25
    conf.temperature_for_policy = 1.0

    conf.puct_config = base_puct_config(dirichlet_noise_pct=0.25)
    conf.evals_per_move = 100

    conf.resign0_score_probability = 0.1
    conf.resign0_pct = 0.99
    conf.resign1_score_probability = 0.025
    conf.resign1_pct = 0.95

    conf.run_to_end_pct = 0.01
    conf.run_to_end_evals = 32
    conf.run_to_end_puct_config = base_puct_config(dirichlet_noise_pct=0.15,
                                                   random_scale=0.75)
    conf.run_to_end_early_score = 0.01
    conf.run_to_end_minimum_game_depth = 30

    conf.abort_max_length = -1

    return conf


def train_config_template(game, gen_prefix):
    conf = confs.TrainNNConfig(game)

    conf.generation_prefix = gen_prefix

    conf.next_step = 0
    conf.starting_step = 0
    conf.use_previous = True
    conf.validation_split = 0.95
    conf.overwrite_existing = False

    conf.epochs = 1
    conf.batch_size = 256
    conf.compile_strategy = "SGD"
    conf.l2_regularisation = 0.00002
    conf.learning_rate = 0.01

    conf.initial_value_weight = 1.0
    conf.max_epoch_size = 1024 * 1024

    conf.resample_buckets = [
        [
            100,
            1.00000
        ]]

    return conf


def server_config_template(game, generation_prefix, prev_states):
    conf = confs.ServerConfig()

    conf.game = game
    conf.generation_prefix = generation_prefix

    conf.port = 9000

    conf.current_step = 0

    conf.num_samples_to_train = 20000
    conf.max_samples_growth = 0.8

    conf.base_generation_description = default_generation_desc(game,
                                                               generation_prefix,
                                                               multiple_policy_heads=True,
                                                               num_previous_states=prev_states)

    from ggpzero.nn.manager import get_manager
    man = get_manager()
    transformer = man.get_transformer(game, conf.base_generation_description)
    conf.base_network_model = nn_model_config_template(game, "small", transformer)

    conf.base_training_config = train_config_template(game, generation_prefix)

    conf.self_play_config = selfplay_config_template()
    return conf
