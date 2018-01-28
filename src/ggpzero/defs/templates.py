from datetime import datetime

from ggpzero.defs import confs, datadesc


def nn_model_config_template(game, network_size_hint, transformer):
    ' helper for creating NNModelConfig templates '

    conf = confs.NNModelConfig()

    # from transformer
    conf.role_count = transformer.role_count

    conf.input_rows = transformer.num_rows
    conf.input_columns = transformer.num_cols
    conf.input_channels = transformer.num_channels

    # policy distribution head
    conf.multiple_policies = len(transformer.policy_dist_count) > 1
    conf.policy_dist_count = transformer.policy_dist_count
    assert isinstance(conf.policy_dist_count, list) and len(conf.policy_dist_count) > 0

    # no regularisation please
    conf.l2_regularisation = False

    # normal defaults
    conf.cnn_kernel_size = 3
    conf.dropout_rate_policy = 0.25
    conf.dropout_rate_value = 0.5

    # residual_layers is the same size as max dimension of board
    conf.residual_layers = min(transformer.num_rows, transformer.num_cols)

    if network_size_hint == "tiny":
        conf.residual_layers = max(4, conf.residual_layers / 2)
        conf.cnn_filter_size = 32
        conf.value_hidden_size = 32

    elif network_size_hint == "smaller":
        conf.cnn_filter_size = 64
        conf.value_hidden_size = 64

    elif network_size_hint == "small":
        conf.cnn_filter_size = 96
        conf.value_hidden_size = 128

    elif network_size_hint == "medium-small":
        conf.cnn_filter_size = 112
        conf.value_hidden_size = 128

    elif network_size_hint == "medium":
        conf.cnn_filter_size = 128
        conf.value_hidden_size = 192

    elif network_size_hint == "medium-large":
        conf.cnn_filter_size = 160
        conf.value_hidden_size = 192

    elif network_size_hint == "large":
        conf.cnn_filter_size = 192
        conf.value_hidden_size = 256

    elif network_size_hint == "larger":
        conf.cnn_filter_size = 224
        conf.value_hidden_size = 384

    elif network_size_hint == "massive":
        # these are alphago sizes
        conf.cnn_filter_size = 256
        conf.value_hidden_size = 512

    else:
        assert False, "network_size_hint %s, not recognised" % network_size_hint

    return conf


def puct_config_template(generation, name="default"):
    configs = dict(
        default=confs.PUCTPlayerConfig(name="default",
                                       verbose=True,
                                       playouts_per_iteration=2400,
                                       playouts_per_iteration_noop=800,
                                       dirichlet_noise_alpha=0.05,

                                       puct_before_expansions=3,
                                       puct_before_root_expansions=5,
                                       puct_constant_before=3.0,
                                       puct_constant_after=0.75,

                                       choose="choose_top_visits",
                                       max_dump_depth=2),

        test=confs.PUCTPlayerConfig(name="test",
                                    verbose=True,
                                    playouts_per_iteration=42,
                                    playouts_per_iteration_noop=0,

                                    dirichlet_noise_alpha=0.03,
                                    puct_before_expansions=3,
                                    puct_before_root_expansions=5,
                                    puct_constant_before=3.0,
                                    puct_constant_after=0.75,

                                    choose="choose_top_visits",
                                    max_dump_depth=2),

        compete=confs.PUCTPlayerConfig(name="compete",
                                       verbose=True,

                                       playouts_per_iteration=400,
                                       playouts_per_iteration_noop=400,

                                       dirichlet_noise_alpha=-1,

                                       puct_before_expansions=3,
                                       puct_before_root_expansions=5,
                                       puct_constant_before=3.0,
                                       puct_constant_after=1.00,


                                       temperature=2.0,
                                       depth_temperature_max=10.0,
                                       depth_temperature_start=4,
                                       depth_temperature_increment=0.25,
                                       depth_temperature_stop=100,
                                       random_scale=0.75,

                                       choose="choose_temperature",
                                       max_dump_depth=2),

        compete2=confs.PUCTPlayerConfig(name="compete2",
                                        verbose=True,

                                        playouts_per_iteration=800,
                                        playouts_per_iteration_noop=800,

                                        root_expansions_preset_visits=7,

                                        resign_score_value=0.05,
                                        playouts_per_iteration_resign=25,

                                        dirichlet_noise_alpha=-1,

                                        puct_before_expansions=3,
                                        puct_before_root_expansions=5,
                                        puct_constant_before=3.0,
                                        puct_constant_after=1.00,

                                        temperature=1.0,
                                        depth_temperature_max=2.5,
                                        depth_temperature_start=8,
                                        depth_temperature_increment=0.1,
                                        depth_temperature_stop=60,
                                        random_scale=0.65,

                                        choose="choose_temperature",
                                        max_dump_depth=2),

        policy=confs.PUCTPlayerConfig(name="policy-test",
                                      verbose=True,
                                      playouts_per_iteration=0,
                                      playouts_per_iteration_noop=0,
                                      dirichlet_noise_alpha=-1,
                                      choose="choose_top_visits",
                                      max_dump_depth=1),

        policy_compete=confs.PUCTPlayerConfig(name="policy_compete",
                                              verbose=True,
                                              playouts_per_iteration=0,
                                              playouts_per_iteration_noop=0,
                                              dirichlet_noise_alpha=-1,

                                              temperature=1.0,
                                              depth_temperature_max=2.5,
                                              depth_temperature_start=8,
                                              depth_temperature_increment=0.1,
                                              depth_temperature_stop=60,
                                              random_scale=0.65,

                                              choose="choose_temperature",
                                              max_dump_depth=1),

        max_score=confs.PUCTPlayerConfig(name="max-score",
                                         verbose=True,
                                         playouts_per_iteration=1,
                                         playouts_per_iteration_noop=0,
                                         dirichlet_noise_alpha=-1,
                                         puct_constant_before=0,
                                         puct_constant_after=0,

                                         choose="choose_top_visits",
                                         max_dump_depth=2),

        compare=confs.PUCTPlayerConfig(name="compare",
                                       verbose=True,
                                       playouts_per_iteration=150,
                                       playouts_per_iteration_noop=1,

                                       dirichlet_noise_alpha=0.03,

                                       puct_before_expansions=3,
                                       puct_before_root_expansions=5,
                                       puct_constant_before=3.0,
                                       puct_constant_after=0.75,

                                       choose="choose_top_visits",
                                       max_dump_depth=2))
    conf = configs[name]
    conf.generation = generation
    return conf


def selfplay_config_template():
    conf = confs.SelfPlayConfig()
    conf.max_number_of_samples = 4
    conf.resign_score_probability = 0.1
    conf.resign_false_positive_retry_percentage = 0.1

    conf.select_iterations = 0
    conf.sample_iterations = 800
    conf.score_iterations = 42

    conf.select_puct_config = confs.PUCTEvaluatorConfig()
    conf.select_puct_config.verbose = False
    conf.select_puct_config.temperature = 0.5
    conf.select_puct_config.depth_temperature_start = 2
    conf.select_puct_config.depth_temperature_increment = 0.25
    conf.select_puct_config.depth_temperature_stop = 40
    conf.select_puct_config.random_scale = 0.85
    conf.select_puct_config.choose = "choose_temperature"
    conf.select_iterations = 0

    conf.sample_puct_config = confs.PUCTEvaluatorConfig()
    conf.sample_puct_config.verbose = False
    conf.sample_iterations = 400

    conf.score_puct_config = confs.PUCTEvaluatorConfig()
    conf.score_puct_config.verbose = False
    conf.score_iterations = 75

    return conf


def default_generation_desc(game, name="default", **kwds):
    desc = datadesc.GenerationDescription(game)

    desc.name = name
    desc.date_created = datetime.now().strftime("%Y/%m/%d %H:%M")

    desc.channel_last = False
    desc.multiple_policy_heads = False
    desc.num_previous_states = 0
    for k, v in kwds.items():
        setattr(desc, k, v)

    return desc


def train_config_template(game, gen_prefix):
    conf = confs.TrainNNConfig("speedChess")

    conf.generation_prefix = gen_prefix

    conf.use_previous = True
    conf.next_step = 0
    conf.validation_split = 0.9
    conf.batch_size = 128
    conf.epochs = 20
    conf.max_sample_count = 300000
    conf.starting_step = 0
    conf.drop_dupes_count = 3
    conf.overwrite_existing = False

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
    conf.base_network_model = nn_model_config_template(game, "smaller", transformer)

    conf.base_training_config = train_config_template(game, generation_prefix)

    conf.self_play_config = selfplay_config_template()
    return conf
