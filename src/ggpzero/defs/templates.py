from ggpzero.defs import confs
from ggpzero.nn.manager import get_manager


def nn_model_config_template(game, network_size_hint="small"):
    ' helper for creating NNModelConfig templates '

    conf = confs.NNModelConfig()

    # from transformer
    transformer = get_manager().get_transformer(game)
    conf.role_count = transformer.role_count

    conf.input_rows = transformer.num_rows
    conf.input_columns = transformer.num_cols
    conf.input_channels = transformer.num_channels

    conf.policy_dist_count = transformer.policy_dist_count

    # ensure no regularisation
    conf.l2_regularisation = False

    # use defaults
    conf.learning_rate = None

    # normal defaults
    conf.cnn_kernel_size = 3
    conf.dropout_rate_policy = 0.25
    conf.dropout_rate_value = 0.5

    # residual_layers is the same size as max dimension of board
    conf.residual_layers = max(transformer.num_rows, transformer.num_cols)

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
                                       playouts_per_iteration=1600,
                                       playouts_per_iteration_noop=800,
                                       dirichlet_noise_alpha=0.03,

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

        test2=confs.PUCTPlayerConfig(name="test2",
                                     verbose=True,
                                     playouts_per_iteration=100,
                                     playouts_per_iteration_noop=100,

                                     dirichlet_noise_alpha=-1,
                                     puct_before_expansions=3,
                                     puct_before_root_expansions=5,
                                     puct_constant_before=3.0,
                                     puct_constant_after=0.75,

                                     choose="choose_top_visits",
                                     max_dump_depth=2),

        policy=confs.PUCTPlayerConfig(name="policy-test",
                                      verbose=True,
                                      playouts_per_iteration=0,
                                      playouts_per_iteration_noop=0,
                                      dirichlet_noise_alpha=-1,
                                      choose="choose_top_visits",
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
                                       playouts_per_iteration=200,
                                       playouts_per_iteration_noop=200,

                                       dirichlet_noise_alpha=0.03,

                                       puct_before_expansions=3,
                                       puct_before_root_expansions=5,
                                       puct_constant_before=3.0,
                                       puct_constant_after=0.75,

                                       temperature=0.8,
                                       depth_temperature_start=4,
                                       depth_temperature_increment=0.5,
                                       depth_temperature_stop=40,
                                       random_scale=0.75,

                                       choose="choose_temperature",
                                       max_dump_depth=2))
    conf = configs[name]
    conf.generation = generation
    return conf
