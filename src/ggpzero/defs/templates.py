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
    conf.dropout_rate_policy = 0.333
    conf.dropout_rate_value = 0.5

    # ideally residual_layers is the same size as board size

    if network_size_hint == "tiny":
        conf.residual_layers = 4
        conf.cnn_filter_size = 32
        conf.value_hidden_size = 32

    elif network_size_hint == "smaller":
        conf.residual_layers = 8
        conf.cnn_filter_size = 64
        conf.value_hidden_size = 64

    elif network_size_hint == "small":
        conf.residual_layers = 8
        conf.cnn_filter_size = 96
        conf.value_hidden_size = 128

    elif network_size_hint == "normal":
        conf.residual_layers = 8
        conf.cnn_filter_size = 112
        conf.value_hidden_size = 128

    elif network_size_hint == "large":
        conf.residual_layers = 8
        conf.cnn_filter_size = 128
        conf.value_hidden_size = 256

    elif network_size_hint == "larger":
        conf.residual_layers = 8
        conf.cnn_filter_size = 192
        conf.value_hidden_size = 256
    else:
        assert False, "network_size_hint %s, not recognised" % network_size_hint

    return conf

