from ggpzero.defs import confs

from ggpzero.util.keras import is_channels_first, keras_models, keras_regularizers
from ggpzero.util.keras import keras_layers as klayers


# kind of thought keras was taking care of this XXX
def get_bn_axis():
    return 1 if is_channels_first() else 3


def Conv2DBlock(*args, **kwds):
    # args, kwds -> passed through to Conv2D

    assert "activation" in kwds
    activation = kwds.pop("activation")

    try:
        name = kwds.pop('name',)
    except KeyError:
        name = 'conv2d_block'

    conv_name = name + '_conv2d'
    bn_name = name + '_bn'
    act_name = name + '_act'

    do_bn = activation != "selu"

    def block(x):
        x = klayers.Conv2D(*args,
                           name=conv_name, **kwds)(x)

        if do_bn:
            x = klayers.BatchNormalization(axis=get_bn_axis(),
                                           name=bn_name)(x)

        if activation == "leakyrelu":
            x = klayers.LeakyReLU(alpha=0.03, name=act_name)(x)
        else:
            x = klayers.Activation(activation, name=act_name)(x)
        return x

    return block


def ResidualBlock(*args, **kwds):
    # args, kwds -> passed through to Conv2D
    assert "padding" not in kwds

    name = kwds.pop('name', 'residual_block')

    name_conv0 = name + "_conv0"
    name_conv1 = name + "_conv1"
    name_bn0 = name + "_bn0"
    name_bn1 = name + "_bn1"
    name_act0 = name + "_act0"
    name_act_after = name + "_act_after"
    name_add = name + "_add"

    activation = kwds.pop('activation', "relu")

    def block(tensor):
        x = klayers.Conv2D(*args,
                           name=name_conv0,
                           padding="same",
                           **kwds)(tensor)

        x = klayers.BatchNormalization(axis=get_bn_axis(),
                                       name=name_bn0)(x)

        if activation == "leakyrelu":
            x = klayers.LeakyReLU(alpha=0.03, name=name_act0)(x)
        else:
            x = klayers.Activation(activation, name=name_act0)(x)

        x = klayers.Conv2D(*args,
                           name=name_conv1,
                           padding="same",
                           **kwds)(x)

        x = klayers.BatchNormalization(axis=get_bn_axis(),
                                       name=name_bn1)(x)

        x = klayers.add([tensor, x], name=name_add)
        if activation == "leakyrelu":
            x = klayers.LeakyReLU(alpha=0.03, name=name_act_after)(x)
        else:
            x = klayers.Activation(activation, name=name_act_after)(x)

        return x

    return block


def get_network_model(conf):
    assert isinstance(conf, confs.NNModelConfig)

    # fancy l2 regularizer stuff
    extra_params = {}
    if conf.l2_regularisation:
        extra_params["kernel_regularizer"] = keras_regularizers.l2(1e-4)

    activation = 'leakyrelu' if conf.leaky_relu else 'relu'

    # inputs:
    if is_channels_first():
        inputs_board = klayers.Input(shape=(conf.input_channels,
                                            conf.input_columns,
                                            conf.input_rows),
                                     name="inputs_board")
    else:
        inputs_board = klayers.Input(shape=(conf.input_columns,
                                            conf.input_rows,
                                            conf.input_channels),
                                     name="inputs_board")

    # initial conv2d/Resnet on cords
    layer = Conv2DBlock(conf.cnn_filter_size, conf.cnn_kernel_size,
                        padding='same',
                        activation=activation,
                        name='initial', **extra_params)(inputs_board)

    for i in range(conf.residual_layers):
        layer = ResidualBlock(conf.cnn_filter_size,
                              conf.cnn_kernel_size,
                              name="ResLayer_%s" % i,
                              activation=activation,
                              **extra_params)(layer)

    number_of_policies = 1
    if conf.multiple_policies:
        number_of_policies = conf.role_count

    assert number_of_policies == len(conf.policy_dist_count)

    policy_heads = []
    for idx, count in enumerate(conf.policy_dist_count):
        # residual net -> flattened for policy head
        to_flatten = Conv2DBlock(2, 1,
                                 name='to_flatten_policy_head_%s' % idx,
                                 padding='valid',
                                 activation=activation,
                                 **extra_params)(layer)

        flat = klayers.Flatten()(to_flatten)

        # output: policy head(s)
        if conf.dropout_rate_policy > 0:
            flat = klayers.Dropout(conf.dropout_rate_policy)(flat)

        head = klayers.Dense(count,
                             activation="softmax",
                             name="policy_%d" % idx,
                             **extra_params)(flat)

        policy_heads.append(head)

    # residual net -> flattened for value head
    to_flatten = Conv2DBlock(1, 1,
                             name='to_flatten_value_head',
                             padding='valid',
                             activation=activation,
                             **extra_params)(layer)

    flat = klayers.Flatten()(to_flatten)

    # output: value head
    hidden = klayers.Dense(conf.value_hidden_size, activation="relu",
                           name="value_hidden_layer", **extra_params)(flat)

    if conf.dropout_rate_value > 0:
        hidden = klayers.Dropout(conf.dropout_rate_value)(hidden)

    value_head = klayers.Dense(conf.role_count,
                               activation="sigmoid", name="value", **extra_params)(hidden)

    # model:
    outputs = policy_heads + [value_head]
    return keras_models.Model(inputs=[inputs_board], outputs=outputs)
