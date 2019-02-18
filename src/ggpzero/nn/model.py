from ggpzero.defs import confs

from ggpzero.util.keras import is_channels_first, keras_models, get_antirectifier
from ggpzero.util.keras import keras_layers as klayers


def get_bn_axis():
    return 1 if is_channels_first() else 3


def act(x, activation, name):
    if activation == "crelu":
        return get_antirectifier(name)(x)
    elif activation == "leakyrelu":
        return klayers.LeakyReLU(alpha=0.03, name=name)(x)
    else:
        return klayers.Activation(activation, name=name)(x)


def bn(x, name):
    return klayers.BatchNormalization(axis=get_bn_axis(),
                                      name=name)(x)


def conv2d_block(*args, **kwds):
    # args, kwds -> passed through to Conv2D

    activation = kwds.pop("activation")
    name = kwds.pop('name')

    do_bn = kwds.pop('do_bn', True)

    def block(x):
        x = klayers.Conv2D(*args,
                           name=name + '_conv2d',
                           use_bias=False,
                           **kwds)(x)

        if do_bn:
            x = bn(x, name + '_bn')

        return act(x, activation, name + '_' + activation)

    return block


def residual_block_v1(filter_size, kernel_size,
                      activation="relu", prefix=""):
    assert prefix

    def conv(x, name):
        name = prefix + name
        return klayers.Conv2D(filter_size, kernel_size,
                              name=name,
                              padding="same",
                              use_bias=False)(x)

    def bn_(x, name):
        return bn(x, prefix + name)

    def act_(x, name):
        return act(x, activation, prefix + name)

    def block(tensor):
        x = conv(tensor, "conv0")
        x = bn_(x, "bn0")
        x = act_(x, "act0")

        x = conv(x, "conv1")
        x = bn_(x, "bn1")

        x = klayers.add([tensor, x], name=prefix + "add")
        return act_(x, "act_add")

    return block


def residual_block_v2(filter_size, kernel_size, num_convs,
                      dropout=None, activation="relu", prefix="",
                      squeeze_excite=False):

    assert prefix

    if squeeze_excite:
        assert dropout is None

    def conv(x, step):
        name = prefix + "conv%s" % step
        x = klayers.Conv2D(filter_size, kernel_size,
                           name=name,
                           padding="same",
                           use_bias=False)(x)
        return x

    def bn_(x, step):
        return bn(x, prefix + "bn_%s" % step)

    def act_(x, step):
        return act(x, activation, prefix + "act_%s" % step)

    def se_block(in_block, ratio=4):
        # XXX code modified:
        # from https://github.com/titu1994/keras-squeeze-excite-network
        assert is_channels_first()

        x = klayers.GlobalAveragePooling2D(name=prefix + "se_average")(in_block)
        x = klayers.Reshape((1, 1, filter_size), name=prefix + "reshape")(x)

        # is there a reason to use kernel_initializer='he_normal' ??? We use default everywhere
        # else?
        x = klayers.Dense(filter_size // ratio,
                          name=prefix + "se_compress",
                          kernel_initializer='he_normal',
                          use_bias=False,
                          activation='relu')(x)
        x = klayers.Dense(filter_size,
                          name=prefix + "se_gating",
                          kernel_initializer='he_normal',
                          use_bias=False,
                          activation='sigmoid')(x)

        x = klayers.Permute((3, 1, 2),
                            name=prefix + "se_permute")(x)
        return klayers.multiply([in_block, x], name=prefix + "se_scale")

    def block(tensor):
        x = tensor
        for i in range(num_convs - 1):
            x = bn_(x, i + 1)
            x = act_(x, i + 1)
            x = conv(x, i + 1)

        x = bn_(x, num_convs)
        x = act_(x, num_convs)

        if dropout is not None:
            x = klayers.Dropout(dropout,
                                name=prefix + "dropout_%s" % num_convs)(x)

        x = conv(x, num_convs)

        if squeeze_excite:
            x = se_block(x, filter_size)

        x = klayers.add([tensor, x], name=prefix + "add")

        return x

    return block


def get_network_model(conf, generation_descr):
    assert isinstance(conf, confs.NNModelConfig)

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

    # XXX lots of config abuse:
    v2 = conf.residual_layers <= 0
    if v2:
        layer = klayers.Conv2D(conf.cnn_filter_size, 1,
                               padding="same",
                               use_bias=False,
                               name='initial-conv')(inputs_board)

        # XXX hacks galore (needs to go into config somehow)
        if conf.residual_layers < 0:
            res_layers = [2] * -conf.residual_layers
            squeeze_excite = False
            dropout = None
        else:
            assert conf.residual_layers == 0
            res_layers = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
            squeeze_excite = True
            dropout = 0.25

        for i, c in enumerate(res_layers):
            if c == 2:
                layer = residual_block_v2(conf.cnn_filter_size,
                                          conf.cnn_kernel_size,
                                          c,
                                          prefix="ResLayer_%s_" % i,
                                          squeeze_excite=squeeze_excite,
                                          dropout=None,
                                          activation=activation)(layer)
            else:
                layer = residual_block_v2(conf.cnn_filter_size,
                                          conf.cnn_kernel_size,
                                          c,
                                          prefix="ResLayer_%s_" % i,
                                          squeeze_excite=False,
                                          dropout=0.25,
                                          activation=activation)(layer)

    else:
        # AG0 way:

        # initial conv2d/Resnet on cords
        layer = conv2d_block(conf.cnn_filter_size, conf.cnn_kernel_size,
                             activation=activation,
                             padding="same",
                             name='initial-conv')(inputs_board)

        # layers
        for i in range(conf.residual_layers):
            layer = residual_block_v1(conf.cnn_filter_size,
                                      conf.cnn_kernel_size,
                                      prefix="ResLayer_%s_" % i,
                                      activation=activation)(layer)

    # policy
    ########
    # similar to AG0, but with multiple policy heads
    assert conf.multiple_policies
    number_of_policies = conf.role_count
    assert number_of_policies == len(conf.policy_dist_count)

    policy_heads = []
    for idx, count in enumerate(conf.policy_dist_count):
        # residual net -> flattened for policy head
        # XXX 2, should be based on size of policy...
        to_flatten = conv2d_block(2, 1,
                                  name='to_flatten_policy_head_%s' % idx,
                                  activation=activation,
                                  padding='valid')(layer)

        flat = klayers.Flatten()(to_flatten)

        # output: policy head(s)
        if conf.dropout_rate_policy > 0:
            flat = klayers.Dropout(conf.dropout_rate_policy)(flat)

        head = klayers.Dense(count, name="policy_%d" % idx,
                             activation="softmax")(flat)

        policy_heads.append(head)

    # value
    #######
    # XXX config abuse:

    if generation_descr.draw_head:
        num_value_heads = 3
    else:
        num_value_heads = 2

    value_v3 = conf.value_hidden_size == 0
    value_v2 = conf.value_hidden_size < 0
    if value_v3:
        assert conf.input_columns == conf.input_rows
        average_layer = layer
        dims = conf.input_columns
        while dims >= 5:
            if dims % 2 == 1:
                average_layer = klayers.AveragePooling2D(4, 1)(average_layer)
                dims -= 3
            else:
                average_layer = klayers.AveragePooling2D(2, 2)(average_layer)
                dims /= 2

        assert dims < conf.input_columns

        to_flatten1 = conv2d_block(32, 1,
                                   name='value_flatten1',
                                   activation=activation,
                                   do_bn=False,
                                   padding='valid')(average_layer)

        to_flatten2 = conv2d_block(1, 1,
                                   name='value_flatten2',
                                   activation=activation,
                                   do_bn=False,
                                   padding='valid')(layer)

        flat = klayers.concatenate([klayers.Flatten()(to_flatten1),
                                    klayers.Flatten()(to_flatten2)])

        # XXX
        # if conf.dropout_rate_value > 0:
        #    flat = klayers.Dropout(conf.dropout_rate_value)(flat)

        hidden = klayers.Dense(256, name="value_hidden")(flat)
        hidden = act(hidden, 'crelu', name="value_hidden_act")

        # XXX dropout here?
        if conf.dropout_rate_value > 0:
            flat = klayers.Dropout(conf.dropout_rate_value)(flat)

        value_head = klayers.Dense(num_value_heads,
                                   activation="sigmoid",
                                   name="value")(hidden)

    elif value_v2:
        assert conf.input_columns == conf.input_rows
        output_layer = layer
        dims = conf.input_columns
        while dims > 5:
            if dims % 2 == 1:
                output_layer = klayers.AveragePooling2D(4, 1)(output_layer)
                dims -= 3
            else:
                output_layer = klayers.AveragePooling2D(2, 2)(output_layer)
                dims /= 2

        # XXX 16 - hardcoded
        to_flatten = klayers.Conv2D(16, 1,
                                    name='to_flatten_value_head',
                                    padding='valid',
                                    activation=activation)(output_layer)

        if conf.dropout_rate_value > 0:
            to_flatten = klayers.Dropout(conf.dropout_rate_value)(to_flatten)

        flat = klayers.Flatten()(to_flatten)

        value_head = klayers.Dense(num_value_heads,
                                   activation="sigmoid", name="value")(flat)

    else:
        # old way, as per AG0
        to_flatten = conv2d_block(1, 1,
                                  name='to_flatten_value_head',
                                  padding='valid',
                                  activation=activation)(layer)
        flat = klayers.Flatten()(to_flatten)

        hidden = klayers.Dense(conf.value_hidden_size, name="value_hidden_layer",
                               activation="relu")(flat)

        if conf.dropout_rate_value > 0:
            hidden = klayers.Dropout(conf.dropout_rate_value)(hidden)

        value_head = klayers.Dense(num_value_heads,
                                   activation="sigmoid", name="value")(hidden)

    # model:
    outputs = policy_heads + [value_head]

    return keras_models.Model(inputs=[inputs_board], outputs=outputs)
