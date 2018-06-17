from ggpzero.defs import confs

from ggpzero.util.keras import is_channels_first, keras_models, keras_regularizers
from ggpzero.util.keras import keras_layers as klayers


def get_bn_axis():
    return 1 if is_channels_first() else 3


def act(x, activation, name):
    if activation == "leakyrelu":
        x = klayers.LeakyReLU(alpha=0.03, name=name)(x)
    else:
        x = klayers.Activation(activation, name=name)(x)
    return x


def bn(x, name):
    return klayers.BatchNormalization(axis=get_bn_axis(),
                                      name=name)(x)


def conv2d_block(*args, **kwds):
    # args, kwds -> passed through to Conv2D

    activation = kwds.pop("activation")
    name = kwds.pop('name')

    def block(x):
        x = klayers.Conv2D(*args,
                           name=name + '_conv2d',
                           **kwds)(x)

        x = bn(x, name + '_bn')
        return act(x, activation, name + '_act')

    return block


def residual_block_v1(cnn_filter_size, cnn_kernel_size,
                      simple=False, dropout=-1, prefix="residual_block_",
                      **kwds):

    # kwds -> passed through to Conv2D
    for n in "padding use_bias name":
        assert n not in kwds

    activation = kwds.pop('activation', "relu")

    def conv(x, name, kernel_size=None):
        name = prefix + name
        if kernel_size is None:
            kernel_size = cnn_kernel_size
        x = klayers.Conv2D(cnn_filter_size, kernel_size,
                           name=name,
                           padding="same",
                           use_bias=False,
                           **kwds)(x)
        return x

    def bn_(x, name):
        return bn(x, prefix + name)

    def act_(x, name):
        return act(x, activation, prefix + name)

    def block(tensor):
        if simple:
            x = conv(tensor, "conv")
            x = bn_(x, "bn")

            if dropout:
                x = klayers.Dropout(dropout)(x)

        else:
            x = conv(tensor, "conv0")
            x = bn_(x, "bn0")
            x = act_(x, "act0")

            x = conv(x, "conv1", kernel_size=1)
            x = bn_(x, "bn1")
            x = act_(x, "act1")

            if dropout:
                x = klayers.Dropout(dropout)(x)

            x = conv(x, "conv2")
            x = bn_(x, "bn2")

        x = klayers.add([tensor, x], name=prefix + "add")
        return act_(x, "act_final")

    return block


def residual_block_v2(filter_size, prev_filter_size, kernel_size, num_convs,
                      dropout=-1, activation="relu", prefix="residual_block_",
                      **kwds):

    # kwds -> passed through to Conv2D
    for n in "padding use_bias name":
        assert n not in kwds

    def conv(x, step):
        name = prefix + "conv%s" % step
        x = klayers.Conv2D(filter_size, kernel_size,
                           name=name,
                           padding="same",
                           use_bias=False,
                           **kwds)(x)
        return x

    def bn_(x, step):
        return bn(x, prefix + "bn_%s" % step)

    def act_(x, step):
        return act(x, activation, prefix + "act_%s" % step)

    def block(tensor):
        x = tensor
        for i in range(num_convs - 1):
            x = bn_(x, i + 1)
            x = act_(x, i + 1)
            x = conv(x, i + 1)

        x = bn_(x, num_convs)
        x = act_(x, num_convs)

        if dropout > 0:
            x = klayers.Dropout(dropout)(x)

        x = conv(x, num_convs)

        if filter_size != prev_filter_size:
            if is_channels_first():
                tensor = klayers.Conv2D(filter_size, (1, 1), activation='linear', padding='same')(tensor)
            else:
                tensor = klayers.Conv2D(filter_size, (1, 1), activation='linear', padding='same')(tensor)

        x = klayers.add([tensor, x], name=prefix + "add")

        return x

    return block


def get_network_model(conf):
    assert isinstance(conf, confs.NNModelConfig)

    activation = 'leakyrelu' if conf.leaky_relu else 'elu'

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

    # XXX config abuse:
    v2 = conf.residual_layers <= 0
    if v2:
        layer = klayers.Conv2D(conf.cnn_filter_size, 1,
                               padding="same",
                               use_bias=False,
                               name='initial-conv')(inputs_board)

        filter_size = prev_filter_size = conf.cnn_filter_size

        # XXX hard coding incr size (to zero)
        incr_size = 0

        # XXX hard coding layers
        for i, c in enumerate([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 2, 2, 3, 1, 1, 1, 1, 1]):
            if i > 0 and i % 3 == 0:
                filter_size += incr_size

            # XXX hard coding dropout
            layer = residual_block_v2(filter_size, prev_filter_size,
                                      conf.cnn_kernel_size,
                                      c,
                                      prefix="ResLayer_%s_" % i,
                                      dropout=0.35,
                                      activation=activation)(layer)
            prev_filter_size = filter_size

        # would make sense to add these, but it seems to hurt more than anything.  Straight line
        # all the way eh?
        # XXX
        # layer = klayers.BatchNormalization(axis=get_bn_axis(),
        #                                    name="final_bn")(layer)
        # layer = act(layer, activation, "final_act")

    else:
        # AG0 way:

        # initial conv2d/Resnet on cords
        layer = conv2d_block(conf.cnn_filter_size, conf.cnn_kernel_size,
                             activation=activation,
                             name='initial')(inputs_board)

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
                                  padding='valid',
                                  activation=activation)(layer)

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

    value_v3 = conf.value_hidden_size == 0
    value_v2 = conf.value_hidden_size < 0
    if value_v3:
        assert conf.input_columns == conf.input_rows
        output_layer = layer
        dims = conf.input_columns
        while dims >= 5:
            if dims % 2 == 1:
                output_layer = klayers.AveragePooling2D(4, 1)(output_layer)
                dims -= 3
            else:
                output_layer = klayers.AveragePooling2D(2, 2)(output_layer)
                dims /= 2

        to_flatten1 = klayers.Conv2D(16, 1,
                                     name='to_flatten1',
                                     padding='valid',
                                     activation=activation)(output_layer)

        to_flatten2 = klayers.Conv2D(2, 1,
                                     name='to_flatten2',
                                     padding='valid',
                                     activation=activation)(layer)

        if conf.dropout_rate_value > 0:
            to_flatten1 = klayers.Dropout(conf.dropout_rate_value)(to_flatten1)
            to_flatten2 = klayers.Dropout(conf.dropout_rate_value)(to_flatten2)

        flat1 = klayers.Flatten()(to_flatten1)
        flat2 = klayers.Flatten()(to_flatten2)
        flat = klayers.concatenate([flat1, flat2])

        value_head = klayers.Dense(conf.role_count,
                                   activation="sigmoid", name="value")(flat)

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

        value_head = klayers.Dense(conf.role_count,
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

        value_head = klayers.Dense(conf.role_count,
                                   activation="sigmoid", name="value")(hidden)

    # model:
    outputs = policy_heads + [value_head]

    model = keras_models.Model(inputs=[inputs_board], outputs=outputs)

    # add in weight decay?  XXX rename conf to reflect it is weight decay and use +ve value instead of hard coded value.
    if conf.l2_regularisation:
        for layer in model.layers:
            # XXX To get global weight decay in keras regularizers have to be added to every layer
            # in the model. In my models these layers are batch normalization (beta/gamma
            # regularizer) and dense/convolutions (W_regularizer/b_regularizer) layers.

            if hasattr(layer, 'kernel_regularizer'):
                # XXX too much?  Is it doubled from paper?  XXX 5e-3 ?
                layer.kernel_regularizer = keras_regularizers.l2(1e-4)

    return model
