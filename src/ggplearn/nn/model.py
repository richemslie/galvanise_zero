from keras import layers as klayers
from keras import models
from keras.regularizers import l2


def Conv2DBlock(*args, **kwds):
    activation = None
    if "activation" in kwds:
        activation = kwds.pop("activation")

    # XXX augment name
    # if "name" in kwds:
    #    res_name = kwds.pop("name")

    def block(x):
        x = klayers.Conv2D(*args, **kwds)(x)
        x = klayers.BatchNormalization()(x)
        x = klayers.Activation(activation)(x)
        return x
    return block


def ResidualBlock(*args, **kwds):
    assert "padding" not in kwds
    kwds["padding"] = "same"

    # XXX augment name
    # if "name" in kwds:
    #    res_name = kwds.pop("name")

    # all other args/kwds passed through to Conv2DBlock

    def block(tensor):
        x = Conv2DBlock(*args, **kwds)(tensor)
        x = Conv2DBlock(*args, **kwds)(x)
        x = klayers.add([tensor, x])
        return klayers.Activation("relu")(x)

    return block


def get_network_model(config, **kwds):

    class AttrDict(dict):
        def __getattr__(self, name):
            return self[name]

    network_size = kwds.get("network_size", "normal")
    if network_size == "tiny":
        params = AttrDict(CNN_FILTERS_SIZE=32,
                          RESIDUAL_BLOCKS=1,
                          MAX_HIDDEN_SIZE_NC=16)

    elif network_size == "smaller":
        params = AttrDict(CNN_FILTERS_SIZE=48,
                          RESIDUAL_BLOCKS=2,
                          MAX_HIDDEN_SIZE_NC=64)

    elif network_size == "small":
        params = AttrDict(CNN_FILTERS_SIZE=64,
                          RESIDUAL_BLOCKS=3,
                          MAX_HIDDEN_SIZE_NC=128)
    elif network_size == "normal":
        params = AttrDict(CNN_FILTERS_SIZE=128,
                          RESIDUAL_BLOCKS=6,
                          MAX_HIDDEN_SIZE_NC=256)
    elif network_size == "big":
        params = AttrDict(CNN_FILTERS_SIZE=192,
                          RESIDUAL_BLOCKS=8,
                          MAX_HIDDEN_SIZE_NC=256)

    params.update(dict(ALPHAZERO_REGULARISATION=kwds.get("a0_reg", False)))
    params.update(dict(DO_DROPOUT=kwds.get("dropout", True)))

    # fancy l2 regularizer stuff I will understand one day
    ######################################################
    reg_params = {}
    if params.ALPHAZERO_REGULARISATION:
        reg_params["kernel_regularizer"] = l2(1e-4)

    # inputs:
    #########
    inputs_board = klayers.Input(shape=(config.num_rows,
                                        config.num_cols,
                                        config.num_channels))

    assert config.number_of_non_cord_states
    inputs_other = klayers.Input(shape=(config.number_of_non_cord_states,))

    # CNN/Resnet on cords
    #####################
    layer = Conv2DBlock(params.CNN_FILTERS_SIZE, 3,
                        padding='same',
                        activation='relu', **reg_params)(inputs_board)

    for _ in range(params.RESIDUAL_BLOCKS):
        layer = ResidualBlock(params.CNN_FILTERS_SIZE, 3)(layer)

    # number of roles + 1
    res_policy_out = Conv2DBlock(config.role_count + 1, 1,
                                 padding='valid', activation='relu', **reg_params)(layer)

    res_score_out = Conv2DBlock(2, 1, padding='valid', activation='relu', **reg_params)(layer)
    res_policy_out = klayers.Flatten()(res_policy_out)
    res_score_out = klayers.Flatten()(res_score_out)

    if params.DO_DROPOUT:
        res_policy_out = klayers.Dropout(0.333)(res_policy_out)
        res_score_out = klayers.Dropout(0.5)(res_score_out)

    # FC on other non-cord states
    #############################
    nc_layer_count = min(config.number_of_non_cord_states * 2, params.MAX_HIDDEN_SIZE_NC)
    nc_layer = klayers.Dense(nc_layer_count, activation="relu", name="nc_layer", **reg_params)(inputs_other)
    nc_layer = klayers.BatchNormalization()(nc_layer)

    # output: policy
    ################
    prelude_policy = klayers.concatenate([res_policy_out, nc_layer], axis=-1)
    output_policy = klayers.Dense(config.policy_dist_count,
                                  activation="softmax", name="policy", **reg_params)(prelude_policy)

    # output: score
    ###############
    prelude_scores = klayers.concatenate([res_score_out, nc_layer], axis=-1)
    prelude_scores = klayers.Dense(32, activation="relu", **reg_params)(prelude_scores)

    output_score = klayers.Dense(config.final_score_count,
                                 activation="sigmoid", name="score", **reg_params)(prelude_scores)

    # model
    #######
    return models.Model(inputs=[inputs_board, inputs_other],
                        outputs=[output_policy, output_score])
