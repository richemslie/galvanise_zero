from keras import layers as klayers
from keras import metrics, models


def top_2_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


def top_3_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def Conv2D_WithBN(*args, **kwds):
    activation = None
    if "activation" in kwds:
        activation = kwds.pop("activation")

    def f(l0):
        l1 = klayers.Conv2D(*args, **kwds)(l0)
        l2 = klayers.BatchNormalization()(l1)
        l3 = klayers.Activation(activation)(l2)
        return l3
    return f

def IdentityBlock(*args, **kwds):
    assert "padding" not in kwds
    kwds["padding"] = "same"

    if "name" in kwds:
        res_name = kwds.pop("name")
        # XXX use the name

    # all other args/kwds passed through to Conv2D_WithBN

    def block(tensor):
        l1 = Conv2D_WithBN(*args, **kwds)(tensor)
        l2 = Conv2D_WithBN(*args, **kwds)(l1)
        res = klayers.add([tensor, l2])
        return klayers.Activation("relu")(res)

    return block


# XXX put number_of_outputs on config
def get_network_model(config, number_of_outputs):
    CNN_FILTERS_SIZE = 128
    RESIDUAL_LAYERS = 3

    inputs_board = klayers.Input(shape=(config.num_rows,
                                        config.num_cols,
                                        config.num_channels))

    assert config.number_of_non_cord_states
    inputs_other = klayers.Input(shape=(config.number_of_non_cord_states,))

    # CNN/Resnet on cords
    #####################

    x = Conv2D_WithBN(CNN_FILTERS_SIZE, 3,
                      padding='same',
                      activation='relu')(inputs_board)

    for _ in range(RESIDUAL_LAYERS):
        x = IdentityBlock(CNN_FILTERS_SIZE, 3)(x)

    # keep adding layers til network gets too small
    size = min(config.num_rows, config.num_cols)
    while True:
        x = Conv2D_WithBN(CNN_FILTERS_SIZE, 3,
                              padding='valid',
                              activation='relu')(x)

        size -= 2
        if size < 5:
            break

    x = Conv2D_WithBN(CNN_FILTERS_SIZE, 1, padding='valid', activation='relu')(x)
    flattened = klayers.Flatten()(x)

    # FC on other non-cord states
    #############################
    nc_layer_count = min(config.number_of_non_cord_states * 2, 256)
    nc_layer = klayers.Dense(nc_layer_count, activation="relu", name="nc_layer")(inputs_other)
    nc_layer = klayers.BatchNormalization()(nc_layer)

    # concatenate the two
    #####################
    resultant_layer = klayers.concatenate([flattened, nc_layer], axis=-1)

    # add in dropout
    ################
    dropout_layer = klayers.Dropout(0.5)(resultant_layer)

    # the policy
    output_policy = klayers.Dense(number_of_outputs, activation="softmax", name="policy")(dropout_layer)

    # two scores for each player.  the score of the node (acts as a prior).  the final score of the
    # game (back propagated)
    scores_prelude0 = klayers.Dense(32, activation="relu")(dropout_layer)
    scores_prelude0 = klayers.Dropout(0.5)(scores_prelude0)

    scores_prelude1 = klayers.Dense(32, activation="relu")(dropout_layer)
    scores_prelude1 = klayers.Dropout(0.5)(scores_prelude1)

    # best
    output_score0 = klayers.Dense(2, activation="sigmoid", name="score0")(scores_prelude0)

    # final
    output_score1 = klayers.Dense(2, activation="sigmoid", name="score1")(scores_prelude1)

    model = models.Model(inputs=[inputs_board, inputs_other], outputs=[output_policy, output_score0, output_score1])

    # XXX kind of want loss_weights to change over time - pretty easy, see https://github.com/fchollet/keras/issues/2595
    # XXX try different optimizers...
    model.compile(loss=['categorical_crossentropy', 'mean_squared_error', 'mean_squared_error'],
                  optimizer='adam', loss_weights=[1.0, 1.0, 0.5], metrics=["acc", top_2_acc, top_3_acc])

    return model
