from keras import layers as klayers
from keras import metrics, models


def top_2_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


def top_3_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def get_network_model(config, number_of_outputs):
    ''' same as above (well nearly, BatchNormalization done after activation...) '''

    inputs_board = klayers.Input(shape=(config.num_rows,
                                        config.num_cols,
                                        config.num_channels))

    assert config.number_of_non_cord_states
    inputs_other = klayers.Input(shape=(config.number_of_non_cord_states,))

    # CNN
    #####
    layer = inputs_board
    for i in range(3):
        layer = klayers.Conv2D(128, 3,
                               padding='same',
                               activation='relu')(layer)

        nlayer = klayers.BatchNormalization()(layer)

    # keep adding layers til network gets too small
    size = min(config.num_rows, config.num_cols)
    while True:
        layer = klayers.Conv2D(256, 3,
                               padding='valid',
                               activation='relu')(nlayer)
        nlayer = klayers.BatchNormalization()(layer)

        size -= 2
        if size < 5:
            break

    flattened = klayers.Flatten()(nlayer)

    # FC
    ####
    fc_layer = klayers.Dense(config.number_of_non_cord_states * 2, activation="relu")(inputs_other)

    # combine
    #########
    resultant_layer = klayers.concatenate([flattened, fc_layer], axis=-1)

    # lots of dropout
    dropout_layer = klayers.Dropout(0.75)(resultant_layer)

    # the policy
    output_policy = klayers.Dense(number_of_outputs, activation="softmax", name="policy")(dropout_layer)

    # four scores.  Best move score, final result score for the game - for each player
    scores_prelude = klayers.Dense(number_of_outputs, activation="sigmoid")(dropout_layer)
    output_scores = klayers.Dense(4, name="scores")(scores_prelude)

    model = models.Model(inputs=[inputs_board, inputs_other], outputs=[output_policy, output_scores])

    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                  optimizer='adam', loss_weights=[1.0, 0.75], metrics=["acc", top_2_accuracy, top_3_accuracy])

    return model