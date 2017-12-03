from keras import layers as klayers
from keras import metrics, models

def top_3_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_5_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

def get_network_model(config, number_of_outputs):
    ''' same as above (well nearly, BatchNormalization done after activation...) '''


    inputs1 = klayers.Input(shape=(config.num_rows,
                                   config.num_cols,
                                   config.num_channels))

    assert config.number_of_non_cord_states
    inputs2 = klayers.Input(shape=(config.number_of_non_cord_states,))

    ####### CNN
    layer = inputs1
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

    ####### FC
    fc_layer = klayers.Dense(config.number_of_non_cord_states * 2, activation="relu")(inputs2)

    ####### combine
    resultant_layer = klayers.concatenate([flattened, fc_layer], axis=-1)

    # lots of dropout
    dropout_layer = klayers.Dropout(0.5)(resultant_layer)

    # and some extra layer to see things out
    output_policy = klayers.Dense(number_of_outputs, activation="softmax", name="policy")(dropout_layer)
    scores_prelude = klayers.Dense(number_of_outputs, activation="sigmoid")(dropout_layer)
    output_scores = klayers.Dense(number_of_outputs, name="scores")(scores_prelude)

    model = models.Model(inputs=[inputs1, inputs2], outputs=[output_policy, output_scores])

    # add a bunch of regression metrics
    # XXX not sure about softmax and binary_crossentropy
    model.compile(loss=['binary_crossentropy', 'mean_squared_error'],
                  optimizer='adam',
                  metrics=["acc", top_3_accuracy, top_5_accuracy])

    #model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
    #              optimizer='adam',
    #              metrics=["acc", metrics.categorical_accuracy, top_2_accuracy, top_3_accuracy, top_5_accuracy])

    return model
