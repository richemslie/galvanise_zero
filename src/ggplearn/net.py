
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import metrics

def get_network_model(config, number_of_outputs=1):
    model = Sequential()
    model.add(Convolution2D(128, 3,
                            padding="same",
                            input_shape=(config.num_rows,
                                         config.num_cols,
                                         config.num_channels)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3,
                            padding="same",
                            input_shape=(config.num_rows,
                                         config.num_cols,
                                         config.num_channels)))

    # add layers until smallish
    size = min(config.num_rows, config.num_cols)
    while size > 3:

        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(256, 3))
        size -= 2

    # lost of dropout
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(number_of_outputs * 2))

    model.add(Activation('sigmoid'))

    model.add(Dense(number_of_outputs))

    def top_2_accuracy(y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

    def top_3_accuracy(y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

    def top_5_accuracy(y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

    # add a bunch of regression metrics
    # metrics.mean_squared_logarithmic_error
    # metrics.mean_absolute_percentage_error
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=[metrics.categorical_accuracy, top_2_accuracy, top_3_accuracy, top_5_accuracy])

    return model
