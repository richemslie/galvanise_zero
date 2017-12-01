import csv

import numpy as np

from galvanise import interface
interface.initialise_k273(1)

from galvanise import log
from galvanise.nn.config import get_samples_path, get_weights_path, get_network_path, get_config
from galvanise.nn.networks import get_network_model
log.initialise()

import keras.optimizers

###############################################################################
# data

# -1 is all of them
MAX_SAMPLES = -1
RELOAD_MODEL = False

ANALYSE_TIME = 3.0
BATCH_SIZE = 64
NB_EPOCHS = 8

def train(config, nn, X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCHS):
    X_train = np.array(X_train)
    nb_input_layers = X_train.shape[1]

    X_train = X_train.reshape(X_train.shape[0], nb_input_layers, config.nb_rows, config.nb_cols)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], nb_input_layers, config.nb_rows, config.nb_cols)
    y_test = np.array(y_test)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    print "first test sample:"
    print X_test[0]

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    nn.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
           verbose=1, validation_data=(X_test, y_test), shuffle=True)

    score = nn.evaluate(X_test, y_test, verbose=1)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def to_np_and_shuffle(X, y):
    X = np.array(X)
    y = np.array(y)
    shuffle = np.arange(len(y))
    np.random.shuffle(shuffle)
    X = X[shuffle]
    y = y[shuffle]
    return X, y


def build_and_train_nn(config):
    if role_index is None:
        do_scores = True
        role_index = 0

    X = []
    y = []

    csv_file = open(get_samples_path(config.game))
    data = TrainingData(config,
                        csv.reader(csv_file, delimiter=','),
                        do_scores)

    # get a bunch of sample to train:
    count = 0
    filter_by_role_index = None if do_scores else role_index
    print "Gathering data"
    for ri, a, b in data.gen(filter_by_role_index):
        X.append(a)
        y.append(b)

        if MAX_SAMPLES > 0 and count > MAX_SAMPLES:
            break
        count += 1

    print "gathered %s samples" % len(X)

    # shuffle
    print "Shuffling data"
    X, y = to_np_and_shuffle(X, y)

    nn_type = "score" if do_scores else "policy"

    nn = get_network_model(config, do_scores, y.shape[1])

    print("Compiling model")
    if do_scores:
        opt = keras.optimizers.RMSprop(lr=0.0001)
        nn.compile(optimizer=opt, loss="mean_squared_error", metrics=['accuracy'])
    else:
        nn.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    print("Compiled model")

    if RELOAD_MODEL:
        try:
            print "Loading pretrained weights"
            nn.load_weights(get_weights_path(nn_type, config.game, role_index))
        except Exception, e:
            print "Failed to reload pretrained weights...", str(e)
            print "No problem... keeping going"

    nb_samples = len(X)

    # split data into training/validation
    nb_training_samples = int(nb_samples * 0.8)

    # actually train things
    train(config, nn,
          X[:nb_training_samples], y[:nb_training_samples],
          X[nb_training_samples:], y[nb_training_samples:])

    f = open(get_network_path(nn_type, config.game, role_index), "w")
    f.write(nn.to_yaml())
    f.close()

    nn.save_weights(get_weights_path(nn_type, config.game, role_index),
                    overwrite=True)

###############################################################################

if __name__ == "__main__":
    import pdb
    import sys
    import traceback

    config = get_config(sys.argv[1])

    try:
        if nn_type == 'score':
            build_and_train_nn(config)

    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
