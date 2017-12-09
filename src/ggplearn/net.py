from keras import layers as klayers
from keras import metrics, models
import keras.callbacks

from ggplib.util import log


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
    while size > 5:
        x = Conv2D_WithBN(CNN_FILTERS_SIZE, 3,
                          padding='valid',
                          activation='relu')(x)
        size -= 2

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
    mass_dropout_layer = klayers.Dropout(0.75)(resultant_layer)

    # the policy
    output_policy = klayers.Dense(number_of_outputs, activation="softmax", name="policy")(dropout_layer)

    scores_prelude = klayers.Dense(32, activation="relu")(mass_dropout_layer)
    scores_prelude = klayers.Dropout(0.5)(scores_prelude)

    # final
    output_score = klayers.Dense(2, activation="sigmoid", name="score")(scores_prelude)

    model = models.Model(inputs=[inputs_board, inputs_other], outputs=[output_policy, output_score])

    # slightly less loss on score until it stops overfitting (which is mostly because of lack of
    # data)
    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                  optimizer='adam', loss_weights=[1.0, 0.75], metrics=["acc", top_2_acc, top_3_acc])

    return model


###############################################################################

class MyCallback(keras.callbacks.Callback):
    ''' custom callbac to do nice logging '''
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        assert logs
        log.debug("Epoch %s/%s" % (epoch, self.epochs))

        def str_by_name(names, dp=3):
            fmt = "%%s = %%.%df" % dp
            strs = [fmt % (k, logs[k]) for k in names]
            return ", ".join(strs)

        loss_names = "loss policy_loss score_loss".split()
        val_loss_names = "val_loss val_policy_loss val_score_loss".split()

        log.info(str_by_name(loss_names, 4))
        log.info(str_by_name(val_loss_names, 4))

        # accuracy:
        for output in "policy score".split():
            acc = []
            val_acc = []
            for k in self.params['metrics']:
                if output not in k or "acc" not in k:
                    continue
                if "score" in output and "top" in k:
                    continue

                if 'val' in k:
                    val_acc.append(k)
                else:
                    acc.append(k)

            log.info("%s : %s" % (output, str_by_name(acc)))
            log.info("%s : %s" % (output, str_by_name(val_acc)))


class MyProgbarLogger(keras.callbacks.Callback):
    ''' simple progress bar.  default was breaking with too much metrics '''
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch %d/%d' % (epoch + 1, self.epochs))

        self.target = self.params['samples']

        from keras.utils.generic_utils import Progbar
        self.progbar = Progbar(target=self.target)
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []
            self.progbar.update(self.seen, self.log_values)

    def on_batch_end(self, batch, logs=None):
        self.seen += logs.get('size')

        for k in logs:
            if "loss" in k and "val" not in k:
                self.log_values.append((k, logs[k]))

        self.progbar.update(self.seen, self.log_values)
