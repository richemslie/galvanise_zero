''' Regularisation tricks (tyvm) credit to :https://github.com/mokemokechicken/reversi-alpha-zero
'''

from keras import layers as klayers
from keras import metrics, models
from keras.regularizers import l2
import keras.callbacks
import keras.backend as K

from ggplib.util import log


def top_2_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


def top_3_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


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


def objective_function_for_policy(y_true, y_pred):
    return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)


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
    else:
        params = AttrDict(CNN_FILTERS_SIZE=128,
                          RESIDUAL_BLOCKS=6,
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


###############################################################################

class MyCallback(keras.callbacks.Callback):
    ''' custom callbac to do nice logging '''
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        assert logs
        log.debug("Epoch %s/%s" % (epoch + 1, self.epochs))

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
