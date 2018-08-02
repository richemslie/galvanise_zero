from __future__ import absolute_import
from ggplib.util import log


from keras.optimizers import SGD, Adam
from keras.utils.generic_utils import Progbar
import keras.callbacks as keras_callbacks
from keras import metrics as keras_metrics
import keras.backend as K

from keras import models as keras_models
from keras import layers as keras_layers
from keras import regularizers as keras_regularizers


def _bla():
    ' i am here to confuse flake8 '
    print SGD, Adam, Progbar, keras_callbacks, keras_metrics
    print keras_models, keras_layers, keras_regularizers


def is_channels_first():
    ' NCHW is cuDNN default, and what tf wants for GPU. '
    return K.image_data_format() == "channels_first"


def antirectifier(inputs):
    inputs -= K.mean(inputs, axis=1, keepdims=True)
    inputs = K.l2_normalize(inputs, axis=1)
    pos = K.relu(inputs)
    neg = K.relu(-inputs)
    return K.concatenate([pos, neg], axis=1)


def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)


def get_antirectifier(name):
    # output_shape=antirectifier_output_shape
    return keras_layers.Lambda(antirectifier, name=name)


def constrain_resources_tf():
    ' constrain resource as tensorflow likes to assimilate your machine rendering it useless '

    import tensorflow as tf
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    gpu_available = [x.name for x in local_device_protos if x.device_type == 'GPU']

    if not gpu_available:
        # this doesn't strictly use just one cpu... but seems it is the best one can do
        config = tf.ConfigProto(device_count=dict(CPU=1),
                                allow_soft_placement=False,
                                log_device_placement=False,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25,
                                    allow_growth=True)

        config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session(config=config)

    K.set_session(sess)


def init(data_format='channels_first'):
    assert K.backend() == "tensorflow"

    if K.image_data_format() != data_format:
        was = K.image_data_format()
        K.set_image_data_format(data_format)
        log.warning("Changing image_data_format: %s -> %s" % (was, K.image_data_format()))

    constrain_resources_tf()
