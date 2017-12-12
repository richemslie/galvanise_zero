from __future__ import absolute_import


def use_one_cpu_please():
    # this doesn't strictly use just one cpu... but seems it is the best one can do
    import tensorflow as tf
    config = tf.ConfigProto(device_count=dict(CPU=1),
                            allow_soft_placement=False,
                            log_device_placement=False,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    sess = tf.Session(config=config)

    from keras import backend
    backend.set_session(sess)
