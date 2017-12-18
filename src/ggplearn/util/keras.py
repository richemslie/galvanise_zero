from __future__ import absolute_import


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

    from keras import backend
    backend.set_session(sess)


def constrain_resources():
    from keras import backend
    if backend.backend() == "tensorflow":
        constrain_resources_tf()
