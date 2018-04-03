''' tests raw speed of predictions on GPU. '''

import gc
import time

import numpy as np

from ggpzero.defs import confs, templates

from ggpzero.nn.manager import get_manager
from ggpzero.nn import train


def config():
    conf = confs.TrainNNConfig("reversi")
    conf.generation_prefix = "x4"
    conf.overwrite_existing = True
    conf.next_step = 20
    conf.validation_split = 1.0
    conf.starting_step = 10
    return conf


class Runner:
    def __init__(self, data, keras_model):
        self.data = data
        self.keras_model = keras_model
        self.sample_count = len(data.inputs)

    def warmup(self):
        res = []
        for i in range(2):
            idx, end_idx = i * batch_size, (i + 1) * batch_size
            print i, idx, end_idx
            inputs = np.array(data.inputs[idx:end_idx])
            res.append(keras_model.predict(inputs, batch_size=batch_size))
        print res


def perf_test(batch_size):
    gc.collect()
    print 'Starting speed run'
    num_batches = sample_count / batch_size + 1
    print "batches %s, batch_size %s, inputs: %s" % (num_batches,
                                                     batch_size,
                                                     len(data.inputs))

def go():
    ITERATIONS = 3

    man = get_manager()

    # get data
    train_config = config()

    # get nn to test speed on
    transformer = man.get_transformer(train_config.game)
    trainer = train.TrainManager(train_config, transformer)

    nn_model_config = templates.nn_model_config_template(train_config.game, "small", transformer)
    generation_descr = templates.default_generation_desc(train_config.game)
    trainer.get_network(nn_model_config, generation_descr)

    data = trainer.gather_data()

    r = Runner(trainer.gather_data(), trainer.nn.get_model())
    r.warmup()


def speed_test():
    ITERATIONS = 3

    man = get_manager()

    # get data
    train_config = config()

    # get nn to test speed on
    transformer = man.get_transformer(train_config.game)
    trainer = train.TrainManager(train_config, transformer)

    nn_model_config = templates.nn_model_config_template(train_config.game, "small", transformer)
    generation_descr = templates.default_generation_desc(train_config.game)
    trainer.get_network(nn_model_config, generation_descr)

    data = trainer.gather_data()

    res = []

    batch_size = 4096
    sample_count = len(data.inputs)
    keras_model = trainer.nn.get_model()

    # warm up
    for i in range(2):
        idx, end_idx = i * batch_size, (i + 1) * batch_size
        print i, idx, end_idx
        inputs = np.array(data.inputs[idx:end_idx])
        res.append(keras_model.predict(inputs, batch_size=batch_size))
        print res[0]

    for _ in range(ITERATIONS):
        res = []
        times = []
        gc.collect()

        print 'Starting speed run'
        num_batches = sample_count / batch_size + 1
        print "batches %s, batch_size %s, inputs: %s" % (num_batches,
                                                         batch_size,
                                                         len(data.inputs))
        for i in range(num_batches):
            idx, end_idx = i * batch_size, (i + 1) * batch_size
            inputs = np.array(data.inputs[idx:end_idx])
            print "inputs", len(inputs)
            s = time.time()
            Y = keras_model.predict(inputs, batch_size=batch_size)
            times.append(time.time() - s)
            print "outputs", len(Y[0])

        print "times taken", times
        print "total_time taken", sum(times)
        print "predictions per second", sample_count / float(sum(times))


if __name__ == "__main__":
    from ggpzero.util.main import main_wrap
    main_wrap(go)
