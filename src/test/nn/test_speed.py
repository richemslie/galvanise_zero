import gc
import os
import time

import numpy as np

from ggplib.db import lookup

from ggpzero.defs import msgs

from ggpzero.nn.manager import get_manager
from ggpzero.training.nn_train import parse


def config():
    conf = msgs.TrainNNRequest("breakthrough")
    conf.store_path = os.path.join(os.environ["GGPZERO_PATH"], "data", "breakthrough", "v5")
    conf.use_previous = False
    conf.next_step = 79
    conf.validation_split = 1.0
    conf.max_sample_count = 500000
    return conf


def speed_test():
    ITERATIONS = 3

    man = get_manager()

    # get data
    conf = config()

    # get nn to test speed on
    generation = "v5_0"
    nn = man.load_network(conf.game, generation)
    nn.summary()
    keras_model = nn.get_model()
    print keras_model

    game_info = lookup.by_name(conf.game)
    train_conf = parse(conf, game_info, man.get_transformer(conf.game))

    res = []

    batch_size = 1024
    input_0 = train_conf.inputs[0]
    sample_count = len(input_0)

    # warm up
    for i in range(2):
        idx, end_idx = i * batch_size, (i + 1) * batch_size
        print i, idx, end_idx
        inputs = np.array(input_0[idx:end_idx])
        res.append(keras_model.predict(inputs, batch_size=conf.batch_size))
        print res[0]

    for _ in range(ITERATIONS):
        res = []
        times = []
        gc.collect()

        print 'Starting speed run'
        num_batches = sample_count / batch_size + 1
        print "batches %s, batch_size %s, inputs: %s" % (num_batches,
                                                         batch_size,
                                                         len(input_0))
        for i in range(num_batches):
            idx, end_idx = i * batch_size, (i + 1) * batch_size
            inputs = np.array(input_0[idx:end_idx])
            print "inputs", len(inputs)
            s = time.time()
            res.append(keras_model.predict(inputs, batch_size=batch_size))
            times.append(time.time() - s)
            print "outputs", len(res), sum(len(r) for r in res), len(res[0])

        print "times taken", times
        print "total_time taken", sum(times)
        print "predictions per second", sample_count / float(sum(times))


if __name__ == "__main__":
    from ggpzero.util.main import main_wrap
    main_wrap(speed_test)
