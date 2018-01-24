''' XXX this is not a test  More like a script? '''

import os
import sys

from ggpzero.defs import msgs

from ggpzero.training.nn_train import parse_and_train


class Configs:
    def breakthrough(self, gen_prefix):
        conf = msgs.TrainNNRequest("breakthrough")

        conf.network_size = "medium"
        conf.generation_prefix = gen_prefix
        conf.store_path = os.path.join(os.environ["GGPZERO_PATH"], "data", "breakthrough", "v5")

        conf.use_previous = False
        conf.next_step = 84

        conf.validation_split = 0.9
        conf.batch_size = 256
        conf.epochs = 30
        conf.max_sample_count = 300000
        conf.starting_step = 12
        conf.drop_dupes_count = 3

        return conf

    def reversi(self, gen_prefix):
        conf = msgs.TrainNNRequest("reversi")

        conf.network_size = "medium-large"
        conf.generation_prefix = gen_prefix
        conf.store_path = os.path.join(os.environ["GGPZERO_PATH"], "data", "reversi", "v7")

        conf.use_previous = False
        conf.next_step = 52

        conf.validation_split = 0.9
        conf.batch_size = 256
        conf.epochs = 30
        conf.max_sample_count = 2500000
        conf.starting_step = 5

        # XXX try increasing this 145
        conf.drop_dupes_count = 7
        conf.max_epoch_samples_count = 900000

        return conf

    def c4(self, gen_prefix):
        conf = msgs.TrainNNRequest("connectFour")

        conf.network_size = "medium-small"
        conf.generation_prefix = gen_prefix
        conf.store_path = os.path.join(os.environ["GGPZERO_PATH"], "data", "connectFour", "v7")

        conf.use_previous = False
        conf.next_step = 33

        conf.validation_split = 0.5
        conf.batch_size = 4096
        conf.epochs = 1
        conf.max_sample_count = 270000
        conf.starting_step = 25
        conf.drop_dupes_count = 3
        conf.overwrite_existing = True

        return conf

    def hex(self, gen_prefix):
        conf = msgs.TrainNNRequest("hex")

        conf.network_size = "small"
        conf.generation_prefix = gen_prefix
        conf.store_path = os.path.join(os.environ["GGPZERO_PATH"], "data", "hex", "v7")

        conf.use_previous = False
        conf.next_step = 20

        conf.validation_split = 0.9
        conf.batch_size = 256
        conf.epochs = 20
        conf.max_sample_count = 300000
        conf.starting_step = 3
        conf.drop_dupes_count = 3
        conf.overwrite_existing = True

        return conf

    def speedChess(self, gen_prefix):
        conf = msgs.TrainNNRequest("speedChess")

        conf.network_size = "medium-small"
        conf.generation_prefix = gen_prefix
        conf.store_path = os.path.join(os.environ["GGPZERO_PATH"], "data", "speedChess", "v9")

        conf.use_previous = False
        conf.next_step = 12

        conf.validation_split = 0.9
        conf.batch_size = 128
        conf.epochs = 20
        conf.max_sample_count = 300000
        conf.starting_step = 3
        conf.drop_dupes_count = 3
        conf.overwrite_existing = True

        return conf


if __name__ == "__main__":
    from ggpzero.util.main import main_wrap

    game = sys.argv[1]
    gen_prefix = sys.argv[2]

    configs = Configs()
    train_conf = getattr(configs, game)(gen_prefix)

    def retrain():
        parse_and_train(train_conf)

    main_wrap(retrain)
