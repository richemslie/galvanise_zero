import sys

from ggpzero.defs import confs, templates

from ggpzero.nn import train
from ggpzero.nn.manager import get_manager


class Configs:
    def breakthrough(self, gen_prefix):
        conf = confs.TrainNNConfig("breakthrough")

        conf.generation_prefix = gen_prefix

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
        conf = confs.TrainNNConfig("reversi")

        conf.generation_prefix = gen_prefix

        conf.use_previous = False
        conf.next_step = 40

        conf.validation_split = 0.9
        conf.batch_size = 256
        conf.epochs = 42
        conf.max_sample_count = 2500000
        conf.starting_step = 10

        conf.drop_dupes_count = 7
        conf.max_epoch_samples_count = 900000

        conf.compile_strategy = "amsgrad"

        return conf


    def c4(self, gen_prefix):
        conf = confs.TrainNNConfig("connectFour")

        conf.generation_prefix = gen_prefix

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
        conf = confs.TrainNNConfig("hex")

        conf.generation_prefix = gen_prefix

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
        conf = confs.TrainNNConfig("speedChess")

        conf.generation_prefix = gen_prefix

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


def get_nn_model(game, transformer, size="medium-small"):
    config = templates.nn_model_config_template(game, size, transformer)

    config.dropout_rate_policy = -1
    config.dropout_rate_value = -1
    config.l2_regularisation = True

    return config


def retrain(args):
    game = sys.argv[1]
    gen_prefix = sys.argv[2]
    gen_prefix_next = sys.argv[3]

    configs = Configs()
    train_config = getattr(configs, game)(gen_prefix)

    generation_descr = templates.default_generation_desc(train_config.game,
                                                         multiple_policy_heads=True)

    # create a transformer
    man = get_manager()

    transformer = man.get_transformer(train_config.game, generation_descr)

    # create the manager
    trainer = train.TrainManager(train_config, transformer, next_generation_prefix=gen_prefix_next)

    nn_model_config = get_nn_model(train_config.game, transformer)
    trainer.get_network(nn_model_config, generation_descr)

    data = trainer.gather_data()

    trainer.do_epochs(data)
    trainer.save()


if __name__ == "__main__":
    from ggpzero.util.main import main_wrap
    main_wrap(retrain)
