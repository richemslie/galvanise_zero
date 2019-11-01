import sys

from ggpzero.defs import confs, templates

from ggpzero.nn.manager import get_manager
from ggpzero.nn import train


def get_train_config(game, gen_prefix, next_step, starting_step):
    config = confs.TrainNNConfig(game)

    config.next_step = next_step
    config.starting_step = starting_step

    config.generation_prefix = gen_prefix
    config.batch_size = 1024
    config.compile_strategy = "SGD"
    config.epochs = 10

    config.learning_rate = 0.01

    config.overwrite_existing = False
    config.use_previous = False
    config.validation_split = 0.95000
    config.resample_buckets = [[200, 1.0]]
    config.max_epoch_size = 1048576 * 2

    return config


def get_nn_model(game, transformer, size="small"):
    config = templates.nn_model_config_template(game, size, transformer, features=True)

    config.cnn_filter_size = 96
    config.residual_layers = 6
    config.value_hidden_size = 512

    config.dropout_rate_policy = 0.25
    config.dropout_rate_value = 0.5

    # config.concat_all_layers = True
    # config.global_pooling_value = False

    config.concat_all_layers = False
    config.global_pooling_value = True

    return config


def do_training(game, gen_prefix, next_step, starting_step, num_previous_states,
                gen_prefix_next, do_data_augmentation=False):

    man = get_manager()

    # create a transformer
    generation_descr = templates.default_generation_desc(game,
                                                         multiple_policy_heads=True,
                                                         num_previous_states=num_previous_states)
    transformer = man.get_transformer(game, generation_descr)

    # create train_config
    train_config = get_train_config(game, gen_prefix, next_step, starting_step)
    trainer = train.TrainManager(train_config, transformer, do_data_augmentation=do_data_augmentation)
    trainer.update_config(train_config, next_generation_prefix=gen_prefix_next)

    # get the nn model and set on trainer
    nn_model_config = get_nn_model(train_config.game, transformer)
    trainer.get_network(nn_model_config, generation_descr)

    trainer.do_epochs()
    trainer.save()


if __name__ == "__main__":

    def main(args):
        gen_prefix_next = sys.argv[1]

        # modify these >>>
        game = "hex_lg_19"
        gen_prefix = "h2"

        next_step = 220
        starting_step = 0
        num_previous_states = 1

        do_training(game, gen_prefix, next_step, starting_step,
                    num_previous_states, gen_prefix_next, do_data_augmentation=True)

    from ggpzero.util.main import main_wrap
    main_wrap(main)
