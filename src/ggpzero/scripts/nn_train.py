import sys

from ggpzero.defs import confs, templates

from ggpzero.nn import train
from ggpzero.nn.manager import get_manager


def get_train_config(game, gen_prefix, next_step, starting_step):
    config = confs.TrainNNConfig(game)

    config.next_step = next_step
    config.starting_step = starting_step

    config.generation_prefix = gen_prefix
    config.batch_size = 512
    config.compile_strategy = "adam"
    config.epochs = 8
    config.learning_rate = None
    config.overwrite_existing = False
    config.use_previous = False
    config.validation_split = 0.90000

    config.resample_buckets = []

    return config


def get_nn_model(game, transformer, size="small"):
    config = templates.nn_model_config_template(game, size, transformer)
    assert config.multiple_policies
    assert config.cnn_kernel_size == 3
    assert not config.l2_regularisation

    config.cnn_filter_size = 64
    config.residual_layers = 6

    config.dropout_rate_policy = 0.25
    config.dropout_rate_value = 0.5

    config.residual_layers = 8
    config.role_count = 2
    config.value_hidden_size = 128
    config.leaky_relu = False

    return config


def do_training(game, gen_prefix, next_step, starting_step, num_previous_states, gen_prefix_next):
    man = get_manager()

    # create a transformer
    generation_descr = templates.default_generation_desc(game,
                                                         multiple_policy_heads=True,
                                                         num_previous_states=num_previous_states)
    transformer = man.get_transformer(game, generation_descr)

    # create train_config
    train_config = get_train_config(game, gen_prefix, next_step, starting_step)
    trainer = train.TrainManager(train_config, transformer)
    trainer.update_config(train_config, next_generation_prefix=gen_prefix_next)

    # get the nn model and set on trainer
    nn_model_config = get_nn_model(train_config.game, transformer)
    trainer.get_network(nn_model_config, generation_descr)

    data = trainer.gather_data()

    trainer.do_epochs(data)
    trainer.save()


if __name__ == "__main__":

    def main(args):
        gen_prefix_next = sys.argv[1]

        # modify these >>>
        game = "hexLG13"
        gen_prefix = "h1"
        next_step = 38
        starting_step = 3
        num_previous_states = 1

        do_training(game, gen_prefix, next_step, starting_step, num_previous_states, gen_prefix_next)

    from ggpzero.util.main import main_wrap
    main_wrap(main)
