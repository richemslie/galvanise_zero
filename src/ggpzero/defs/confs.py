import attr

from ggpzero.util.attrutil import register_attrs


# DO NOT IMPORT msgs.py


@register_attrs
class PUCTPlayerConfig(object):
    # XXX split this into PUCTPlayerConfig & PUCTEvaluatorConfig

    name = attr.ib("PUCTPlayer")
    verbose = attr.ib(True)

    # XXX player only attributes
    generation = attr.ib("latest")
    resign_score_value = attr.ib(-1)
    playouts_per_iteration = attr.ib(800)
    playouts_per_iteration_noop = attr.ib(1)
    playouts_per_iteration_resign = attr.ib(1)

    # root level minmax ing, an old galvanise nn idea.  Expands the root node, and presets visits.
    # -1 off.
    root_expansions_preset_visits = attr.ib(-1)

    # applies different constant until the following expansions are met
    puct_before_expansions = attr.ib(4)
    puct_before_root_expansions = attr.ib(4)

    # the puct constant.  before expansions, and after expansions are met
    puct_constant_before = attr.ib(0.75)
    puct_constant_after = attr.ib(0.75)

    # added to root child policy pct (less than 0 is off)
    dirichlet_noise_pct = attr.ib(0.25)
    dirichlet_noise_alpha = attr.ib(0.1)

    # looks up method() to use
    choose = attr.ib("choose_top_visits")

    # debug, only if verbose is true
    max_dump_depth = attr.ib(2)

    random_scale = attr.ib(0.5)
    temperature = attr.ib(1.0)
    depth_temperature_start = attr.ib(5)
    depth_temperature_increment = attr.ib(0.5)
    depth_temperature_stop = attr.ib(10)
    depth_temperature_max = attr.ib(5.0)


@register_attrs
class SelfPlayConfig(object):
    with_generation = attr.ib("latest")

    # -1 is off, and defaults to alpha-zero style
    max_number_of_samples = attr.ib(4)

    # if the probability of losing drops below - then resign
    resign_score_probability = attr.ib(0.9)

    # ignore resignation - and continue to end
    resign_false_positive_retry_percentage = attr.ib(0.1)

    # select will get to the point where we start sampling
    select_puct_config = attr.ib(default=attr.Factory(PUCTPlayerConfig))
    select_iterations = attr.ib(100)

    # sample is the actual sample we take to train for.  The focus is on good policy distribution.
    sample_puct_config = attr.ib(default=attr.Factory(PUCTPlayerConfig))
    sample_iterations = attr.ib(800)

    # after samples, will play to the end using this config
    score_puct_config = attr.ib(default=attr.Factory(PUCTPlayerConfig))
    score_iterations = attr.ib(100)


@register_attrs
class NNModelConfig(object):
    role_count = attr.ib(2)

    input_rows = attr.ib(8)
    input_columns = attr.ib(8)
    input_channels = attr.ib(8)

    residual_layers = attr.ib(8)
    cnn_filter_size = attr.ib(64)
    cnn_kernel_size = attr.ib(3)

    value_hidden_size = attr.ib(256)
    multiple_policies = attr.ib(False)

    l2_regularisation = attr.ib(False)

    # < 0 - no dropout
    dropout_rate_policy = attr.ib(0.333)
    dropout_rate_value = attr.ib(0.5)

    learning_rate = attr.ib(0.001)


@register_attrs
class TrainNNConfig(object):
    game = attr.ib("breakthrough")

    network_model = attr.ib(default=attr.Factory(NNModelConfig))

    # the generation prefix is what defines our models (along with step). Be careful not to
    # overwrite these.
    generation_prefix = attr.ib("v2_")

    # this is where the generations are stored
    store_path = attr.ib("/home/me/somewhere")

    # uses previous network?
    use_previous = attr.ib(True)
    next_step = attr.ib(42)
    overwrite_existing = attr.ib(False)
    validation_split = attr.ib(0.8)
    batch_size = attr.ib(32)
    epochs = attr.ib(10)

    # if the total number of samples is met, will trim the oldest samples
    max_sample_count = attr.ib(250000)

    # <= 0 off.  Idea is we have more samples in the data, we take a sample a different
    # subset each epoch.
    max_epoch_samples_count = attr.ib(250000)

    # this is applied even if max_sample_count can't be reached
    starting_step = attr.ib(0)

    # if we see duplicate states of mre than > n, drop them, keeping the most recent.  < 0 is off.
    drop_dupes_count = attr.ib(-1)


@register_attrs
class WorkerConfig(object):
    connect_port = attr.ib(9000)
    connect_ip_addr = attr.ib("127.0.0.1")
    do_training = attr.ib(False)
    do_self_play = attr.ib(False)
    self_play_batch_size = attr.ib(1)

    # slow things down
    sleep_between_poll = attr.ib(-1)

    # send back whatever samples we have gather at this - sort of application level keep alive
    server_poll_time = attr.ib(10)

    # the minimum number of samples gathered before sending to the server
    min_num_samples = attr.ib(128)

    # if this is set, no threads will be set up to poll
    inline_manager = attr.ib(False)

    # run system commands to get the network model
    run_cmds_if_no_model = attr.ib(default=attr.Factory(list))


@register_attrs
class ServerConfig(object):
    port = attr.ib(9000)

    game = attr.ib("breakthrough")

    current_step = attr.ib(0)
    network_size = attr.ib("normal")

    generation_prefix = attr.ib("v2_")

    generation_size = attr.ib(1024)
    max_growth_while_training = attr.ib(0.2)

    validation_split = attr.ib(0.8)
    batch_size = attr.ib(32)
    epochs = attr.ib(10)

    max_sample_count = attr.ib(250000)
    drop_dupes_count = attr.ib(3)

    # this is applied even if max_sample_count can't be reached
    starting_step = attr.ib(0)

    retrain_network = attr.ib(False)

    # the self play config
    self_play_config = attr.ib(default=attr.Factory(SelfPlayConfig))

    # save the samples every n seconds
    checkpoint_interval = attr.ib(60.0 * 5)
