from ggpzero.util.attrutil import register_attrs, attribute, attr_factory


# DO NOT IMPORT msgs.py


@register_attrs
class PUCTPlayerConfig(object):
    # XXX split this into PUCTPlayerConfig & PUCTEvaluatorConfig

    name = attribute("PUCTPlayer")
    verbose = attribute(True)

    # XXX player only attributes
    generation = attribute("latest")
    resign_score_value = attribute(-1)
    playouts_per_iteration = attribute(800)
    playouts_per_iteration_noop = attribute(1)
    playouts_per_iteration_resign = attribute(1)

    # root level minmax ing, an old galvanise nn idea.  Expands the root node, and presets visits.
    # -1 off.
    root_expansions_preset_visits = attribute(-1)

    # applies different constant until the following expansions are met
    puct_before_expansions = attribute(4)
    puct_before_root_expansions = attribute(4)

    # the puct constant.  before expansions, and after expansions are met
    puct_constant_before = attribute(0.75)
    puct_constant_after = attribute(0.75)

    # added to root child policy pct (less than 0 is off)
    dirichlet_noise_pct = attribute(0.25)
    dirichlet_noise_alpha = attribute(0.1)

    # looks up method() to use
    choose = attribute("choose_top_visits")

    # debug, only if verbose is true
    max_dump_depth = attribute(2)

    random_scale = attribute(0.5)
    temperature = attribute(1.0)
    depth_temperature_start = attribute(5)
    depth_temperature_increment = attribute(0.5)
    depth_temperature_stop = attribute(10)
    depth_temperature_max = attribute(5.0)


@register_attrs
class SelfPlayConfig(object):
    with_generation = attribute("latest")

    # -1 is off, and defaults to alpha-zero style
    max_number_of_samples = attribute(4)

    # if the probability of losing drops below - then resign
    resign_score_probability = attribute(0.9)

    # ignore resignation - and continue to end
    resign_false_positive_retry_percentage = attribute(0.1)

    # select will get to the point where we start sampling
    select_puct_config = attribute(default=attr_factory(PUCTPlayerConfig))
    select_iterations = attribute(100)

    # sample is the actual sample we take to train for.  The focus is on good policy distribution.
    sample_puct_config = attribute(default=attr_factory(PUCTPlayerConfig))
    sample_iterations = attribute(800)

    # after samples, will play to the end using this config
    score_puct_config = attribute(default=attr_factory(PUCTPlayerConfig))
    score_iterations = attribute(100)


@register_attrs
class NNModelConfig(object):
    role_count = attribute(2)

    input_rows = attribute(8)
    input_columns = attribute(8)
    input_channels = attribute(8)

    residual_layers = attribute(8)
    cnn_filter_size = attribute(64)
    cnn_kernel_size = attribute(3)

    value_hidden_size = attribute(256)

    multiple_policies = attribute(False)

    # the size of policy distribution.  The size of the list will be 1 if not multiple_policies.
    policy_dist_count = attribute(default=attr_factory(list))

    l2_regularisation = attribute(False)

    # < 0 - no dropout
    dropout_rate_policy = attribute(0.333)
    dropout_rate_value = attribute(0.5)


@register_attrs
class TrainNNConfig(object):
    game = attribute("breakthrough")

    # the generation prefix is what defines our models (along with step). Be careful not to
    # overwrite these.
    generation_prefix = attribute("v2_")

    # uses previous network?
    use_previous = attribute(True)
    next_step = attribute(42)
    overwrite_existing = attribute(False)
    validation_split = attribute(0.8)
    batch_size = attribute(32)
    epochs = attribute(10)

    # if the total number of samples is met, will trim the oldest samples
    max_sample_count = attribute(250000)

    # <= 0 off.  Idea is we have more samples in the data, we take a sample a different
    # subset each epoch.
    max_epoch_samples_count = attribute(250000)

    # this is applied even if max_sample_count can't be reached
    starting_step = attribute(0)

    # if we see duplicate states of mre than > n, drop them, keeping the most recent.  < 0 is off.
    drop_dupes_count = attribute(-1)

    # one of adam / amsgrad/ SGD
    compile_strategy = attribute("adam")
    learning_rate = attribute(None)


@register_attrs
class WorkerConfig(object):
    connect_port = attribute(9000)
    connect_ip_addr = attribute("127.0.0.1")
    do_training = attribute(False)
    do_self_play = attribute(False)
    self_play_batch_size = attribute(1)

    # slow things down
    sleep_between_poll = attribute(-1)

    # send back whatever samples we have gather at this - sort of application level keep alive
    server_poll_time = attribute(10)

    # the minimum number of samples gathered before sending to the server
    min_num_samples = attribute(128)

    # if this is set, no threads will be set up to poll
    inline_manager = attribute(False)

    # run system commands to get the neural network isn't in data
    run_cmds_if_no_nn = attribute(default=attr_factory(list))


@register_attrs
class ServerConfig(object):
    port = attribute(9000)

    game = attribute("breakthrough")

    current_step = attribute(0)
    network_size = attribute("normal")

    generation_prefix = attribute("v2_")

    generation_size = attribute(1024)
    max_growth_while_training = attribute(0.2)

    validation_split = attribute(0.8)
    batch_size = attribute(32)
    epochs = attribute(10)

    max_sample_count = attribute(250000)
    drop_dupes_count = attribute(3)

    # this is applied even if max_sample_count can't be reached
    starting_step = attribute(0)

    retrain_network = attribute(False)

    # the self play config
    self_play_config = attribute(default=attr_factory(SelfPlayConfig))

    # save the samples every n seconds
    checkpoint_interval = attribute(60.0 * 5)
