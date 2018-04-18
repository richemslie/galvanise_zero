from ggpzero.util.attrutil import register_attrs, attribute, attr_factory

from ggpzero.defs.datadesc import GenerationDescription


# DO NOT IMPORT msgs.py

@register_attrs
class PUCTEvaluatorConfig(object):
    verbose = attribute(False)

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

    # popular leela-zero feature: First Play Urgency.  When the policy space is large - this might
    # be neccessary.  If > 0, applies the prior of the parent, minus a discount to unvisited nodes
    # < 0 is off.
    fpu_prior_discount = attribute(-1)


@register_attrs
class PUCTPlayerConfig(object):
    # XXX split this into PUCTPlayerConfig & PUCTEvaluatorConfig

    name = attribute("PUCTPlayer")
    verbose = attribute(True)

    # XXX player only attributes
    generation = attribute("latest")
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

    # popular leela-zero feature: First Play Urgency.  When the policy space is large - this might
    # be neccessary.  If > 0, applies the prior of the parent, minus a discount to unvisited nodes
    # < 0 is off.
    fpu_prior_discount = attribute(-1)

@register_attrs
class SelfPlayConfig(object):
    # -1 is off, and defaults to alpha-zero style
    max_number_of_samples = attribute(4)

    # uses all the same machinery, but instead of scoring, will go play all the way to end
    # -1 off
    sample_to_end_pct = attribute(-1)

    # select will get to the point where we start sampling
    select_puct_config = attribute(default=attr_factory(PUCTEvaluatorConfig))
    select_iterations = attribute(100)

    # sample is the actual sample we take to train for.  The focus is on good policy distribution.
    sample_puct_config = attribute(default=attr_factory(PUCTEvaluatorConfig))
    sample_iterations = attribute(800)

    # after samples, will play to the end using this config
    score_puct_config = attribute(default=attr_factory(PUCTEvaluatorConfig))
    score_iterations = attribute(100)

    # if the probability of losing drops below - then resign
    # and ignore resignation - and continue to end
    # two levels, resign0 should have more freedom than resign1
    resign0_score_probability = attribute(0.9)
    resign0_false_positive_retry_percentage = attribute(0.5)
    resign1_score_probability = attribute(0.975)
    resign1_false_positive_retry_percentage = attribute(0.1)


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

    leaky_relu = attribute(False)


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

    # this is applied even if max_sample_count can't be reached
    starting_step = attribute(0)

    # one of adam / amsgrad/ SGD
    compile_strategy = attribute("adam")
    learning_rate = attribute(None)

    # experimental:
    # list of tuple.  Idea is that at epoch we take a percentage of the samples to train.
    # [(5, 1.0), (10, 0.8), (0, 0.5), (-5, 0.2)]
    # which translates into, take all samples of first 5, 80% of next 10, 50% of next n, and 20% of
    # the last 5.  also assert number of gens is more than sum(abs(k) for k,_ in resample_buckets)
    resample_buckets = attribute(default=attr_factory(list))


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

    # if this is set to zero, will do inline
    num_workers = attribute(0)

    # run system commands to get the neural network isn't in data
    run_cmds_if_no_nn = attribute(default=attr_factory(list))

    # will exit if there is an update to the config
    exit_on_update_config = attribute(False)


@register_attrs
class ServerConfig(object):
    game = attribute("breakthrough")
    generation_prefix = attribute("v42")

    port = attribute(9000)

    current_step = attribute(0)

    # number of samples to acquire before starting to train
    num_samples_to_train = attribute(1024)

    # maximum growth while training
    max_samples_growth = attribute(0.2)

    # the starting generation description
    base_generation_description = attribute(default=attr_factory(GenerationDescription))

    # the base network model
    base_network_model = attribute(default=attr_factory(NNModelConfig))

    # the starting training config
    base_training_config = attribute(default=attr_factory(TrainNNConfig))

    # the self play config
    self_play_config = attribute(default=attr_factory(SelfPlayConfig))

    # save the samples every n seconds
    checkpoint_interval = attribute(60.0 * 5)
