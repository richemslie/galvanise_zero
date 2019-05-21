from ggpzero.util.attrutil import register_attrs, attribute, attr_factory

from ggpzero.defs.datadesc import GenerationDescription


# DO NOT IMPORT msgs.py


@register_attrs
class PUCTEvaluatorConfig(object):
    verbose = attribute(False)

    # root level minmax ing, an old galvanise nn idea.  Expands the root node, and presets visits.
    # -1 off.
    root_expansions_preset_visits = attribute(-1)

    puct_constant = attribute(0.85)

    # added to root child policy pct (< 0 is off)
    dirichlet_noise_pct = attribute(0.25)

    # XXX experimental, feature likely to go away
    # policy squashing during noise will squash any probabilities in policy over
    # noise_policy_squash_prob to noise_policy_squash_prob.
    # the pct is whether it will activate or not during setting noise (< 0 is off)
    noise_policy_squash_pct = attribute(-1)
    noise_policy_squash_prob = attribute(0.05)

    # looks up method() to use.  one of (choose_top_visits | choose_temperature)
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
    fpu_prior_discount = attribute(-1.0)
    fpu_prior_discount_root = attribute(-1.0)

    top_visits_best_guess_converge_ratio = attribute(0.85)
    evaluation_multipler_to_convergence = attribute(2.0)


@register_attrs
class PUCTEvaluatorV2Config(object):
    verbose = attribute(False)

    puct_constant = attribute(0.75)
    puct_constant_root = attribute(2.5)

    # added to root child policy pct (< 0 is off)
    dirichlet_noise_pct = attribute(0.25)

    # XXX experimental, feature likely to go away
    # policy squashing during noise will squash any probabilities in policy over
    # noise_policy_squash_prob to noise_policy_squash_prob.
    # the pct is whether it will activate or not during setting noise (< 0 is off)
    noise_policy_squash_pct = attribute(-1)
    noise_policy_squash_prob = attribute(0.05)

    # looks up method() to use.  one of (choose_top_visits | choose_temperature)
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
    fpu_prior_discount = attribute(-1.0)
    fpu_prior_discount_root = attribute(-1.0)

    minimax_backup_ratio = attribute(0.75)

    top_visits_best_guess_converge_ratio = attribute(0.8)

    think_time = attribute(10.0)
    converged_visits = attribute(5000)

    # batches to GPU.  number of greenlets to run, along with virtual lossesa
    batch_size = attribute(32)

    # experimental for long running searches: extra_uct_exploration
    extra_uct_exploration = attribute(-1.0)


@register_attrs
class PUCTPlayerConfig(object):
    name = attribute("Player")

    verbose = attribute(False)

    # XXX these should be renamed, and values less abused (0, -1 have special meaning)
    playouts_per_iteration = attribute(800)
    playouts_per_iteration_noop = attribute(1)

    generation = attribute("latest")

    # one of PUCTEvaluatorConfig/PUCTEvaluatorV2Config
    evaluator_config = attribute(default=attr_factory(PUCTEvaluatorV2Config))


@register_attrs
class SelfPlayConfig(object):
    # In each full game played out will oscillate between using sample_iterations and n <
    # evals_per_move.  so if set to 25% will take 25% of samples, and 75% will be skipped using n
    # evals.  This idea is adopted from KataGo and is NOT a full implementation of the idea there.
    # This is just the simplest way to introduce concept without changing much code.  < 0, off.
    oscillate_sampling_pct = attribute(0.25)

    # temperature for policy
    temperature_for_policy = attribute(1.0)

    # sample is the actual sample we take to train for.  The focus is on good policy distribution.
    puct_config = attribute(default=attr_factory(PUCTEvaluatorConfig))
    evals_per_move = attribute(800)

    # resign
    # two levels, resign0 should have more freedom than resign1
    resign0_score_probability = attribute(0.9)
    resign0_pct = attribute(0.5)

    resign1_score_probability = attribute(0.975)
    resign1_pct = attribute(0.1)

    # run to end after resign - pct -> chance to actually run, score to exit on
    run_to_end_pct = attribute(0.2)
    run_to_end_evals = attribute(42)
    run_to_end_puct_config = attribute(default=attr_factory(PUCTEvaluatorConfig))
    run_to_end_early_score = attribute(0.01)
    run_to_end_minimum_game_depth = attribute(30)

    # aborts play if play depth exceeds this max_length (-1 off)
    abort_max_length = attribute(-1)

    # look back to see if states are draw
    number_repeat_states_draw = attribute(-1)

    # score to back prop, to try and avoid repeat states
    repeat_states_score = attribute(0.49)


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

    # the size of policy distribution.
    policy_dist_count = attribute(default=attr_factory(list))

    # < 0 - no dropout
    dropout_rate_policy = attribute(0.333)
    dropout_rate_value = attribute(0.5)

    leaky_relu = attribute(False)
    squeeze_excite_layers = attribute(False)
    resnet_v2 = attribute(False)
    global_pooling_value = attribute(False)


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
    compile_strategy = attribute("SGD")
    learning_rate = attribute(0.03)
    l2_regularisation = attribute(0.0001)

    # list of tuple.  This is the replay buffer.

    # [(5, 1.0), (10, 0.8)]
    # Will take the first 5 generations with all data and 80% of the next 10 generations.  Every
    # generation after is ignored.

    # [(-1, 1.0)]
    # Will take all generations with 100% data.

    resample_buckets = attribute(default=attr_factory(list))

    # set the maximum size for an epoch.  buckets will be scaled accordingly.
    max_epoch_size = attribute(-1)

    # set the initial weight before for the first epoch between training on the next epoch the
    # value weight will automatically adjust based on whether overfitting occurs
    initial_value_weight = attribute(1.0)


@register_attrs
class WorkerConfig(object):
    connect_port = attribute(9000)
    connect_ip_addr = attribute("127.0.0.1")
    do_training = attribute(False)
    do_self_play = attribute(False)
    self_play_batch_size = attribute(1)

    # passed into Supervisor, used instead of hard coded value.
    number_of_polls_before_dumping_stats = attribute(1024)

    # use to create SelfPlayManager.
    unique_identifier = attribute("pleasesetme")

    # number of threads to use during self play.  if this is set to zero, will do inline (no threads).
    num_workers = attribute(0)

    # slow things down (this is to prevent overheating GPU) [only if inline, ie num_workers == 0]
    sleep_between_poll = attribute(-1)

    # send back all the samples we have gathered after n seconds -
    # can also act like an application level keep alive
    server_poll_time = attribute(10)

    # the minimum number of samples gathered before sending to the server
    min_num_samples = attribute(128)

    # will exit if there is an update to the config
    exit_on_update_config = attribute(False)

    # dont replace the network every new generation, instead wait n generations
    # Note: lease this at 1.  XXX Remove this?  Not sure how useful it is.
    replace_network_every_n_gens = attribute(1)


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

    # XXX remove
    # this forces the network to be reset to random weights, every n generations
    reset_network_every_n_generations = attribute(-1)
