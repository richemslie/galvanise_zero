import attr

from ggpzero.util.attrutil import register_attrs

# DO NOT IMPORT msgs.py


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
    policy_dist_count = attr.ib(2)

    l2_regularisation = attr.ib(False)

    # < 0 - no dropout
    dropout_rate_policy = attr.ib(0.333)
    dropout_rate_value = attr.ib(0.5)

    learning_rate = attr.ib(0.001)


@register_attrs
class PolicyPlayerConfig(object):
    name = attr.ib("PolicyPlayer")
    verbose = attr.ib(True)
    generation = attr.ib("latest")

    # a training optimisation
    skip_prediction_single_move = attr.ib(True)

    # a random number is chosen between 0 and random_scale, this is used to choose the move by
    # iterative over accumulative probability in the poilicy distribution.
    random_scale = attr.ib(0.5)

    # < 0 is off, a lower temperature (or in other words, a temperture tending to zero) will
    # encourage more random play.  temperature is applied to the probabilities of the policy (in
    # alphago zero paper it talks about applying to the number of visits.  This is essentially the
    # same thing.  It is unlikely this will be ever to set to anything other than 1.  The
    # temperature is instead controlled via incrementing the temperature game as the game is
    # expanded, with a lower bound set with depth_temperature_start (so in other words :
    # new_probability = probability * (1 / conf.temperature * depth)
    # and depth = max(1, (game_depth - conf.depth_temperature_start) * conf.depth_temperature_increment)
    temperature = attr.ib(-1)
    depth_temperature_start = attr.ib(5)
    depth_temperature_increment = attr.ib(0.5)
    depth_temperature_stop = attr.ib(-1)


@register_attrs
class PUCTPlayerConfig(object):
    name = attr.ib("PUCTPlayer")
    verbose = attr.ib(True)
    generation = attr.ib("latest")

    playouts_per_iteration = attr.ib(800)
    playouts_per_iteration_noop = attr.ib(1)

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


# XXX not sure this should be here?
@register_attrs
class TrainData(object):
    input_channels = attr.ib(attr.Factory(list))
    output_policies = attr.ib(attr.Factory(list))
    output_final_scores = attr.ib(attr.Factory(list))

    validation_input_channels = attr.ib(attr.Factory(list))
    validation_output_policies = attr.ib(attr.Factory(list))
    validation_output_final_scores = attr.ib(attr.Factory(list))

    batch_size = attr.ib(512)
    epochs = attr.ib(24)


@register_attrs
class Sample(object):
    # state policy trained on.  This is a tuple of 0/1s.  Effectively a bit array.
    state = attr.ib([0, 0, 0, 1])

    # previous state
    prev_state = attr.ib([1, 0, 0, 1])

    # polict distribution - should sum to 1.
    policy = attr.ib([0, 0, 0.5, 0.5])

    # list of final scores for value head of network - list has same number as number of roles
    final_score = attr.ib([0, 1])

    # game depth at which point sample is taken
    depth = attr.ib(42)

    # total length of game
    game_length = attr.ib(42)

    # conceptually who's turn it is.  It is the role index (into sm.roles) if game has concept of
    # 'turn'.  If not -1.  XXX this is not a GGP concept.  Each player makes a move each turn (it
    # may be a noop).  XXX we should remove this entirely.  Unfortnately we need to keep this to
    # index into the probability distribution.
    lead_role_index = attr.ib(-1)

    match_identifier = attr.ib("agame_421")
    has_resigned = attr.ib(False)
    resign_false_positive = attr.ib(False)
    starting_sample_depth = attr.ib(42)


@register_attrs
class Generation(object):
    game = attr.ib("game")

    # trained with this generation
    with_generation = attr.ib("gen0")

    # number of samples in this generation
    num_samples = attr.ib(1024)

    # the samples (list of Sample)
    samples = attr.ib(attr.Factory(list))


@register_attrs
class SelfPlayConfig(object):
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

    # run system commands after training (copy files to machines etc)
    run_post_training_cmds = attr.ib(default=attr.Factory(list))

    # the self play config
    self_play_config = attr.ib(default=attr.Factory(SelfPlayConfig))
