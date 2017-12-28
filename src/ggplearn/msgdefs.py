import attr

# XXX separate this into confs and msgs


@attr.s
class TrainData(object):
    inputs = attr.ib()
    outputs = attr.ib()
    validation_inputs = attr.ib()
    validation_outputs = attr.ib()
    batch_size = attr.ib(512)
    epochs = attr.ib(24)


@attr.s
class PolicyPlayerConf(object):
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
    # same thing.  It is unlikely this will be ever to set to anything other than 1.  The temperature is
    # instead controlled via incrementing the temperature game as the game is expanded, with a lower bound set with
    # depth_temperature_start (so in other words :
    # new_probability = probability * (1 / conf.temperature * depth)
    # and depth = max(1, (game_depth - conf.depth_temperature_start) * conf.depth_temperature_increment)
    temperature = attr.ib(-1)
    depth_temperature_start = attr.ib(5)
    depth_temperature_increment = attr.ib(0.5)


@attr.s
class PUCTPlayerConf(object):
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

    # tunes the puct_constant with the initial (predicted) score of the node
    puct_constant_tune = attr.ib(False)

    # added to root child policy pct (less than 0 is off)
    dirichlet_noise_pct = attr.ib(0.25)
    dirichlet_noise_alpha = attr.ib(0.1)

    # MAYBE useful for when small number of iterations.  otherwise pretty much the same
    expand_root = attr.ib(-1)

    # looks up method() to use
    choose = attr.ib("choose_top_visits")

    # debug, only if verbose is true
    max_dump_depth = attr.ib(2)


@attr.s
class ServerConfig(object):
    port = attr.ib(9000)

    game = attr.ib("breakthrough")

    current_step = attr.ib(0)
    network_size = attr.ib("normal")

    generation_prefix = attr.ib("v2_")
    store_path = attr.ib("somewhere")

    player_select_conf = attr.ib(default=attr.Factory(PolicyPlayerConf))
    player_policy_conf = attr.ib(default=attr.Factory(PUCTPlayerConf))
    player_score_conf = attr.ib(default=attr.Factory(PUCTPlayerConf))

    generation_size = attr.ib(1024)
    max_growth_while_training = attr.ib(0.2)

    validation_split = attr.ib(0.8)
    batch_size = attr.ib(32)
    epochs = attr.ib(10)
    max_sample_count = attr.ib(250000)
    retrain_network = attr.ib(False)

    # run system commands after training (copy files to machines etc)
    run_post_training_cmds = attr.ib(default=attr.Factory(list))


@attr.s
class WorkerConf(object):
    connect_port = attr.ib(9000)
    connect_ip_addr = attr.ib("127.0.0.1")
    do_training = attr.ib(False)
    do_self_play = attr.ib(False)
    concurrent_plays = attr.ib(1)


@attr.s
class Ping(object):
    pass


@attr.s
class Pong(object):
    pass


@attr.s
class RequestConfig(object):
    pass


@attr.s
class Ok(object):
    message = attr.ib("ok")


@attr.s
class WorkerConfigMsg(object):
    conf = attr.ib(default=attr.Factory(WorkerConf))


@attr.s
class ConfigureApproxTrainer(object):
    game = attr.ib("game")
    generation = attr.ib("gen0")
    player_select_conf = attr.ib(default=attr.Factory(PolicyPlayerConf))
    player_policy_conf = attr.ib(default=attr.Factory(PUCTPlayerConf))
    player_score_conf = attr.ib(default=attr.Factory(PUCTPlayerConf))


@attr.s
class RequestSample(object):
    new_states = attr.ib(default=attr.Factory(list))


@attr.s
class Sample(object):
    # store just the previous states
    prev_state = attr.ib()

    # state policy trained on
    state = attr.ib()

    # polict distribution
    policy = attr.ib()

    final_score = attr.ib()
    depth = attr.ib()
    game_length = attr.ib()
    lead_role_index = attr.ib()


@attr.s
class RequestSampleResponse(object):
    samples = attr.ib(default=attr.Factory(list))
    duplicates_seen = attr.ib(0)


@attr.s
class TrainNNRequest(object):
    game = attr.ib("game")

    network_size = attr.ib("small")
    generation_prefix = attr.ib("v2_")
    store_path = attr.ib("somewhere")

    # uses previous netwrok
    use_previous = attr.ib("42")
    next_step = attr.ib("42")

    validation_split = attr.ib(0.8)
    batch_size = attr.ib(32)
    epochs = attr.ib(10)

    max_sample_count = attr.ib(250000)


@attr.s
class Generation(object):
    game = attr.ib("game")
    with_generation = attr.ib("gen0")
    num_samples = attr.ib(1024)
    samples = attr.ib(attr.Factory(list))
