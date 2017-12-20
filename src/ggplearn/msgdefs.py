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

    # < 0 is off
    choose_exponential_scale = attr.ib(-1.0)
    random_scale = attr.ib(0.5)

    # < 0 is off
    temperature_value = attr.ib(0.9)
    temperature_pct = attr.ib(0.65)
    temperature_depth = attr.ib(20)


@attr.s
class PUCTPlayerConf(object):
    name = attr.ib("PUCTPlayer")
    verbose = attr.ib(True)
    generation = attr.ib("latest")

    playouts_per_iteration = attr.ib(800)
    playouts_per_iteration_noop = attr.ib(1)
    cpuct_constant_first_4 = attr.ib(0.75)
    cpuct_constant_after_4 = attr.ib(0.75)

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
    policy_network_size = attr.ib("small")
    score_network_size = attr.ib("smaller")

    generation_prefix = attr.ib("v2_")
    store_path = attr.ib("somewhere")

    policy_player_conf = attr.ib(default=attr.Factory(PUCTPlayerConf))
    score_player_conf = attr.ib(default=attr.Factory(PUCTPlayerConf))

    generation_size = attr.ib(1024)
    max_growth_while_training = attr.ib(0.2)

    validation_split = attr.ib(0.8)
    batch_size = attr.ib(32)
    epochs = attr.ib(10)
    max_sample_count = attr.ib(250000)

    # run system commands after training (copy files to machines etc)
    run_post_training_cmds = attr.ib(default=attr.Factory(list))


@attr.s
class WorkerConf(object):
    connect_port = attr.ib(9000)
    connect_ip_addr = attr.ib("127.0.0.1")


@attr.s
class Ping(object):
    pass


@attr.s
class Pong(object):
    worker_type = attr.ib()


@attr.s
class Ok(object):
    message = attr.ib("ok")


@attr.s
class ConfigureApproxTrainer(object):
    game = attr.ib("breakthrough")
    policy_generation = attr.ib("gen0_small")
    score_generation = attr.ib("gen0_smaller")
    temperature = attr.ib(1.0)
    player_select_conf = attr.ib(default=attr.Factory(PolicyPlayerConf))
    player_policy_conf = attr.ib(default=attr.Factory(PUCTPlayerConf))
    player_score_conf = attr.ib(default=attr.Factory(PolicyPlayerConf))


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
    sample = attr.ib(default=attr.Factory(Sample))
    duplicates_seen = attr.ib(0)


@attr.s
class TrainNNRequest(object):
    game = attr.ib("breakthrough")

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
    game = attr.ib("breakthrough")
    with_policy_generation = attr.ib("gen0")
    with_score_generation = attr.ib("gen0")
    num_samples = attr.ib(1024)
    samples = attr.ib(attr.Factory(list))
