import attr

from ggplearn.training import approximate_play as ap


@attr.s
class Sample(object):
    # 3 previous states, if any
    prev_state2 = attr.ib()
    prev_state1 = attr.ib()
    prev_state0 = attr.ib()

    # state policy trained on
    state = attr.ib()

    # polict distribution
    policy = attr.ib()

    final_score = attr.ib()
    depth = attr.ib()
    game_length = attr.ib()
    lead_role_index = attr.ib()


@attr.s
class Generation(object):
    game = attr.ib("breakthrough")
    with_policy_generation = attr.ib("gen0")
    with_score_generation = attr.ib("gen0")
    num_samples = attr.ib(1024)
    samples = attr.ib(attr.Factory(list))


@attr.s
class Ping(object):
    pass


@attr.s
class Pong(object):
    pass


@attr.s
class Hello(object):
    pass


@attr.s
class HelloResponse(object):
    worker_type = attr.ib()


@attr.s
class SelfPlayQuery(object):
    game = attr.ib("breakthrough")
    policy_generation = attr.ib("gen0")
    score_generation = attr.ib("gen0")


@attr.s
class SelfPlayResponse(object):
    send_generation = attr.ib(False)

@attr.s
class SendGenerationFiles(object):
    model_data = attr.ib()
    weight_data = attr.ib()
    generations = attr.ib("gen0")


@attr.s
class Ok(object):
    message = attr.ib("ok")


@attr.s
class ConfigureApproxTrainer(object):
    conf = attr.ib(default=attr.Factory(ap.RunnerConf))


@attr.s
class RequestSample(object):
    new_states = attr.ib(default=attr.Factory(list))


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
