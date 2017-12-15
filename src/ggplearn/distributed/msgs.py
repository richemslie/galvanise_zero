import attr

from collections import OrderedDict
from ggplearn.training import approximate_play as ap

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
    copy_cmd = attr.ib()


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
    sample = attr.ib(default=attr.Factory(ap.Sample))
    duplicates_seen = attr.ib(0)


@attr.s
class RequestFitting(object):
    game = attr.ib("breakthrough")
    validation_split = attr.ib(0.8)
    batch_size = attr.ib(32)
    epochs = attr.ib(10)
    network_size = attr.ib("tiny")
    generation = attr.ib(default=attr.Factory(Generation))

@attr.s
class RequestFittingResponse(object):
    pass
