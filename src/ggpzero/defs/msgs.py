import attr

from ggpzero.util.attrutil import register_attrs

from ggpzero.defs import confs


@register_attrs
class Ping(object):
    pass


@register_attrs
class Pong(object):
    pass


@register_attrs
class Ok(object):
    message = attr.ib("ok")


@register_attrs
class RequestConfig(object):
    pass


@register_attrs
class WorkerConfigMsg(object):
    conf = attr.ib(default=attr.Factory(confs.WorkerConfig))


@register_attrs
class ConfigureSelfPlay(object):
    game = attr.ib("game")
    generation = attr.ib("gen0")
    self_play_conf = attr.ib(default=attr.Factory(confs.SelfPlayConfig))


@register_attrs
class RequestSamples(object):
    # list of states (0/1 tuples) - to reduce duplicates
    new_states = attr.ib(default=attr.Factory(list))


@register_attrs
class RequestSampleResponse(object):
    # list of def.confs.Sample
    samples = attr.ib(default=attr.Factory(list))
    duplicates_seen = attr.ib(0)


@register_attrs
class RequestTrainnNN(object):
    game = attr.ib("game")
    train_conf = attr.ib(default=attr.Factory(confs.TrainNNConfig))
