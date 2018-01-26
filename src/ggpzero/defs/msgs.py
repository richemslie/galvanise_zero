from ggpzero.util.attrutil import register_attrs, attribute, attr_factory

from ggpzero.defs import confs


@register_attrs
class Ping(object):
    pass


@register_attrs
class Pong(object):
    pass


@register_attrs
class Ok(object):
    message = attribute("ok")


@register_attrs
class RequestConfig(object):
    pass


@register_attrs
class WorkerConfigMsg(object):
    conf = attribute(default=attr_factory(confs.WorkerConfig))


@register_attrs
class ConfigureSelfPlay(object):
    game = attribute("game")
    generation = attribute("gen0")
    self_play_conf = attribute(default=attr_factory(confs.SelfPlayConfig))


@register_attrs
class RequestSamples(object):
    # list of states (0/1 tuples) - to reduce duplicates
    new_states = attribute(default=attr_factory(list))


@register_attrs
class RequestSampleResponse(object):
    # list of def.confs.Sample
    samples = attribute(default=attr_factory(list))
    duplicates_seen = attribute(0)


@register_attrs
class RequestNetworkTrain(object):
    game = attribute("game")
    train_conf = attribute(default=attr_factory(confs.TrainNNConfig))
    network_model = attribute(default=attr_factory(confs.NNModelConfig))
