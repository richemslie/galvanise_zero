import attr

from ggpzero.util.attrutil import register_attrs

from ggpzero.defs.confs import PUCTPlayerConfig, PolicyPlayerConfig, WorkerConfig


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
    conf = attr.ib(default=attr.Factory(WorkerConfig))


@register_attrs
class ConfigureApproxTrainer(object):
    game = attr.ib("game")
    generation = attr.ib("gen0")
    player_select_conf = attr.ib(default=attr.Factory(PolicyPlayerConfig))
    player_policy_conf = attr.ib(default=attr.Factory(PUCTPlayerConfig))
    player_score_conf = attr.ib(default=attr.Factory(PUCTPlayerConfig))


@register_attrs
class RequestSample(object):
    # list of states (0/1 tuples) - to reduce duplicates
    new_states = attr.ib(default=attr.Factory(list))


@register_attrs
class RequestSampleResponse(object):
    # list of def.confs.Sample
    samples = attr.ib(default=attr.Factory(list))
    duplicates_seen = attr.ib(0)


@register_attrs
class TrainNNRequest(object):
    game = attr.ib("game")

    # XXX replace with NNModelConfig
    network_size = attr.ib("small")

    # the generation prefix is what defines our models (along with step). Be careful not to
    # overwrite these.
    generation_prefix = attr.ib("v2_")

    # this is where the generations are stored
    store_path = attr.ib("/home/me/somewhere")

    # uses previous network?
    use_previous = attr.ib(True)
    next_step = attr.ib("42")
    overwrite_existing = attr.ib(False)
    validation_split = attr.ib(0.8)
    batch_size = attr.ib(32)
    epochs = attr.ib(10)

    # if the total number of samples is met, will trim the oldest samples
    max_sample_count = attr.ib(250000)

    # this is applied even if max_sample_count can't be reached
    starting_step = attr.ib(0)

    # if we see duplicate states of mre than > n, drop them, keeping the most recent.  < 0 is off.
    drop_dupes_count = attr.ib(-1)
