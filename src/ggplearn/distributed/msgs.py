import attr

from collections import OrderedDict

@attr.s
class Message(object):
    name = attr.ib()
    payload = attr.ib()


@attr.s
class Ping(object):
    pass


@attr.s
class PingResponse(object):
    ip_addr = attr.ib()
    worker_type = attr.ib()


@attr.s
class RequestSample(object):
    new_states = attr.ib()

@attr.s
class ConfigureRunner(object):
    new_states = attr.ib()
