import attr
from pprint import pprint

from ggplearn.util import attrutil, func, broker


def test_chunks():
    res = list(func.chunks(range(10), 2))
    assert res[0] == [0, 1]
    assert res[4] == [8, 9]


def test_challenge():
    m = broker.challenge(128)

    print m
    assert len(m) == 128

    m = broker.response(m)

    print m
    assert len(m) == 128


def test_broker():
    @attr.s
    class DummyMsg(object):
        what = attr.ib()

    class BrokerX(broker.Broker):
        the_client = None
        got = None

        def on_call_me(self, client, msg):
            self.the_client = client
            self.got = msg.what

    b = BrokerX()

    b.register(DummyMsg, b.on_call_me)
    m = DummyMsg("hello")
    b.onMessage(42, broker.Message(broker.clz_to_name(DummyMsg), payload=m))

    assert b.the_client == 42
    assert b.got == "hello"


@attr.s
class DummyMsg(object):
    what = attr.ib()


def test_client():
    client = broker.Client(None)
    client.logical_connection = True

    data = client.format_msg(DummyMsg("hello world!"))
    client.rxd.append(data)

    msg = list(client.unbuffer_data())[0]

    assert msg.name == "test_util.DummyMsg"
    assert msg.payload.what == "hello world!"


def test_broker_frag_msg():
    client = broker.Client(None)
    client.logical_connection = True

    data = client.format_msg(DummyMsg("hello world2!"))
    assert len(data) > 14
    client.rxd.append(data[:10])

    empty = list(client.unbuffer_data())
    assert empty == []

    client.rxd.append(data[10:])
    buflens = [len(x) for x in client.rxd]
    assert buflens[0] == 10
    assert buflens[1] > 10

    msg = list(client.unbuffer_data())[0]
    assert msg is not None

    assert msg.name == "test_util.DummyMsg"
    assert msg.payload.what == "hello world2!"


@attr.s
class Container(object):
    x = attr.ib()
    y = attr.ib()
    z = attr.ib()


def test_attrs_recursive():
    print 'test_attrs_recursive.1'

    c = Container(DummyMsg('a'),
                  DummyMsg('b'),
                  DummyMsg('c'))

    m = Container(DummyMsg('o'),
                  DummyMsg('p'),
                  DummyMsg(c))

    d = attrutil.asdict_plus(m)
    pprint(d)

    r = attrutil.fromdict_plus(d)
    assert isinstance(r, Container)

    assert r.x.what == 'o'
    assert r.z.what.x.what == 'a'

    json_str = attrutil.attr_to_json(m, indent=4)
    print json_str

    k = attrutil.json_to_attr(json_str)
    assert k.x.what == 'o'
    assert k.z.what.x.what == 'a'
