import attr
from ggplearn.distributed import common, msgs


def test_chunks():
    res = list(common.chunks(range(10), 2))
    assert res[0] == [0, 1]
    assert res[4] == [8, 9]


def test_challenge():
    m = common.challenge(128)

    print m
    assert len(m) == 128

    m = common.response(m)

    print m
    assert len(m) == 128


def test_broker():
    @attr.s
    class DummyMsg(object):
        what = attr.ib()

    class BrokerX(common.Broker):
        the_client = None
        got = None

        def on_call_me(self, client, msg):
            self.the_client = client
            self.got = msg.what

    b = BrokerX()
    b.register("callme", DummyMsg, b.on_call_me)

    b.onMessage(42, msgs.Message("callme", payload=dict(what="hello")))

    assert b.the_client == 42
    assert b.got == "hello"


def test_client():
    client = common.Client(None)
    client.logical_connection = True

    data = client.format_msg("echo_me", "hello world!")
    client.rxd.append(data)

    msg = list(client.unbuffer_data())[0]

    assert msg.name == "echo_me"
    assert msg.payload == "hello world!"


def test_broker_frag_msg():
    client = common.Client(None)
    client.logical_connection = True

    data = client.format_msg("echo", "hello world2!")
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

    assert msg.name == "echo"
    assert msg.payload == "hello world2!"


def test_attrs():
    # awesome - just tested attrs module, and well I started writing something similar and swapped
    # out for this instead!

    m1 = msgs.Message("echo", "hello")
    import attr
    desc = attr.asdict(m1)
    print desc
    m2 = msgs.Message(**desc)
    assert m2.name == m1.name
    assert m2.payload == m1.payload


def test_attrs_in_action():
    res = msgs.HelloResponse(ip_addr="127.0.0.1",
                             worker_type="trainer")

    client = common.Client(None)
    client.logical_connection = True

    data = client.format_msg("hello_response", res)

    client.rxd.append(data)
    for msg in client.unbuffer_data():
        print msg
