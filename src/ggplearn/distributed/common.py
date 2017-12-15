import json
import zlib
import codecs
import random
import string
import struct

import attr
from twisted.internet import protocol, reactor

from ggplib.util import log

from ggplearn.distributed import msgs


# XXX isnt this in collections??
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def challenge(n):
    return "".join(random.choice(string.printable) for i in range(n))


def response(s):
    ' ok this arbitrary - just a lot of port scanning on my server, and this is a detterent '
    l = []
    res = codecs.encode(s, 'rot_13')
    swap_me = True
    for c0, c1 in chunks(res, 2):
        x = (ord(c0) + ord(c1)) % 100
        while True:
            if chr(x) in string.printable:
                break
            x -= 1
            if x < 20:
                if swap_me:
                    x = ord(c0)
                    swap_me = False
                else:
                    x = ord(c1)
                    swap_me = True

        l.append(chr(x))
    res = "".join(l)
    return res + res[::-1]


class Broker(object):
    def __init__(self):
        self.handlers = {}

    def register(self, name, msg_clz, cb):
        self.handlers[name] = (msg_clz, cb)

    def onMessage(self, client, msg):
        if msg.name not in self.handlers:
            log.error("%s : unknown msg %s" % (client, str(msg.name)))
            client.disconnect()
            return

        try:
            clz, cb = self.handlers[msg.name]
            the_message = clz(**msg.payload)
            cb(client, the_message)

        except Exception, exc:
            log.error("%s : exception calling method %s" % (client, str(msg.name)))
            # XXX add tb
            log.error(str(exc))
            client.disconnect()


class Client(protocol.Protocol):
    CHALLENGE_SIZE = 512

    def __init__(self, broker):
        self.broker = broker
        self.logical_connection = False

        # this buffer is only used until logical_connection made
        self.start_buf = ""

        self.rxd = []
        self.header = struct.Struct("=i")

    def disconnect(self):
        self.transport.loseConnection()

    def connectionMade(self):
        self.logical_connection = False
        log.debug("Client::connectionMade()")

    def connectionLost(self, reason=""):
        self.logical_connection = False
        log.debug("Client::connectionLost() : %s" % reason)

    def unbuffer_data(self):
        # flatten
        buf = ''.join(self.rxd)

        while True:
            buf_len = len(buf)
            if buf_len < self.header.size:
                break

            payload_len = self.header.unpack_from(buf[:self.header.size])[0]
            if buf_len < payload_len + self.header.size:
                break

            # good, we have a message
            offset = self.header.size
            compressed_data = buf[offset:offset + payload_len]
            offset += payload_len

            data = zlib.decompress(compressed_data)
            desc = json.loads(data)
            # bit too much magic here to just convert to attr/Message... across the wire so adding some checks
            assert isinstance(desc, dict)
            assert len(desc) == 2
            assert 'name' in desc
            assert 'payload' in desc

            # ok let attrs do its magic
            msg = msgs.Message(**desc)
            yield msg

            buf = buf[offset:]

        # compact
        self.rxd = []
        if len(buf):
            self.rxd.append(buf)

    def dataReceived(self, data):
        if self.logical_connection:
            self.rxd.append(data)
            for msg in self.unbuffer_data():
                self.broker.onMessage(self, msg)
        else:
            self.init_data_rxd(data)

    def format_msg(self, name, payload):
        msg = msgs.Message(name, payload)

        data = json.dumps(attr.asdict(msg))
        compressed_data = zlib.compress(data)

        preamble = self.header.pack(len(compressed_data))
        assert len(preamble) == self.header.size
        return preamble + compressed_data

    def write_msg(self, name, payload):
        # payload can be anything json like, including an attr object
        self.transport.write(self.format_msg(name, payload))


class WorkerClient(Client):
    def init_data_rxd(self, data):
        self.start_buf += data
        if len(self.start_buf) == self.CHALLENGE_SIZE:
            self.transport.write(response(self.start_buf))
            self.logical_connection = True
            log.info("Logical Connection Made")


class WorkerFactory(protocol.ClientFactory):
    ' client side factory, connects to server '
    delay = initialDelay = 1
    factor = 1.2

    def __init__(self, broker):
        self.broker = broker

    def buildProtocol(self, addr):
        log.debug("WorkerFactory::buildProtocol: %s" % addr)
        return WorkerClient(self.broker)


class ServerClient(Client):

    def init_data_rxd(self, data):
        self.start_buf += data
        if len(self.start_buf) == self.CHALLENGE_SIZE:
            if self.expected_response == self.start_buf:
                self.logical_connection = True
                log.info("Logical connection made")
                self.broker.new_worker(self)
            else:
                self.logical_connection = True
                log.error("Logical connection failed")
                self.disconnect()

    def connectionMade(self):
        Client.connectionMade(self)
        msg = challenge(self.CHALLENGE_SIZE)
        self.transport.write(msg)
        self.expected_response = response(msg)

    def connectionLost(self, reason):
        if self.logical_connection:
            self.broker.remove_worker(self)
        Client.connectionLost(self, reason)


class ServerFactory(protocol.Factory):
    def __init__(self, broker):
        self.broker = broker

    def buildProtocol(self, addr):
        log.debug("ServerFactory::buildProtocol() %s" % addr)
        return ServerClient(self.broker)

