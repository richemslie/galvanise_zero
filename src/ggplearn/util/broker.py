''' XXX move out of this repo '''

import zlib
import codecs
import random
import string
import struct
import traceback

import attr

from twisted.internet import protocol

from ggplib.util import log
from ggplearn.util import attrutil, func


@attr.s
class Message(object):
    name = attr.ib()
    payload = attr.ib()


def challenge(n):
    return "".join(random.choice(string.printable) for i in range(n))


def response(s):
    ' ok this arbitrary - just a lot of port scanning on my server, and this is a detterent '
    buf = []
    res = codecs.encode(s, 'rot_13')
    swap_me = True
    for c0, c1 in func.chunks(res, 2):
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

        buf.append(chr(x))
    res = "".join(buf)
    return res + res[::-1]


def clz_to_name(clz):
    return "%s.%s" % (clz.__module__, clz.__name__)


class Broker(object):
    def __init__(self):
        self.handlers = {}

    def register(self, attr_clz, cb):
        ' registers a attr class to a callback '
        assert attr.has(attr_clz)
        self.handlers[clz_to_name(attr_clz)] = attr_clz, cb

    def onMessage(self, caller, msg):
        if msg.name not in self.handlers:
            log.error("%s : unknown msg %s" % (caller, str(msg.name)))
            caller.disconnect()
            return

        try:
            clz, cb = self.handlers[msg.name]
            response = cb(caller, msg.payload)

            # doesn't necessarily need to have a response
            if response is not None:
                caller.send_msg(response)

        except Exception as e:
            log.error("%s : exception calling method %s.  " % (caller, str(msg.name)))
            log.error("%s" % e)
            log.error(traceback.format_exc())

            # do this last as might raise also...
            caller.disconnect()


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
            msg = attrutil.json_to_attr(data)
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

    def format_msg(self, payload):
        assert attr.has(payload)
        name = clz_to_name(payload.__class__)

        msg = Message(name, payload)

        data = attrutil.attr_to_json(msg)
        compressed_data = zlib.compress(data)

        preamble = self.header.pack(len(compressed_data))
        assert len(preamble) == self.header.size
        return preamble + compressed_data

    def send_msg(self, payload):
        self.transport.write(self.format_msg(payload))


class WorkerClient(Client):
    def init_data_rxd(self, data):
        self.start_buf += data
        if len(self.start_buf) == self.CHALLENGE_SIZE:
            self.transport.write(response(self.start_buf))
            self.logical_connection = True
            log.info("Logical connection established")


class WorkerFactory(protocol.ReconnectingClientFactory):
    ' client side factory, connects to server '

    # maximum number of seconds between connection attempts
    maxDelay = 30

    # delay for the first reconnection attempt
    initialDelay = 2

    # a multiplicitive factor by which the delay grows
    factor = 1.5

    def __init__(self, broker):
        self.broker = broker

    def buildProtocol(self, addr):
        log.debug("Connection made to: %s" % addr)
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
        log.debug("Connection made from: %s" % addr)
        return ServerClient(self.broker)
