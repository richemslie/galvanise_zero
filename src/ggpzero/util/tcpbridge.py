#!/usr/bin/env python2
''' simple as it gets: bridge from stdio <---> tcp. '''

import os
import sys

from twisted.protocols import basic
from twisted.internet import protocol

from twisted.internet import stdio, reactor

try:
    port = int(sys.argv[1])
except:
    port = 2222

class TCPBridge(basic.LineReceiver):
    def __init__(self, client):
        self.stdio_client = client
        self.stdio_client.tcp_bridge = self

    def connectionMade(self):
        for l in self.stdio_client.buf:
            self.sendLine(l)

    def lineReceived(self, line):
        self.stdio_client.requestSendLine(line)

    def requestSendLine(self, line):
        self.sendLine(line)


class Factory(protocol.ClientFactory):
    def __init__(self, client):
        self.client = client

    def buildProtocol(self, addr):
        return TCPBridge(self.client)


class StdioBridgeClient(basic.LineReceiver):
    delimiter = os.linesep.encode("ascii")
    buf = []
    tcp_bridge = None
    def connectionMade(self):
        reactor.connectTCP("localhost",
                           port,
                           Factory(self))

    def lineReceived(self, line):
        # print >>sys.stderr, "HERE/lineReceived()", self.tcp_bridge, line
        if self.tcp_bridge is None:
            self.buf.append(line)
        else:
            self.tcp_bridge.requestSendLine(line)

    def requestSendLine(self, line):
        self.sendLine(line)


if __name__ == "__main__":
    stdio.StandardIO(StdioBridgeClient())
    reactor.run()
