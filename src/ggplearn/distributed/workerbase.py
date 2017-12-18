from builtins import super

from twisted.internet import reactor

from ggplearn.util.broker import Broker, WorkerFactory

from ggplearn import msgdefs


class WorkerBrokerBase(Broker):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        self.register(msgdefs.Ping, self.on_ping)

        # connect to server
        reactor.callLater(0, self.connect)

    def connect(self):
        reactor.connectTCP(self.conf.connect_ip_addr,
                           self.conf.connect_port,
                           WorkerFactory(self))

    def on_ping(self, server, msg):
        return msgdefs.Pong(worker_type=self.worker_type)
