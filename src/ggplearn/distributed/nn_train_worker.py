from twisted.internet import reactor

from ggplib.util import log

from ggplearn.util.broker import Broker, WorkerFactory

from ggplearn import msgdefs
from ggplearn.training import nn_train


class WorkerBroker(Broker):
    worker_type = "nn_train"

    def __init__(self, conf):
        self.conf = conf
        Broker.__init__(self)

        self.register(msgdefs.Ping, self.on_ping)
        self.register(msgdefs.Hello, self.on_hello)
        self.register(msgdefs.TrainNNRequest, self.on_train_request)

    def on_ping(self, server, msg):
        return msgdefs.Pong()

    def on_hello(self, server, msg):
        return msgdefs.HelloResponse(worker_type=self.worker_type)

    def on_train_request(self, server, msg):
        log.warning("request to train %s" % msg)
        nn_train.parse_and_train(msg)
        return msgdefs.Ok("network_trained")


###############################################################################

def start_worker_factory():
    from ggplib.util.init import setup_once
    setup_once("nn_train_worker")

    from ggplearn.util.keras import constrain_resources
    constrain_resources()

    conf = msgdefs.WorkerConf()

    broker = WorkerBroker(conf)
    reactor.connectTCP(conf.connect_ip_addr,
                       conf.connect_port,
                       WorkerFactory(broker))
    reactor.run()


if __name__ == "__main__":
    start_worker_factory()
