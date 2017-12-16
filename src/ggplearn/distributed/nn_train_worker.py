import attr

from twisted.internet import reactor

from ggplib.util import log

from ggplearn.util.broker import Broker, WorkerFactory

from ggplearn.training import nn_train

from ggplearn.distributed import msgs


@attr.s
class WorkerConf(object):
    connect_ip_addr = attr.ib("127.0.0.1")
    connect_port = attr.ib(9000)


class WorkerBroker(Broker):
    worker_type = "nn_train"

    def __init__(self, conf):
        self.conf = conf
        Broker.__init__(self)

        self.register(msgs.Ping, self.on_ping)
        self.register(msgs.Hello, self.on_hello)
        self.register(msgs.TrainNNRequest, self.on_train_request)

    def on_ping(self, server, msg):
        return msgs.Pong()

    def on_hello(self, server, msg):
        return msgs.HelloResponse(worker_type=self.worker_type)

    def on_train_request(self, server, msg):
        log.warning("request to train %s" % msg)
        nn_train.parse_and_train(msg)
        return msgs.Ok("network_trained")


###############################################################################

def start_worker_factory(conf=None):
    if conf is None:
        conf = WorkerConf()

    broker = WorkerBroker(conf)
    reactor.connectTCP(conf.connect_ip_addr,
                       conf.connect_port,
                       WorkerFactory(broker))
    reactor.run()


if __name__ == "__main__":
    from ggplib.util.init import setup_once
    setup_once("nn_train_worker")

    from ggplearn.util.keras import constrain_resources
    constrain_resources()

    start_worker_factory()
