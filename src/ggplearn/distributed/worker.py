import attr

from twisted.internet import reactor

from ggplib.util import log

from ggplearn.util.broker import Broker, WorkerFactory
from ggplearn.distributed import msgs
from ggplearn.training import approximate_play as ap


@attr.s
class WorkerConf(object):
    connect_ip_addr = attr.ib("127.0.0.1")
    connect_port = attr.ib(9000)

    # server will call this on each generation (only once - as this string may not be unique per
    # worker) if empty, server doesn't do anything.  eg
    # copy_cmd = attr.ib("cp %s /dev/null")
    copy_cmd = attr.ib("")


class WorkerApproxPlayerBroker(Broker):
    worker_type = "player"

    def __init__(self, conf):
        self.conf = conf
        Broker.__init__(self)

        self.register(msgs.Ping, self.on_ping)
        self.register(msgs.Hello, self.on_hello)

        self.register(msgs.ConfigureApproxTrainer, self.on_configure)
        self.register(msgs.RequestSample, self.on_request_sample)

        self.approx_player = None

    def on_ping(self, server, msg):
        return msgs.Pong()

    def on_hello(self, server, msg):
        return msgs.HelloResponse(worker_type=self.worker_type,
                                  copy_cmd=self.conf.copy_cmd)

    def on_configure(self, server, msg):
        self.approx_player = ap.Runner(msg.conf)
        return msgs.Ok("configured")

    def on_request_sample(self, server, msg):
        for s in msg.new_states:
            self.approx_player.add_to_unique_states(tuple(s))

        sample, duplicates_seen = self.approx_player.generate_sample()

        m = msgs.RequestSampleResponse(sample, duplicates_seen)
        server.send_msg(m)


###############################################################################

def start_worker_factory(conf=None):
    if conf is None:
        conf = WorkerConf()

    broker = WorkerApproxPlayerBroker(conf)
    reactor.connectTCP(conf.connect_ip_addr,
                       conf.connect_port,
                       WorkerFactory(broker))
    reactor.run()


if __name__ == "__main__":
    from ggplib.util.init import setup_once
    setup_once("worker")

    from ggplearn.util.keras import constrain_resources
    constrain_resources()

    start_worker_factory()
