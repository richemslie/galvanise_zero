from twisted.internet import reactor

from ggplib.util import log

from ggplearn.util.broker import Broker, WorkerFactory

from ggplearn import msgdefs
from ggplearn.training import approximate_play as ap


# XXX code here is very similar to nn_train_worker.py
class WorkerBroker(Broker):
    worker_type = "approx_self_play"

    def __init__(self, conf):
        self.conf = conf
        Broker.__init__(self)

        self.register(msgdefs.Ping, self.on_ping)
        self.register(msgdefs.Hello, self.on_hello)
        self.register(msgdefs.SelfPlayQuery, self.on_self_play_query)
        self.register(msgdefs.SendGenerationFiles, self.on_send_generation_files)

        self.register(msgdefs.ConfigureApproxTrainer, self.on_configure)
        self.register(msgdefs.RequestSample, self.on_request_sample)

        self.approx_player = None

    def on_ping(self, server, msg):
        return msgdefs.Pong()

    def on_hello(self, server, msg):
        return msgdefs.HelloResponse(self.worker_type)

    def on_configure(self, server, msg):
        self.approx_player = ap.Runner(msg)
        return msgdefs.Ok("configured")

    def on_request_sample(self, server, msg):
        log.debug("Got request for sample with number unique states %s" % len(msg.new_states))
        for s in msg.new_states:
            self.approx_player.add_to_unique_states(tuple(s))

        sample, duplicates_seen = self.approx_player.generate_sample()

        log.verbose("Done sample")
        m = msgdefs.RequestSampleResponse(sample, duplicates_seen)
        server.send_msg(m)


###############################################################################

def start_worker_factory():
    from ggplib.util.init import setup_once
    setup_once("approx_self_play")

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
