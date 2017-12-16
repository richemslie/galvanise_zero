import attr

from twisted.internet import reactor

from ggplib.util import log
from ggplib.db import lookup

from ggplearn.util.broker import Broker, WorkerFactory
from ggplearn.distributed import msgs
from ggplearn.training import approximate_play as ap
from ggplearn.nn import network


@attr.s
class WorkerConf(object):
    connect_ip_addr = attr.ib("127.0.0.1")
    connect_port = attr.ib(9000)

    # server will call this on each generation (only once - as this string may not be unique per
    # worker) if empty, server doesn't do anything.  eg
    # copy_cmd = attr.ib("cp %s /dev/null")
    copy_cmd = attr.ib("")


class WorkerBroker(Broker):
    worker_type = "approx_self_play"

    def __init__(self, conf):
        self.conf = conf
        Broker.__init__(self)

        self.register(msgs.Ping, self.on_ping)
        self.register(msgs.Hello, self.on_hello)
        self.register(msgs.SelfPlayQuery, self.on_self_play_query)
        self.register(msgs.SendGenerationFiles, self.on_send_generation_files)

        self.register(msgs.ConfigureApproxTrainer, self.on_configure)
        self.register(msgs.RequestSample, self.on_request_sample)

        self.approx_player = None
        self.game_info = None

    def on_ping(self, server, msg):
        return msgs.Pong()

    def on_hello(self, server, msg):
        return msgs.HelloResponse(self.worker_type)

    def on_self_play_query(self, server, msg):
        # check we have current generation
        self.game_info = lookup.by_name(msg.game)
        m = msgs.SelfPlayResponse(False)
        for g in (msg.policy_generation, msg.score_generation):
            if not network.create(g, self.game_info).can_load():
                log.warning("did not find generation %s" % g)
                m.send_generation = True
                break
        return m

    def on_send_generation_files(self, server, msg):
        XXX

    def on_configure(self, server, msg):
        self.approx_player = ap.Runner(msg.conf)
        return msgs.Ok("configured")

    def on_request_sample(self, server, msg):
        log.debug("Got request for sample with number unique states %s" % len(msg.new_states))
        for s in msg.new_states:
            self.approx_player.add_to_unique_states(tuple(s))

        sample, duplicates_seen = self.approx_player.generate_sample()

        log.verbose("Done sample")
        m = msgs.RequestSampleResponse(sample, duplicates_seen)
        server.send_msg(m)


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
    setup_once("approx_self_play")

    from ggplearn.util.keras import constrain_resources
    constrain_resources()

    start_worker_factory()
