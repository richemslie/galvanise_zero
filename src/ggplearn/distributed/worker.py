from twisted.internet import protocol, reactor

from ggplib.util import log

from ggplearn.distributed import msgs, common


class WorkerBroker(common.Broker):
    def __init__(self):
        common.Broker.__init__(self)
        self.register("ping", msgs.Ping, self.on_ping)

    def on_ping(self, server, msg):
        print "hello world", msg
        m = msgs.PingResponse("127.0.0.1", "player")
        server.write_msg("ping_response", m)

def start_worker_factory(ip_addr, port):
    broker = WorkerBroker()
    factory = common.WorkerFactory(broker)
    reactor.connectTCP(ip_addr, port, factory)
    reactor.run()


if __name__ == "__main__":
    from ggplib.util.init import setup_once
    setup_once("worker")
    start_worker_factory("127.0.0.1", 9000)
