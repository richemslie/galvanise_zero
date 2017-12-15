import csv
import time

import attrs

from twisted.internet import reactor

from ggplib.util import log

from ggplearn.distributed import msgs, common

from ggplearn.player.mc import PUCTPlayerConf

@attr.s
class ServerConfig:
    game = attr.ib()
    path_ext = attr.ib()

    policy_player_conf = attr.ib()
    score_player_conf = attr.ib()

    start_generation = attr.ib()
    end_generation = attr.ib()

    generation_size = attr.ib()
    max_growth_while_training = attr.ib()


def default_conf():
    conf = ServerConfig()
    conf.game = "breakthrough"
    conf.path_ext = "train_v2"

    policy_player_conf = PUCTPlayerConf()
    score_player_conf = PUCTPlayerConf()
    policy_generation_ext = "_tiny"
    score_generation_ext = "_tiny"

    start_generation = 0
    end_generation = 10

    generation_size = 1024
    max_growth_while_training = 0.2


class WorkerInfo(object):
    def __init__(self, worker):
        self.worker = worker

    def cleanup(self):
        pass


class ServerBroker(common.Broker):
    def __init__(self):
        common.Broker.__init__(self)
        self.workers = {}
        self.free_players = []
        self.conf = default_conf()

        self.register("ping_response", msgs.PingResponse, self.on_ping_response)

        self.csv_file = open('', 'a')
        self.csv_writer = csv.writer(self.csv_file)

    def new_worker(self, worker):
        self.workers[worker] = WorkerInfo(worker)
        worker.write_msg("ping", {})
        log.debug("New worker %s" % worker)

    def remove_worker(self, worker):
        if worker not in self.workers:
            worker.cleanup()
            log.critical("worker removed, but not in workers %s" % worker)
        del self.workers[worker]

    def on_ping_response(self, worker, msg):
        print worker, msg
        info = self.workers[worker]
        if msg.worker_type == "player":
            info.ip_addr = msg.ip_addr
            info.worker_type = msg.worker_type
            self.free_players.append(worker)

        reactor.callLater(0, lambda: self.schedule_players())

    def on_sample_response(self, worker, sample_msg):
        fields = []
        fields.append("STATE")
        fields += sample_msg.state
        fields.append("END")
        fields.append(sample_msg.lead_role_index)
        fields.append(sample_msg.final_score["white"])
        fields.append(sample_msg.final_score["black"])
        fields.append("POLICY")
        for x,y in sample_msg.policy:
            fields.append(x)
            fields.append(y)

        self.csv_writer.writerow(fields)
        self.csv_file.flush()
        self.free_players.append(worker)
        reactor.callLater(0, lambda: self.schedule_players())

    def schedule_players(self):
        if not self.free_players:
            print "no free players"
            reactor.callLater(1.0, lambda: self.schedule_players())
            return

        for w in self.free_players:
            w.write_msg("request_sample", {})

def start_server_factory(port):
    broker = ServerBroker()
    factory = common.ServerFactory(broker)
    reactor.listenTCP(port, factory)
    reactor.run()


if __name__ == "__main__":
    from ggplib.util.init import setup_once
    setup_once("worker")

    start_server_factory(9000)
