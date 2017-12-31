from builtins import super

import os
import sys
import time
import shutil

from twisted.internet import reactor

from ggplib.util import log

from ggplearn.util import attrutil

from ggplearn.nn.scheduler import create_scheduler

from ggplearn import msgdefs

from ggplearn.util.broker import Broker, WorkerFactory

from ggplearn.training import nn_train
from ggplearn.training import approximate_play


def default_conf():
    conf = msgdefs.WorkerConf(9000, "127.0.0.1")
    conf.do_training = False
    conf.do_self_play = True
    conf.concurrent_plays = 1
    return conf


class Worker(Broker):
    def __init__(self, conf_filename):
        super().__init__()

        self.conf_filename = conf_filename
        if os.path.exists(conf_filename):
            conf = attrutil.json_to_attr(open(conf_filename).read())
            assert isinstance(conf, msgdefs.WorkerConf)
        else:
            conf = default_conf()

        self.conf = conf
        print "CONF", attrutil.pprint(conf)
        self.save_our_config()

        self.register(msgdefs.Ping, self.on_ping)
        self.register(msgdefs.RequestConfig, self.on_request_config)

        self.register(msgdefs.ConfigureApproxTrainer, self.on_configure_approx_trainer)
        self.register(msgdefs.RequestSample, self.on_request_sample)
        self.register(msgdefs.TrainNNRequest, self.on_train_request)

        self.approx_players = None
        self.self_play_session = None

        # connect to server
        reactor.callLater(0, self.connect)

    def save_our_config(self):
        if os.path.exists(self.conf_filename):
            shutil.copy(self.conf_filename, self.conf_filename + "-bak")

        with open(self.conf_filename, 'w') as open_file:
            open_file.write(attrutil.attr_to_json(self.conf, indent=4))

    def connect(self):
        reactor.connectTCP(self.conf.connect_ip_addr,
                           self.conf.connect_port,
                           WorkerFactory(self))

    def on_ping(self, server, msg):
        server.send_msg(msgdefs.Pong())

    def on_request_config(self, server, msg):
        return msgdefs.WorkerConfigMsg(self.conf)

    def on_configure_approx_trainer(self, server, msg):
        attrutil.pprint(msg)

        # crate a new session and scheduler
        self.self_play_session = approximate_play.Session()
        self.scheduler = create_scheduler(msg.game, msg.generation, batch_size=1024)

        for player_conf in msg.player_select_conf, msg.player_policy_conf, msg.player_score_conf:
            assert player_conf.generation == msg.generation

        if self.approx_players is None:
            self.approx_players = [approximate_play.Runner(msg) for _ in range(self.conf.concurrent_plays)]

        for r in self.approx_players:
            r.patch_players(self.scheduler)

        return msgdefs.Ok("configured")

    def on_request_sample(self, server, msg):
        assert self.self_play_session is not None
        assert self.scheduler is not None

        log.debug("Got request for sample with number unique states %s" % len(msg.new_states))

        # update duplicates
        self.self_play_session.reset()
        for s in msg.new_states:
            self.self_play_session.add_to_unique_states(tuple(s))

        for r in self.approx_players:
            self.scheduler.add_runnable(self.self_play_session.start_and_record, r)

        log.info("Running scheduler")

        s = time.time()
        self.scheduler.run()

        log.info("Number of samples %s, predictions %d" % (len(self.self_play_session.samples),
                                                           self.scheduler.num_predictions))
        log.info("time takens python/predict/all %.2f / %.2f / %.2f" % (self.scheduler.acc_python_time,
                                                                        self.scheduler.acc_predict_time,
                                                                        time.time() - s))

        log.info("Done all samples")

        m = msgdefs.RequestSampleResponse(self.self_play_session.samples,
                                          self.self_play_session.duplicates_seen)
        server.send_msg(m)

    def on_train_request(self, server, msg):
        log.warning("request to train %s" % msg)
        nn_train.parse_and_train(msg)
        return msgdefs.Ok("network_trained")


def start_worker_factory():
    from ggplib.util.init import setup_once
    setup_once("worker")

    from ggplearn.util.keras import init
    init(data_format='channels_last')

    broker = Worker(sys.argv[1])
    broker.start()


if __name__ == "__main__":
    start_worker_factory()
