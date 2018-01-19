from builtins import super

import os
import sys
import time
import shutil

from twisted.internet import reactor

from ggplib.util import log
from ggplib.db import lookup

from ggpzero.util import attrutil

from ggpzero.defs import msgs, confs

from ggpzero.util.broker import Broker, WorkerFactory
from ggpzero.util import cppinterface

from ggpzero.training import nn_train
from ggpzero.nn.manager import get_manager


def default_conf():
    conf = confs.WorkerConfig(9000, "127.0.0.1")
    conf.do_training = False
    conf.do_self_play = True
    conf.self_play_batch_size = 1024
    return conf


class Worker(Broker):
    def __init__(self, conf_filename):
        super().__init__()

        self.conf_filename = conf_filename
        if os.path.exists(conf_filename):
            conf = attrutil.json_to_attr(open(conf_filename).read())
            assert isinstance(conf, confs.WorkerConfig)
        else:
            conf = default_conf()

        self.conf = conf
        print "CONF", attrutil.pprint(conf)
        self.save_our_config()

        self.register(msgs.Ping, self.on_ping)
        self.register(msgs.RequestConfig, self.on_request_config)

        self.register(msgs.ConfigureSelfPlay, self.on_configure)
        self.register(msgs.RequestSamples, self.on_request_samples)
        self.register(msgs.TrainNNRequest, self.on_train_request)

        self.game_info = None
        self.sm = None
        self.nn = None
        self.supervisor = None

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
        server.send_msg(msgs.Pong())

    def on_request_config(self, server, msg):
        return msgs.WorkerConfigMsg(self.conf)

    def on_configure(self, server, msg):
        attrutil.pprint(msg)

        if self.game_info is None:
            self.game_info = lookup.by_name(msg.game)
            self.sm = self.game_info.get_sm()
        else:
            self.game_info.game == msg.game

        self.nn = get_manager().load_network(msg.game, msg.generation)

        if self.supervisor is None:
            self.supervisor = cppinterface.Supervisor(self.sm, self.nn,
                                                      batch_size=self.conf.self_play_batch_size,
                                                      sleep_between_poll=self.conf.sleep_between_poll)
            self.supervisor.start_self_play(msg.self_play_conf, self.conf.inline_manager)
        else:
            self.supervisor.update_nn(self.nn)
            self.supervisor.clear_unique_states()

        return msgs.Ok("configured")

    def cb_from_superviser(self):
        self.samples += self.supervisor.fetch_samples()
        # XXX self.conf.min_num_samples
        return len(self.samples) > 128

    def on_request_samples(self, server, msg):
        assert self.supervisor is not None
        self.samples = []
        self.supervisor.reset_stats()

        log.debug("Got request for sample with number unique states %s" % len(msg.new_states))

        # update duplicates
        for s in msg.new_states:
            self.supervisor.add_unique_state(s)

        start_time = time.time()
        self.supervisor.poll_loop(do_stats=True, cb=self.cb_from_superviser)

        log.info("Number of samples %s, prediction calls %d, predictions %d" % (len(self.samples),
                                                                                self.supervisor.num_predictions_calls,
                                                                                self.supervisor.total_predictions))
        log.info("time takens python/predict/all %.2f / %.2f / %.2f" % (self.supervisor.acc_time_polling,
                                                                        self.supervisor.acc_time_prediction,
                                                                        time.time() - start_time))

        log.info("Done all samples")

        m = msgs.RequestSampleResponse(self.samples, 0)
        server.send_msg(m)

    def on_train_request(self, server, msg):
        log.warning("request to train %s" % msg)

        nn_train.parse_and_train(msg)
        return msgs.Ok("network_trained")


def start_worker_factory():
    from ggplib.util.init import setup_once
    setup_once("worker")

    from ggpzero.util.keras import init
    init()

    broker = Worker(sys.argv[1])
    broker.start()


if __name__ == "__main__":
    start_worker_factory()
