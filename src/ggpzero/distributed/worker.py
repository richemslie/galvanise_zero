from builtins import super

import os
import sys
import time
import shutil
import traceback

from twisted.internet import reactor

from ggplib.util import log
from ggplib.db import lookup

from ggpzero.util import attrutil, runprocs

from ggpzero.defs import msgs, confs

from ggpzero.util.broker import Broker, BrokerClientFactory
from ggpzero.util import cppinterface
from ggpzero.util.state import encode_state, decode_state

from ggpzero.nn.manager import get_manager

from ggpzero.nn.train import TrainManager


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
        self.register(msgs.RequestNetworkTrain, self.on_train_request)

        self.nn = None
        self.sm = None
        self.game_info = None
        self.supervisor = None
        self.self_play_conf = None

        # will be created on demand
        self.trainer = None

        self.cmds_running = []

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
                           BrokerClientFactory(self))

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

        self.self_play_conf = msg.self_play_conf
        self.with_generation_name = msg.generation_name

        # refresh the neural network.  May have to run some commands to get it.
        self.nn = None
        try:
            self.nn = get_manager().load_network(self.game_info.game,
                                                 self.with_generation_name)
            self.configure_self_play()

        except Exception as exc:
            log.error("error in on_configure(): %s" % exc)
            for l in traceback.format_exc().splitlines():
                log.error(l)

            self.cmds_running = runprocs.RunCmds(self.conf.run_post_training_cmds,
                                                 cb_on_completion=self.finished_cmds_running,
                                                 max_time=180.0)
            self.cmds_running.spawn()

        return msgs.Ok("configured")

    def finished_cmds_running(self):
        self.cmds_running = None
        log.info("commands done")
        self.configure_self_play()

    def configure_self_play(self):
        assert self.self_play_conf is not None

        if self.nn is None:
            self.nn = get_manager().load_network(self.game_info.game,
                                                 self.self_play_conf.with_generation)

        if self.supervisor is None:
            self.supervisor = cppinterface.Supervisor(self.sm, self.nn,
                                                      batch_size=self.conf.self_play_batch_size,
                                                      sleep_between_poll=self.conf.sleep_between_poll)

            self.supervisor.start_self_play(self.self_play_conf, self.conf.num_workers)

        else:
            # force exit of the worker if there was an update to the config
            if self.conf.exit_on_update_config:
                os._exit(0)

            self.supervisor.update_nn(self.nn)
            self.supervisor.clear_unique_states()

    def cb_from_superviser(self):
        samples = self.supervisor.fetch_samples()
        for sample in samples:
            sample.state = encode_state(sample.state)
            sample.prev_states = [encode_state(s) for s in sample.prev_states]

        self.samples += samples

        # keeps the tcp connection active for remote workers
        if time.time() > self.on_request_samples_time + self.conf.server_poll_time:
            return True

        return len(self.samples) > self.conf.min_num_samples

    def on_request_samples(self, server, msg):
        self.on_request_samples_time = time.time()

        assert self.supervisor is not None
        self.samples = []
        self.supervisor.reset_stats()

        log.debug("Got request for sample with number unique states %s" % len(msg.new_states))

        # update duplicates
        for s in msg.new_states:
            self.supervisor.add_unique_state(decode_state(s))

        start_time = time.time()
        self.supervisor.poll_loop(do_stats=True, cb=self.cb_from_superviser)

        msg = "#samp %d, pred()s %d/%d, py/pred/all %.1f/%.1f/%.1f"
        time_since_last = time.time() - start_time
        log.info(msg % (len(self.samples),
                        self.supervisor.num_predictions_calls,
                        self.supervisor.total_predictions,
                        self.supervisor.acc_time_polling,
                        self.supervisor.acc_time_prediction,
                        time_since_last))

        predicts_per_sec = self.supervisor.total_predictions / time_since_last
        log.info("Average pred p/s %.1f" % predicts_per_sec)

        m = msgs.RequestSampleResponse(self.samples, 0)
        server.send_msg(m)

    def on_train_request(self, server, msg):
        log.warning("request to train %s")
        print attrutil.pprint(msg)
        self.train(msg.game, msg.train_conf, msg.network_model, msg.generation_description)
        return msgs.Ok("network_trained")

    def train(self, game, train_config, network_model, generation_description):
        assert train_config.game == game

        if self.trainer is None:
            # create a transformer
            man = get_manager()

            transformer = man.get_transformer(game, generation_description)

            # create the manager
            self.trainer = TrainManager(train_config, transformer)

        self.trainer.update_config(train_config)
        self.trainer.get_network(network_model, generation_description)

        data = self.trainer.gather_data()
        self.trainer.do_epochs(data)
        self.trainer.save()


def start_worker_factory():
    from ggplib.util.init import setup_once
    setup_once("worker")

    from ggpzero.util.keras import init
    init()

    broker = Worker(sys.argv[1])
    broker.start()


if __name__ == "__main__":
    start_worker_factory()
