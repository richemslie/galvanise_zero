import os

import time

import attr
import json

from twisted.internet import reactor

from ggplib.util import log
from ggplib.db import lookup

from ggplearn.util import broker, attrutil
from ggplearn.util.broker import Broker, ServerFactory

from ggplearn.distributed import msgs

from ggplearn.nn import bases

from ggplearn.player import mc


def runner_conf_default_puct():
    from ggplearn.training.approximate_play import RunnerConf
    conf = RunnerConf()
    conf.score_puct_player_conf = mc.PUCTPlayerConf(name="score_puct",
                                                    verbose=False,
                                                    num_of_playouts_per_iteration=4,
                                                    # num_of_playouts_per_iteration=32,
                                                    num_of_playouts_per_iteration_noop=0,
                                                    expand_root=5,
                                                    dirichlet_noise_alpha=0.1,
                                                    cpuct_constant_first_4=0.75,
                                                    cpuct_constant_after_4=0.75,
                                                    choose="choose_converge")

    conf.policy_puct_player_conf = mc.PUCTPlayerConf(name="policy_puct",
                                                     verbose=False,
                                                     num_of_playouts_per_iteration=42,
                                                     # num_of_playouts_per_iteration=800,
                                                     num_of_playouts_per_iteration_noop=0,
                                                     expand_root=5,
                                                     dirichlet_noise_alpha=-1,
                                                     cpuct_constant_first_4=3.0,
                                                     cpuct_constant_after_4=0.75,
                                                     choose="choose_converge")
    return conf


@attr.s
class ServerConfig(object):
    port = attr.ib(9000)

    game = attr.ib("breakthrough")

    current_step = attr.ib(0)
    policy_network_size = attr.ib("small")
    score_network_size = attr.ib("smaller")

    generation_prefix = attr.ib("v2_")
    store_path = attr.ib("v2")

    policy_player_conf = attr.ib(default=attr.Factory(mc.PUCTPlayerConf))
    score_player_conf = attr.ib(default=attr.Factory(mc.PUCTPlayerConf))

    generation_size = attr.ib(42)
    max_growth_while_training = attr.ib(0.2)


class WorkerInfo(object):
    def __init__(self, worker, create_time):
        self.worker = worker
        self.valid = True
        self.create_time = create_time
        self.worker_type = None
        self.copy_cmd = None
        self.reset()

    def reset(self):
        self.configured = False

        # sent out up to this amount
        self.unique_state_index = 0

    def get_and_update(self, unique_states):
        assert self.configured
        new_states = unique_states[self.unique_state_index:]
        self.unique_state_index += len(new_states)
        return new_states

    def cleanup(self):
        self.valid = False


class ServerBroker(Broker):
    def __init__(self, conf):
        Broker.__init__(self)

        self.conf = conf

        self.game_info = lookup.by_name(self.conf.game)

        self.workers = {}
        self.free_players = []

        self.accumulated_samples = []
        self.unique_states_set = set()
        self.unique_states = []

        self.generation = None

        self.register(msgs.Pong, self.on_pong)
        self.register(msgs.HelloResponse, self.on_hello_response)
        self.register(msgs.Ok, self.on_ok)
        self.register(msgs.RequestSampleResponse, self.on_sample_response)

        if self.conf.current_step == 0:
            self.create_networks()

    def create_networks(self):
        print self.get_policy_generation(self.conf.current_step)
        print self.get_score_generation(self.conf.current_step)

        assert self.conf.current_step == 0
        policy_gen = self.get_policy_generation(self.conf.current_step)
        bases_config = bases.get_config(self.game_info.game,
                                        self.game_info.model,
                                        generation=policy_gen)

        policy_nn = bases_config.create_network(a0_reg=False, dropout=True,
                                                network_size=self.conf.policy_network_size)
        policy_nn.save()

        score_gen = self.get_score_generation(self.conf.current_step)
        bases_config = bases.get_config(self.game_info.game,
                                        self.game_info.model,
                                        generation=score_gen)
        score_nn = bases_config.create_network(a0_reg=False, dropout=True,
                                               network_size=self.conf.score_network_size)
        score_nn.save()

        # create networks

    def get_policy_generation(self, step):
        return "%sgen_%s_%s" % (self.conf.generation_prefix,
                                self.conf.policy_network_size,
                                step)

    def get_score_generation(self, step):
        return "%sgen_%s_%s" % (self.conf.generation_prefix,
                                self.conf.score_network_size,
                                step)

    def get_data_path(self, filename):
        return os.path.join(os.environ["GGPLEARN_PATH"],
                            "data",
                            self.conf.game,
                            self.conf.store_path,
                            filename)

    def need_more_samples(self):
        return len(self.accumulated_samples) < (self.conf.generation_size +
                                                self.conf.generation_size * self.conf.max_growth_while_training)

    def get_copy_cmds(self):
        all_copy_cmds = set()
        for info in self.workers.values():
            if info.copy_cmd:
                all_copy_cmds.add(info.copy_cmd)
        return all_copy_cmds

    def new_worker(self, worker):
        self.workers[worker] = WorkerInfo(worker, time.time())
        log.debug("New worker %s" % worker)
        worker.send_msg(msgs.Ping())

    def remove_worker(self, worker):
        if worker not in self.workers:
            log.critical("worker removed, but not in workers %s" % worker)
        self.workers[worker].cleanup()
        del self.workers[worker]

    def on_pong(self, worker, msg):
        info = self.workers[worker]
        log.info("worker %s, ping/pong time %.3f msecs" % (worker,
                                                           (time.time() - info.create_time) * 1000))
        worker.send_msg(msgs.Hello())

    def on_hello_response(self, worker, msg):
        info = self.workers[worker]
        if msg.worker_type == "player":
            info.worker_type = msg.worker_type
            info.copy_cmd = msg.copy_cmd
        else:
            raise Exception("TODO")

        # ok we need to configure player
        self.free_players.append(info)
        reactor.callLater(0, self.schedule_players)

    def on_ok(self, worker, msg):
        info = self.workers[worker]
        if msg.message == "configured":
            info.configured = True
        self.free_players.append(info)
        reactor.callLater(0, self.schedule_players)

    def on_sample_response(self, worker, msg):
        # need to check it isn't a duplicate and drop if so XXX
        info = self.workers[worker]
        state = tuple(msg.sample.state)
        if state not in self.unique_states_set:
            self.unique_states_set.add(state)
            self.unique_states.append(state)

        self.accumulated_samples.append(msg.sample)

        print "len accumulated_samples", len(self.accumulated_samples)
        print "worker saw %s duplicates" % msg.duplicates_seen

        self.free_players.append(info)
        reactor.callLater(0, self.schedule_players)

    def new_generation(self):
        assert len(self.accumulated_samples) > self.conf.generation_size

        if self.generation is not None:
            return

        gen = msgs.Generation()
        gen.game = self.conf.game
        gen.with_score_generation = self.get_score_generation(self.conf.current_step)
        gen.with_policy_generation = self.get_policy_generation(self.conf.current_step)
        gen.num_samples = self.conf.generation_size
        gen.samples = self.accumulated_samples[:self.conf.generation_size]

        # write json file
        json.encoder.FLOAT_REPR = lambda f: ("%.5f" % f)

        filename = self.get_data_path("gendata_%s.json" % self.conf.current_step)
        with open(filename, 'w') as open_file:
            open_file.write(attrutil.attr_to_json(gen, indent=4))

        self.generation = gen

    def roll_generation(self):
        # do lots of stuff
        # like copy networks
        # reconfigure
        pass

    def schedule_players(self):
        if not self.free_players:
            return

        new_free_players = []
        for worker_info in self.free_players:
            if not worker_info.valid:
                continue

            # print worker_info, "IS FREE and configured", worker_info.configured

            if not worker_info.configured:
                runner_conf = runner_conf_default_puct()
                runner_conf.game = self.conf.game
                runner_conf.policy_generation = self.get_policy_generation(self.conf.current_step)
                runner_conf.score_generation = self.get_score_generation(self.conf.current_step)

                m = msgs.ConfigureApproxTrainer(conf=runner_conf)
                worker_info.worker.send_msg(m)

            else:
                if self.need_more_samples():
                    updates = worker_info.get_and_update(self.unique_states)
                    m = msgs.RequestSample(updates)
                    print "sending request with %s updates" % len(updates)
                    worker_info.worker.send_msg(m)

                else:
                    print worker_info, "full!", len(self.accumulated_samples)
                    new_free_players.append(worker_info)

        self.free_players = new_free_players

        if len(self.accumulated_samples) > self.conf.generation_size:
            self.new_generation()


def start_server_factory(conf=None):
    if conf is None:
        conf = ServerConfig()

    broker = ServerBroker(conf)

    reactor.listenTCP(conf.port, ServerFactory(broker))
    reactor.run()


if __name__ == "__main__":
    from ggplib.util.init import setup_once
    setup_once("worker")
    start_server_factory()
