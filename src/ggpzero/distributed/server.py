import os
import sys
import time
import shutil

import json

from twisted.internet import reactor

from ggplib.util import log
from ggplib.db import lookup

from ggpzero.util import attrutil, runprocs
from ggpzero.util.broker import Broker, ServerFactory

from ggpzero.defs import msgs, confs, templates

from ggpzero.nn.manager import get_manager


def critical_error(msg):
    log.critical(msg)
    reactor.stop()
    sys.exit(1)


def default_conf():
    conf = confs.ServerConfig()

    conf.port = 9000
    conf.game = "breakthrough"
    conf.current_step = 0

    conf.network_size = "normal"

    conf.generation_prefix = "v5"
    conf.store_path = os.path.join(os.environ["GGPZERO_PATH"], "data", "breakthrough", "v5")

    # generation set on server
    conf.player_select_conf = confs.PolicyPlayerConfig(verbose=False,
                                                       depth_temperature_start=8,
                                                       depth_temperature_increment=0.25,
                                                       random_scale=0.85)

    conf.player_policy_conf = confs.PUCTPlayerConfig(name="policy_puct",
                                                     verbose=False,
                                                     playouts_per_iteration=800,
                                                     playouts_per_iteration_noop=1,
                                                     dirichlet_noise_alpha=0.2,
                                                     puct_constant_after=3.0,
                                                     puct_constant_before=0.75,
                                                     choose="choose_converge")

    conf.player_score_conf = confs.PolicyPlayerConfig(verbose=False,
                                                      depth_temperature_start=8,
                                                      depth_temperature_increment=0.25,
                                                      random_scale=0.85)

    conf.generation_size = 5000
    conf.max_growth_while_training = 0.25

    conf.validation_split = 0.8
    conf.batch_size = 64
    conf.epochs = 20
    conf.max_sample_count = 250000
    conf.run_post_training_cmds = []

    return conf


class WorkerInfo(object):
    def __init__(self, worker, ping_time):
        self.worker = worker
        self.valid = True
        self.ping_time_sent = ping_time
        self.conf = None
        self.reset()

    def reset(self):
        if self.conf is not None and self.conf.do_self_play:
            self.self_play_configured = False

            # sent out up to this amount
            self.unique_state_index = 0

    def get_and_update(self, unique_states):
        assert self.self_play_configured
        new_states = unique_states[self.unique_state_index:]
        self.unique_state_index += len(new_states)
        return new_states

    def cleanup(self):
        self.valid = False


class ServerBroker(Broker):
    def __init__(self, conf_filename):
        Broker.__init__(self)

        self.conf_filename = conf_filename
        if os.path.exists(conf_filename):
            conf = attrutil.json_to_attr(open(conf_filename).read())
            assert isinstance(conf, confs.ServerConfig)
        else:
            conf = default_conf()

        attrutil.pprint(conf)

        self.conf = conf

        self.game_info = lookup.by_name(self.conf.game)

        self.workers = {}
        self.free_players = []
        self.the_nn_trainer = None

        self.accumulated_samples = []
        self.unique_states_set = set()
        self.unique_states = []

        # when a generation object is around, we are in the processing of training
        self.generation = None
        self.cmds_running = None

        self.register(msgs.Pong, self.on_pong)

        self.register(msgs.Ok, self.on_ok)
        self.register(msgs.WorkerConfigMsg, self.on_worker_config)

        self.register(msgs.RequestSampleResponse, self.on_sample_response)

        self.training_in_progress = False

        self.check_nn_generations_exist()
        self.create_approx_config()
        self.save_our_config()

        # finally start listening on port
        reactor.listenTCP(conf.port, ServerFactory(self))

    def check_nn_generations_exist(self):
        game = self.conf.game
        gen = self.get_generation(self.conf.current_step)
        log.debug("current gen %s" % gen)

        man = get_manager()
        if not man.can_load(game, gen):
            if self.conf.current_step == 0:
                # create a random network and save it
                nn_model_conf = templates.nn_model_config_template(game, self.conf.network_size)
                nn = man.create_new_network(game, nn_model_conf)
                man.save_network(nn, game, gen)

            else:
                critical_error("Did not find network %s.  exiting." % gen)

    def save_our_config(self, rolled=False):
        if os.path.exists(self.conf_filename):
            if rolled:
                shutil.copy(self.conf_filename, self.conf_filename + "-%00d" % (self.conf.current_step - 1))
            else:
                shutil.copy(self.conf_filename, self.conf_filename + "-bak")

        with open(self.conf_filename, 'w') as open_file:
            open_file.write(attrutil.attr_to_json(self.conf,
                                                  sort_keys=True,
                                                  separators=(',', ': '),
                                                  indent=4))

    def get_master_by_ip(self):
        ''' spin through self play workers, and gets the first worker for a new ip.  returns list '''
        pass

    def get_generation(self, step):
        return "%s_%s" % (self.conf.generation_prefix,
                          step)

    def need_more_samples(self):
        return len(self.accumulated_samples) < (self.conf.generation_size +
                                                self.conf.generation_size * self.conf.max_growth_while_training)

    def new_worker(self, worker):
        self.workers[worker] = WorkerInfo(worker, time.time())
        log.debug("New worker %s" % worker)
        worker.send_msg(msgs.Ping())
        worker.send_msg(msgs.RequestConfig())

    def remove_worker(self, worker):
        if worker not in self.workers:
            log.critical("worker removed, but not in workers %s" % worker)
        self.workers[worker].cleanup()
        if self.workers[worker] == self.the_nn_trainer:
            self.the_nn_trainer = None

        del self.workers[worker]

    def on_pong(self, worker, msg):
        info = self.workers[worker]
        log.info("worker %s, ping/pong time %.3f msecs" % (worker,
                                                           (time.time() - info.ping_time_sent) * 1000))

    def on_worker_config(self, worker, msg):
        info = self.workers[worker]

        # can be both
        if not (msg.conf.do_training or msg.conf.do_self_play):
            msg = "worker not configured properly (neither self play or trainer)"
            raise Exception(msg)

        info.conf = msg.conf
        if info.conf.do_training:
            # protection against > 1 the_nn_trainer
            if self.the_nn_trainer is not None:
                raise Exception("the_nn_trainer already set")

            log.info("worker trainer set %s" % worker)
            self.the_nn_trainer = info

        if info.conf.do_self_play:
            if info.conf.concurrent_plays < 1:
                raise Exception("self play and concurrent_plays < 1 (%d)" % self.concurrent_plays)

            info.reset()

            self.free_players.append(info)

            log.info("worker added as self play %s" % worker)

            # configure player will happen in schedule_players
            reactor.callLater(0, self.schedule_players)

    def on_ok(self, worker, msg):
        info = self.workers[worker]
        if msg.message == "configured":
            info.self_play_configured = True
            self.free_players.append(info)
            reactor.callLater(0, self.schedule_players)

        if msg.message == "network_trained":
            if self.conf.run_post_training_cmds:
                self.cmds_running = runprocs.RunCmds(self.conf.run_post_training_cmds,
                                                     cb_on_completion=self.finished_cmds_running,
                                                     max_time=15.0)
                self.cmds_running.spawn()
            else:
                self.roll_generation()

    def finished_cmds_running(self):
        self.cmds_running = None
        log.info("commands done")
        self.roll_generation()

    def on_sample_response(self, worker, msg):
        assert len(msg.samples) > 0

        info = self.workers[worker]
        dupe_count = 0
        for sample in msg.samples:
            state = tuple(sample.state)

            # need to check it isn't a duplicate and drop it
            if state in self.unique_states_set:
                dupe_count += 1

            else:
                self.unique_states_set.add(state)
                self.unique_states.append(state)
                self.accumulated_samples.append(sample)

                assert len(self.unique_states_set) == len(self.accumulated_samples)

        if dupe_count:
            log.warning("dropping %s inflight duplicate state" % dupe_count)

        log.info("len accumulated_samples: %s" % len(self.accumulated_samples))
        log.info("worker saw %s duplicates" % msg.duplicates_seen)

        self.free_players.append(info)
        reactor.callLater(0, self.schedule_players)

    def new_generation(self):
        assert len(self.accumulated_samples) > self.conf.generation_size

        if self.generation is not None:
            return

        log.info("new_generation()")

        gen = confs.Generation()
        gen.game = self.conf.game
        gen.with_generation = self.get_generation(self.conf.current_step)
        gen.num_samples = self.conf.generation_size
        gen.samples = self.accumulated_samples[:self.conf.generation_size]

        # write json file
        json.encoder.FLOAT_REPR = lambda f: ("%.5f" % f)

        log.info("writing json")
        filename = os.path.join(self.conf.store_path, "gendata_%s_%s.json" % (self.conf.game,
                                                                              self.conf.current_step))
        with open(filename, 'w') as open_file:
            open_file.write(attrutil.attr_to_json(gen, indent=4))

        self.generation = gen
        if self.the_nn_trainer is None:
            critical_error("There is no nn trainer to create network - exiting")

        next_step = self.conf.current_step + 1
        log.info("create TrainNNRequest() for step %s" % next_step)

        m = msgs.TrainNNRequest()
        m.game = self.conf.game
        m.network_size = self.conf.network_size
        m.generation_prefix = self.conf.generation_prefix
        m.store_path = self.conf.store_path

        m.use_previous = self.conf.retrain_network

        m.next_step = next_step
        m.overwrite_existing = False
        m.validation_split = self.conf.validation_split
        m.batch_size = self.conf.batch_size
        m.epochs = self.conf.epochs
        m.max_sample_count = self.conf.max_sample_count
        m.starting_step = self.conf.starting_step
        m.drop_dupes_count = self.conf.drop_dupes_count

        # send out message to train
        log.info("sent to the_nn_trainer")
        self.the_nn_trainer.worker.send_msg(m)
        self.training_in_progress = True

    def roll_generation(self):
        # training is done
        self.conf.current_step += 1
        self.check_nn_generations_exist()

        # reconfigure player workers
        for _, info in self.workers.items():
            info.reset()

        self.create_approx_config()

        # rotate these
        self.accumulated_samples = self.accumulated_samples[self.conf.generation_size:]
        self.unique_states = self.unique_states[self.conf.generation_size:]
        self.unique_states_set = set(self.unique_states)

        assert len(self.accumulated_samples) == len(self.unique_states)
        assert len(self.unique_states) == len(self.unique_states_set)

        # store the server config
        self.save_our_config(rolled=True)

        self.generation = None
        self.training_in_progress = False
        self.free_players.append(self.the_nn_trainer)

        log.warning("roll_generation() complete.  We have %s samples leftover" % len(self.accumulated_samples))

        self.schedule_players()

    def create_approx_config(self):
        # we use score_gen for select also XXX we should probably just go to one
        generation = self.get_generation(self.conf.current_step)

        self.conf.player_select_conf.generation = generation
        self.conf.player_policy_conf.generation = generation
        self.conf.player_score_conf.generation = generation

        conf = msgs.ConfigureApproxTrainer(self.conf.game, generation)
        conf.player_select_conf = self.conf.player_select_conf
        conf.player_policy_conf = self.conf.player_policy_conf
        conf.player_score_conf = self.conf.player_score_conf

        self.approx_play_config = conf

    def schedule_players(self):
        if len(self.accumulated_samples) > self.conf.generation_size:
            self.new_generation()

        if not self.free_players:
            return

        new_free_players = []
        for worker_info in self.free_players:
            if not worker_info.valid:
                continue

            if self.training_in_progress and worker_info.conf.do_training:
                # will be added back in at the end of training
                continue

            if not worker_info.self_play_configured:
                worker_info.worker.send_msg(self.approx_play_config)

            else:
                if self.need_more_samples():
                    updates = worker_info.get_and_update(self.unique_states)
                    m = msgs.RequestSample(updates)
                    log.debug("sending request with %s updates" % len(updates))
                    worker_info.worker.send_msg(m)
                else:
                    log.warning("capacity full! %d" % len(self.accumulated_samples))
                    new_free_players.append(worker_info)

        self.free_players = new_free_players

        if self.the_nn_trainer is None:
            log.warning("There is no nn trainer - please start")


def start_server_factory():
    from ggplib.util.init import setup_once
    setup_once("server")

    from ggpzero.util.keras import init
    init()

    ServerBroker(sys.argv[1])

    reactor.run()


if __name__ == "__main__":
    start_server_factory()
