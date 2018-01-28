import os
import sys
import gzip
import time
import shutil

import json

from twisted.internet import reactor

from ggplib.util import log
from ggplib.db import lookup

from ggpzero.util import attrutil
from ggpzero.util.broker import Broker, ServerFactory

from ggpzero.defs import msgs, confs, datadesc, templates

from ggpzero.nn.manager import get_manager


def get_date_string():
    import datetime
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M")


def critical_error(msg):
    log.critical(msg)
    reactor.stop()
    sys.exit(1)


class WorkerInfo(object):
    def __init__(self, worker, ping_time_sent):
        self.worker = worker
        self.valid = True
        self.ping_time_sent = ping_time_sent
        self.conf = None
        self.self_play_configured = None
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
    def __init__(self, conf_filename, conf=None):
        Broker.__init__(self)

        self.conf_filename = conf_filename
        if conf is None:
            assert os.path.exists(conf_filename)
            conf = attrutil.json_to_attr(open(conf_filename).read())

        assert isinstance(conf, confs.ServerConfig)
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
        self.pending_gen_samples = None
        self.training_in_progress = False

        # register messages:
        self.register(msgs.Pong, self.on_pong)

        self.register(msgs.Ok, self.on_ok)
        self.register(msgs.WorkerConfigMsg, self.on_worker_config)

        self.register(msgs.RequestSampleResponse, self.on_sample_response)

        self.check_files_exist()
        self.load_and_check_data()

        self.create_self_play_config()
        self.save_our_config()

        # finally start listening on port
        reactor.listenTCP(conf.port, ServerFactory(self))

        # save the samples periodically
        self.checkpoint_cb = reactor.callLater(self.conf.checkpoint_interval, self.checkpoint)

    def check_files_exist(self):
        # first check that the directories exist
        man = get_manager()
        for p in (man.model_path(self.conf.game),
                  man.weights_path(self.conf.game),
                  man.generation_path(self.conf.game),
                  man.samples_path(self.conf.game, self.conf.generation_prefix)):

            if os.path.exists(p):
                if not os.path.isdir(p):
                    critical_error("Path exists and not directory: %s")
            else:
                log.warning("Attempting to create path: %s" % p)
                os.makedirs(p)
                if not os.path.exists(p) or not os.path.isdir(p):
                    critical_error("Failed to create directory: %s" % p)

        self.check_nn_files_exist()

    def check_nn_files_exist(self):
        game = self.conf.game
        gen_name = self.get_generation_name(self.conf.current_step)

        man = get_manager()
        if not man.can_load(game, gen_name):
            if self.conf.current_step == 0:
                log.warning("Creating the initial network.")

                # create a random network and save it
                nn = man.create_new_network(game,
                                            self.conf.base_network_model,
                                            self.conf.base_generation_description)
                man.save_network(nn, gen_name)

            else:
                critical_error("Did not find network %s.  exiting." % gen_name)

        # for debugging purposes
        nn = man.load_network(game, gen_name)
        nn.summary()

    def save_our_config(self, rolled=False):
        if os.path.exists(self.conf_filename):
            if rolled:
                shutil.copy(self.conf_filename, self.conf_filename + "-%00d" % (self.conf.current_step - 1))
            else:
                shutil.copy(self.conf_filename, self.conf_filename + "-bak")

        with open(self.conf_filename, 'w') as open_file:
            open_file.write(attrutil.attr_to_json(self.conf, pretty=True))

    def get_generation_name(self, step):
        ''' this is the name of the current generation '''
        return "%s_%s" % (self.conf.generation_prefix, step)

    def need_more_samples(self):
        return len(self.accumulated_samples) < (self.conf.num_samples_to_train +
                                                self.conf.num_samples_to_train * self.conf.max_samples_growth)

    def new_broker_client(self, worker):
        self.workers[worker] = WorkerInfo(worker, time.time())
        log.debug("New worker %s" % worker)
        worker.send_msg(msgs.Ping())
        worker.send_msg(msgs.RequestConfig())

    def remove_broker_client(self, worker):
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
            if info.conf.self_play_batch_size < 1:
                raise Exception("self play and self_play_batch_size < 1 (%d)" % self.concurrent_plays)

            info.reset()
            self.free_players.append(info)

            log.info("worker added as self play %s" % worker)

            # configure player will happen in schedule_players
            reactor.callLater(0, self.schedule_players)

    def on_ok(self, worker, msg):
        info = self.workers[worker]
        if msg.message == "configured":
            info.self_play_configured = True
            if info not in self.free_players:
                self.free_players.append(info)
            reactor.callLater(0, self.schedule_players)

        if msg.message == "network_trained":
            self.roll_generation()

    def add_new_samples(self, samples):
        dupe_count = 0
        for sample in samples:
            state = tuple(sample.state)

            # need to check it isn't a duplicate and drop it
            if state in self.unique_states_set:
                dupe_count += 1

            else:
                self.unique_states_set.add(state)
                self.unique_states.append(state)
                self.accumulated_samples.append(sample)

                assert len(self.unique_states_set) == len(self.accumulated_samples)

        return dupe_count

    def on_sample_response(self, worker, msg):
        info = self.workers[worker]
        if len(msg.samples) > 0:
            dupe_count = self.add_new_samples(msg.samples)
            if dupe_count:
                log.warning("dropping %s inflight duplicate state(s)" % dupe_count)

            if msg.duplicates_seen:
                log.info("worker saw %s duplicates" % msg.duplicates_seen)

            log.info("len accumulated_samples: %s" % len(self.accumulated_samples))

        self.free_players.append(info)
        reactor.callLater(0, self.schedule_players)

    @property
    def sample_data_filename(self):
        man = get_manager()
        p = man.samples_path(self.conf.game, self.conf.generation_prefix)
        return os.path.join(p, "gendata_%s_%s.json.gz" % (self.conf.game,
                                                          self.conf.current_step))

    def save_sample_data(self):
        gen_samples = datadesc.GenerationSamples()
        gen_samples.game = self.conf.game
        gen_samples.date_created = get_date_string()

        gen_samples.with_generation = self.get_generation_name(self.conf.current_step)

        # only save the minimal number for this run
        gen_samples.num_samples = min(len(self.accumulated_samples), self.conf.num_samples_to_train)
        gen_samples.samples = self.accumulated_samples[:gen_samples.num_samples]

        # write json file
        json.encoder.FLOAT_REPR = lambda f: ("%.5f" % f)

        log.info("writing json (gzipped): %s" % self.sample_data_filename)
        with gzip.open(self.sample_data_filename, 'w') as f:
            f.write(attrutil.attr_to_json(gen_samples, indent=4))

        return gen_samples

    def load_and_check_data(self):
        log.info("checking if generation data available")
        try:
            gen_samples = attrutil.json_to_attr(gzip.open(self.sample_data_filename).read())
            log.info("data exists, with generation: %s, adding %s samples" % (gen_samples.with_generation,
                                                                              gen_samples.num_samples))

            self.add_new_samples(gen_samples.samples)

        except IOError as exc:
            log.info("Not such file for generation: %s" % exc)

    def checkpoint(self):
        num_samples = len(self.accumulated_samples)
        log.verbose("entering checkpoint with %s sample accumulated" % num_samples)
        if num_samples > 0:
            gen_samples = self.save_sample_data()

            if num_samples > self.conf.num_samples_to_train:
                if self.pending_gen_samples is None:
                    log.info("data done for: %s" % self.get_generation_name(self.conf.current_step + 1))
                    self.pending_gen_samples = gen_samples

                if not self.training_in_progress:
                    if self.the_nn_trainer is None:
                        log.error("There is no trainer - please start")
                    else:
                        self.send_request_to_train_nn()

        # cancel any existing cb
        if self.checkpoint_cb is not None and self.checkpoint_cb.active():
            self.checkpoint_cb.cancel()

        # call checkpoint again in n seconds
        self.checkpoint_cb = reactor.callLater(self.conf.checkpoint_interval, self.checkpoint)

    def send_request_to_train_nn(self):
        assert not self.training_in_progress
        next_step = self.conf.current_step + 1

        log.verbose("send_request_to_train_nn() @ step %s" % next_step)

        train_conf = self.conf.base_training_config
        train_conf.use_previous = next_step % 5 != 0
        assert train_conf.game == self.conf.game
        assert train_conf.generation_prefix == self.conf.generation_prefix

        train_conf.next_step = next_step

        m = msgs.RequestNetworkTrain()
        m.game = self.conf.game
        m.train_conf = train_conf
        m.network_model = self.conf.base_network_model
        m.generation_description = self.conf.base_generation_description

        # send out message to train
        self.the_nn_trainer.worker.send_msg(m)

        log.info("sent out request to the_nn_trainer!")

        self.training_in_progress = True

    def roll_generation(self):
        # training is done
        self.conf.current_step += 1
        self.check_nn_files_exist()

        # reconfigure player workers
        for _, info in self.workers.items():
            info.reset()

        self.create_self_play_config()

        # rotate these
        self.accumulated_samples = self.accumulated_samples[self.conf.num_samples_to_train:]
        self.unique_states = self.unique_states[self.conf.num_samples_to_train:]
        self.unique_states_set = set(self.unique_states)

        self.checkpoint()

        assert len(self.accumulated_samples) == len(self.unique_states)
        assert len(self.unique_states) == len(self.unique_states_set)

        # store the server config
        self.save_our_config(rolled=True)

        self.pending_gen_samples = None
        self.training_in_progress = False

        if self.the_nn_trainer.conf.do_self_play:
            if self.the_nn_trainer not in self.free_players:
                self.free_players.append(self.the_nn_trainer)

        log.warning("roll_generation() complete.  We have %s samples leftover" % len(self.accumulated_samples))

        self.schedule_players()

    def create_self_play_config(self):
        # we use score_gen for select also XXX we should probably just go to one
        generation_name = self.get_generation_name(self.conf.current_step)

        self.configure_selfplay_msg = msgs.ConfigureSelfPlay(self.conf.game,
                                                             generation_name,
                                                             self.conf.self_play_config)

    def schedule_players(self):
        if len(self.accumulated_samples) > self.conf.num_samples_to_train:
            # if we haven't started training yet, lets speed things up...
            if not self.training_in_progress:
                self.checkpoint()

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
                worker_info.worker.send_msg(self.configure_selfplay_msg)

            else:
                if self.need_more_samples():
                    updates = worker_info.get_and_update(self.unique_states)
                    m = msgs.RequestSamples(updates)
                    if updates:
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

    if sys.argv[1] == "-c":
        game, generation_prefix = sys.argv[2], sys.argv[3]
        num_prev_states = int(sys.argv[4])
        filename = sys.argv[5]
        conf = templates.server_config_template(game, generation_prefix, num_prev_states)
        ServerBroker(filename, conf)

    else:
        filename = sys.argv[1]
        assert os.path.exists(filename)
        ServerBroker(filename)

    reactor.run()


if __name__ == "__main__":
    start_server_factory()
