from builtins import super

from ggplib.util import log
from ggplib.player.base import MatchPlayer

from ggpzero.defs import confs

from ggpzero.util.cppinterface import joint_move_to_ptr, basestate_to_ptr, PlayPoller

from ggpzero.nn.manager import get_manager


class PUCTPlayer(MatchPlayer):
    poller = None
    last_probability = -1
    last_node_count = -1

    def __init__(self, conf):
        assert isinstance(conf, (confs.PUCTPlayerConfig, confs.PUCTEvaluatorConfig))

        self.conf = conf
        if conf.playouts_per_iteration > 0:
            self.identifier = "%s_%s_%s" % (self.conf.name, conf.playouts_per_iteration, conf.generation)
        else:
            self.identifier = "%s_%s" % (self.conf.name, conf.generation)

        super().__init__(self.identifier)
        self.sm = None

    def cleanup(self):
        log.info("PUCTPlayer.cleanup() called")
        if self.poller is not None:
            self.poller.player_reset(0)

    def on_meta_gaming(self, finish_time):
        if self.conf.verbose:
            log.info("PUCTPlayer, match id: %s" % self.match.match_id)

        if self.sm is None or "*" in self.conf.generation:
            if "*" in self.conf.generation:
                log.warning("Using recent generation %s" % self.conf.generation)

            game_info = self.match.game_info
            self.sm = game_info.get_sm()

            man = get_manager()
            gen = self.conf.generation

            self.nn = man.load_network(game_info.game, gen)
            self.poller = PlayPoller(self.sm, self.nn, self.conf.evaluator_config)

            def get_noop_idx(actions):
                for idx, a in enumerate(actions):
                    if "noop" in a:
                        return idx
                assert False, "did not find noop"

            self.role0_noop_legal, self.role1_noop_legal = map(get_noop_idx, game_info.model.actions)

        self.poller.player_reset(self.match.game_depth)

    def on_apply_move(self, joint_move):
        self.poller.player_apply_move(joint_move_to_ptr(joint_move))
        self.poller.poll_loop()

    def on_next_move(self, finish_time):
        log.info("PUCTPlayer.on_next_move(), %s" % self.get_name())
        current_state = self.match.get_current_state()
        self.sm.update_bases(current_state)

        if (self.sm.get_legal_state(0).get_count() == 1 and
            self.sm.get_legal_state(0).get_legal(0) == self.role0_noop_legal):
            lead_role_index = 1

        else:
            assert (self.sm.get_legal_state(1).get_count() == 1 and
                    self.sm.get_legal_state(1).get_legal(0) == self.role1_noop_legal)
            lead_role_index = 0

        if lead_role_index == self.match.our_role_index:
            max_iterations = self.conf.playouts_per_iteration
        else:
            max_iterations = self.conf.playouts_per_iteration_noop

        current_state = self.match.get_current_state()

        self.poller.player_move(basestate_to_ptr(current_state), max_iterations, finish_time)
        self.poller.poll_loop()

        move, prob, node_count = self.poller.player_get_move(self.match.our_role_index)
        self.last_probability = prob
        self.last_node_count = node_count
        return move

    def update_config(self, *args, **kwds):
        self.poller.player_update_config(*args, **kwds)

    def __repr__(self):
        return self.get_name()


