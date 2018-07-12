from builtins import super

import sys
import attr

from ggplib.util import log
from ggplib.player.base import MatchPlayer

from ggpzero.util import attrutil
from ggpzero.defs import confs

from ggpzero.util.cppinterface import joint_move_to_ptr, basestate_to_ptr, PlayPoller, PlayPollerV2

from ggpzero.nn.manager import get_manager


class PUCTPlayer(MatchPlayer):
    poller_clz = PlayPoller
    last_probability = -1
    last_node_count = -1

    def __init__(self, conf):
        assert isinstance(conf, (confs.PUCTPlayerConfig, confs.PUCTEvaluatorConfig))

        self.conf = conf
        self.identifier = "%s_%s_%s" % (self.conf.name,
                                        self.conf.playouts_per_iteration,
                                        conf.generation)
        super().__init__(self.identifier)
        self.sm = None

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
            self.poller = self.poller_clz(self.sm, self.nn, self.conf.evaluator_config)

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


class PUCTPlayerV2(PUCTPlayer):
    poller_clz = PlayPollerV2

    def update_config(self, *args, **kwds):
        self.poller.update_config(*args, **kwds)


###############################################################################

def get_default_conf(gen, **kwds):
    eval_config = confs.PUCTEvaluatorConfig(verbose=True,
                                            choose="choose_top_visits",

                                            dirichlet_noise_alpha=0.03,
                                            dirichlet_noise_pct=0.1,

                                            puct_before_expansions=3,
                                            puct_before_root_expansions=10,
                                            puct_constant_before=2.0,
                                            puct_constant_after=0.85,

                                            temperature=1.00,
                                            depth_temperature_max=2.0,
                                            depth_temperature_start=1,
                                            depth_temperature_increment=0.25,
                                            depth_temperature_stop=1,
                                            random_scale=1.0,

                                            fpu_prior_discount=0.25,
                                            max_dump_depth=3)

    config = confs.PUCTPlayerConfig(name="puct",
                                    verbose=True,
                                    generation=gen,
                                    playouts_per_iteration=100,
                                    playouts_per_iteration_noop=0,
                                    evaluator_config=eval_config)

    for k, v in kwds.items():
        actually_set = False
        if attrutil.has(eval_config, k):
            setattr(eval_config, k, v)
            actually_set = True

        if attrutil.has(config, k):
            setattr(config, k, v)
            actually_set = True

        if not actually_set:
            print '*** Attribute not found:', k

    return config


def main():
    from ggpzero.util.keras import init

    init()

    args = sys.argv[1:]
    port = int(args[0])

    generation = args[1]
    conf = get_default_conf(generation)

    if len(args) >= 2:
        playouts_multiplier = int(args[2])
        if playouts_multiplier == 0:
            conf.playouts_per_iteration = 0
        else:
            conf.playouts_per_iteration *= playouts_multiplier

    player = PUCTPlayer(conf=conf)

    from ggplib.play import play_runner
    play_runner(player, port)


if __name__ == "__main__":
    main()
