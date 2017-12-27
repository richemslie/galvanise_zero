import os
import math
import random
from operator import itemgetter

from ggplib.player.base import MatchPlayer
from ggplib.util import log

from ggplearn.util.bt import pretty_print_board

from ggplearn import msgdefs
from ggplearn.nn import bases

models_path = os.path.join(os.environ["GGPLEARN_PATH"], "src", "ggplearn", "models")


class PolicyPlayer(MatchPlayer):
    def __init__(self, conf):
        MatchPlayer.__init__(self, "%s_%s" % (conf.name, conf.generation))
        self.conf = conf

        self.nn = None
        self.bases_config = None

    def on_meta_gaming(self, finish_time):
        if self.conf.verbose:
            log.info("%s, match id: %s" % (self.get_name(), self.match.match_id))

        game_info = self.match.game_info

        # This is a performance hack, where once we get the nn/config we don't reload it.
        # Obviously now will only play that game.  For production should use 'latest' as will
        # always get the latest nn for that game.

        if self.conf.generation == 'latest' or self.nn is None:
            self.bases_config = bases.get_config(game_info.game, game_info.model, self.conf.generation)
            self.nn = self.bases_config.create_network()
            self.nn.load()

    def policy_to_actions(self, policy):
        # only do our moves
        ls = self.match.sm.get_legal_state(self.match.our_role_index)
        legals = set(ls.to_list())

        actions = self.match.game_info.model.actions[self.match.our_role_index]

        if self.match.our_role_index == 1:
            start_pos = len(self.match.game_info.model.actions[0])
        else:
            start_pos = 0

        actions = [(idx, move, policy[start_pos + idx])
                   for idx, move in enumerate(actions) if idx in legals]
        actions.sort(key=itemgetter(2), reverse=True)

        total_prob = sum(p for _, _, p in actions)
        if self.conf.verbose:
            if not (0.99 < total_prob < 1.01):
                log.debug("total probability %.2f != 1.00" % total_prob)

                # XXX log out anything (non-legal)with more 0.02 probabilty ?

        # normalise
        actions = [(idx, move, p / total_prob) for idx, move, p in actions]

        # apply temperature
        if self.conf.temperature > 0:
            before_actions = actions
            depth = max(1, (self.match.game_depth - self.conf.depth_temperature_start) *
                        self.conf.depth_temperature_increment)
            temp = 1.0 / self.conf.temperature * depth
            if self.conf.verbose:
                log.debug("depth %s, temperature %s " % (depth, temp))

            actions = [(idx, move, math.pow(p, temp)) for idx, move, p in actions]

            # try to renormalise - if can...
            total_prob = sum(p for _, _, p in actions)
            actions = [(idx, move, p / total_prob) for idx, move, p in actions]

            if self.conf.verbose:
                for (legal, move, prob), (_, _, before) in zip(actions, before_actions):
                    log.verbose("%s \t %.2f, %.2f" % (move, prob * 100, before * 100))
        else:
            if self.conf.verbose:
                for legal, move, prob in actions:
                    log.verbose("%s \t %.2f" % (move, prob * 100))

        return actions

    def on_next_move(self, finish_time):
        self.match.sm.update_bases(self.match.get_current_state())

        if self.conf.verbose:
            log.verbose("on_next_move() with gen %s" % self.conf.generation)

        bs = self.match.get_current_state()
        state = bs.to_list()

        if self.conf.verbose:
            if self.match.game_info.game == "breakthrough":
                pretty_print_board(self.match.sm, state)
                print

        # don't bother to predict if only one move.
        if self.conf.skip_prediction_single_move:
            ls = self.match.sm.get_legal_state(self.match.our_role_index)
            if ls.get_count() == 1:
                return ls.get_legal(0)

        policy, network_score = self.nn.predict_1(state, self.match.our_role_index)
        normalise_actions = self.policy_to_actions(policy)

        expected_prob = random.random() * self.conf.random_scale

        seen_prob = 0
        pos = 0
        for best_legal, best_move, best_prob in normalise_actions:
            seen_prob += best_prob
            if seen_prob > expected_prob:
                break
            pos += 1

        assert best_legal is not None
        if self.conf.verbose:
            log.verbose("Expected prob: %.2f, choice is %s @ %s" % (expected_prob,
                                                                    best_move, pos))

            log.info("Finals %.3f / %.3f" % tuple(network_score))
            log.info("Choice is %s with %.2f" % (best_move, best_prob * 100))

        return best_legal


###############################################################################


def main():
    import sys
    from ggplib.play import play_runner
    from ggplearn.util.keras import constrain_resources

    constrain_resources()

    port = int(sys.argv[1])
    generation = sys.argv[2]

    try:
        conf_name = sys.argv[3]
    except IndexError:
        conf_name = "default"

    confs = {
        'default' : msgdefs.PolicyPlayerConf(name="default", generation=generation,
                                             temperature=-1, random_scale=0.001),

        'abc' : msgdefs.PolicyPlayerConf(name="abc", generation=generation,
                                         random_scale=0.85,
                                         temperature=1.0,
                                         depth_temperature_start=6,
                                         depth_temperature_increment=0.25),
    }

    player = PolicyPlayer(confs[conf_name])
    play_runner(player, port)


if __name__ == "__main__":
    main()
