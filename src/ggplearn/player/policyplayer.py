import os
from operator import itemgetter

import numpy as np

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

        self.game_depth = 0

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
        total_prob = sum(p for _, _, p in actions)
        assert (0.99 < total_prob < 1.01)

        # if self.conf.temperature_value > 0:
        #    actions = self.apply_temperature(actions)

        if self.conf.verbose:
            for legal, move, prob in actions:
                log.verbose("%s \t %.2f" % (move, prob * 100))

        return actions

    def apply_temperature(self, actions):
        # WIP ... somehow normalised isn't working, and too tired to figure out right now
        # XXX how many parameters are required here?  I dont want to overload conf with this.
        assert 0 < self.conf.temperature_value < 1

        pct = self.conf.temperature_pct
        value = self.conf.temperature_value
        depth = self.conf.temperature_depth - min(self.conf.temperature_depth, self.game_depth + 1)
        temp = (1 - pct) + pct * (value ** depth)
        temp = 1 / temp

        if self.conf.verbose:
            log.info("depth %d, temperature is %.3f" % (depth, temp))

        temps = [(l, m, p ** temp) for l, m, p in actions]

        from pprint import pprint
        pprint(actions)
        pprint(temps)
        temps_tot = sum(p for _, _, p in temps)
        print "XXX temps_tot", temps_tot
        normalise = [(l, m, p / temps_tot) for l, m, p in actions]
        print "XXX2 temps_tot", sum(p for _, _, p in normalise)

        normalise.sort(key=itemgetter(2), reverse=True)
        pprint(normalise)

        return normalise

    def on_apply_move(self, joint_move):
        self.game_depth += 1

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
            print "all states"

        policy, network_score = self.nn.predict_1(state, self.match.our_role_index)
        normalise_actions = self.policy_to_actions(policy)

        # choose()
        if self.conf.choose_exponential_scale > 0:
            expected_prob = np.random.exponential(self.conf.choose_exponential_scale, 1)
            expected_prob *= self.conf.random_scale
            if self.conf.verbose:
                log.verbose("expected_prob: %.2f" % expected_prob)

            seen_prob = 0
            for best_legal, best_move, best_prob in normalise_actions:
                seen_prob += best_prob
                if seen_prob > expected_prob:
                    break
        else:
            best_legal, best_move, best_prob = normalise_actions[0]

        assert best_legal is not None
        if self.conf.verbose:
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

    conf = msgdefs.PolicyPlayerConf(name="0.25", generation=generation, choose_exponential_scale=0.25)
    player = PolicyPlayer(conf)

    play_runner(player, port)


if __name__ == "__main__":
    main()
