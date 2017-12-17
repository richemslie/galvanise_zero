import os

from ggplib.player.base import MatchPlayer
from ggplib.util import log

from ggplearn.util.bt import pretty_print_board

from ggplearn.nn import bases

models_path = os.path.join(os.environ["GGPLEARN_PATH"], "src", "ggplearn", "models")


class NNPlayerOneShot(MatchPlayer):

    def __init__(self, generation="latest"):
        MatchPlayer.__init__(self, "NNPlayerOneShot-" + generation)
        self.generation = generation
        self.nn = None
        self.generation = generation

    def on_meta_gaming(self, finish_time):
        log.info("NNPlayerOneShot, match id: %s" % self.match.match_id)

        game_info = self.match.game_info

        # this is a performance hack, where once we get the nn/config we don't reget it.
        # if latest is set will always get the latest

        if self.generation == 'latest' or self.nn is None:
            self.base_config = bases.get_config(game_info.game, game_info.model, self.generation)
            self.nn = self.base_config.create_network()
            self.nn.load()

    def on_next_move(self, finish_time):
        sm = self.match.sm
        sm.update_bases(self.match.get_current_state())

        # only do our moves
        ls = sm.get_legal_state(self.match.our_role_index)
        if ls.get_count() == 1:
            return ls.get_legal(0)

        print len(self.generation) * "="
        print self.generation
        print len(self.generation) * "="

        legals = set(ls.to_list())

        bs = self.match.get_current_state()
        state = bs.to_list()

        policy, score = self.nn.predict_1(state, self.match.our_role_index)

        if self.match.our_role_index == 1:
            start_pos = len(self.match.game_info.model.actions[0])
        else:
            start_pos = 0

        actions = self.match.game_info.model.actions[self.match.our_role_index]

        best = -1
        best_idx = None
        best_move = None

        if self.match.game_info.game == "breakthrough":
            pretty_print_board(sm, state)
            print

        weirds = []
        print "all states"
        actions = list(enumerate(actions))
        actions.sort(key=lambda c: policy[start_pos + c[0]], reverse=True)

        for idx, move in actions:
            ridx = start_pos + idx
            pvalue = policy[ridx] * 100

            if idx in legals:
                print move, "%.2f" % pvalue

                if pvalue > best:
                    best = pvalue
                    best_idx = idx
                    best_move = move

            else:
                if pvalue > 2:
                    weirds.append((move, pvalue))

        print
        if weirds:
            print "WIERDS:"
            for move, pvalue in weirds:
                print move, "%.2f" % pvalue
            print

        if best is not None:
            print "============="
            print "Finals %.3f / %.3f" % tuple(score)
            print "Choice is %s" % best_move
            print "============="
            return best_idx

        return ls.get_legal(0)


###############################################################################

def main():
    import sys
    from ggplib.play import play_runner
    from ggplearn.util.keras import constrain_resources

    constrain_resources()

    port = int(sys.argv[1])
    generation = sys.argv[2]

    player = NNPlayerOneShot(generation)
    play_runner(player, port)


if __name__ == "__main__":
    main()
