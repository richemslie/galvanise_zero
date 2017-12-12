import os
import numpy as np

from keras.models import model_from_json

from ggplib.player.base import MatchPlayer
from ggplib.util import log

from ggplearn.utils.bt import pretty_print_board
from ggplearn import net_config

models_path = os.path.join(os.environ["GGPLEARN_PATH"], "src", "ggplearn", "models")


class NNPlayerOneShot(MatchPlayer):

    def __init__(self, generation):
        MatchPlayer.__init__(self, "NNPlayerOneShot-" + generation)
        self.generation = generation
        self.nn_config = None

    def on_meta_gaming(self, finish_time):
        log.info("NNPlayerOneShot, match id: %s" % self.match.match_id)

        game_info = self.match.game_info

        if self.nn_config is None:
            self.nn_config = net_config.get_bases_config(game_info.game)
            self.base_infos = net_config.create_base_infos(self.nn_config, game_info.model)

            # load neural network model and weights
            model_filename = os.path.join(models_path, "model_nn_%s_%s.json" % (game_info.game, self.generation))
            weights_filename = os.path.join(models_path, "weights_nn_%s_%s.h5" % (game_info.game, self.generation))

            with open(model_filename, "r") as f:
                self.nn_model = model_from_json(f.read())

            self.nn_model.load_weights(weights_filename)

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

        legals = set([ls.get_legal(i) for i in range(ls.get_count())])

        bs = self.match.get_current_state()
        state = [bs.get(i) for i in range(bs.len())]

        X_0 = net_config.state_to_channels(state,
                                           self.match.our_role_index,
                                           self.nn_config,
                                           self.base_infos)

        X_0 = X_0.reshape(1,
                          self.nn_config.num_rows,
                          self.nn_config.num_cols,
                          self.nn_config.num_channels)

        X_1 = np.array([[v for v, base_info in zip(state,
                                                   self.base_infos)
                         if base_info.channel is None]])

        result = self.nn_model.predict([X_0, X_1], batch_size=1)

        x = [result[i][0] for i in range(2)]
        policy, scores = x[0], x[1]

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
            print "Finals %.3f / %.3f" % tuple(scores)
            print "Choice is %s" % best_move
            print "============="
            return best_idx

        return ls.get_legal(0)


###############################################################################

def main():
    import sys
    from twisted.internet import reactor
    from twisted.web import server

    from ggplib.util import log
    from ggplib.server import GGPServer
    from ggplib import interface
    from ggplearn.utils.keras import use_one_cpu_please

    port = int(sys.argv[1])
    generation = sys.argv[2]

    player_name = NNPlayerOneShot.__class__.__name__ + "_" + generation
    interface.initialise_k273(1, log_name_base=player_name)
    log.initialise()

    use_one_cpu_please()

    player = NNPlayerOneShot(generation)

    ggp = GGPServer()
    ggp.set_player(player)
    site = server.Site(ggp)

    reactor.listenTCP(port, site)
    reactor.run()


if __name__ == "__main__":
    main()
