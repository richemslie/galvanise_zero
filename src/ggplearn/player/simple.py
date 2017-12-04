import numpy as np
from keras.models import model_from_json
from ggplearn import process_games
from ggplib.player.base import MatchPlayer
from ggplib.util import log

class NNPlayerOneShot(MatchPlayer):

    def __init__(self, postfix):
        self.postfix = postfix
        MatchPlayer.__init__(self, "NNPlayerOneShot-" + self.postfix)

    def on_meta_gaming(self, finish_time):
        log.info("NNPlayerOneShot, match id: %s" % self.match.match_id)

        sm = self.match.sm
        game_info = self.match.game_info

        # look up game...
        with open("model_nn_%s_%s.json" % (game_info.game, self.postfix), "r") as f:
            self.nn_model = model_from_json(f.read())

        self.nn_model.load_weights("weights_nn_%s_%s.h5" % (game_info.game, self.postfix))
        self.nn_config = process_games.get_bases_config(game_info.game)
        self.base_infos = process_games.create_base_infos(self.nn_config, game_info.model)

    def on_next_move(self, finish_time):
        sm = self.match.sm
        sm.update_bases(self.match.get_current_state())

        # only do our moves
        ls = sm.get_legal_state(self.match.our_role_index)
        if ls.get_count() == 1:
            return ls.get_legal(0)

        print len(self.postfix) * "="
        print self.postfix
        print len(self.postfix) * "="

        legals = set([ls.get_legal(i) for i in range(ls.get_count())])

        bs = self.match.get_current_state()
        state = [bs.get(i) for i in range(bs.len())]

        X_0 = process_games.state_to_channels(state, self.match.our_role_index, self.nn_config, self.base_infos)
        X_0 = X_0.reshape(1, self.nn_config.num_rows, self.nn_config.num_cols, self.nn_config.num_channels)
        X_1 = np.array([[v for v, base_info in zip(state, self.base_infos) if base_info.channel is None]])
        result = self.nn_model.predict([X_0, X_1], batch_size=1)

        policy, av_scores = result[0][0], result[1][0]

        if self.match.our_role_index == 1:
            start_pos = len(self.match.game_info.model.actions[0])
        else:
            start_pos = 0

        actions = self.match.game_info.model.actions[1]

        best = -1
        best_idx = None
        best_move = None

        weirds = []
        for idx, move in enumerate(actions):
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
            s = 2 if self.match.our_role_index else 0
            print "============="
            print "Choice is %s, %.2f / %.2f / %.2f" % (best_move, best, av_scores[s], av_scores[s+1])
            return best_idx

        return ls.get_legal(0)

def main():
    import sys
    from twisted.internet import reactor
    from twisted.web import server

    from ggplib.util import log
    from ggplib.server import GGPServer
    from ggplib import interface

    port = int(sys.argv[1])

    player_name = NNPlayerOneShot
    interface.initialise_k273(1, log_name_base=player_name)
    log.initialise()

    player = NNPlayerOneShot(sys.argv[2])

    ggp = GGPServer()
    ggp.set_player(player)
    site = server.Site(ggp)

    reactor.listenTCP(port, site)
    reactor.run()


if __name__ == "__main__":
    main()
