import numpy as np
from keras.models import model_from_json
from ggplearn import process_games
from ggplib.player.base import MatchPlayer
from ggplib.util import log

class NNPlayerOneShot(MatchPlayer):

    def on_meta_gaming(self, finish_time):
        log.info("NNPlayerOneShot, match id: %s" % self.match.match_id)

        sm = self.match.sm
        game_info = self.match.game_info

        # look up game...
        with open("model_nn_%s.json" % game_info.game, "r") as f:
            self.nn_model = model_from_json(f.read())

        self.nn_model.load_weights("weights_nn_%s.h5" % game_info.game)
        self.nn_config = process_games.get_bases_config(game_info.game)
        self.base_infos = process_games.create_base_infos(self.nn_config, game_info.model)

    def on_next_move(self, finish_time):
        sm = self.match.sm
        sm.update_bases(self.match.get_current_state())

        # only do our moves
        ls = sm.get_legal_state(self.match.our_role_index)
        if ls.get_count() == 1:
            return ls.get_legal(0)

        legals = set([ls.get_legal(i) for i in range(ls.get_count())])

        bs = self.match.get_current_state()
        state = [bs.get(i) for i in range(bs.len())]

        X_0 = process_games.state_to_channels(state, self.match.our_role_index, self.nn_config, self.base_infos)
        X_0 = X_0.reshape(1, self.nn_config.num_rows, self.nn_config.num_cols, self.nn_config.num_channels)
        X_1 = np.array([[v for v, base_info in zip(state, self.base_infos) if base_info.channel is None]])
        result = self.nn_model.predict([X_0, X_1], batch_size=1)
        policy, scores = result[0][0], result[1][0]

        if self.match.our_role_index == 1:
            start_pos = len(self.match.game_info.model.actions[0])
        else:
            start_pos = 0

        use_policy = True
        best = -1
        best_idx = None
        best_ridx = None
        for idx in range(len(self.match.game_info.model.actions[1])):
            ridx = start_pos + idx
            if idx in legals:
                print sm.legal_to_move(0, idx), "%.2f" % (policy[ridx] * 100), "%.2f" % scores[ridx]

                if use_policy:
                    if policy[ridx] > best:
                        best = policy[ridx]
                        best_idx = idx
                        best_ridx = ridx

                else:
                    if scores[ridx] > best:
                        best = scores[ridx]
                        best_idx = idx
                        best_ridx = ridx

        print "WEIRD NOT LEGAL:"
        for idx in range(len(self.match.game_info.model.actions[1])):
            ridx = start_pos + idx
            if idx not in legals:
                #if policy[ridx] * 100 > 0:
                print sm.legal_to_move(0, idx), "%.2f" % (policy[ridx] * 100), "%.2f" % scores[ridx]


        if best_idx is not None:
            print "============="
            print "Choice is %s, %.2f / %.2f" % (sm.legal_to_move(0, best_idx),
                                                 (policy[best_ridx] * 100),
                                                 scores[best_ridx])
            print "============="


            return best_idx
        ZZZ

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

    player = NNPlayerOneShot(player_name)

    ggp = GGPServer()
    ggp.set_player(player)
    site = server.Site(ggp)

    reactor.listenTCP(port, site)
    reactor.run()


if __name__ == "__main__":
    main()
