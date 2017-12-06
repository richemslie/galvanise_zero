import os
import math
import random
import tempfile

from collections import OrderedDict

import json

from ggplib.util import log
from ggplib.player.mcs import MCSPlayer
from ggplib.player.gamemaster import GameMaster
from ggplib.db.lookup import get_database
from ggplib.db.helper import get_gdl_for_game


class MCSPlayer2(MCSPlayer):
    choose_percentage = 0.25
    choose_temperature = 0.35
    max_run_time = 1.0

    def choose(self):
        assert self.root is not None

        # calculate total visits
        total_visits = 0
        for stat in self.root.values():
            total_visits += stat.visits

        # take top % of visits
        left_visits = int(total_visits * self.choose_percentage)

        # use boltzman score to divide scores up some more
        our_role_index = self.match.our_role_index

        def get_boltzman_score(stat):
            return math.exp(stat.get(our_role_index) / self.choose_temperature)

        # effectively split up into buckets
        total_sample_expected_value = 0
        for stat in sorted(self.root.values(),
                           key=lambda x: x.get(self.match.our_role_index),
                           reverse=True):
            visits = min(stat.visits, left_visits)
            left_visits -= visits
            print "X", stat.move, get_boltzman_score(stat) * visits
            total_sample_expected_value += get_boltzman_score(stat) * visits

            if left_visits == 0:
                break

        choose_expected_value = random.random() * total_sample_expected_value
        best_selection = None
        for stat in sorted(self.root.values(),
                           key=lambda x: x.get(self.match.our_role_index),
                           reverse=True):
            best_selection = stat
            choose_expected_value -= get_boltzman_score(stat) * stat.visits
            if choose_expected_value < 0:
                break

        assert best_selection is not None
        log.debug("choice move2 = %s" % best_selection.move)
        return best_selection.choice

    def before_apply_info(self):
        assert self.root is not None

        candidates = []
        # ok - now we dump everything for debug, and return the best score
        for stat in sorted(self.root.values(), key=lambda x: x.get(self.match.our_role_index), reverse=True):
            d = OrderedDict()
            d['choice'] = stat.choice
            d['move'] = stat.move
            d['visits'] = stat.visits
            d['score'] = stat.get(self.match.our_role_index)
            candidates.append(d)

        return json.dumps(dict(candidates=candidates), indent=4)


def run_game(game_name):
    # create gamemaster and add players
    gm = GameMaster(get_gdl_for_game(game_name))
    for role in gm.sm.get_roles():
        gm.add_player(MCSPlayer2(role), role)

    gm.start(meta_time=30, move_time=5)

    depth = 0
    last_move = None

    game = OrderedDict()
    game["game"] = game_name

    current = game["depth_%s" % depth] = OrderedDict()

    bs = gm.sm.new_base_state()

    def bases():
        gm.sm.get_current_state(bs)
        return [bs.get(i) for i in range(bs.len())]

    current["state"] = bases()

    while not gm.finished():
        last_move = gm.play_single_move(last_move=last_move)
        current["move"] = last_move

        candidates = current["candidates"] = OrderedDict()
        for player, role in gm.players:
            info = json.loads(player.before_apply_info())
            actions = []
            for c in info["candidates"]:
                actions.append([c['choice'], c['score'], c['visits']])
            candidates[role] = actions

        depth += 1
        current = game["depth_%s" % depth] = OrderedDict()
        current["state"] = bases()

    gm.play_to_end(last_move)
    game["final_scores"] = gm.scores

    return game


def main(game_name):
    from ggplib import interface
    interface.initialise_k273(1)

    # XXX reduce log 
    # import ggplib.util.log
    # ggplib.util.log.initialise()

    # pre-initialise database - used in match for remapping
    get_database()

    json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

    game_name = sys.argv[1]
    while True:
        games = []
        for i in range(25):
            games.append(run_game(game_name))

        fd, path = tempfile.mkstemp(suffix='.json', prefix="mcs_%s_" % game_name, dir=".")

        with os.fdopen(fd, 'w') as open_file:
            open_file.write(json.dumps(games))


if __name__ == "__main__":
    import pdb
    import sys
    import traceback

    try:
        main(sys.argv[1])

    except Exception as exc:
        print exc
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
