import os
import time
import uuid
import random

from ggplib.player import get
from ggplib.player.gamemaster import GameMaster

from ggplib.db import lookup
from ggplib.db.helper import get_gdl_for_game

from ggpzero.util import attrutil

from ggpzero.defs import templates

from ggpzero.player.cpuctplayer import CppPUCTPlayer

from ggpzero.tournament.confs import TiltyardMatchSummary, TiltyardMatch, MatchSummaries


from ggplib.util.symbols import SymbolFactory
symbol_factory = SymbolFactory()


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()


games_to_url = {
    "breakthrough" : "http://games.ggp.org/base/games/breakthrough/v0/",
    "cittaceot" : "http://games.ggp.org/base/games/cittaceot/v0/",
    "checkers" : "http://games.ggp.org/base/games/checkers/v1/",
    "connectFour" : "http://games.ggp.org/base/games/connectFour/v0/",
    "escortLatch" : "http://games.ggp.org/base/games/escortLatch/v0/",
    "hex" : "http://games.ggp.org/base/games/hex/v0/",
    "reversi" : "http://games.ggp.org/base/games/reversi/v0/",
}


def create_match_info(start_clock, play_clock, role_count,
                      game, player_names, is_completed):
    assert len(player_names) == role_count

    match_info = TiltyardMatch()
    match_info.randomToken = str(uuid.uuid4())

    match_info.gameMetaURL = games_to_url[game]
    match_info.startClock = start_clock
    match_info.playClock = play_clock
    match_info.scrambled = False
    match_info.playerNamesFromHost = player_names
    match_info.tournamentNameFromHost = "ggpzero fun times"

    # at end:
    match_info.isAborted = False
    match_info.isCompleted = False

    match_info.startTime = time.time()

    # populate during match
    # match_info.stateTimes = []
    # match_info.states = []
    # match_info.moves = []
    # match_info.errors = []
    # match_info.goalValues = []

    return match_info


def basestate_to_str(sm_model, bs):
    def to_str(symbol):
        s = str(symbol)
        s = s.replace('(', ' ( ').replace(')', ' ) ')
        return s

    symbols = [symbol_factory.symbolize(s)[1] for idx, s in enumerate(sm_model.bases) if bs.get(idx)]

    res = "(%s)" % (" ".join(to_str(s) for s in symbols))
    return res


def set_initial_state(match_info, state_str):
    match_info.stateTimes.append(time.time())
    match_info.states.append(state_str)


def add_to_next(match_info, move, state_str):
    match_info.stateTimes.append(time.time())
    match_info.states.append(state_str)
    match_info.moves.append(move)
    match_info.errors.append("")


def set_goals(match_info, goal_values):
    match_info.goalValues = goal_values
    match_info.isCompleted = True


def matches_path(game):
    return os.path.join(os.environ["GGPZERO_PATH"], "data", "tournament", game)


def summary_path():
    return os.path.join(os.environ["GGPZERO_PATH"], "data", "tournament", "summary.json")


def update_summaries(game, match_info):
    # generate a TiltyardMatchSummary
    summary = TiltyardMatchSummary()
    for k in ("randomToken playerNamesFromHost scrambled startTime "
              "playClock tournamentNameFromHost startClock matchId gameMetaURL "
              "isAborted isCompleted goalValues".split()):
        setattr(summary, k, getattr(match_info, k))
    summary.matchURL = "http://localhost:8888/%s/%s" % (game, match_info.randomToken)

    try:
        the_summaries = attrutil.json_to_attr(open(summary_path()).read())
    except:
        the_summaries = MatchSummaries()

    # find the summary:
    for idx, s in enumerate(the_summaries.queryMatches):
        if s.randomToken == summary.randomToken:
            the_summaries.queryMatches[idx] = summary
            return the_summaries

    # add to front
    the_summaries.queryMatches = [summary] + the_summaries.queryMatches
    return the_summaries


def save_summaries(the_summaries):
    with open(summary_path(), 'w') as f:
        f.write(attrutil.attr_to_json(the_summaries,
                                      sort_keys=True,
                                      separators=(',', ': '),
                                      indent=4))


def update_match_info(game, match_info):
    # overwrites match info file
    path_to_matches = matches_path(game)
    match_info_path = os.path.join(path_to_matches,
                                   match_info.randomToken + ".json")
    with open(match_info_path, 'w') as f:
        f.write(attrutil.attr_to_json(match_info))

    the_summaries = update_summaries(game, match_info)

    # overwrites summaries file
    save_summaries(the_summaries)


def do_game(game, gen, players, meta_time, move_time):
    # ensure we have a network

    gm = GameMaster(get_gdl_for_game(game), verbose=True)
    info = lookup.by_name(game)

    for player, role in zip(players, gm.sm.get_roles()):
        gm.add_player(player, role)

    roles = gm.sm.get_roles()
    match_info = create_match_info(meta_time, move_time, len(roles),
                                   game, [p.get_name() for p, r in gm.players], False)

    # save match_info
    update_match_info(game, match_info)

    gm.start(meta_time=meta_time, move_time=move_time)

    the_bs = gm.sm.new_base_state()
    gm.sm.get_current_state(the_bs)
    set_initial_state(match_info, basestate_to_str(info.model, the_bs))

    last_move = None
    while not gm.finished():
        last_move = gm.play_single_move(last_move)

        gm.sm.get_current_state(the_bs)
        add_to_next(match_info, list(last_move), basestate_to_str(info.model, the_bs))

        update_match_info(game, match_info)

    gm.play_to_end(last_move)
    set_goals(match_info, [gm.get_score(r) for _, r in gm.players])

    update_match_info(game, match_info)


def main():
    setup()

    mapping = {
        #'breakthrough' : ["v5_%s" % s for s in [80, 92]],
        'cittaceot' : ["v8_%s" % s for s in [10, 12]],
        #'checkers' : ["v7_%s" % s for s in [10, 16]],
        #'connectFour' : ["v7_%s" % s for s in [25, 38]],
        #'escortLatch' : ["v7_%s" % s for s in [5, 10]],
        'hex' : ["v7_%s" % s for s in [15, 20]],
        #'reversi' : ["v7_%s" % s for s in [25, 39]]
    }

    simplemcts = get.get_player("simplemcts")
    players = {}
    for game in mapping:
        players.setdefault(game, [simplemcts])

        path_to_matches = matches_path(game)
        if not os.path.exists(path_to_matches):
            os.makedirs(path_to_matches)

    for game, gens in mapping.items():
        for gen in gens:
            conf = templates.puct_config_template(gen, "compete")
            players[game].append(CppPUCTPlayer(conf=conf))

    meta_time = 30
    move_time = 10
    while True:
        game = random.choice(mapping.keys())
        players_available = players[game]
        random.shuffle(players_available)

        do_game(game, gen, players_available[:2], meta_time, move_time)


###############################################################################

if __name__ == "__main__":
    debug = True
    try:
        assert debug
        from ipdb import launch_ipdb_on_exception
        with launch_ipdb_on_exception():
            main()
    except Exception as _:
        main()
