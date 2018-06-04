'''
Copyright Richard Emslie 2018.

Original from hippolyta - from with license:
Copyright Tom Plick 2010, 2014.

This file is released under the GPL version 3.
'''

import os
import re
import sys
import time
import pprint
import urllib
import urllib2
import functools

from bs4 import BeautifulSoup

from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game
from ggplib.util import log

from ggpzero.util import attrutil as at
from ggpzero.defs import confs
from ggpzero.player import puctplayer

from ggpzero.scripts.battle.hex import hex_get_state


@at.register_attrs
class GameConfig(object):
    game_name = at.attribute("reversi")
    generation = at.attribute("genx")

    # 100 * n
    sims_multiplier = at.attribute(8)

    depth_temperature_max = at.attribute(1.5)
    depth_temperature_stop = at.attribute(16)


@at.register_attrs
class LGConfig(object):
    # LG convention is to postfix non humans with "_bot"
    whoami = at.attribute("gzero_bot")

    # fetch these from your browser, after logged in
    cookie = at.attribute("login2=.......; JSESSIONID=.......")

    # dry run, dont actually send moves
    dry_run = at.attribute(True)

    # list of GameConfig
    play_games = at.attribute(default=at.attr_factory(list))

    # store game path
    store_path = at.attribute("/home/rxe/working/ggpzero/data/lg/")

    # only let these games play, some manual control
    allow_match_ids = at.attribute(default=at.attr_factory(list))


def template_config():
    c = LGConfig()
    c.play_games = [GameConfig(g)
                    for g in "breakthrough reversi reversi_10x10".split()]
    return c


class GameMasterByGame(object):
    def __init__(self, game_config):
        assert isinstance(game_config, GameConfig)
        self.game_config = game_config

        game_name = self.game_config.game_name
        puct_conf = self.get_puct_config()

        self.gm = GameMaster(get_gdl_for_game(game_name))

        # add players
        for role in self.gm.sm.get_roles():
            self.gm.add_player(puctplayer.PUCTPlayer(conf=puct_conf), role)

    def get_puct_config(self):
        multiplier = self.game_config.sims_multiplier
        if multiplier == 0:
            playouts_per_iteration = 1
        else:
            playouts_per_iteration = 100 * multiplier

        conf = confs.PUCTPlayerConfig(name="clx",
                                      generation=self.game_config.generation,
                                      verbose=True,

                                      playouts_per_iteration=playouts_per_iteration,
                                      playouts_per_iteration_noop=0,

                                      dirichlet_noise_alpha=0.03,
                                      root_expansions_preset_visits=-1,
                                      fpu_prior_discount=0.25,

                                      puct_before_expansions=3,
                                      puct_before_root_expansions=4,
                                      puct_constant_before=3.0,
                                      puct_constant_after=0.75,

                                      choose="choose_temperature",
                                      temperature=2.0,
                                      depth_temperature_max=self.game_config.depth_temperature_max,
                                      depth_temperature_start=4,
                                      depth_temperature_increment=0.75,
                                      depth_temperature_stop=self.game_config.depth_temperature_stop,
                                      random_scale=0.35,

                                      max_dump_depth=1)

        return conf

    def get_move(self, state, depth, lead_role_index):
        log.info("GameMasterByGame.get_move: %s" % state)

        if isinstance(state, str):
            state = self.gm.convert_to_base_state(state)

        self.gm.reset()
        self.gm.start(meta_time=180, move_time=300,
                      initial_basestate=state,
                      game_depth=depth)

        move = self.gm.play_single_move(last_move=None)
        player = self.gm.get_player(lead_role_index)
        return move[lead_role_index], player.last_probability, self.gm.finished()


class LittleGolemConnection(object):
    MIN_WAIT_TIME = 5
    MAX_WAIT_TIME = 60
    WAIT_TIME_FOR_UPDATE = 5
    WAIT_TIME_COUNT = 3

    def __init__(self, config):
        self.config = config
        self.games_by_gamemaster = {}

        for c in self.config.play_games:
            gm_by_game = GameMasterByGame(c)
            self.games_by_gamemaster[c.game_name] = gm_by_game

        # check store path exists
        if not os.path.exists(self.config.store_path):
            os.makedirs(self.config.store_path)

    def get_page(self, url):
        url = "http://www.littlegolem.net/" + url

        c = self.config
        req = urllib2.Request(url.replace(" ", "+"),
                              headers={"Cookie" : c.cookie})
        return urllib2.urlopen(req).read()

    def post_page(self, url, data=None):
        url = "http://www.littlegolem.net/" + url

        assert data is not None
        url_values = urllib.urlencode(data)

        c = self.config
        req = urllib2.Request(url.replace(" ", "+"),
                              data=url_values,
                              headers={"Cookie" : c.cookie})

        return urllib2.urlopen(req).read()

    def games_waiting(self):
        text = self.get_page("jsp/game/index.jsp")
        soup = BeautifulSoup(text)

        for div in soup.find_all("div", attrs={"class": "row"}):
            if "On Move" in str(div):
                for tbody in div.find_all("tbody"):
                    for tr in tbody.find_all("tr"):
                        cols = tr.find_all("td")
                        match_id = re.findall(r'\d+', str(cols[0]))[0]
                        depth = re.findall(r'\d+', str(cols[5]))[0]
                        opponent = cols[2].string
                        if isinstance(opponent, unicode):
                            opponent = opponent.encode('ascii', 'ignore')
                        s = "game waiting: '%s', against '%s' @ depth %s (match_id: %s)" % (cols[4].get_text(),
                                                                                            opponent,
                                                                                            depth,
                                                                                            match_id)
                        log.verbose(len(s) * "=")
                        log.verbose(s)
                        log.verbose(len(s) * "=")
                        match_id = int(match_id)
                        if match_id in self.config.allow_match_ids:
                            yield match_id, opponent, int(depth)

    def answer_invitation(self):
        text = self.get_page("jsp/invitation/index.jsp")
        if "Refuse</a>" not in text:
            return False

        # get the first invite
        text = text[text.find("<td>Your decision</td>") : text.find("Refuse</a>")]
        invite_id = re.search(r'invid=(\d+)">Accept</a>', text).group(1)

        log.info("GOT INVITATION: %s" % invite_id)

        # accept all games - whether we play or not is decided by config.allow_match_ids
        response = "accept"
        self.get_page("ng/a/Invitation.action?%s=&invid=%s" % (response,
                                                               invite_id))
        return True

    def play_move(self, game, *args):
        return self.games_by_gamemaster[game].get_move(*args)

    def handle_hex(self, board_size, match_id, depth, sgf, text):
        log.verbose("handle_hex, board_size %d, match_id:%s, depth:%d" % (board_size, match_id, depth))
        role_index, state = hex_get_state(board_size, sgf)
        game = "hexLG%d" % board_size
        move, prob, finished = self.play_move(game, state, depth, role_index)

        if move != "swap":
            move = move.replace("(place", "").replace(")", "")
            parts = move.split()
            move = parts[0] + "_abcdefghijklm"[int(parts[1])]

        return move, prob, finished

    def handle_breakthrough(self, match_id, depth, sgf, text):
        cords = []
        for s in re.findall(r"alt='\w*'", text):
            color = re.search(r"alt='(\w*)'", s).group(1)
            val = {"" : '0', "white" : '1', "black" : '2', "W" : '2', "B" : '1', "S" : '3'}[color]
            cords.append(val)

        assert len(cords) == 64
        log.verbose("handle_breakthrough, match_id:%s, depth:%d\n%s" % (match_id,
                                                                        depth,
                                                                        cords))

        # group into rows
        rows = []
        for i in range(8):
            s, e = i * 8, (i + 1) * 8
            rows.append(cords[s:e])

        # get role to play, and rotate board (LG flips the board)
        if text.find("<b>g</b>") < text.find("<b>h</b>"):
            rows = [r[::-1] for r in rows[::-1]]
            our_role = "white"
            our_lead_role_index = 0
        else:
            our_role = "black"
            our_lead_role_index = 1

        log.verbose("role to play %s" % our_role)

        # convert to gdl trues string
        trues = []
        trues.append("(true (control %s))" % our_role)

        for y, row in enumerate(rows):
            for x, num in enumerate(row):
                if num == '0':
                    continue
                role = 'white' if num == '1' else 'black'
                trues.append("(true (cellHolds %s %s %s))" % (x + 1,
                                                              y + 1,
                                                              role))

        # actually play the move
        move, prob, finished = self.play_move("breakthrough", " ".join(trues), depth, our_lead_role_index)

        log.info("move %s with %s / %s" % (move, prob, "finished" if finished else ""))

        # convert to what little golem expects
        move = move.replace("(move", "").replace(")", "")
        a, b, c, d = move.split()
        a = 8 - int(a)
        b = int(b) - 1
        c = 8 - int(c)
        d = int(d) - 1
        return "%s%s%s%s" % (a, b, c, d), prob, finished

    def handle_reversi(self, match_id, depth, sgf, text):
        EMPTY, WHITE, BLACK = 0, 1, 2

        player_black = re.search(r"PB\[[a-zA-Z0-9_ ]*\]", sgf).group(0)[3:-1]
        player_white = re.search(r"PW\[[a-zA-Z0-9_ ]*\]", sgf).group(0)[3:-1]

        print player_black, player_white

        if player_black == self.config.whoami:
            our_role = "black"
            our_lead_role_index = 0
        else:
            our_role = "red"
            our_lead_role_index = 1

        print our_role, our_lead_role_index

        cords = []
        for line in text.splitlines():
            if "amazon/0.gif" in line:
                cords.append(EMPTY)
            elif "reversi/b.gif" in line:
                cords.append(BLACK)
            elif "reversi/w.gif" in line:
                cords.append(WHITE)

        # group into rows
        rows = []
        for i in range(8):
            s, e = i * 8, (i + 1) * 8
            rows.append(cords[s:e])
        pprint.pprint(rows)

        trues = []
        trues.append("(true (control %s))" % our_role)

        # weird GGP reversi implementation maps things differently:
        #   white -> red
        #   black -> black
        #   reflect rows

        for y, row in enumerate(rows):
            row = row[::-1]
            for x, what in enumerate(row):
                if what == EMPTY:
                    continue
                stone = 'black' if what == BLACK else 'red'
                trues.append("(true (cell %s %s %s))" % (x + 1,
                                                         y + 1,
                                                         stone))

        # actually play the move
        move, prob, finished = self.play_move("reversi", " ".join(trues), depth, our_lead_role_index)
        print "move", move, prob, "finished" if finished else ""

        move = move.replace("(move", "").replace(")", "")

        # noop means pass in GGP
        if move == "noop":
            return "pass", prob, finished

        # convert to what little golem expects
        a, b = move.split()
        a = "hgfedcba"[int(a) - 1]
        b = "abcdefgh"[int(b) - 1]
        return "%s%s" % (a, b), prob, finished

    def handle_reversi_10x10(self, match_id, depth, sgf, text):
        EMPTY, WHITE, BLACK = 0, 1, 2

        player_black = re.search(r"PB\[[a-zA-Z0-9_ ]*\]", sgf).group(0)[3:-1]
        player_white = re.search(r"PW\[[a-zA-Z0-9_ ]*\]", sgf).group(0)[3:-1]

        print player_black, player_white

        if player_black == self.config.whoami:
            our_role = "black"
            our_lead_role_index = 0
        else:
            our_role = "white"
            our_lead_role_index = 1

        print our_role, our_lead_role_index

        cords = []
        for line in text.splitlines():
            if "amazon/0.gif" in line:
                cords.append(EMPTY)
            elif "reversi/b.gif" in line:
                cords.append(BLACK)
            elif "reversi/w.gif" in line:
                cords.append(WHITE)

        # group into rows
        rows = []
        for i in range(10):
            s, e = i * 10, (i + 1) * 10
            rows.append(cords[s:e])
        pprint.pprint(rows)

        trues = []
        trues.append("(true (control %s))" % our_role)

        for y, row in enumerate(rows):
            for x, what in enumerate(row):
                if what == EMPTY:
                    continue
                stone = 'black' if what == BLACK else 'white'
                trues.append("(true (cell %s %s %s))" % (x + 1,
                                                         y + 1,
                                                         stone))

        # actually play the move
        move, prob, finished = self.play_move("reversi_10x10", " ".join(trues), depth, our_lead_role_index)
        print "move", move, prob, "finished" if finished else ""
        move = move.replace("(move", "").replace(")", "")

        # noop means pass in GGP
        if move == "noop":
            return "pass", prob, finished

        # convert to what little golem expects
        a, b = move.split()
        a = "abcdefghij"[int(a) - 1]
        b = "abcdefghij"[int(b) - 1]
        return "%s%s" % (a, b), prob, finished

    def handle_game(self, match_id, opponent, depth):
        log.info("Handling game %s / %s, against '%s'" % (match_id, depth, opponent))

        # [note this may be different for different game, will need to fish for it in text XXX]
        orig_sgf = self.get_page("servlet/sgf/%s/game%s.txt" % (match_id, match_id))

        # save a copy of sgf per game... if it doesn't exist then send a welcome message!
        fn = os.path.join(self.config.store_path, "game_%s.txt" % match_id)
        send_welcome = not os.path.exists(fn)
        with open(fn, "w") as fw:
            fw.write(orig_sgf)

        text = self.get_page("jsp/game/game.jsp?gid=%s" % match_id)
        soup = BeautifulSoup(text)

        penny_for_thoughts = False
        for x in soup.find_all("div", attrs={"class": "portlet-body"}):
            msgs = x.find_all("p")
            if len(msgs):
                for m in list(msgs):
                    if "penny" in str(m).lower():
                        penny_for_thoughts = True

                fn = os.path.join(self.config.store_path, "msgs_%s.txt" % match_id)
                with open(fn, "a") as fw:
                    fw.write(str(msgs))
                    fw.write("\n")

        meth = None
        if "Breakthrough-Size 8" in text:
            meth = self.handle_breakthrough

        elif "Reversi-Size 8x8" in text:
            meth = self.handle_reversi

        elif "Reversi 10x10-Size 10x10" in text:
            meth = self.handle_reversi_10x10

        elif "Hex-Size 11" in text:
            meth = functools.partial(self.handle_hex, 11)

        elif "Hex-Size 13" in text:
            meth = functools.partial(self.handle_hex, 13)

        else:
            assert False, "unknown game: '%s'" % text

        move, prob, finished = meth(match_id, depth, orig_sgf, text)
        if self.config.dry_run:
            print orig_sgf
            print "Would of sent", move
            print "Probability", prob
            print "Is finished", finished
            sys.exit(0)

        # if the move is invalid, we wont get any indication of it.  Instead let's load the sgf.
        # If the game doesnt advance, abort.  This will stop us from hammering the server with
        # invalid moves, and probably causing a ban.

        log.info("sending move '%s' for match_id: %s" % (move, match_id))

        if prob > -0.005 and prob < 0.015:
            print "Resigning as probability is %.3f" % prob
            msg = "GG :)  Thanks- I resigned as my probability of winning was %.3f" % prob
            self.post_page("jsp/game/game.jsp", dict(sendgame=match_id,
                                                     sendmove="resign",
                                                     message=msg))

            # dont bother checking if game advances.  It doesn't.

        else:
            msg = ""
            if send_welcome:
                assert not finished
                msg = "Hi there!  My name is gzero, I am a bot.  I hope you a fun game! :)."

            elif finished:
                msg = "Thanks for playing."

            if penny_for_thoughts:
                if msg:
                    msg += "  ....  "
                msg += "gzero thinks its probability of winning is %.3f" % prob

            if msg:
                self.post_page("jsp/game/game.jsp", dict(sendgame=match_id,
                                                         sendmove=move,
                                                         message=msg))
            else:
                self.post_page("jsp/game/game.jsp", dict(sendgame=match_id,
                                                         sendmove=move))

            # check that game advances...
            for i in range(self.WAIT_TIME_COUNT):
                time.sleep(self.WAIT_TIME_FOR_UPDATE)
                new_sgf = self.get_page("servlet/sgf/%s/game%s.txt" % (match_id, match_id))
                if new_sgf != orig_sgf:
                    return

            log.error("Game didn't advance on sending move.  Aborting.")
            sys.exit(1)

    def poll_gen(self):
        sleep_time = self.MIN_WAIT_TIME
        last_answer_invites = time.time() - 1

        # forever:
        while True:
            if time.time() > last_answer_invites:
                last_answer_invites += 30

                while self.answer_invitation():
                    pass

            handled = False
            for match_id, opponent, depth in self.games_waiting():
                handled = True
                sleep_time = self.MIN_WAIT_TIME
                self.handle_game(match_id, opponent, depth)

            if not handled:
                # backoff, lets not hammer LG server
                sleep_time = min(sleep_time + 10, self.MAX_WAIT_TIME)
                yield sleep_time

    def loop_forever(self):
        for sleep_time in self.poll_gen():
            print "sleeping for %s seconds" % sleep_time
            time.sleep(sleep_time)


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == "__main__":
    ''' to create a template config file, use -c config_filename.
        otherwise to run, provide the config_filename '''

    setup()

    conf_filename = sys.argv[1]
    if os.path.exists(conf_filename):
        config = at.json_to_attr(open(conf_filename).read())
    else:
        print "Creating config"
        config = template_config()

    # save it - pick up new features
    with open(conf_filename, "w") as f:
        contents = at.attr_to_json(config, pretty=True)
        f.write(contents)

    lg = LittleGolemConnection(config)
    lg.loop_forever()
