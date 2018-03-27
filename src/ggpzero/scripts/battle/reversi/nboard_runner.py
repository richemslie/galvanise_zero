''' code from mokemokechicken XXX add license '''

import sys
import time

from twisted.protocols import basic
from twisted.internet import protocol

from twisted.internet import reactor

from ggplib.db import lookup
from ggplib.util import log


from ggpzero.defs import confs
from ggpzero.player.puctplayer import PUCTPlayer


import re
from collections import namedtuple

GGF = namedtuple("GGF", "BO MOVES")
BO = namedtuple("BO", "board_type, square_cont, color")  # color: {O, *}  (O is white, * is black)
MOVE = namedtuple("MOVE", "color pos")  # color={B, W} pos: like 'F5'


def parse_ggf(ggf):
    """https://skatgame.net/mburo/ggsa/ggf

    :param ggf:
    :rtype: GGF
    """
    tokens = re.split(r'([a-zA-Z]+\[[^\]]+\])', ggf)
    moves = []
    bo = None
    for token in tokens:
        match = re.search(r'([a-zA-Z]+)\[([^\]]+)\]', token)
        if not match:
            continue
        key, value = re.search(r'([a-zA-Z]+)\[([^\]]+)\]', token).groups()
        key = key.upper()
        if key == "BO":
            bo = BO(*value.split(" "))
        elif key in ("B", "W"):
            moves.append(MOVE(key, value))
    return GGF(bo, moves)


def convert_move_to_gdl_move(move):
    mapping_x_cord = {x0 : x1 for x0, x1 in zip('12345678', '87654321')}
    mapping_y_cord = {x0 : x1 for x0, x1 in zip('abcdefgh', '12345678')}

    mapping_colour = dict(B='black', W='red')
    turn_move = mapping_colour[move.color]
    move_str = move.pos.lower()

    if move_str == "pa":
        return turn_move, 'noop'
    else:
        return turn_move, "(move %s %s)" % (mapping_y_cord[move_str[0]], mapping_x_cord[move_str[1]])


def convert_move_to_gdl_move2(move):
    mapping_x_cord = {x0 : x1 for x0, x1 in zip('12345678', '87654321')}
    mapping_y_cord = {x0 : x1 for x0, x1 in zip('abcdefgh', '12345678')}

    move_str = move.lower()

    if move_str == "pa":
        return 'noop'
    else:
        return "(move %s %s)" % (mapping_y_cord[move_str[0]], mapping_x_cord[move_str[1]])


def convert_gdl_move_to_move(player, gdl_move):
    mapping_x_cord = {x0 : x1 for x0, x1 in zip('87654321', '12345678')}
    mapping_y_cord = {x0 : x1 for x0, x1 in zip('12345678', 'abcdefgh')}

    gdl_move_parts = gdl_move.replace("(", "").replace(")", "").split()

    if gdl_move_parts[0] == "noop":
        return "pa"

    return "%s%s" % (mapping_y_cord[gdl_move_parts[1]], mapping_x_cord[gdl_move_parts[2]])


class NBoardProtocolVersion2(object):
    def __init__(self, engine):
        self.engine = engine
        self.handlers = [
            (re.compile(r'nboard ([0-9]+)'), self.nboard),
            (re.compile(r'set depth ([0-9]+)'), self.set_depth),
            (re.compile(r'set game (.+)'), self.set_game),
            (re.compile(r'move ([^/]+)(/[^/]*)?(/[^/]*)?'), self.move),
            (re.compile(r'hint ([0-9]+)'), self.hint),
            (re.compile(r'go'), self.go),
            (re.compile(r'ping ([0-9]+)'), self.ping),
            (re.compile(r'learn'), self.learn),
            (re.compile(r'analyze'), self.analyze),
        ]

    def handle_message(self, message):
        for regexp, func in self.handlers:
            if self.scan(message, regexp, func):
                return
        log.warning("ignoring message: %s" % message)

    def scan(self, message, regexp, func):
        match = regexp.match(message)
        if match:
            func(*match.groups())
            return True
        return False

    def nboard(self, version):
        if version != "2":
            log.warning("UNKNOWN NBoard Version %s!!!" % version)
        self.engine.send("set myname %s" % self.engine.get_name())
        self.tell_status("waiting")

    def set_depth(self, depth):
        """Set engine midgame search depth.
        Optional: Set midgame depth to {maxDepth}. Endgame depths are at the engine author's discretion.
        :param depth:
        """
        self.engine.set_depth(int(depth))

    def set_game(self, ggf_str):
        """Tell the engine that all further commands relate to the position at the end of the given game, in GGF format.
        Required:The engine must update its stored game state.
        :param ggf_str: see https://skatgame.net/mburo/ggsa/ggf . important info are BO, B+, W+
        """
        ggf = parse_ggf(ggf_str)

        assert int(ggf.BO.board_type) == 8
        for i in range(8):
            start = i * 8
            end = start + 8
            print ggf.BO.square_cont[start:end]

        # (O is white, * is black) -> gdl (O is red, * is black)
        # gdl_color = 'black' if ggf.BO.color == '*' else 'red'

        gdl_moves = [convert_move_to_gdl_move(m) for m in ggf.MOVES]

        # set engine to initial state and apply moves
        self.engine.reset()
        for m in gdl_moves:
            self.engine.apply_move(m[1])

    def move(self, move, evaluation, time_sec):
        """Tell the engine that all further commands relate to the position after the given move.
        The move is 2 characters e.g. "F5". Eval is normally in centi-disks. Time is in seconds.
        Eval and time may be omitted. If eval is omitted it is assumed to be "unknown";
        if time is omitted it is assumed to be 0.
        Required:Update the game state by making the move. No response required.
        """
        gdl_move = convert_move_to_gdl_move2(move)
        self.engine.apply_move(gdl_move)

    def hint(self, n):
        """Tell the engine to give evaluations for the given position. n tells how many moves to evaluate,
        e.g. 2 means give evaluations for the top 2 positions. This is used when the user is analyzing a game.
        With the "hint" command the engine is not CONSTRained by the time remaining in the game.
        Required: The engine sends back an evaluation for at its top move
        Best: The engine sends back an evaluation for approximately the top n moves.
        If the engine searches using iterative deepening it should also send back evaluations during search,
        which makes the GUI feel more responsive to the user.
        Depending on whether the evalation came from book or a search, the engine sends back
        search {pv: PV} {eval:Eval} 0 {depth:Depth} {freeform text}
        or
        book {pv: PV} {eval:Eval} {# games:long} {depth:Depth} {freeform text:string}
        PV: The pv must begin with two characters representing the move considered (e.g. "F5" or "PA") and
        must not contain any whitespace. "F5d6C3" and "F5-D6-C3" are valid PVs but "F5 D6 C3" will
        consider D6 to be the eval.
        Eval: The eval is from the point-of-view of the player to move and is a double.
        At the engine's option it can also be an ordered pair of doubles separated by a comma:
        {draw-to-black value}, {draw-to-white value}.
        Depth: depth is the search depth. It must start with an integer but can end with other characters;
        for instance "100%W" is a valid depth. The depth cannot contain spaces.
        Two depth codes have special meaning to NBoard: "100%W" tells NBoard that the engine has solved
        for a win/loss/draw and the sign of the eval matches the sign of the returned eval.
        "100%" tells NBoard that the engine has done an exact solve.
        The freeform text can be any other information that the engine wants to convey.
        NBoard 1.1 and 2.0 do not display this information but later versions or other GUIs may.
        :param n:
        """
        self.tell_status("thinking hint...")
        self.engine.hint(int(n))
        self.tell_status("waiting")

    def report_hint(self, hint_list):
        pass
        # for hint in reversed(hint_list):  # there is a rule that the last is best?
        #     move = convert_action_to_move(hint.action)
        #     self.engine.send(f"search %s  0 {int(hint.visit)}" % (move,
        #                                                            hint.value,
        #     )

    def go(self):
        """Tell the engine to decide what move it would play.
        This is used when the engine is playing in a game.
        With the "go" command the computer is limited by both the maximum search depth and
        the time remaining in the game.
        Required: The engine responds with "=== {move}" where move is e.g. "F5"
        Best: The engine responds with "=== {move:String}/{eval:float}/{time:float}".
        Eval may be omitted if the move is forced. The engine also sends back thinking output
        as in the "hint" command.
        Important: The engine does not update the board with this move,
        instead it waits for a "move" command from NBoard.
        This is because the user may have modified the board while the engine was thinking.
        Note: To make it easier for the engine author,
        The NBoard gui sets the engine's status to "" when it receives the response.
        The engine can override this behaviour by sending a "status" command immediately after the response.
        """
        self.tell_status("thinking...")

        # return the opposite of engine.apply_move() - just for symmetry
        colour, gdl_move_str = self.engine.go()

        move = convert_gdl_move_to_move(colour, gdl_move_str)
        self.engine.send("=== %s" % move)
        self.tell_status("waiting")

    def ping(self, n):
        """Ensure synchronization when the board position is about to change.
        Required: Stop thinking and respond with "pong n".
        If the engine is analyzing a position it must stop analyzing before sending "pong n"
        otherwise NBoard will think the analysis relates to the current position.
        :param n:
        :return:
        """
        # self.engine.stop_thinkng()  # not implemented
        self.engine.send("pong %s" % n)

    def learn(self):
        """Learn the current game.
        Required: Respond "learned".
        Best: Add the current game to book.
        Note: To make it easier for the engine author,
        The NBoard gui sets the engine's status to "" when it receives the "learned" response.
        The engine can override this behaviour by sending a "status" command immediately after the response.
        """
        self.engine.send("learned")

    def analyze(self):
        """Perform a retrograde analysis of the current game.
        Optional: Perform a retrograde analysis of the current game.
        For each board position occurring in the game,
        the engine sends back a line of the form analysis {movesMade:int} {eval:double}.
        movesMade = 0 corresponds to the start position. Passes count towards movesMade,
        so movesMade can go above 60.
        """
        pass

    def tell_status(self, status):
        self.engine.send("status %s" % (status))


class Match:
    match_id = "nboard_runner_xx1"

    def __init__(self, game_info, basestate):
        self.game_depth = 0
        self.game_info = game_info
        self.basestate = basestate

    def get_current_state(self):
        return self.basestate


class Engine(basic.LineReceiver):
    def __init__(self, addr, puct_player):
        self.addr = addr
        self.puct_player = puct_player

        self.nboard = NBoardProtocolVersion2(self)
        self.sm = None
        self.depth = 1
        self.reset()

    def connectionMade(self):
        log.debug("Connection made from: %s" % self.addr)
        basic.LineReceiver.connectionMade(self)

    def connectionLost(self, reason):
        log.debug("Connection (%s) lost reason: %s" % (self.addr, reason))
        basic.LineReceiver.connectionMade(self)

    def lineReceived(self, line):
        log.verbose("Engine.lineReceived(): %s" % line)
        self.nboard.handle_message(line)

    def send(self, line):
        log.verbose("Engine.send(): %s" % line)
        self.sendLine(line)

    def get_name(self):
        return "%s_%d" % (self.puct_player.conf.name, self.depth)

    def reset(self):
        # get reversi game info and initialise puct player
        game_info = lookup.by_name("reversi")
        if self.sm is None:
            sm = game_info.get_sm()
            self.joint_move = sm.get_joint_move()
            self.basestate = sm.new_base_state()
            self.basestate.assign(sm.get_initial_state())

            self.puct_player.match = Match(game_info, self.basestate)
            self.puct_player.on_meta_gaming(-1)
            self.sm = self.puct_player.sm
            self.game_depth = 0

        else:
            self.basestate.assign(self.sm.get_initial_state())
            self.puct_player.on_meta_gaming(-1)

    def determine_legal_role_index(self):
        if (self.sm.get_legal_state(0).get_count() == 1 and
            self.sm.get_legal_state(0).get_legal(0) == self.puct_player.role0_noop_legal):
            lead_role_index = 1
            other_role_index = 0

        else:
            assert (self.sm.get_legal_state(1).get_count() == 1 and
                    self.sm.get_legal_state(1).get_legal(0) == self.puct_player.role1_noop_legal)
            lead_role_index = 0
            other_role_index = 1

        return lead_role_index, other_role_index

    def apply_move(self, move):
        log.debug("apply_move: %s" % move)

        # current state
        self.sm.update_bases(self.basestate)
        lead_role_index, other_role_index = self.determine_legal_role_index()

        # noop:
        ls = self.sm.get_legal_state(other_role_index)
        assert ls.get_count() == 1
        self.joint_move.set(other_role_index, ls.get_legal(0))

        # find our move
        ls = self.sm.get_legal_state(lead_role_index)
        choices = [ls.get_legal(ii) for ii in range(ls.get_count())]

        matched = False
        for choice in choices:
            choice_move = self.sm.legal_to_move(lead_role_index, choice)
            if choice_move == move:
                self.joint_move.set(lead_role_index, choice)
                matched = True
                break

        assert matched

        self.puct_player.on_apply_move(self.joint_move)

        # move the statemachine forward
        self.sm.update_bases(self.basestate)
        self.sm.next_state(self.joint_move, self.basestate)

        self.game_depth += 1

    def go(self):
        # current state
        self.sm.update_bases(self.basestate)
        lead_role_index, _ = self.determine_legal_role_index()
        self.puct_player.match.our_role_index = lead_role_index

        # the match should be in the right state
        choice = self.puct_player.on_next_move(time.time() + 60)

        colour = "black" if lead_role_index == 0 else "red"
        move_str = self.sm.legal_to_move(lead_role_index, choice)
        return colour, move_str

    def set_depth(self, depth):
        self.depth = depth
        conf = self.puct_player.conf
        conf.playouts_per_iteration = depth * 100
        if conf.playouts_per_iteration == 0:
            conf.playouts_per_iteration = 1
        log.warning("setting playouts to %s" % conf.playouts_per_iteration)


class ServerFactory(protocol.Factory):
    def __init__(self, pp):
        self.puct_player = pp

    def buildProtocol(self, addr):
        return Engine(addr, self.puct_player)


###############################################################################

compete = confs.PUCTPlayerConfig(name="gzero",

                                 choose="choose_temperature",
                                 #choose="choose_top_visits",

                                 puct_constant_before=3.0,
                                 puct_constant_after=0.75,
                                 puct_before_expansions=3,
                                 puct_before_root_expansions=4,

                                 temperature=1.0,
                                 depth_temperature_max=10.0,
                                 depth_temperature_start=1,
                                 depth_temperature_increment=0.5,
                                 depth_temperature_stop=4,
                                 random_scale=1.0,

                                 playouts_per_iteration=100,
                                 playouts_per_iteration_noop=0,

                                 root_expansions_preset_visits=-1,
                                 dirichlet_noise_alpha=0.3,
                                 dirichlet_noise_pct=0.25,

                                 verbose=True,
                                 max_dump_depth=2)


def main(args):
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    port = int(args[0])
    generation = args[1]

    compete.generation = generation
    puct_player = PUCTPlayer(compete)
    puct_player.tiered = False

    factory = ServerFactory(puct_player)
    reactor.listenTCP(port, factory)
    reactor.run()


###############################################################################

if __name__ == "__main__":
    main(sys.argv[1:])
