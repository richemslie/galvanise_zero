'''
The idea is to automate much of this.
'''

from collections import Counter

import numpy as np

from ggplib.util import log
from ggplib.util.symbols import SymbolFactory, Term

# only importing for checking type (otherwise will go insane)
from ggplib.db.lookup import GameInfo


class BaseInfo(object):
    def __init__(self, index, gdl_str, symbols):
        self.index = index
        self.gdl_str = gdl_str
        self.symbols = symbols

        # drops true ie (true (control black)) -> (control black)
        self.terms = symbols[1]

        if isinstance(self.terms, Term):
            self.terms = [self.terms]

        # populated in create_base_infos()
        self.channel = None

        # if cord_state, will use these cords
        self.cord_state = False
        self.x_idx = None
        self.y_idx = None

        # if control_state. will set set channel to value
        self.control_state = None
        self.control_state_value = None

    def terms_to_piece(self, terms_indices):
        if isinstance(terms_indices, tuple):
            return tuple([self.terms[idx] for idx in terms_indices])
        else:
            return self.terms[terms_indices]


class GdlBasesTransformer(object):
    game = None

    role_count = 2

    # the following are defined subclass
    base_term = pieces = piece_term = x_term = y_term = None
    x_cords = y_cords = []

    control_base_term = None
    control_base_terms = None

    # these are set in create_base_infos
    base_infos = None
    num_unhandled_states = 0
    ignore_terms = []

    def __init__(self, game_info, channel_last=False):
        assert isinstance(game_info, GameInfo)
        self.game_info = game_info

        # for the number of outputs of the network
        sm_model = self.game_info.model
        self.policy_dist_count = sum(len(l) for l in sm_model.actions)
        self.final_score_count = len(sm_model.roles)

        # used for when we have a single policy_distribution being shared
        assert self.role_count == 2
        self.policy_1_index_start = len(sm_model.actions[0])

        # this is the 'image' data ordering for tensorflow/keras
        self.channel_last = channel_last

        self.num_of_base_controls = 0
        self.create_base_infos()

    @property
    def num_rows(self):
        return len(self.x_cords)

    @property
    def num_cols(self):
        return len(self.y_cords)

    @property
    def channel_size(self):
        return self.num_cols * self.num_rows

    @property
    def raw_channels_per_state(self):
        # one for each role to indicate turn, one for each pieces
        return len(self.pieces)

    @property
    def num_channels(self):
        # one for each role to indicate turn, one for each pieces
        return self.num_of_base_controls + len(self.pieces)

    def create_base_infos(self):
        # ############ XXX
        # base_infos (this part anyway) should probably be done in the StateMachineModel
        symbol_factory = SymbolFactory()
        sm_model = self.game_info.model
        self.base_infos = [BaseInfo(idx, s, symbol_factory.symbolize(s)) for idx, s in enumerate(sm_model.bases)]
        # ########### XXX

        all_cords = []
        for y_cord in self.x_cords:
            for x_cord in self.y_cords:
                all_cords.append((y_cord, x_cord))

        count = Counter()
        for b_info in self.base_infos:
            if b_info.terms[0] != self.base_term:
                continue

            piece = b_info.terms_to_piece(self.piece_term)

            if piece not in self.pieces:
                continue

            b_info.channel = self.pieces.index(piece)

            x_cord = b_info.terms[self.x_term]
            y_cord = b_info.terms[self.y_term]
            b_info.x_idx = self.x_cords.index(x_cord)
            b_info.y_idx = self.y_cords.index(y_cord)

            # for debug
            count[b_info.channel] += 1

        for i, piece in enumerate(self.pieces):
            log.info("found %s states for channel %s" % (count[i], piece))

        if self.control_base_terms is None:
            self.control_base_terms = []

        if self.control_base_term is not None:
            self.control_base_terms.append(self.control_base_term)

        assert self.control_base_terms

        self.num_of_base_controls = 0
        self.control_states = []

        for b in self.base_infos:
            if b.terms[0] in self.control_base_terms:
                self.num_of_base_controls += 1
                self.control_states.append(b.index)
                b.control_state = True

        log.info("Number of control states %s" % self.num_of_base_controls)

        # warn about any unhandled states
        self.num_unhandled_states = 0
        for b_info in self.base_infos:
            if b_info.channel is None and b_info.control_state is None:
                self.num_unhandled_states += 1

        self.channel_base_infos = [bi for bi in self.base_infos if bi.channel is not None]

        if self.num_unhandled_states:
            log.warning("Number of unhandled states %d" % self.num_unhandled_states)

    def state_to_channels(self, state, prev_states=None):
        # prev_states -> list of states
        assert prev_states is None, "unhandled for now"

        # create a bunch of zero channels
        channels = [np.zeros((self.num_cols, self.num_rows))
                    for _ in range(self.num_channels)]

        # simply add to channel
        for b_info in self.channel_base_infos:
            if state[b_info.index]:
                channels[b_info.channel][b_info.y_idx][b_info.x_idx] = 1

        # set a control state by setting entire channel to 1
        channel_idx = len(self.pieces)
        for idx in self.control_states:
            if state[idx]:
                channels[channel_idx] += 1
            channel_idx += 1

        assert len(channels) == self.num_channels

        channels = np.array(channels, dtype='float32')
        if self.channel_last:
            orig = channels
            channels = np.rollaxis(channels, -1)
            channels = np.rollaxis(channels, -1)
            assert channels.shape == (orig.shape[1], orig.shape[2], orig.shape[0])

        return channels

    def policy_to_array(self, policy, lead_role_index):
        index_start = 0 if lead_role_index == 0 else self.policy_1_index_start
        policy_outputs = np.zeros(self.policy_dist_count, dtype='float32')

        for idx, prob in policy:
            policy_outputs[index_start + idx] = prob

        return policy_outputs

    def check_sample(self, sample):
        assert len(sample.state) == len(self.base_infos)
        assert len(sample.final_score) == self.final_score_count

        for legal, p in sample.policy:
            assert 0 <= legal < self.policy_dist_count
            assert -0.01 < p < 1.01

        assert 0 <= sample.lead_role_index <= self.role_count

    def sample_to_nn(self, sample, inputs, policies, finals):
        # transform samples -> numpy arrays as inputs/outputs to nn

        # input - planes
        inputs.append(self.state_to_channels(sample.state))

        # output - policy
        policies.append(self.policy_to_array(sample.policy,
                                             sample.lead_role_index))

        # output - best/final scores
        finals.append(np.array(sample.final_score, dtype='float32'))


###############################################################################

class AtariGo_7x7(GdlBasesTransformer):
    game = 'atariGo_7x7'
    x_cords = '1 2 3 4 5 6 7'.split()
    y_cords = '1 2 3 4 5 6 7'.split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['black', 'white']
    ignore_terms = ['group']
    control_base_term = 'control'


class Breakthrough(GdlBasesTransformer):
    game = "breakthrough"
    x_cords = "1 2 3 4 5 6 7 8".split()
    y_cords = "1 2 3 4 5 6 7 8".split()

    base_term = "cellHolds"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['white', 'black']
    control_base_term = 'control'


class Reversi(GdlBasesTransformer):
    game = "reversi"
    x_cords = "1 2 3 4 5 6 7 8".split()
    y_cords = "1 2 3 4 5 6 7 8".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['black', 'red']
    control_base_term = 'control'


class Connect4(GdlBasesTransformer):
    game = "connectFour"
    x_cords = "1 2 3 4 5 6 7 8".split()
    y_cords = "1 2 3 4 5 6".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['red', 'black']
    control_base_term = 'control'


class Hex(GdlBasesTransformer):
    game = "hex"
    x_cords = "a b c d e f g h i".split()
    y_cords = "1 2 3 4 5 6 7 8 9".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['red', 'blue']
    control_base_term = 'control'


class BreakthroughSmall(GdlBasesTransformer):
    game = "breakthroughSmall"
    x_cords = "1 2 3 4 5 6".split()
    y_cords = "1 2 3 4 5 6".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['white', 'black']
    control_base_term = 'control'


class CitTacEot(GdlBasesTransformer):
    game = "cittaceot"
    x_cords = "1 2 3 4 5".split()
    y_cords = "1 2 3 4 5".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['x', 'o']
    control_base_term = 'control'


class Checkers(GdlBasesTransformer):
    game = "checkers"
    x_cords = "a b c d e f g h".split()
    y_cords = "1 2 3 4 5 6 7 8".split()
    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['bp', 'bk', 'wk', 'wp']
    control_base_term = 'control'


class EscortLatch(GdlBasesTransformer):
    game = "escortLatch"
    x_cords = "a b c d e f g h".split()
    y_cords = "1 2 3 4 5 6 7 8".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3
    pieces = ['wp', 'wk', 'bp', 'bk']

    control_base_terms = ["blackKingCaptured", "whiteKingCaptured", "control"]


class SpeedChess(GdlBasesTransformer):
    game = "speedChess"

    x_cords = "a b c d e f g h".split()
    y_cords = "1 2 3 4 5 6 7 8".split()

    base_term = "cell"
    x_term = 1
    y_term = 2

    piece_term = 3, 4
    pieces = [(p0, p1) for p1 in ['king', 'rook', 'pawn', 'knight', 'queen', 'bishop']
              for p0 in ['white', 'black']]

    control_base_terms = ["kingHasMoved", "hRookHasMoved",
                          "aRookHasMoved", "aRookHasMoved", "control"]


###############################################################################

def init():
    from ggpzero.nn.manager import get_manager
    for clz in (AtariGo_7x7, BreakthroughSmall, Breakthrough, Reversi, Connect4,
                Hex, CitTacEot, Checkers, EscortLatch, SpeedChess):
        get_manager().register_transformer(clz)
