'''
The idea is to automate much of this.
'''

from collections import Counter

import numpy as np

from ggplib.util import log
from ggplib.util.symbols import SymbolFactory, Term

# only importing for checking type (otherwise will go insane)
from ggplib.db.lookup import GameInfo

from ggpzero.defs import datadesc


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

    # XXX horrible way to say we have multiple_policy_heads
    policy_1_index_start = None

    def __init__(self, game_info, generation_descr):
        assert isinstance(game_info, GameInfo)
        assert isinstance(generation_descr, datadesc.GenerationDescription)
        self.game_info = game_info

        # for the number of outputs of the network
        sm_model = self.game_info.model
        if generation_descr.multiple_policy_heads:
            self.policy_dist_count = [len(l) for l in sm_model.actions]
        else:
            assert self.role_count == 2
            self.policy_dist_count = [sum(len(l) for l in sm_model.actions)]

            # used for when we have a single policy_distribution being shared
            self.policy_1_index_start = len(sm_model.actions[0])

        self.final_score_count = len(sm_model.roles)

        # this is the 'image' data ordering for tensorflow/keras
        self.channel_last = generation_descr.channel_last

        self.num_previous_states = generation_descr.num_previous_states
        assert self.num_previous_states >= 0

        self.num_of_base_controls = 0

        def get_noop_idx(actions):
            for idx, a in enumerate(actions):
                if "noop" in a:
                    return idx
            assert False, "did not find noop"

        self.noop_legals = map(get_noop_idx, game_info.model.actions)

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
        total_states = self.num_previous_states + 1
        return self.num_of_base_controls + self.raw_channels_per_state * total_states

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

    def null_state(self):
        return tuple(0 for _ in range(len(self.base_infos)))

    def identifier(self, state, prev_states=None):
        if self.num_previous_states > 0:
            if prev_states is None:
                prev_states = [self.null_state() for _ in range(self.num_previous_states)]
            else:
                assert len(prev_states) <= self.num_previous_states
                while len(prev_states) < self.num_previous_states:
                    prev_states.append(self.null_state())
        else:
            assert not prev_states
            prev_states = []

        identifier = state[:]
        for l in prev_states:
            identifier += l

        assert len(identifier) == len(self.base_infos) * (1 + self.num_previous_states)
        return tuple(identifier)

    def check_sample(self, sample):
        assert len(sample.state) == len(self.base_infos)
        assert len(sample.final_score) == self.final_score_count

        total = 0.0
        for legal, p in sample.policy:
            assert -0.01 < p < 1.01
            total += p

        assert 0.9999 < total < 1.001

        if isinstance(sample, datadesc.SampleOld):
            assert 0 <= sample.lead_role_index <= self.role_count

        else:
            assert isinstance(sample, datadesc.Sample)

            # ZZZ XXX deprecate single policy heads
            # assert len(sample.policies) == len(self.policy_dist_count)

            for idx, policy in enumerate(sample.policies):
                total = 0.0
                for legal, p in policy:

                    # ZZZ XXX deprecate single policy heads
                    if self.policy_1_index_start is None:
                        assert 0 <= legal < self.policy_dist_count[idx]

                    assert -0.01 < p < 1.01
                    total += p

                assert 0.9999 < total < 1.00001

        return sample

    def state_to_channels(self, state, prev_states=None):
        assert prev_states is None or len(prev_states) <= self.num_previous_states
        if prev_states is None:
            prev_states = []

        # create a bunch of zero channels
        channels = [np.zeros((self.num_cols, self.num_rows))
                    for _ in range(self.num_channels)]

        # add the state to channels
        for b_info in self.channel_base_infos:
            if state[b_info.index]:
                channels[b_info.channel][b_info.y_idx][b_info.x_idx] = 1

        # add any previous states to the channels
        channel_incr = self.raw_channels_per_state
        for ii in range(self.num_previous_states):
            try:
                prev_state = prev_states[ii]

                for b_info in self.channel_base_infos:
                    if prev_state[b_info.index]:
                        channel_idx = b_info.channel + channel_incr
                        channels[channel_idx][b_info.y_idx][b_info.x_idx] = 1

            except IndexError:
                pass

            channel_incr += self.raw_channels_per_state

        # set a control state by setting entire channel to 1
        channel_idx = channel_incr
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
        # ZZZ XXX deprecate single policy heads
        if self.policy_1_index_start is not None:
            index_start = 0 if lead_role_index == 0 else self.policy_1_index_start
            array = np.zeros(self.policy_dist_count[0], dtype='float32')

            for idx, prob in policy:
                array[index_start + idx] = prob

        else:
            array = np.zeros(self.policy_dist_count[lead_role_index], dtype='float32')
            for idx, prob in policy:
                array[idx] = prob

        return array

    def sample_to_nn(self, sample, inputs, outputs):
        # transform samples -> numpy arrays as inputs/outputs to nn

        # input - planes
        inputs.append(self.state_to_channels(sample.state, sample.prev_states))

        output = []
        # output - policies
        if self.policy_1_index_start is not None:
            if isinstance(sample, datadesc.SampleOld):
                output.append(self.policy_to_array(sample.policy,
                                                   sample.lead_role_index))
            else:
                assert False, "TODO"

        else:
            if isinstance(sample, datadesc.SampleOld):
                assert self.role_count == 2
                assert len(self.policy_dist_count) == 2
                # ZZZ XXX deprecate single policy heads
                # XXX HACKSHACKSHACKSHACKS

                if sample.lead_role_index == 0:
                    array = self.policy_to_array(sample.policy, 0)
                else:
                    array = self.policy_to_array(self.noop_policy(0), 0)

                output.append(array)

                if sample.lead_role_index == 1:
                    array = self.policy_to_array(sample.policy, 1)
                else:
                    array = self.policy_to_array(self.noop_policy(1), 1)

                output.append(array)

            else:
                assert self.role_count == 2
                assert len(self.policy_dist_count) == 2
                for i in range(self.role_count):
                    array = self.policy_to_array(sample.policies[i], i)
                    output.append(array)

        # output - best/final scores
        output.append(np.array(sample.final_score, dtype='float32'))
        outputs.append(output)

    def noop_policy(self, ri):
        assert self.policy_1_index_start is None
        return [(self.noop_legals[ri], 1.0)]


###############################################################################

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


class Tron(GdlBasesTransformer):
    TODOXXX = False
    game = "tron_10x10"
    x_cords = "1 2 3 4 5 6 7 8 9 10".split()
    y_cords = "1 2 3 4 5 6 7 8 9 10".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3
    pieces = ['v']

    control_base_terms = []


class AtariGo_7x7(GdlBasesTransformer):
    TODOXXX = False
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
