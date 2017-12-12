
'''
        if self.extra_term:
            channel_count += 1

leftover
    # hacks here for amazonssuicide_10x10
    # control_base_term = None
    # extra_term = None

        # # XXX hack in progress for amazons
        # channel_count += 1
        # if self.extra_term:
        #     count = 0
        #     for board_pos, (x_cord, y_cord) in enumerate(all_cords):
        #         # this is slow.  Will go through all the bases and match up terms.
        #         for b_info in self.base_infos:
        #             if b_info.terms[0] != self.extra_term:
        #                 continue

        #             if b_info.terms[self.x_term] == x_cord and \
        #                b_info.terms[self.y_term] == y_cord:

        #                 count += 1
        #                 b_info.channel = channel_count
        #                 b_info.cord_idx = board_pos
        #                 break

        # # here we add in who's turn it is, by adding a layer for each role and then setting
        # # everything to 1.
        # # XXX this needs to be control states...

        # # XXX this was just for amazones... WIP/HACK
        # if self.control_base_term is not None:
        #     for idx, b_info in enumerate(self.base_infos):
        #         if b_info.terms[0] == self.control_base_term:
        #             if state[idx]:
        #                 channels.append(np.ones(self.channel_size))
        #             else:
        #                 channels.append(np.zeros(self.channel_size))
        # else:


class AmazonsSuicide_10x10(BasesConfig):
    game = "amazonsSuicide_10x10"
    x_cords = "1 2 3 4 5 6 7 8 9 10".split()
    y_cords = "1 2 3 4 5 6 7 8 9 10".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['white', 'black', 'arrow']

    extra_term = "justMoved"
    x_term = 1
    y_term = 2

    control_base_term = 'turn'

    @property
    def num_channels(self):
        # one for each role to indicate turn, one for each pieces
        return 4 + len(self.pieces) + 1

'''

from collections import Counter

import numpy as np

from ggplib.symbols import SymbolFactory
from ggplib.util import log


class BaseInfo(object):
    def __init__(self, gdl_str, symbols):
        self.gdl_str = gdl_str
        self.symbols = symbols

        # drops true ie (true (control black)) -> (control black)
        self.terms = symbols[1]

        # populated in create_base_infos()
        self.channel = None
        self.x_idx = None
        self.y_idx = None


class BasesConfig(object):
    game = None

    role_count = 2

    # will be updated later
    number_of_non_cord_states = 0

    # set in create_base_infos
    base_infos = None

    # defined subclass
    base_term = pieces = piece_term = x_cords = y_cords = x_term = y_term = None

    def __init__(self, sm_model, generation):
        self.sm_model = sm_model
        self.generation = generation
        self.create_base_infos()

        # for the number of outputs of the network
        self.policy_dist_count = sum(len(l) for l in sm_model.actions)
        self.final_score_count = len(sm_model.roles)

    def create_network(self, **kwds):
        from ggplearn import net
        return net.NeuralNetwork(self, net.get_network_model(self, **kwds))

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
    def num_channels(self):
        # one for each role to indicate turn, one for each pieces
        return self.role_count + len(self.pieces)

    def create_base_infos(self):
        symbol_factory = SymbolFactory()
        self.base_infos = [BaseInfo(s, symbol_factory.symbolize(s)) for s in self.sm_model.bases]

        all_cords = []
        for y_cord in self.x_cords:
            for x_cord in self.y_cords:
                all_cords.append((y_cord, x_cord))

        count = Counter()
        for b_info in self.base_infos:
            if b_info.terms[0] != self.base_term:
                continue

            piece = b_info.terms[self.piece_term]
            x_cord = b_info.terms[self.x_term]
            y_cord = b_info.terms[self.y_term]

            b_info.channel = self.pieces.index(piece)
            b_info.x_idx = self.x_cords.index(x_cord)
            b_info.y_idx = self.y_cords.index(y_cord)

        for i, piece in enumerate(self.pieces):
            log.info("found %s states for channel %s" % (count[i], piece))

        # update the  non cord states
        self.number_of_non_cord_states = 0
        for b_info in self.base_infos:
            if b_info.channel is None:
                self.number_of_non_cord_states += 1
        log.info("Number of number_of_non_cord_states %d" % self.number_of_non_cord_states)

    def state_to_channels(self, state, lead_role_index, channel_last=True):
        assert 0 <= lead_role_index <= self.role_count
        # create a bunch of zero channels
        channels = [np.zeros((self.num_cols, self.num_rows))
                    for _ in range(self.num_channels)]

        # simply add to channel
        for b_info, base_value in zip(self.base_infos, state):
            assert isinstance(base_value, int)

            if base_value and b_info.channel is not None:
                channels[b_info.channel][b_info.y_idx][b_info.x_idx] = 1

        # set who's turn it is by setting entire channel to 1
        turn_idx = lead_role_index + len(self.pieces)
        channels[turn_idx] += 1

        assert len(channels) == self.num_channels

        channels = np.array(channels)
        if channel_last:
            orig = channels
            channels = np.rollaxis(channels, -1)
            channels = np.rollaxis(channels, -1)
            assert channels.shape == (orig.shape[1], orig.shape[2], orig.shape[0])
        return channels

    def get_non_cord_input(self, state):
        return [v for v, base_info in zip(state, self.base_infos)
                if base_info.channel is None]


###############################################################################

class AtariGo_7x7(BasesConfig):
    game = "atariGo_7x7"
    x_cords = "1 2 3 4 5 6 7".split()
    y_cords = "1 2 3 4 5 6 7".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['black', 'white']


class Breakthrough(BasesConfig):
    game = "breakthrough"
    x_cords = "1 2 3 4 5 6 7 8".split()
    y_cords = "1 2 3 4 5 6 7 8".split()

    base_term = "cellHolds"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['white', 'black']


class Reversi(BasesConfig):
    game = "reversi"
    x_cords = "1 2 3 4 5 6 7 8".split()
    y_cords = "1 2 3 4 5 6 7 8".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['black', 'red']


class Connect4(BasesConfig):
    game = "connectFour"
    x_cords = "1 2 3 4 5 6 7 8".split()
    y_cords = "1 2 3 4 5 6".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['red', 'black']


class Hex(BasesConfig):
    game = "hex"
    x_cords = "a b c d e f g h i".split()
    y_cords = "1 2 3 4 5 6 7 8 9".split()

    base_term = "cell"
    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = ['red', 'blue']


###############################################################################

def get_bases_config(game, sm_model, generation="latest"):
    classes = [v for _, v in globals().items() if isinstance(v, type) and issubclass(v, BasesConfig)]
    for clz in classes:
        if clz.game == game:
            return clz(sm_model, generation)
