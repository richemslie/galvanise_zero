# XXX i call these channels.  But I think the correct term is plane.
# XXX this is all getting a bit hacky.

import numpy as np

from ggplib.symbols import SymbolFactory
from ggplib.util import log


class BasesConfig(object):
    role_count = 2

    # will be updated later
    number_of_non_cord_states = 0

    # hacks here for amazonssuicide_10x10
    control_base_term = None
    extra_term = None

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


def get_bases_config(game_name):
    for clz in AtariGo_7x7, Breakthrough, Reversi, Connect4, Hex, AmazonsSuicide_10x10:
        if clz.game == game_name:
            return clz()


###############################################################################

class BaseInfo(object):
    def __init__(self, gdl_str, symbols):
        self.gdl_str = gdl_str
        self.symbols = symbols

        # drops true ie (true (control black)) -> (control black)
        self.terms = symbols[1]

        # populated in update_base_infos()
        self.channel = None
        self.cord_idx = None


def create_base_infos(config, sm_model):
    symbol_factory = SymbolFactory()
    base_infos = [BaseInfo(s, symbol_factory.symbolize(s)) for s in sm_model.bases]

    all_cords = []
    for x_cord in config.x_cords:
        for y_cord in config.y_cords:
            all_cords.append((x_cord, y_cord))

    # need to match up there terms.  There will be one channel each.  We don't care what
    # the order is, the NN doesn't care either.
    BASE_TERM = 0

    def match_terms(b_info, arg):
        return b_info.terms[config.piece_term] == arg

    for channel_count, arg in enumerate(config.pieces):
        count = 0
        for board_pos, (x_cord, y_cord) in enumerate(all_cords):
            # this is slow.  Will go through all the bases and match up terms.
            for b_info in base_infos:
                if b_info.terms[BASE_TERM] != config.base_term:
                    continue

                if b_info.terms[config.x_term] == x_cord and \
                   b_info.terms[config.y_term] == y_cord:

                    if match_terms(b_info, arg):
                        count += 1
                        b_info.channel = channel_count
                        b_info.cord_idx = board_pos
                        break

        log.info("init_state() found %s states for channel %s" % (count, channel_count))

    # XXX hack in progress for amazons
    channel_count += 1
    if config.extra_term:
        count = 0
        for board_pos, (x_cord, y_cord) in enumerate(all_cords):
            # this is slow.  Will go through all the bases and match up terms.
            for b_info in base_infos:
                if b_info.terms[0] != config.extra_term:
                    continue

                if b_info.terms[config.x_term] == x_cord and \
                   b_info.terms[config.y_term] == y_cord:

                    count += 1
                    b_info.channel = channel_count
                    b_info.cord_idx = board_pos
                    break

    # update the config for non cord states
    config.number_of_non_cord_states = 0
    for b_info in base_infos:
        if b_info.channel is None:
            config.number_of_non_cord_states += 1
        print b_info.gdl_str, b_info.channel, b_info.cord_idx
    log.info("Number of number_of_non_cord_states %d" % config.number_of_non_cord_states)
    return base_infos


def state_to_channels(basestate, lead_role_index, config, base_infos):
    # create a bunch of zero channels
    channel_count = len(config.pieces)
    if config.extra_term:
        channel_count += 1
    channels = [np.zeros(config.channel_size) for _ in range(channel_count)]

    # simply add to channel
    for b_info, base_value in zip(base_infos, basestate):
        # XXX sanity
        assert isinstance(base_value, int) and abs(base_value) <= 1

        if base_value and b_info.channel is not None:
            channels[b_info.channel][b_info.cord_idx] = 1

    # here we add in who's turn it is, by adding a layer for each role and then setting
    # everything to 1.
    # XXX this needs to be control states...

    if config.control_base_term is not None:
        for idx, b_info in enumerate(base_infos):
            if b_info.terms[0] == config.control_base_term:
                if basestate[idx]:
                    channels.append(np.ones(config.channel_size))
                else:
                    channels.append(np.zeros(config.channel_size))
    else:
        for ii in range(config.role_count):
            if lead_role_index == ii:
                channels.append(np.ones(config.channel_size))
            else:
                channels.append(np.zeros(config.channel_size))

    assert len(channels) == config.num_channels

    X_0 = np.array(channels)
    X_0 = np.rollaxis(X_0, -1)
    X_0 = np.reshape(X_0, (config.num_rows, config.num_cols, len(channels)))

    return X_0
