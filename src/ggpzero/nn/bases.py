import numpy as np

from ggplib.util import log

# only importing for checking type (otherwise will go insane)
from ggplib.db.lookup import GameInfo

from ggpzero.util.state import decode_state
from ggpzero.defs import datadesc, gamedesc


class BaseInfo(object):
    def __init__(self, index, symbols):
        from ggplib.util.symbols import Term
        self.index = index

        # drops true ie (true (control black)) -> (control black)
        self.terms = symbols[1]

        if isinstance(self.terms, Term):
            self.terms = [self.terms]

        self.used = False


def create_base_infos(game_model):
    from ggplib.util.symbols import SymbolFactory
    symbol_factory = SymbolFactory()
    return [BaseInfo(idx, symbol_factory.symbolize(s)) for idx, s in enumerate(game_model.bases)]


class BaseToBoardSpace(object):
    def __init__(self, base_indx, channel_id, x_cord_idx, y_cord_idx):
        # which base it is
        self.base_indx = base_indx

        # index into the board channel (relative to start channel for state (there can be multiple
        # of them with prev_states)
        self.channel_id = channel_id

        # the x/y cords
        self.x_idx = x_cord_idx
        self.y_idx = y_cord_idx

    def __repr__(self):
        return "BaseToBoardSpace(bs=%s, c=%s, x=%s, y=%s)" % (self.base_indx, self.channel_id, self.x_idx, self.y_idx)


class BaseToChannelSpace(object):
    def __init__(self, base_indx, channel_id, value):
        # which base it is
        self.base_indx = base_indx

        # which channel it is (relative to end of states)
        self.channel_id = channel_id

        # the value to set the entire channel (flood fill)
        self.value = value

    def __repr__(self):
        return "BaseToChannelSpace(b=%s, c=%s, v=%s)" % (self.base_indx, self.channel_id, self.value)


class GdlBasesTransformer(object):
    def __init__(self, game_info, generation_descr, game_desc=None, verbose=False):
        assert isinstance(game_info, GameInfo)
        assert isinstance(generation_descr, datadesc.GenerationDescription)
        self.game_info = game_info

        if game_desc is None:
           game_desc = getattr(gamedesc.Games(), self.game)()

        assert isinstance(game_desc, gamedesc.GameDesc)
        self.game_desc = game_desc

        self.verbose = verbose

        # for the number of outputs of the network
        sm_model = self.game_info.model
        self.role_count = len(sm_model.roles)

        assert generation_descr.multiple_policy_heads
        self.policy_dist_count = [len(l) for l in sm_model.actions]
        self.final_score_count = len(sm_model.roles)

        # this is the 'image' data ordering for tensorflow/keras
        self.channel_last = generation_descr.channel_last
        self.num_previous_states = generation_descr.num_previous_states

        assert self.num_previous_states >= 0
        self.init_spaces()

    @property
    def game(self):
        return self.game_info.game

    @property
    def x_cords(self):
        return self.game_desc.x_cords

    @property
    def y_cords(self):
        return self.game_desc.y_cords

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
    def num_bases(self):
        return len(self.game_info.model.bases)

    @property
    def num_channels(self):
        # one for each role to indicate turn, one for each pieces
        total_states = self.num_previous_states + 1
        return self.num_of_controls_channels + self.raw_channels_per_state * total_states

    def init_spaces(self):
        base_infos = create_base_infos(self.game_info.model)

        self.board_space = self.create_board_space(base_infos)
        self.raw_channels_per_state = max(b.channel_id for b in self.board_space) + 1

        self.control_space = self.create_control_space(base_infos)

        self.num_of_controls_channels = len(self.game_desc.control_channels)

        # warn about any unhandled states
        self.num_unhandled_states = 0
        for b_info in base_infos:
            if not b_info.used:
                self.num_unhandled_states += 1

        if self.num_unhandled_states:
            log.warning("Number of unhandled states %d" % self.num_unhandled_states)

        # sort by channel
        self.by_channel = {}
        for b in self.board_space:
            self.by_channel.setdefault(b.channel_id, []).append(b)

        for cs in self.control_space:
            self.by_channel.setdefault(cs.channel_id + self.raw_channels_per_state, []).append(cs)

        if self.verbose:
            for channel_id, all in self.by_channel.items():
                print
                print "channel_id", channel_id
                for x in all:
                    print base_infos[x.base_indx].terms, "->", x

    def create_board_space(self, base_infos):
        board_space = []

        all_cords = []
        for y_cord in self.game_desc.x_cords:
            for x_cord in self.game_desc.y_cords:
                all_cords.append((y_cord, x_cord))

        # first do board space
        channel_mapping = {}

        def find_board_channel(b_info):
            for bc in self.game_desc.board_channels:
                if b_info.terms[0] == bc.base_term:
                    return bc

        for b_info in base_infos:
            assert not b_info.used

            # is this a board channel?
            bc = find_board_channel(b_info)
            if bc is None:
                continue

            matched = []
            for bt in bc.board_terms:
                if b_info.terms[bt.term_idx] in bt.terms:
                    matched.append(b_info.terms[bt.term_idx])

            if len(matched) != len(bc.board_terms):
                continue

            # create a BaseToBoardSpace
            key = tuple([b_info.terms[0]] + matched)

            if key not in channel_mapping:
                channel_mapping[key] = len(channel_mapping)

            channel_id = channel_mapping[key]

            x_cord = b_info.terms[bc.x_term_idx]
            y_cord = b_info.terms[bc.y_term_idx]

            x_idx = self.game_desc.x_cords.index(x_cord)
            y_idx = self.game_desc.y_cords.index(y_cord)

            assert not b_info.used
            board_space.append(BaseToBoardSpace(b_info.index, channel_id, x_idx, y_idx))
            b_info.used = True

        return board_space

    def create_control_space(self, base_infos):
        # now do channel space
        control_space = []
        for channel_id, cc in enumerate(self.game_desc.control_channels):
            look_for = set([tuple(cb.arg_terms) for cb in cc.control_bases])

            for b_info in base_infos:
                the_terms = tuple(b_info.terms)
                if the_terms in look_for:

                    done = False
                    for cb in cc.control_bases:
                        if the_terms == tuple(cb.arg_terms):
                            assert not b_info.used
                            control_space.append(BaseToChannelSpace(b_info.index, channel_id, cb.value))
                            b_info.used = True
                            done = True
                            break

                    assert done
        return control_space

    def state_to_channels(self, state, prev_states=None):
        assert prev_states is None or len(prev_states) <= self.num_previous_states
        if prev_states is None:
            prev_states = []

        # create a bunch of zero channels
        channels = [np.zeros((self.num_cols, self.num_rows))
                    for _ in range(self.num_channels)]

        # add the state to channels
        for b in self.board_space:
            if state[b.base_indx]:
                channels[b.channel_id][b.y_idx][b.x_idx] = 1

        # add any previous states to the channels
        channel_incr = self.raw_channels_per_state
        for ii in range(self.num_previous_states):
            try:
                prev_state = prev_states[ii]

                for b in self.board_space:
                    if prev_state[b.base_indx]:
                        channel_idx = b.channel_id + channel_incr
                        channels[channel_idx][b.y_idx][b.x_idx] = 1

            except IndexError:
                pass

            channel_incr += self.raw_channels_per_state

        # set a control state by setting entire channel to 1
        for c in self.control_space:
            channel_idx = c.channel_id + channel_incr

            if state[c.base_indx]:
                # the value to set the entire channel (flood fill)
                channels[channel_idx] += c.value

        channels = np.array(channels, dtype='float32')
        if self.channel_last:
            orig = channels
            channels = np.rollaxis(channels, -1)
            channels = np.rollaxis(channels, -1)
            assert channels.shape == (orig.shape[1], orig.shape[2], orig.shape[0])

        return channels

    def check_sample(self, sample):
        # XXX this should be ==.  But since our encode/decode can end up padding
        assert len(decode_state(sample.state)) >= self.num_bases
        assert len(sample.final_score) == self.final_score_count

        assert isinstance(sample, datadesc.Sample)
        for policy in sample.policies:
            total = 0.0
            for legal, p in policy:
                assert -0.01 < p < 1.01
                total += p

            assert 0.99 < total < 1.01

        return sample

    def policy_to_array(self, policy, role_index):
        array = np.zeros(self.policy_dist_count[role_index], dtype='float32')
        for idx, prob in policy:
            array[idx] = prob

        return array

    def value_to_array(self, values):
        assert len(values) == self.role_count
        return np.array(values, dtype='float32')

    def get_symmetries_desc(self):
        gds = gamedesc.GameSymmetries()
        if hasattr(gds, self.game):
            return getattr(gds, self.game)()
        return None


class GdlBasesTransformer_Draws(GdlBasesTransformer):
    def value_to_array(self, values):
        assert len(values) == 2
        if abs(values[0] - 0.5) < 0.01:
            assert abs(values[1] - 0.5) < 0.01
            new_values = [0.0, 0.0, 1.0]
        else:
            new_values = [values[0], values[1], 0.0]

        return np.array(new_values, dtype='float32')
