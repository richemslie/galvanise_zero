class Config:
    game = None

    nb_roles = 2

    x_term = 1
    y_term = 2
    piece_term = 3

    pieces = []

    def __init__(self):
        self.first_time_map_state = True

    def get_model(self):
        from ggplib.db import lookup
        lookup.xxx

    @property
    def nb_rows(self):
        return len(self.x_cords)

    @property
    def nb_cols(self):
        return len(self.y_cords)

    @property
    def nb_input_layers(self):
        return len(self.search_for_terms) + self.nb_roles

    @property
    def search_for_terms(self):
        return [(p,) for p in self.pieces]

    def match_terms(self, b_info, args):
        return b_info.terms[self.piece_term] == args[0]

    def init_state(self, bs_info):
        all_cords = []
        for x_cord in self.x_cords:
            for y_cord in self.y_cords:
                all_cords.append((x_cord, y_cord))

        for b_info in bs_info.bases:
            b_info.channel = None

        # need to match up there terms.  There will be one channel each.  We don't care what
        # the order is, the NN doesn't care either.
        for (channel_count, args) in enumerate(self.search_for_terms):
            count = 0
            for board_pos, (x_cord, y_cord) in enumerate(all_cords):
                # this is slow.  Will go through all the bases and match up terms.
                for b_info in bs_info.bases:
                    if b_info.terms[BASE_TERM] != self.base_term:
                        continue

                    if (b_info.terms[self.x_term] == x_cord and
                        b_info.terms[self.y_term] == y_cord):

                        if self.match_terms(b_info, args):
                            count += 1
                            b_info.channel = channel_count
                            b_info.cord_idx = board_pos
                            break

            print "init_state() found %s states for %s" % (count, args)

    def map_state_with_roles(self, state, bs_info, role_index):
        if self.first_time_map_state:
            self.init_state(bs_info)
            self.first_time_map_state = False

        # create a bunch of zero channels.  Perhaps we should use numpy here? XXX
        rr = range(self.nb_rows * self.nb_cols)
        channels = [[0 for _ in rr] for ii in range(len(self.search_for_terms))]

        for b_info, _ in bs_info.smart_iter(state):
            if b_info.channel is not None:
                channels[b_info.channel][b_info.cord_idx] = 1

        # here we add in who's turn it is, by adding a layer for each role and then setting
        # everything to 1.  This is crude, but seems to be as effective as having two score
        # networks.
        for ii in range(self.nb_roles):
            if role_index == ii:
                channels.append([1 for ii in range(self.nb_rows * self.nb_cols)])
            else:
                channels.append([0 for ii in range(self.nb_rows * self.nb_cols)])
        return channels

class Breakthrough(Config):
    game = "breakthrough"
    x_cords = "1 2 3 4 5 6 7 8".split()
    y_cords = "1 2 3 4 5 6 7 8".split()
    base_term = "cellHolds"
    pieces = ['white', 'black']


    net.compile(loss="mean_squared_error", optimizer='rmsprop')
