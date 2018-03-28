
class BaseToBoardSpace90(BaseToBoardSpace):
    def reflect_cols(self, num_cols):
        return num_cols - self.x_idx - 1, self.y_idx

    def reflect_rows(self, num_rows):
        return self.x_idx, num_rows - self.y_cords - 1

    def rotate_90(self, num_cols):
        x, y = self.reflect_cols(num_cols)
        return y, x


class GdlBasesTransformer4(object):

    @property
    def num_rows(self):
        n = len(self.game_desc.x_cords)
        return n // 2 + 1

    @property
    def num_cols(self):
        n = len(self.game_desc.x_cords)
        return n // 2 + 1

    @property
    def num_channels(self):
        total_states = self.num_previous_states + 1
        raw = self.raw_channels_per_state
        return self.num_of_controls_channels + raw * 4 * total_states

    def create_board_space(self, base_infos):
        board_space = GdlBasesTransformer.create_board_space(self, base_infos)

        self.raw_channels_per_state = max(b.channel_id for b in self.board_space) + 1

        # to rotate 90 degrees, reverse the row, then transpose
        # REVERSE x_cord then swap x_cord<->y_cord

        for b in board_space:
            new_

        channel_id_incr = 0
        for i in range(4):
            for y in range(self.num_cols):
                for x in range(self.num_rows):
                    # only add in any range
                    pass

            channel_id_incr += 1
        return board_space
