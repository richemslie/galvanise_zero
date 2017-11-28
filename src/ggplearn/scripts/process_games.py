import os
import json

games = []
class Game:
    dupes = 0
    all_states = set()

    def __init__(self, d):
        depth = 0
        while True:
            key = "depth_%d" % depth
            if key not in d:
                break
            info = d[key]
            state = tuple(info['state'])
            assert len(state) == 130
            if depth > 2:
                if state in self.all_states:
                    Game.dupes += 1
                else:
                    Game.all_states.add(state)
            depth += 1


def get_all(path, game_name):
    files = os.listdir(path)
    for f in files:
        print "processing", f
        if f.endswith(".json"):
            buf = open(os.path.join(path, f)).read()
            data = json.loads(buf)
            for g in data:
                if g["game"] == game_name:
                    games.append(g)
                else:
                    print f, "NOT GAME", g["game"]
                    break
    return games


def stat(path, game_name="reversi"):
    all = get_all(path, game_name)

    games = []
    for g in all:
        games.append(Game(g))

    assert games

    print "#games", len(games)
    print "#dupe states", Game.dupes
    print "#unique states", len(Game.all_states)
    print "% of dupe states", Game.dupes / float(Game.dupes + len(Game.all_states))

###############################################################################

class TrainingData:
    def __init__(self, config, data):
        self.config = config
        self.model = config.get_model()
        self.data = data

    def gen(self, filter_by_role_index):
        for row in self.data:
            # numerics please:
            role_index = int(row[0])

            # filtering?
            if not self.do_scores:
                if role_index != filter_by_role_index:
                    continue

            game_depth = int(row[1])
            if game_depth > 100:
                continue

            scores = [float(row[3]), float(row[4])]

            # XXX THESE HACKS WERE HERE FOR POLICY NETWORK...
            #if scores[0] > 0.499 and scores[1] < 0.499 and game_depth > 20:
            #    print "Playing randomly since game is already a draw"
            #    continue

            #if scores[role_index] < 0.01:
            #    print "Skip low score move - would be playing randomly"
            #    continue

            state = [1 if x == "1" else 0 for x in row[5:]]
            X_0 = self.config.map_state_with_roles(state, self.analyser.base_state_info, role_index)

            if self.do_scores:
                y_0 = scores

            yield role_index, X_0, y_0

if __name__ == "__main__":
    import sys
    try:
        stat(sys.argv[1], sys.argv[2])
    except IndexError as exc:
        print exc
        stat(sys.argv[1])
