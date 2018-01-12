import time

from ggplib.db import lookup

from ggpzero.defs import confs

from ggpzero.player.puctplayer import PUCTEvaluator, PUCTPlayer
from ggpzero.nn.scheduler import create_scheduler


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()


def next_time():
    return time.time() + 100


class SelfPlay(object):
    # game = "breakthrough"
    # gen = "v5_84"
    # game = "connectFour"
    # gen = "v6_27"
    game = "reversi"
    gen = "v6_65"

    def __init__(self, num_evaluators=1):
        self.num_evaluators = num_evaluators

        self.conf = confs.PUCTPlayerConfig(generation=self.gen,
                                           max_dump_depth=1,
                                           choose="choose_top_visits",
                                           verbose=False,
                                           dirichlet_noise_pct=0.25,
                                           dirichlet_noise_alpha=0.5,
                                           puct_before_expansions=3,
                                           puct_before_root_expansions=3,
                                           puct_constant_before=5.0,
                                           puct_constant_after=0.75,
                                           random_scale=0.7,
                                           temperature=1.0,
                                           depth_temperature_start=8,
                                           depth_temperature_stop=8)

        self.game_info = lookup.by_name(self.game)
        self.scheduler = create_scheduler(self.game, self.gen, batch_size=self.num_evaluators)
        self.evaluators = []
        self.sm = self.game_info.get_sm()

        # we can reuse this basestate throughout test
        self.initial_state = self.sm.get_initial_state()
        self.create_evaluators()

    def create_evaluators(self):
        # create evaluator
        self.evaluators = [PUCTEvaluator(self.conf)
                           for _ in range(self.num_evaluators)]
        for pe in self.evaluators:
            pe.nn = self.scheduler
            pe.init(self.game_info)

    def run(self, fn):
        for pe in self.evaluators:
            self.scheduler.add_runnable(fn, pe)

        self.scheduler.run()

    def go0(self, pe):
        self.game_depths = []
        ITERATIONS = 100
        ITERATIONS_SAMPLE = 800

        pe.reset()

        game_depth = 0
        root = pe.establish_root(self.initial_state, game_depth)
        while not root.is_terminal:
            iterations = ITERATIONS
            if game_depth >= 14 and game_depth <= 15:
                iterations = ITERATIONS_SAMPLE

            choice = pe.on_next_move(iterations, next_time())
            root = pe.fast_apply_move(choice)
            game_depth += 1

        assert root.is_terminal
        self.game_depths.append(game_depth)

    def go1(self, pe):
        while len(self.game_depths) < self.reqd_depths:
            pe.reset()

            game_depth = 0
            root = pe.establish_root(self.initial_state, game_depth)
            while not root.is_terminal:
                if len(self.game_depths) >= self.reqd_depths:
                    break

                choice = pe.on_next_move(self.iterations, next_time())
                root = pe.fast_apply_move(choice)
                game_depth += 1

            if root.is_terminal:
                self.game_depths.append(game_depth)

    def go2(self, pe):
        while len(self.game_depths) < self.reqd_depths:
            pe.reset()

            game_depth = 0
            cur = pe.establish_root(self.initial_state, game_depth)
            game_done = False

            while len(self.game_depths) < self.reqd_depths:
                game_done = cur.is_terminal

                if not game_done and cur.mc_score is not None:
                    if cur.mc_score[cur.lead_role_index] < 0.2:
                        print "resign", game_depth, cur.mc_score
                        game_done = True

                if game_done:
                    break

                if game_depth in (10, 11):
                    choice = pe.on_next_move(self.long_iterations, next_time())
                else:
                    choice = pe.on_next_move(self.iterations, next_time())

                cur = pe.fast_apply_move(choice)
                game_depth += 1

            if game_done:
                self.game_depths.append(game_depth)


def test_0_evaluator():
    lines = []
    for k in 1, 8, 64, 128:
        s = SelfPlay(num_evaluators=k)

        start_time = time.time()

        s.run(s.go0)
        total_time = time.time() - start_time

        s.game_depths.sort()
        lines.append("")
        lines.append("")
        lines.append("depths %s" % str(s.game_depths))
        lines.append("NUMBER OF GAMES %s" % len(s.game_depths))
        lines.append("predictions %d" % s.scheduler.num_predictions)
        m = "python / predict / total ... %.2f / %.2f / %.2f"
        lines.append(m % (s.scheduler.acc_python_time,
                          s.scheduler.acc_predict_time,
                          total_time))

        lines.append("")
        lines.append("average per game %.2f" % (total_time / len(s.game_depths)))

        lines.append("")
        lines.append("_______________________")

        print lines
        print "XXXX more"

    for l in lines:
        print l


def test_1_evaluator():
    s = SelfPlay(num_evaluators=128)
    s.inittest1(iterations=800)

    start_time = time.time()
    s.run(s.go1)

    print s.game_depths
    print "NUMBER OF GAMES", len(s.game_depths)
    print "predictions %d" % s.scheduler.num_predictions
    m = "python / predict / total ... %.2f / %.2f / %.2f"
    print m % (s.scheduler.acc_python_time,
               s.scheduler.acc_predict_time,
               time.time() - start_time)
    import pprint
    pprint.pprint(s.scheduler.predict_sizes.most_common())


def test_2_evaluator():
    s = SelfPlay(num_evaluators=256)
    s.inittest2(reqd_depths=256, long_iterations=42, iterations=8)

    start_time = time.time()
    s.run(s.go2)

    print s.game_depths
    print "NUMBER OF GAMES", len(s.game_depths)
    print "predictions %d" % s.scheduler.num_predictions
    m = "python / predict / total ... %.2f / %.2f / %.2f"
    print m % (s.scheduler.acc_python_time,
               s.scheduler.acc_predict_time,
               time.time() - start_time)
    import pprint
    pprint.pprint(s.scheduler.predict_sizes.most_common())
