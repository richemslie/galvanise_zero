import sys

from ggplib.play import play_runner
from ggpzero.battle.common import get_player, run


def main():
    args = sys.argv[1:]

    # by defuault use v2
    v2 = True
    if args[0] == "-v1":
        v2 = False
        args = args[1:]

    port = int(args[0])
    gen = args[1]

    opts_v1 = dict(playouts_per_iteration=800)
    opts_v2 = dict(think_time=5.0)

    evaluations = 800
    if len(args) > 2:
        if v2:
            opts_v2.update(think_time=int(args[2]),
                           puct_constant_init=1.25,
                           puct_constant_min=0.75,
                           puct_constant_max=2.5,
                           puct_constant_min_root=1.0,
                           puct_constant_max_root=3.0,
                           minimax_backup_ratio=0.75,
                           minimax_required_visits=150,
                           minimax_threshold_visits=100000,
                            # for transpositons off XXX hack
                           policy_dilution_visits=42,

                           choose="choose_temperature",
                           temperature=1.0,
                           depth_temperature_max=5.0,
                           depth_temperature_start=2,
                           depth_temperature_increment=0.5,
                           depth_temperature_stop=100,
                           random_scale=0.75)
        else:
            opts_v1.update(playouts_per_iteration=int(args[2]))

    move_time = 60.0
    if v2:
        player = get_player("p2", move_time, gen, **opts_v2)
    else:
        player = get_player("p1", move_time, gen, **opts_v1)

    print player
    play_runner(player, port)


###############################################################################

if __name__ == "__main__":
    run(main)
