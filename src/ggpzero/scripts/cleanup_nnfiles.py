''' go through all data for all games, and ask to remove spurious data files.  We want to keep
every 7th iteration for evaluation. '''

import os
from pathlib import Path


def go(game_path):
    game_name = game_path.name

    models_path = weights_path = None
    for p in game_path.iterdir():
        if p.name == 'weights' and p.is_dir():
            weights_path = p
        elif p.name == 'models' and p.is_dir():
            models_path = p

    if models_path and weights_path:
        print game_path, "is valid game"

    # go through each of the models_path, weights_path
    marked_for_deletion = []
    valid_generations_by_prefix = {}
    for root in (models_path, weights_path):
        for p in root.iterdir():
            if game_name not in p.name:
                print "**NOT** a nn file", p
                continue

            if not p.name.startswith(game_name):
                print "**NOT** a valid nn filename", p
                continue

            name = p.name.replace(game_name + "_", "")
            parts = name.split(".")

            try:
                generation = parts[0]

                gen_split = generation.split("_")
                if gen_split[-1] == "prev":
                    marked_for_deletion.append(p)
                    continue

                step = int(gen_split[-1])
                generation_prefix = "_".join(generation.split("_")[:-1])

                valid_generations_by_prefix.setdefault(generation_prefix, []).append((p, step))

            except Exception as exc:
                print exc
                print "**NOT** a valid nn filename", p
                continue

    print "_____________"

    for gp, gens in valid_generations_by_prefix.items():
        max_step = max(s for _, s in gens)
        print "FOUND:", gp, "max_step", max_step

        keep_step_gt = (max_step / 5) * 5
        for p, s in gens:
            if s < keep_step_gt and s % 5 != 0:
                marked_for_deletion.append(p)
            else:
                print "WILL KEEP", p

    if marked_for_deletion:
        print ""
        print "game:", game_name
        print "marked_for_deletion", [m.name for m in marked_for_deletion]
        print ""
        print ""
        print "delete?"
        raw = raw_input()
        if raw == "Y":
            for p in marked_for_deletion:
                print 'bye', p
                p.unlink()

    print "======================"
    print


def main():
    data_path = Path(os.path.join(os.environ["GGPZERO_PATH"], "data"))
    game_paths = []
    for p in data_path.iterdir():
        if p.is_dir() and p.name not in "confs tournament":
            game_paths.append(p)

    # check we have models and weights in there
    for game_path in game_paths:
        go(game_path)


if __name__ == "__main__":
    main()
