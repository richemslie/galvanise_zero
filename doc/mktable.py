import json
import tabulate


def to_player_name(*args):
    if len(args) == 3:
        n, v0, v1 = args
        return "%s_%s_%s" % (n, v0, v1)

    if len(args) == 2:
        n, v0 = args
        return "%s_%s" % (n, v0)

    if len(args) == 1:
        n = args[0]

    return n


def fmt_result(w, l, d):
    return "%s/%s/%s" % (w, l, d)


def format_table(json_table):
    headers = "player opponent black_result white_result".split()

    table = []
    for p, o, w, b in reversed(json_table):
        table.append((to_player_name(*p), to_player_name(*o), fmt_result(*w), fmt_result(*b)))

    print tabulate.tabulate(table, headers, tablefmt="pipe")


if __name__ == "__main__":
    import sys
    json_table = json.loads(open(sys.argv[1]).read())
    format_table(json_table["results"])
