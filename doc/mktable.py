import json
import tabulate


def to_player_name(n, v0, v1):
    if v1 != -1:
        return "%s_%s_%s" % (n, v0, v1)

    if v0 != -1:
        return "%s_%s" % (n, v0)

    return n


def fmt_result(w, l, d):
    return "%s/%s/%s" % (w, l, d)


def format_table(json_table):
    headers = "player opponent black_result white_result".split()

    table = []
    for p, o, w, b in reversed(json_table):
        table.append((to_player_name(*p), to_player_name(*o), fmt_result(*w), fmt_result(*b)))

    print tabulate.tabulate(table, headers, tablefmt="pipe")


json_table = json.loads("""
{ "results" : [
              [ ["gzero", 5, 1], ["random", -1, -1], [3, 2, 0], [3, 2, 0] ],

              [ ["gzero", 10, 1], ["random", -1, -1], [5, 0, 0], [3, 2, 0] ],
              [ ["gzero", 10, 1], ["pymcs", -1, -1], [0, 2, 0], [0, 2, 0] ],

              [ ["gzero", 15, 2], ["pymcs", -1, -1], [2, 3, 0], [1, 2, 1] ],

              [ ["gzero", 20, 1], ["random", -1, -1], [5, 0, 0], [5, 0, 0] ],
              [ ["gzero", 20, 2], ["pymcs", -1, -1], [3, 1, 1], [2, 2, 0] ],
              [ ["gzero", 20, 4], ["simplecmts", -1, -1], [2, 0, 0], [1, 1, 0] ],

              [ ["gzero", 30, 1], ["pymcs", -1, -1], [5, 0, 0], [4, 1, 0] ],
              [ ["gzero", 30, 4], ["ntest", 1, -1], [5, 0, 0], [4, 1, 0] ],
              [ ["gzero", 30, 4], ["ntest", 2, -1], [0, 6, 4], [1, 9, 0] ],
              [ ["gzero", 30, 8], ["ntest", 2, -1], [1, 4, 0], [1, 4, 0] ],

              [ ["gzero", 35, 1], ["pymcs", -1, -1], [5, 0, 0], [4, 1, 0] ],
              [ ["gzero", 35, 1], ["simplecmts", -1, -1], [4, 1, 0], [3, 2, 0] ],
              [ ["gzero", 35, 2], ["simplecmts", -1, -1], [5, 0, 0], [4, 1, 0] ],

              [ ["gzero", 35, 4], ["ntest", 1, -1], [5, 0, 0], [4, 1, 0] ],
              [ ["gzero", 35, 4], ["ntest", 2, -1], [0, 5, 0], [1, 4, 0] ],
              [ ["gzero", 35, 8], ["ntest", 2, -1], [2, 3, 0], [3, 2, 0] ],

              [ ["gzero", 40, 4], ["ntest", 2, -1], [1, 4, 0], [3, 2, 0] ],

              [ ["gzero", 45, 4], ["ntest", 1, -1], [4, 1, 0], [5, 0, 0] ],
              [ ["gzero", 45, 4], ["ntest", 2, -1], [21, 26, 0], [35, 14, 0] ],
              [ ["gzero", 45, 8], ["ntest", 3, -1], [5, 0, 0], [5, 0, 0] ],
              [ ["gzero", 45, 8], ["ntest", 4, -1], [5, 0, 0], [3, 0, 2] ],

              [ ["gzero", 50, 1], ["pymcs", 1, -1], [4, 1, 0], [4, 1, 0] ],
              [ ["gzero", 50, 1], ["simplecmts", -1, -1], [4, 1, 0], [4, 1, 0] ],
              [ ["gzero", 50, 4], ["ntest", 1, -1], [3, 0, 0], [3, 0, 0] ],
              [ ["gzero", 50, 4], ["ntest", 2, -1], [6, 4, 1], [10, 1, 0] ],
              [ ["gzero", 50, 8], ["ntest", 5, -1], [2, 3, 0], [3, 2, 0] ]

] }

""")

format_table(json_table["results"])
