from ggplib.util.symbols import SymbolFactory


def pretty_print_board(sm, state):
    lines = []
    for i, value in enumerate(state):
        if value:
            lines.append(sm.get_gdl(i))

    sf = SymbolFactory()
    states = sf.to_symbols("\n".join(lines))
    mapping = {}
    control = None
    for s in list(states):
        s = s[1]
        if s[0] == "control":
            control = s[1][1]

        elif s[0] == "cell":
            key = int(s[1]), int(s[2])
            mapping[key] = s[3]

    lines = []
    line_len = 8 * 4 + 1
    lines.append("-" * line_len)
    for i in range(1, 9):
        ll = ["|"]
        for j in range(1, 9):
            key = j, i
            if key in mapping:
                if mapping[key] == "black":
                    ll.append(" X |")
                else:
                    assert mapping[key] == "white"
                    ll.append(" O |")
            else:
                ll.append("   |")

        lines.append("".join(ll))
        lines.append("-" * line_len)
    print "CONTROL", control
    print "\n".join(lines)
