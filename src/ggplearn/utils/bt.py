from ggplib.symbols import SymbolFactory

def pretty_print_board(sm, state):
    lines = []
    for i,v in enumerate(state):
        if v:
            lines.append(sm.get_gdl(i))

    sf = SymbolFactory()
    states = sf.to_symbols("\n".join(lines))
    mapping = {}
    control = None
    for s in list(states):
        if s[1][0] == "control":
            control = s[1][1]
        else:
            assert s[1][0] == "cellHolds"
            key = int(s[1][1]), int(s[1][2])
            mapping[key] = s[1][3]

    lines = []
    line_len = 8 * 4 + 1
    lines.append("-" * line_len)
    for i in range(1, 9):
        l = ["|"]
        for j in range(1, 9):
            key = j, i
            if key in mapping:
                if mapping[key] == "black":
                    l.append(" b |")
                else:
                    assert mapping[key] == "white"
                    l.append(" w |")
            else:
                l.append("   |")

        lines.append("".join(l))
        lines.append("-" * line_len)
    print "CONTROL", control
    print "\n".join(lines)
