''' XXX move out of this repo '''

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)


def get_from_json(path, includes=None, excludes=None):
    # XXX fn is a bit too generic to be here '

    includes = includes or []
    excludes = excludes or []
    files = os.listdir(path)
    for the_file in files:
        if not the_file.endswith(".json"):
            continue

        if len([ii for ii in includes if ii not in the_file]):
            continue

        if len([ii for ii in excludes if ii in the_file]):
            continue

        for ii in excludes:
            if ii in the_file:
                continue

        filename = os.path.join(path, the_file)
        buf = open(filename).read()
        yield json.loads(buf), filename
