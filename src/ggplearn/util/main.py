import os
from ggplib.util.init import setup_once
from ggplib.util import log
from ggplearn.util.keras import constrain_resources


def main_wrap(main_fn, logfile_name=None):

    if logfile_name is None:
        # if logfile_name not set, derive it from main_fn
        fn = main_fn.func_code.co_filename
        logfile_name = os.path.splitext(os.path.basename(fn))[0]

    setup_once(logfile_name)

    try:
        # we might be running under python with no keras/numpy support
        constrain_resources()
    except ImportError as exc:
        log.warning("ImportError: %s" % exc)

    import pdb
    import sys
    import traceback

    try:
        if main_fn.func_code.co_argcount == 0:
            return main_fn()
        else:
            return main_fn(sys.argv[1:])

    except Exception as exc:
        print exc
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
