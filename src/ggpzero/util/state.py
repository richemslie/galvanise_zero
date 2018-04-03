import base64
import numpy as np


def encode_state(s):
    assert isinstance(s, (list, tuple))
    a = np.array(s)
    aa = np.packbits(a)
    s = aa.tostring()
    return base64.encodestring(s)


def decode_state(s):
    s = base64.decodestring(s)
    aa = np.fromstring(s, dtype=np.uint8)

    # if the state is not a multiple of 8, will grow by that
    # XXX horrible.  We really should have these functions as methods to do encode/decode on some
    # smart Basestate object... )
    a = np.unpackbits(aa)
    return tuple(a)
