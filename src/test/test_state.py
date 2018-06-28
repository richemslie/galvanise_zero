import base64
import random

from ggplib.db import lookup

from ggpzero.util.state import encode_state, decode_state

games = ["breakthrough", "speedChess", "hex"]

def advance_state(sm, basestate):
    sm.update_bases(basestate)

    # leaks, but who cares, it is a test
    joint_move = sm.get_joint_move()
    base_state = sm.new_base_state()

    for role_index in range(len(sm.get_roles())):
        ls = sm.get_legal_state(role_index)
        choice = ls.get_legal(random.randrange(0, ls.get_count()))
        joint_move.set(role_index, choice)

    # play move, the base_state will be new state
    sm.next_state(joint_move, base_state)
    return base_state


def test_simple():
    for game in games:
        sm = lookup.by_name(game).get_sm()
        bs_0 = sm.get_initial_state()

        bs_1 = sm.new_base_state()
        bs_1.assign(bs_0)
        for i in range(3):
            advance_state(sm, bs_1)

        assert bs_0 != bs_1

        l0 = decode_state(encode_state(bs_0.to_list()))
        l1 = decode_state(encode_state(bs_1.to_list()))

        decode_bs_0 = sm.new_base_state()
        decode_bs_1 = sm.new_base_state()
        decode_bs_0.from_list(l0)
        decode_bs_1.from_list(l1)

        assert bs_0.to_string() == bs_0.to_string()

        assert decode_bs_0 == bs_0
        assert decode_bs_0.hash_code() == bs_0.hash_code()

        print len(decode_bs_0.to_string())
        print len(bs_0.to_string())

        #assert decode_bs_0.to_string() == bs_0.to_string()

        assert decode_bs_1 == bs_1
        assert decode_bs_1.hash_code() == bs_1.hash_code()
        assert decode_bs_1.to_string() == bs_1.to_string()


def test_more():
    for game in games:
        print "doing", game
        sm = lookup.by_name(game).get_sm()
        bs_0 = sm.get_initial_state()

        bs_1 = sm.new_base_state()
        bs_1.assign(bs_0)
        for i in range(5):
            advance_state(sm, bs_1)

        assert bs_0 != bs_1

        # states to compare
        decode_bs_0 = sm.new_base_state()
        decode_bs_1 = sm.new_base_state()
        decode_direct_bs_0 = sm.new_base_state()
        decode_direct_bs_1 = sm.new_base_state()

        # encode as before
        en_0 = encode_state(bs_0.to_list())
        en_1 = encode_state(bs_1.to_list())

        # decode as before
        l0 = decode_state(en_0)
        l1 = decode_state(en_1)
        decode_bs_0.from_list(l0)
        decode_bs_1.from_list(l1)

        # decode directly
        decode_direct_bs_0.from_string(base64.decodestring(en_0))
        decode_direct_bs_1.from_string(base64.decodestring(en_1))

        # all checks
        assert decode_bs_0 == bs_0
        assert decode_bs_0.hash_code() == bs_0.hash_code()
        assert decode_bs_0.to_string() == bs_0.to_string()

        assert decode_direct_bs_0 == bs_0
        assert decode_direct_bs_0.hash_code() == bs_0.hash_code()
        assert decode_direct_bs_0.to_string() == bs_0.to_string()

        assert decode_bs_1 == bs_1
        assert decode_bs_1.hash_code() == bs_1.hash_code()
        assert decode_bs_1.to_string() == bs_1.to_string()

        assert decode_direct_bs_1 == bs_1
        assert decode_direct_bs_1.hash_code() == bs_1.hash_code()
        assert decode_direct_bs_1.to_string() == bs_1.to_string()

        print "good", game




# 488/16: bs = hex.get_sm().get_initial_state()
# 488/17: sm = hex.get_sm()
# 488/18: bs.hash_code()
# 488/19: bs = advance_state(sm, bs)
# 488/20: import random
# 488/21: bs = advance_state(sm, bs)
# 488/22: bs.hash_code()
# 488/23: bs = advance_state(sm, bs)
# 488/24: bs.hash_code()
# 488/25: bs = advance_state(sm, bs)
# 488/26: bs.hash_code()
# 488/27: bs = advance_state(sm, bs)
# 488/28: bs.hash_code()
# 488/29: bs = advance_state(sm, bs)
# 488/30: bs.hash_code()
# 490/1: from ggpzero.util.state import encode_state, decode_state
# 490/2: from ggplib.db import lookup
# 490/3: bt = lookup.by_name("breakthrough")
# 490/4: sm = bt.get_sm()
# 490/5: s = sm.get_initial_state()
# 490/6: s.to_list()
# 490/7: encode_state(bs.to_list())
# 490/8: encode_state(ss.to_list())
# 490/9: encode_state(s.to_list())
# 490/10: s.to_string()
# 490/11: s.to_string()
# 490/12: import base64
# 490/13: base64.encodestring(s.to_string())
# 490/14: len(encode_state(s.to_list()))
# 490/15: len(base64.encodestring(s.to_string()))
# 491/1: history
# 491/2: %history -g
# 491/3: from ggpzero.util.state import encode_state, decode_state
# 491/4: from ggplib.db import lookup
# 491/5: bt = lookup.by_name("breakthrough")
# 491/6: sm = bt.get_sm()
# 491/7: s = sm.get_initial_state()
# 491/8: encode_state(bs.to_list())
# 491/9: encode_state(s.to_list())
# 491/10: import base64
# 491/11:
# base64.encodestring(s.to_string())
# base64.encodestring(s.to_string())
# 491/12: %timeit base64.encodestring(s.to_string())
# 491/13: %timeit encode_state(s.to_list())
# 491/14: x = encode_state(s.to_list())
# 491/15: %timeit decode_state(x)
#    1: %history -g
        sm = lookup.by_name(game).get_sm()
        bs_0 = sm.get_initial_state()

        bs_1 = sm.new_base_state()
        bs_1.assign(bs_0)
        for i in range(3):
            advance_state(sm, bs_1)
