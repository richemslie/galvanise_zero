import base64
import random

from ggplib.db import lookup

from ggpzero.util.state import encode_state, decode_state, fast_decode_state

games = ["breakthrough", "hex", "hexLG13"]

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


def test_speed():
    import time

    for game in games:
        print "doing", game
        sm = lookup.by_name(game).get_sm()

        # a couple of states
        bs_0 = sm.get_initial_state()

        bs_1 = sm.new_base_state()
        bs_1.assign(bs_0)
        for i in range(5):
            advance_state(sm, bs_1)

        # encode states
        encoded_0 = encode_state(bs_0.to_list())
        encoded_1 = encode_state(bs_1.to_list())

        assert decode_state(encoded_0) == fast_decode_state(encoded_0)
        assert decode_state(encoded_1) == fast_decode_state(encoded_1)

        s = time.time()
        for i in range(10000):
            l0 = decode_state(encoded_0)
            l1 = decode_state(encoded_1)

        print "time taken %.3f msecs" % ((time.time() - s) * 1000)

        s = time.time()
        for i in range(10000):
            l0 = fast_decode_state(encoded_0)
            l1 = fast_decode_state(encoded_1)

        print "time taken %.3f msecs" % ((time.time() - s) * 1000)
