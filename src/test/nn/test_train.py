import os
import numpy as np

import tensorflow as tf

from ggplib.util.init import setup_once
from ggplib.db import lookup

from ggpzero.util import keras

from ggpzero.defs import confs, templates
from ggpzero.nn import train
from ggpzero.nn.manager import get_manager


def setup():
    # set up ggplib
    setup_once()

    # ensure we have database with ggplib
    lookup.get_database()

    # initialise keras/tf
    keras.init()

    # just ensures we have the manager ready
    get_manager()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    np.set_printoptions(threshold=100000)


def get_conf():
    conf = confs.TrainNNConfig(game="cittaceot",
                               generation_prefix="v8",
                               use_previous=False,
                               next_step=3,
                               overwrite_existing=False,
                               validation_split=0.9,
                               batch_size=32,
                               max_sample_count=1000000,
                               max_epoch_samples_count=100000,
                               starting_step=0,
                               drop_dupes_count=2,
                               compile_strategy="adam",
                               learning_rate=None)
    return conf

def get_conf_reversi():
    conf = confs.TrainNNConfig(game="reversi",
                               generation_prefix="v7",
                               use_previous=False,
                               next_step=11,
                               overwrite_existing=True,
                               validation_split=0.9,
                               batch_size=32,
                               epochs=3,
                               max_sample_count=10000,
                               max_epoch_samples_count=2000,
                               starting_step=10,
                               drop_dupes_count=1,
                               compile_strategy="adam",
                               learning_rate=None)
    return conf



def run_with_transformer(buf, conf, transformer):

    for fn, data in buf.files_to_sample_data(conf):
        print fn, data
        assert data.game == conf.game

        # actually testing integrity of files...
        assert data.num_samples == len(data.samples)

        assert not data.transformed

        data.transform_all(transformer)

    # second time around - cached
    for fn, data in buf.files_to_sample_data(conf):
        print fn, data
        assert data.game == conf.game

        # actually testing integrity of files...
        assert data.num_samples == len(data.state_identifiers)

        assert data.transformed



def test_cache():
    buf = train.SamplesBuffer()

    # we need the data for this test
    conf = get_conf()

    # create a transformer
    man = get_manager()
    generation_descr = templates.default_generation_desc(conf.game)
    generation_descr.num_previous_states = 0
    generation_descr.multiple_policy_heads = False
    transformer = man.get_transformer(conf.game)

    run_with_transformer(buf, conf, transformer)


def test_cache_multiple_policy_heads():
    buf = train.SamplesBuffer()

    # we need the data for this test
    conf = get_conf()

    # create a transformer
    man = get_manager()
    generation_descr = templates.default_generation_desc(conf.game)
    generation_descr.num_previous_states = 0
    generation_descr.multiple_policy_heads = True
    transformer = man.get_transformer(conf.game, generation_descr)

    run_with_transformer(buf, conf, transformer)


def test_cache_prev_states():
    buf = train.SamplesBuffer()

    # we need the data for this test
    conf = get_conf()

    # create a transformer
    man = get_manager()
    generation_descr = templates.default_generation_desc(conf.game)
    generation_descr.num_previous_states = 2
    generation_descr.multiple_policy_heads = False
    transformer = man.get_transformer(conf.game, generation_descr)

    run_with_transformer(buf, conf, transformer)

def test_trainer():

    # we need the data for this test
    conf = get_conf_reversi()

    # create a transformer
    man = get_manager()
    generation_descr = templates.default_generation_desc(conf.game)
    generation_descr.num_previous_states = 2
    generation_descr.multiple_policy_heads = True
    transformer = man.get_transformer(conf.game, generation_descr)

    # create the manager
    trainer = train.TrainManager(conf, transformer, next_generation_prefix="x2test")

    nn_model_config = templates.nn_model_config_template(conf.game, "small", transformer)
    trainer.get_network(nn_model_config, generation_descr)

    data = trainer.gather_data()

    trainer.do_epochs(data)
    trainer.save()
