
# ggplib imports
from ggplib.util import log

# ggpzero imports
from ggplib.db import lookup
from ggpzero.nn import datacache
from ggpzero.defs import templates

from ggpzero.nn.manager import get_manager


def setup_and_get_cache(game, prev_states, gen):
    # cp some files to a test area

    lookup.get_database()

    generation_descr = templates.default_generation_desc(game,
                                                         multiple_policy_heads=True,
                                                         num_previous_states=prev_states)
    man = get_manager()
    transformer = man.get_transformer(game, generation_descr)
    return datacache.DataCache(transformer, gen)


def test_summary():
    from ggplib.util.init import setup_once
    setup_once()

    game = "amazons_10x10"
    # game = "hexLG13"

    cache = setup_and_get_cache(game, 1, "h3")

    for x in cache.list_files():
        print x

    cache.sync()

    # good to see some outputs
    for index in (10, 420, 42):
        channels = cache.db[index]["channels"]
        log.info('train input, shape: %s.  Example: %s' % (channels.shape, channels))

        for name in cache.db.names[1:]:
            log.info("Outputs: %s" % name)
            output = cache.db[index]["channels"]
            log.info('train output, shape: %s.  Example: %s' % (output.shape, output))


def test_chunking():
    game = "breakthroughSmall"
    cache = setup_and_get_cache(game, 1, "h3")
    cache.sync()

    buckets_def = [(5, 0.8), (10, 0.5), (-1, 0.1)]
    buckets = datacache.Buckets(buckets_def)

    # max_training_count=None, max_validation_count=None
    indexer = cache.create_chunk_indexer(buckets)

    first = len(cache.summary.step_summaries) - 1
    print indexer.create_indices_for_level(first, validation=False, max_size=42)
    print indexer.create_indices_for_level(first, validation=True, max_size=42)

    for s in cache.db[indexer.create_indices_for_level(first, validation=True, max_size=10)]:
        print s

    print len(indexer.training_epoch(100000))
    print len(indexer.training_epoch(50000))
    print len(indexer.training_epoch(5000))
