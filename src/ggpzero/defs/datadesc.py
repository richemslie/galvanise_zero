import attr

from ggpzero.util.attrutil import register_attrs


@register_attrs
class Sample(object):
    # state policy trained on.  This is a tuple of 0/1s.  Effectively a bit array.
    state = attr.ib([0, 0, 0, 1])

    # list of previous state (first element is immediate parent of 'state')
    prev_states = attr.ib([1, 0, 0, 1])

    # list of policy distributions - all should sum to 1.
    policies = attr.ib([[0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]])

    # list of final scores for value head of network - list has same number as number of roles
    final_score = attr.ib([0, 1])

    # game depth at which point sample is taken
    depth = attr.ib(42)

    # total length of game
    game_length = attr.ib(42)

    # these are for debug.  The match_identifier can be used to extract contigous samples from the
    # same match.
    match_identifier = attr.ib("agame_421")
    has_resigned = attr.ib(False)
    resign_false_positive = attr.ib(False)
    starting_sample_depth = attr.ib(42)

    # the results after running the puct iterations
    resultant_puct_score = attr.ib(attr.Factory(list))
    resultant_puct_visits = attr.ib(800)


@register_attrs
class GenerationSamples(object):
    game = attr.ib("game")
    date_created = attr.ib('2018-01-24 22:28')

    # trained with this generation
    with_generation = attr.ib("v6_123")

    # number of samples in this generation
    num_samples = attr.ib(1024)

    # the samples (of type Sample)
    samples = attr.ib(attr.Factory(list))


@register_attrs
class GenerationDescription(object):
    ''' this describes the inputs/output to the network, provide information how the gdl
        transformations to input/outputs. and other meta information.  It does not describe the
        internals of the neural network, which is provided by the keras json model file.

    It will ripple through everything:
      * network model creation
      * reloading and using a trained network
      * the inputs/outputs from GdlTransformer
      * the channel ordering
        * network model creation
        * loading
    '''

    game = attr.ib("breakthrough")
    name = attr.ib("v6_123")
    date_created = attr.ib('2018-01-24 22:28')

    # whether the network expects channel inputs to have channel last format
    channel_last = attr.ib(False)

    # whether the network uses multiple policy heads (False - there is one)
    multiple_policy_heads = attr.ib(False)

    # number of previous states expected (default is 0).
    num_previous_states = attr.ib(0)

    # XXX todo
    transformer_description = attr.ib(None)

    # the training config attributes - for debugging, historical purposes
    # the number of samples trained on, etc
    # the number losses, validation losses, accurcacy
    trained_losses = attr.ib('not set')
    trained_validation_losses = attr.ib('not set')
    trained_policy_accuracy = attr.ib('not set')
    trained_value_accuracy = attr.ib('not set')
