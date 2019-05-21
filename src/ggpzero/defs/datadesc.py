import attr

from ggpzero.util.attrutil import register_attrs


# XXX rename?
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


# XXX rename?
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

    draw_head = attr.ib(False)

    # the training config attributes - for debugging, historical purposes
    # the number of samples trained on, etc
    # the number losses, validation losses, accurcacy
    trained_losses = attr.ib('not set')
    trained_validation_losses = attr.ib('not set')
    trained_policy_accuracy = attr.ib('not set')
    trained_value_accuracy = attr.ib('not set')


@register_attrs
class StepSummary(object):
    step = attr.ib(42)
    filename = attr.ib("gendata_hexLG11_6.json.gz")
    with_generation = attr.ib("h_5")
    num_samples = attr.ib(50000)

    md5sum = attr.ib("93d6ce4b812d353c73f4a8ca5b605d37")

    stats_unique_matches = attr.ib(2200)
    stats_draw_ratio = attr.ib(0.03)

    # when all policies lengths = 1
    stats_bare_policies_ratio = attr.ib(0.03)

    stats_av_starting_depth = attr.ib(5.5)
    stats_av_ending_depth = attr.ib(5.5)
    stats_av_resigns = attr.ib(0.05)
    stats_av_resign_false_positive = attr.ib(0.2)

    stats_av_puct_visits = attr.ib(2000)

    # if len(policy dist) > 1 and len(other_policy dist) == 1: +1 / #samples
    # [0.45, 0.55]
    stats_ratio_of_roles = attr.ib(attr.Factory(list))

    # average score by role
    stats_av_final_scores = attr.ib(attr.Factory(list))

    # score up to for lead_role_index (or role_index 0), number of samples
    # [(0.1, 200), (0.1, 200),... (0.9, 200)]
    stats_av_puct_score_dist = attr.ib(attr.Factory(list))


@register_attrs
class GenDataSummary(object):
    game = attr.ib("game")
    gen_prefix = attr.ib("x1")
    last_updated = attr.ib('2018-01-24 22:28')
    total_samples = attr.ib(10**10)

    # isinstance StepSummary
    step_summaries = attr.ib(attr.Factory(list))
