import attr

from ggpzero.util.attrutil import register_attrs

@register_attrs
class CanPlayGame(object):
    game = attr.ib("breakthrough")

    # can be empty for non-model players
    generations = attr.ib(default=attr.Factory(list))

@register_attrs
class RegisterPlayer(object):
    player_name = attr.ib("gurgeh")
    player_version = attr.ib("0.1-pre-alpha")

    # list of CanPlayGame
    games = attr.ib(default=attr.Factory(list))


@register_attrs
class PlayerReadyRequest(object):
    pass


@register_attrs
class PlayerReadyResponse(object):
    ready = attr.ib(False)


@register_attrs
class PlayNewGame(object):
    game = attr.ib("breakthrough")
    with_generation = attr.ib("v7_42")
    match_id = attr.ib("1234567890")

    # always good to know who the opponents are (list of strings)
    players = attr.ib(default=attr.Factory(list))

    # maximum time allowed for first move (allows for intialisation etc)
    play_time_first = attr.ib(60.0)

    # maximum time allowed for subsequent moves
    play_time = attr.ib(10.0)

    # empty means normal game state
    initial_game_state = attr.ib(default=attr.Factory(list))


@register_attrs
class PlayMatchNext(object):
    game = attr.ib("breakthrough")
    match_id = attr.ib("1234567890")
    moves_made = attr.ib(default=attr.Factory(list))


@register_attrs
class PlayMatchMove(object):
    game = attr.ib("hex")
    match_id = attr.ib("1234567890")
    move = attr.ib("(place f 4)")


@register_attrs
class PlayMatchEnd(object):
    game = attr.ib("breakthrough")
    match_id = attr.ib("1234567890")
    reason = attr.ib("game ended normally")
