import attr
from ggpzero.util.attrutil import register_attrs


@register_attrs
class TiltyardMatchSummary(object):
    randomToken = attr.ib("2WelgaEYJ7ZE44CO5BswtsDl7XKSl2q4")
    # list of player names ["SteadyEddieDev","Random"],
    playerNamesFromHost = attr.ib(attr.Factory(list))
    hasErrors = attr.ib(False)
    lastUpdated = attr.ib(1515825183071)
    moveCount = attr.ib(42)
    scrambled = attr.ib(False)
    allErrors = attr.ib(False)

    # list of bools [false,false]
    isPlayerHuman = attr.ib(attr.Factory(list))

    matchURL = attr.ib("http://matches.ggp.org/matches/70854218199346c0b55a5dc01cfc75ab4f8a049a/")

    startTime = attr.ib(1515080601715)
    playClock = attr.ib(15)
    tournamentNameFromHost = attr.ib("tiltyard_continuous")
    matchLength = attr.ib(382329)

    # list of bools [false,false]
    allErrorsForPlayer = attr.ib(attr.Factory(list))
    hashedMatchHostPK = attr.ib("90bd08a7df7b8113a45f1e537c1853c3974006b2")

    # list of bools [false,false]
    hasErrorsForPlayer = attr.ib(attr.Factory(list))

    startClock = attr.ib(170)
    matchId = attr.ib("tiltyard.breakthroughv0.1515080601516")
    matchRoles = attr.ib(2)

    gameMetaURL = attr.ib("http://games.ggp.org/base/games/breakthrough/v0/")
    isAborted = attr.ib(False)
    isCompleted = attr.ib(True)
    allErrorsForSomePlayer = attr.ib(True)
    # list of ints [100,0]
    goalValues = attr.ib([])


@register_attrs
class MatchSummaries(object):
    # list of TiltyardMatchSummary
    queryMatches = attr.ib(attr.Factory(list))


@register_attrs
class TiltyardMatch(object):
    randomToken = attr.ib("2WelgaEYJ7ZE44CO5BswtsDl7XKSl2q4")

    # list of player names ["SteadyEddieDev","Random"],
    playerNamesFromHost = attr.ib(attr.Factory(list))
    scrambled = attr.ib(False)


    # list of ints [100,0]
    goalValues = attr.ib([])

    # list of bools [false,false]
    isPlayerHuman = attr.ib(attr.Factory(list))

    playClock = attr.ib(15)
    startClock = attr.ib(170)
    matchId = attr.ib("tiltyard.breakthroughv0.1515080601516")
    tournamentNameFromHost = attr.ib("tiltyard_continuous")
    matchHostSignature = attr.ib("")

    gameMetaURL = attr.ib("http://games.ggp.org/base/games/breakthrough/v0/")
    matchHostPK = attr.ib("")
    isAborted = attr.ib(False)
    isCompleted = attr.ib(False)
    previewClock = attr.ib(-1)

    startTime = attr.ib(1515080601715)

    # list of string
    #states = ["( ( cellHolds 8 1 white ) ( cellHolds 3 8 black ) )",
    #          "( ( cellHolds 8 1 white ) ( cellHolds 3 8 black ) "
    states = attr.ib(attr.Factory(list))

    # list of list of strings
    # moves = [["( move 2 5 2 6 )", "noop"], ["noop", "( move 4 8 4 7 )"]]
    moves = attr.ib(attr.Factory(list))

    # list of ints
    # stateTimes = [1515080601732, 1515080784677]
    stateTimes = attr.ib(attr.Factory(list))

    # list of list of strings
    # error = [["", ""], ["", ""]]
    errors = attr.ib(attr.Factory(list))
