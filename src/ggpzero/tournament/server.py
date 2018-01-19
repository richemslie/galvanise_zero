'''

* run a quick few with games with gamemaster
* build up a TiltyardMatchSummary / TiltyardMatch
* save them to files
* query web server will then will load files
* optimize by caching some stuff

'''

import os
import sys
import json
import time
import pprint


import attr

from twisted.web import server
from twisted.internet import reactor
from twisted.web.resource import Resource, NoResource

# from twisted.web.static import File

from ggplib.util import log
from ggplib import interface

from ggpzero.util import attrutil

# for side effects
from ggpzero.tournament import confs


def matches_path(game):
    return os.path.join(os.environ["GGPZERO_PATH"], "data", "tournament", game)


def augment_header(headers):
    headers.addRawHeader('Access-Control-Allow-Origin', '*')
    headers.addRawHeader('Access-Control-Allow-Methods', 'GET')
    headers.addRawHeader('Access-Control-Allow-Headers',
                         'x-prototype-version,x-requested-with')
    headers.addRawHeader('Access-Control-Max-Age', 2520) # 42 hours
    headers.addRawHeader('Content-type', 'application/json')


class GetMatchInfo(Resource):
    isLeaf = True

    def __init__(self, game, identifier):
        Resource.__init__(self)
        path_to_matches = matches_path(game)
        self.match_path = os.path.join(path_to_matches, "%s.json" % identifier)

    def render_GET(self, request):
        obj = attrutil.json_to_attr(open(self.match_path).read())
        augment_header(request.responseHeaders)
        return json.dumps(attr.asdict(obj))


class SummaryForGame(Resource):
    isLeaf = False

    def __init__(self, game):
        Resource.__init__(self)
        self.game = game

    def getChild(self, identifier, request):
        if identifier == "":
            return self

        return GetMatchInfo(self.game, identifier)

    def render_GET(self, request):
        augment_header(request.responseHeaders)
        return json.dumps(attr.asdict(self.obj))


class WebServer(Resource):
    isLeaf = False

    def __init__(self):
        Resource.__init__(self)

    def getChild(self, game, request):
        print 'here', game
        log.debug("Got GET request from: %s" % request.getClientIP())
        log.debug("HEADERS : %s" % pprint.pformat(request.getAllHeaders()))

        if game == 'summary':
            return self

        games = "breakthrough cittaceot checkers connectFour escortLatch hex reversi "
        if game in games.split():
            return SummaryForGame(game)
        return NoResource()

    def render_GET(self, request):
        summary_path = matches_path("summary.json")
        print "summary_path %s" % summary_path

        self.obj = attrutil.json_to_attr(open(summary_path).read())

        augment_header(request.responseHeaders)
        return json.dumps(attr.asdict(self.obj))

    def render_POST(self, request):
        print 'POST', request
        self.render_GET(request)


###############################################################################

def main(args):
    port = int(args[0])
    interface.initialise_k273(1, log_name_base="viewer")
    log.initialise()

    root = WebServer()
    site = server.Site(root)

    log.info("Running WebServer on port %d" % port)

    reactor.listenTCP(port, site)
    reactor.run()


###############################################################################

if __name__ == "__main__":
    debug = False
    try:
        assert debug
        from ipdb import launch_ipdb_on_exception
        with launch_ipdb_on_exception():
            main(sys.argv[1:])
    except Exception as _:
        main(sys.argv[1:])
