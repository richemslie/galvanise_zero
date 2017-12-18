from builtins import super

from ggplib.util import log

from ggplearn import msgdefs
from ggplearn.training import approximate_play as ap
from ggplearn.distributed.workerbase import WorkerBrokerBase


class WorkerBroker(WorkerBrokerBase):
    worker_type = "approx_self_play"

    def __init__(self, conf):
        super().__init__(conf)

        self.register(msgdefs.ConfigureApproxTrainer, self.on_configure)
        self.register(msgdefs.RequestSample, self.on_request_sample)

        self.approx_player = None

    def on_configure(self, server, msg):
        self.approx_player = ap.Runner(msg)
        return msgdefs.Ok("configured")

    def on_request_sample(self, server, msg):
        log.debug("Got request for sample with number unique states %s" % len(msg.new_states))
        for s in msg.new_states:
            self.approx_player.add_to_unique_states(tuple(s))

        sample, duplicates_seen = self.approx_player.generate_sample()

        log.verbose("Done sample")
        m = msgdefs.RequestSampleResponse(sample, duplicates_seen)
        server.send_msg(m)


###############################################################################

def start_worker_factory():
    from ggplib.util.init import setup_once
    setup_once("approx_self_play")

    from ggplearn.util.keras import constrain_resources
    constrain_resources()

    broker = WorkerBroker(msgdefs.WorkerConf())
    broker.start()


if __name__ == "__main__":
    start_worker_factory()
