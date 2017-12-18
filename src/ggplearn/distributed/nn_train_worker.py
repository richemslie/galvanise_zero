from builtins import super

from ggplib.util import log

from ggplearn import msgdefs
from ggplearn.training import nn_train
from ggplearn.distributed.workerbase import WorkerBrokerBase


class WorkerBroker(WorkerBrokerBase):
    worker_type = "nn_train"

    def __init__(self, conf):
        super().__init__(conf)
        self.register(msgdefs.TrainNNRequest, self.on_train_request)

    def on_train_request(self, server, msg):
        log.warning("request to train %s" % msg)
        nn_train.parse_and_train(msg)
        return msgdefs.Ok("network_trained")


###############################################################################

def start_worker_factory():
    from ggplib.util.init import setup_once
    setup_once("nn_train_worker")

    from ggplearn.util.keras import constrain_resources
    constrain_resources()

    broker = WorkerBroker(msgdefs.WorkerConf())
    broker.start()


if __name__ == "__main__":
    start_worker_factory()
