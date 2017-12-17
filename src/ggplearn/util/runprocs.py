import os
import time
import shlex
from signal import SIGKILL
from subprocess import PIPE, Popen

from twisted.internet import reactor

from ggplib.util import log


class RunCmds(object):
    def __init__(self, cmds, cb_on_completion=None, max_time=2.0):
        self.cmds = cmds
        self.cb_on_completion = cb_on_completion
        self.max_time = max_time

        self.timeout_time = None

    def spawn(self):
        self.procs = [(cmd, Popen(shlex.split(cmd),
                                  shell=False, stdout=PIPE, stderr=PIPE)) for cmd in self.cmds]
        self.timeout_time = time.time() + self.max_time
        reactor.callLater(0.1, self.check_running_processes)

    def check_running_processes(self):
        procs, self.procs = self.procs, []
        for cmd, proc in procs:
            retcode = proc.poll()
            if retcode is not None:
                log.debug("cmd '%s' exited with return code: %s" % (cmd, retcode))
                stdout, stderr = proc.stdout.read().strip(), proc.stderr.read().strip()
                if stdout:
                    log.verbose("stdout:%s" % stdout)
                if stderr:
                    log.warning("stderr:%s" % stderr)
                continue

            self.procs.append((cmd, proc))

        if time.time() > self.timeout_time:
            for cmd, proc in self.procs:
                log.warning("cmd '%s' taking too long, killing" % cmd)
                os.kill(proc.pid, SIGKILL)

        if self.procs:
            reactor.callLater(0.1, self.check_running_processes)
        else:
            self.cb_on_completion()

