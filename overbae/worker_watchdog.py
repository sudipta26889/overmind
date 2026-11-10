"""Exit a worker whose consumer has silently stopped, so its container restarts.

A broker connection can die without the socket ever reporting it. The worker then
answers ping and logs nothing while its queue grows, and no socket option recovers it.
"""

import logging
import os
import threading
import time

import redis
from celery.signals import celeryd_after_setup, task_received
from celery.worker import state

logger = logging.getLogger(__name__)

CHECK_SECONDS = 30
STALL_SECONDS = 300


class Watchdog:
    def __init__(self, client, keys, *, stall_seconds, clock=time.monotonic):
        self._client = client
        self._keys = keys
        self._stall_seconds = stall_seconds
        self._clock = clock
        self._progress = clock()

    def received(self):
        self._progress = self._clock()

    def stalled(self, *, running):
        try:
            waiting = any(self._client.llen(key) for key in self._keys)
        except redis.RedisError:
            # An outage is not a stall: the consumer gets a full window once it ends.
            waiting = False
        if running or not waiting:
            self.received()
            return False
        return self._clock() - self._progress > self._stall_seconds


def _watch(watchdog):
    while True:
        time.sleep(CHECK_SECONDS)
        if watchdog.stalled(running=bool(state.active_requests or state.reserved_requests)):
            logger.critical("Worker consumer stalled with tasks waiting; exiting to restart")
            logging.shutdown()
            os._exit(1)


# Only the `celery worker` command sends this, so a worker embedded in a test process is exempt.
@celeryd_after_setup.connect
def _start(sender, instance, **_):
    conf = instance.app.conf
    if not conf.broker_url.startswith(("redis://", "rediss://")):
        return
    prefix = conf.broker_transport_options.get("global_keyprefix", "")
    watchdog = Watchdog(
        redis.from_url(conf.broker_url, socket_connect_timeout=5, socket_timeout=5),
        [f"{prefix}{queue}" for queue in instance.app.amqp.queues.consume_from],
        stall_seconds=STALL_SECONDS,
    )
    task_received.connect(lambda **_: watchdog.received(), weak=False)
    threading.Thread(target=_watch, args=(watchdog,), name="consumer watchdog", daemon=True).start()
