import os
import uuid

import pytest
import redis

from overbae.worker_watchdog import Watchdog

pytestmark = pytest.mark.covers("recovery")

STALL = 300


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


class Broker:
    down = True

    def __init__(self, client):
        self.client = client

    def llen(self, key):
        if self.down:
            raise redis.ConnectionError
        return self.client.llen(key)


@pytest.fixture
def queue():
    client = redis.from_url(os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:6379/15"))
    key = f"overmind:watchdog-test:{uuid.uuid4()}"
    yield client, key
    client.delete(key)


def test_tasks_waiting_with_nothing_received_is_a_stall(queue):
    client, key = queue
    clock = Clock()
    watchdog = Watchdog(client, [key], stall_seconds=STALL, clock=clock)
    client.rpush(key, "task")
    clock.now += STALL + 1
    assert watchdog.stalled(running=False)


def test_a_received_task_restarts_the_window(queue):
    client, key = queue
    clock = Clock()
    watchdog = Watchdog(client, [key], stall_seconds=STALL, clock=clock)
    client.rpush(key, "task")
    clock.now += STALL
    watchdog.received()
    clock.now += STALL
    assert not watchdog.stalled(running=False)


def test_a_long_running_task_is_not_a_stall(queue):
    client, key = queue
    clock = Clock()
    watchdog = Watchdog(client, [key], stall_seconds=STALL, clock=clock)
    client.rpush(key, "task")
    clock.now += STALL + 1
    assert not watchdog.stalled(running=True)


def test_an_empty_queue_keeps_an_idle_worker_healthy(queue):
    client, key = queue
    clock = Clock()
    watchdog = Watchdog(client, [key], stall_seconds=STALL, clock=clock)
    clock.now += STALL + 1
    assert not watchdog.stalled(running=False)
    client.rpush(key, "task")
    clock.now += STALL
    assert not watchdog.stalled(running=False)


def test_a_broker_outage_gives_the_consumer_a_full_window_after_it(queue):
    client, key = queue
    clock = Clock()
    broker = Broker(client)
    watchdog = Watchdog(broker, [key], stall_seconds=STALL, clock=clock)
    clock.now += STALL + 1
    assert not watchdog.stalled(running=False)
    broker.down = False
    client.rpush(key, "task")
    clock.now += STALL
    assert not watchdog.stalled(running=False)
