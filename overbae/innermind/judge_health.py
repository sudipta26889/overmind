"""Back-off while the shared local judge is down or slow."""

from __future__ import annotations

import os
from contextlib import suppress

import redis

from overbae.core.llms import cache_redis

KEY = "overmind:judge:unavailable"


def backoff_ms() -> int:
    return int(float(os.environ.get("JUDGE_BACKOFF_SECONDS", 300)) * 1000)


def slow_seconds() -> float:
    """The judge's GPU is shared: a call this slow means other work needs it, so judging pauses."""
    return float(os.environ.get("JUDGE_SLOW_SECONDS", 20))


def mark_unavailable() -> None:
    client = cache_redis()
    if client is not None:
        with suppress(redis.RedisError):
            client.set(KEY, "1", px=backoff_ms())


def backed_off() -> bool:
    client = cache_redis()
    if client is None:
        return False
    try:
        return bool(client.exists(KEY))
    except redis.RedisError:
        return False


def clear() -> None:
    """Ends the back-off, as an operator does once the judge is known to be back."""
    client = cache_redis()
    if client is not None:
        with suppress(redis.RedisError):
            client.delete(KEY)
