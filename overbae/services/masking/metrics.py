"""Daily counters for what masking could not do. They count reasons, never text."""

from __future__ import annotations

from contextlib import suppress

import redis

from overbae.core.llms import cache_redis


def _bump(key: str) -> None:
    client = cache_redis()
    if client is None:
        return
    with suppress(redis.RedisError):
        if client.incr(key) == 1:
            client.expire(key, 48 * 3600)


def count_fallback(reason: str, today: str, *, leak: bool = False) -> None:
    _bump(f"overmind:masking:fallback:{reason}:{today}")
    if leak:
        _bump(f"overmind:masking:leak:{today}")
