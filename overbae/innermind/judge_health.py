"""Back-off while the shared local judge is down or slow."""

from __future__ import annotations

import os
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import redis

from overbae.core.llms import cache_redis

KEY = "overmind:judge:unavailable"


def backoff_ms() -> int:
    return int(float(os.environ.get("JUDGE_BACKOFF_SECONDS", 300)) * 1000)


def slow_seconds() -> float:
    """The judge's GPU is shared: a call this slow means other work needs it, so judging pauses."""
    return float(os.environ.get("JUDGE_SLOW_SECONDS", 20))


def now() -> datetime:
    return datetime.now(UTC)


def daily_limit() -> int:
    """Judge calls allowed per UTC day: the shared GPU's budget. A bad value fails loudly."""
    raw = os.environ.get("JUDGE_DAILY_LIMIT") or "3000"
    try:
        limit = int(raw)
    except ValueError:
        limit = 0
    if limit <= 0:
        raise ValueError(f"JUDGE_DAILY_LIMIT={raw!r} must be a positive integer")
    return limit


def mark_unavailable(px: int | None = None) -> None:
    client = cache_redis()
    if client is not None:
        with suppress(redis.RedisError):
            client.set(KEY, "1", px=px or backoff_ms())


def spend_call() -> bool:
    """Counts one judge request against today's limit. False once the limit is passed, and
    judging then backs off until the next UTC midnight. Without a cache Redis, uncapped."""
    client = cache_redis()
    if client is None:
        return True
    moment = now()
    key = f"overmind:judge:calls:{moment.date().isoformat()}"
    try:
        count = client.incr(key)
        if count == 1:
            client.expire(key, 48 * 3600)
    except redis.RedisError:
        return True
    if count <= daily_limit():
        return True
    midnight = datetime.combine(moment.date() + timedelta(days=1), datetime.min.time(), UTC)
    mark_unavailable(px=max(1000, int((midnight - moment).total_seconds() * 1000)))
    return False


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
