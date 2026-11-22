"""Back-off while the shared local judge is down or slow."""

from __future__ import annotations

import os
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import redis

from overbae.core.llms import cache_redis

KEY = "overmind:judge:unavailable"
# A remote judge fails on its own account; it never pauses the local one.
CLOUD_KEY = "overmind:judge:unavailable:cloud"
CLOUD_BACKOFF_MS = 60_000


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


def cloud_daily_limit() -> int:
    """Remote judge and explanation calls per UTC day: a bound on spend. A bad value fails."""
    raw = os.environ.get("CLOUD_JUDGE_DAILY_LIMIT") or "20000"
    try:
        limit = int(raw)
    except ValueError:
        limit = 0
    if limit <= 0:
        raise ValueError(f"CLOUD_JUDGE_DAILY_LIMIT={raw!r} must be a positive integer")
    return limit


def mark_unavailable(px: int | None = None, *, cloud: bool = False) -> None:
    client = cache_redis()
    if client is not None:
        with suppress(redis.RedisError):
            if cloud:
                client.set(CLOUD_KEY, "1", px=px or CLOUD_BACKOFF_MS)
            else:
                client.set(KEY, "1", px=px or backoff_ms())


def _spend(prefix: str, limit: int, *, cloud: bool) -> bool:
    """One call against a daily limit. The local judge is uncapped when its counter is out of
    reach; a remote judge fails closed, because that cap bounds spend and nothing else does."""
    client = cache_redis()
    if client is None:
        return not cloud
    moment = now()
    key = f"overmind:judge:{prefix}:{moment.date().isoformat()}"
    try:
        count = client.incr(key)
        if count == 1:
            client.expire(key, 48 * 3600)
    except redis.RedisError:
        return not cloud
    if count <= limit:
        return True
    midnight = datetime.combine(moment.date() + timedelta(days=1), datetime.min.time(), UTC)
    mark_unavailable(px=max(1000, int((midnight - moment).total_seconds() * 1000)), cloud=cloud)
    return False


def calls_today() -> int:
    """Local judge requests counted so far today (UTC); 0 without a cache Redis."""
    client = cache_redis()
    if client is None:
        return 0
    try:
        return int(client.get(f"overmind:judge:calls:{now().date().isoformat()}") or 0)
    except (redis.RedisError, ValueError):
        return 0


def spend_call() -> bool:
    """Counts one local judge request against today's limit. False once the limit is passed, and
    judging then backs off until the next UTC midnight. Without a cache Redis, uncapped."""
    return _spend("calls", daily_limit(), cloud=False)


def spend_cloud_call() -> bool:
    """The same for a remote judge or explanation call, against its own limit and back-off.
    Without a cache Redis, or when it errors, the call is refused: that cap bounds spend."""
    return _spend("cloud_calls", cloud_daily_limit(), cloud=True)


def backed_off(*, cloud: bool = False) -> bool:
    client = cache_redis()
    if client is None:
        return False
    try:
        return bool(client.exists(CLOUD_KEY if cloud else KEY))
    except redis.RedisError:
        return False


def clear() -> None:
    """Ends the back-off, as an operator does once the judge is known to be back."""
    client = cache_redis()
    if client is not None:
        with suppress(redis.RedisError):
            client.delete(KEY, CLOUD_KEY)
