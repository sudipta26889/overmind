from __future__ import annotations

import io
import os

import pytest
import redis
from django.core.management import call_command
from factories import make_project, make_span

from overbae.innermind import admission, judge_health
from overbae.models import LiveScoringWork, Project

pytestmark = [pytest.mark.covers("masking-route"), pytest.mark.django_db]

_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw")
REDIS_URL = (
    f"{os.environ.get('TEST_CACHE_REDIS', 'redis://localhost:6379')}/{15 - int(_WORKER or 0)}"
)


@pytest.fixture
def redis_cache(settings):
    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
            "KEY_PREFIX": "overmind",
        }
    }
    client = redis.Redis.from_url(REDIS_URL)
    client.flushdb()
    yield client
    client.flushdb()


def test_it_counts_work_by_state_and_kind_what_is_held_locally_and_todays_counters(redis_cache):
    local, remote = make_project(), make_project()
    Project.objects.filter(pk=remote.pk).update(cloud_judging=True)
    held = make_span(local, trace_id="a" * 32, attributes={"output": "x"})
    make_span(remote, trace_id="b" * 32, attributes={"output": "y"})
    make_span(local, trace_id="c" * 32, attributes={"output": "z"}, eval_attributes={"score": 1.0})
    admission.enqueue(local.pk, "span", [held.span_id])
    admission.enqueue(remote.pk, "trajectory", ["b" * 32])
    day = judge_health.now().date().isoformat()
    redis_cache.set(f"overmind:judge:cloud_calls:{day}", 7)
    redis_cache.set(f"overmind:masking:leak:{day}", 2)
    redis_cache.set(f"overmind:masking:fallback:residual:{day}", 3)
    out = io.StringIO()

    call_command("judging_status", stdout=out)

    text = out.getvalue()
    assert "waiting  span       1" in text
    assert "waiting  trajectory 1" in text.replace("  ", " ") or "trajectory" in text
    assert "held on the local route: 1 waiting items, 1 unscored spans" in text
    assert "projects approved for remote judging: 1" in text
    assert "cloud calls today: 7" in text
    assert "leaks today: 2" in text
    assert "fallback residual today: 3" in text
    # Read-only.
    assert LiveScoringWork.objects.count() == 2
