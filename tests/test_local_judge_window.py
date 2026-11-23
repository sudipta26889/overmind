from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest
import redis
import time_machine
from asgiref.sync import async_to_sync
from factories import make_project, make_span
from fakes import masking as fake_masking

from overbae.core import local_hours
from overbae.core.model_registry import ROLE_CHAINS, Role
from overbae.innermind import admission, judge_health, span_processor
from overbae.models import LiveScoringWork, Project

pytestmark = [pytest.mark.covers("masking-route"), pytest.mark.django_db(transaction=True)]

_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw")
REDIS_URL = (
    f"{os.environ.get('TEST_CACHE_REDIS', 'redis://localhost:6379')}/{15 - int(_WORKER or 0)}"
)
LOCAL = "local/judge"
CLOUD = "overmind/cloud-judge-masked"
EXPECTATION = "The answer addresses the question."

# India time is UTC+5:30: 03:00 IST is 21:30 UTC the day before; 12:00 IST is 06:30 UTC.
NIGHT = datetime(2026, 11, 24, 21, 30, tzinfo=UTC)  # 03:00 IST on the 25th
NOON = datetime(2026, 11, 25, 6, 30, tzinfo=UTC)  # 12:00 IST
LATE = datetime(2026, 11, 25, 18, 30, tzinfo=UTC)  # 00:00 IST on the 26th
END = datetime(2026, 11, 24, 2, 30, tzinfo=UTC)  # 08:00 IST: the end hour, exclusive


@pytest.fixture(autouse=True)
def _setup(settings, fake_llm, tmp_path, monkeypatch):
    settings.TRACE_SCORING_SAMPLE_RATE = 1
    settings.LOCAL_JUDGE_HOURS_IST = "1-8"
    monkeypatch.setitem(ROLE_CHAINS, Role.CLOUD_JUDGE, ("cloud-judge-masked",))
    fake_masking.enable_remote(settings, fake_llm, tmp_path)
    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
            "KEY_PREFIX": "overmind",
        }
    }
    client = redis.Redis.from_url(REDIS_URL)
    client.flushdb()
    yield
    client.flushdb()


def span_of(project):
    return make_span(
        project,
        trace_id=uuid.uuid4().hex,
        attributes={
            "input": "Where is my parcel?",
            "output": "It ships tomorrow.",
            "overmind.expectations": [EXPECTATION],
        },
    )


def score(project, span):
    async_to_sync(span_processor.score_project_spans)(project.pk, [span])
    span.refresh_from_db()
    return span


@pytest.mark.parametrize(
    ("value", "inside", "outside"),
    [
        ("1-8", [1, 4, 7], [0, 8, 12, 23]),
        ("23-2", [23, 0, 1], [2, 12, 22]),
        ("20-24", [20, 23], [0, 19]),
    ],
)
def test_the_window_is_start_inclusive_end_exclusive_and_may_wrap_midnight(value, inside, outside):
    window = local_hours.parse(value)

    def at(hour):  # an instant whose India hour is `hour`
        return datetime(2026, 11, 25, hour, 15, tzinfo=local_hours.IST)

    assert all(local_hours.is_open(window, at(h)) for h in inside)
    assert not any(local_hours.is_open(window, at(h)) for h in outside)


def test_empty_means_the_local_judge_is_open_at_every_hour(settings):
    settings.LOCAL_JUDGE_HOURS_IST = ""

    assert local_hours.parse("") is None
    for moment in (NIGHT, NOON, LATE):
        assert judge_health.local_judge_open(moment)


@pytest.mark.parametrize("bad", ["8", "a-b", "1-", "25-3", "1-25", "5-5", "1-8-9", "1:8"])
def test_a_malformed_window_raises_with_a_clear_message(bad):
    with pytest.raises(ValueError, match="LOCAL_JUDGE_HOURS_IST"):
        local_hours.parse(bad)


def test_a_malformed_window_stops_the_process_at_start_up():
    import subprocess
    import sys

    env = {**os.environ, "DJANGO_SETTINGS_MODULE": "tests.settings", "LOCAL_JUDGE_HOURS_IST": "bad"}
    started = subprocess.run(
        [sys.executable, "-c", "import django; django.setup()"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert started.returncode != 0
    assert "LOCAL_JUDGE_HOURS_IST" in started.stderr


def test_inside_the_window_a_local_item_is_judged(fake_llm):
    project = make_project()
    with time_machine.travel(NIGHT, tick=False):
        span = score(project, span_of(project))

    assert span.eval_attributes["score"] == 1.0
    assert {r.model for r in fake_llm.requests} == {LOCAL}


def test_outside_the_window_a_local_item_is_held_with_no_request_and_no_attempt(fake_llm):
    project = make_project()
    with time_machine.travel(NOON, tick=False):
        span = score(project, span_of(project))

    assert "score" not in span.eval_attributes
    assert span.eval_attributes.get("attempts", 0) == 0
    assert span.eval_claimed_at is None
    assert fake_llm.requests == []
    assert not judge_health.backed_off()  # nothing to outlast when the window opens


def test_the_end_hour_is_closed_and_a_window_that_wraps_midnight_is_open_at_midnight(
    fake_llm, settings
):
    project = make_project()
    with time_machine.travel(END, tick=False):
        assert "score" not in score(project, span_of(project)).eval_attributes
    settings.LOCAL_JUDGE_HOURS_IST = "23-2"
    with time_machine.travel(LATE, tick=False):
        assert score(project, span_of(project)).eval_attributes["score"] == 1.0


def test_admission_holds_local_work_outside_the_window_and_releases_it_inside():
    project = make_project()
    span = span_of(project)
    admission.enqueue(project.pk, "span", [span.span_id])
    sent = []

    with time_machine.travel(NOON, tick=False):
        admission.dispatch(lambda i, t: sent.append(i))
    assert sent == []
    assert LiveScoringWork.objects.get(key=span.span_id).state == "waiting"

    with time_machine.travel(NIGHT, tick=False):
        admission.dispatch(lambda i, t: sent.append(i))
    assert len(sent) == 1


def test_a_remote_item_is_judged_at_any_hour(fake_llm):
    project = make_project()
    Project.objects.filter(pk=project.pk).update(cloud_judging=True)
    with time_machine.travel(NOON, tick=False):
        span = score(project, span_of(project))

    assert span.eval_attributes["score"] == 1.0
    assert {r.model for r in fake_llm.requests} == {CLOUD}


def test_a_call_that_waited_for_the_stream_does_not_start_after_the_window_closes(fake_llm):
    from overbae.innermind import evals

    with time_machine.travel(NOON, tick=False), pytest.raises(evals.JudgeUnavailableError):
        evals.local_call("hello", model="local-judge")
    assert fake_llm.requests == []


def test_the_status_command_shows_the_window_and_whether_it_is_open(settings):
    import io

    from django.core.management import call_command

    out = io.StringIO()
    with time_machine.travel(NOON, tick=False):
        call_command("judging_status", stdout=out)
    assert "local judge window (IST): 1-8, closed now" in out.getvalue()
