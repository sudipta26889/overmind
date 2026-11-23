from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import openai
import pytest
import redis
from asgiref.sync import async_to_sync
from celery.exceptions import SoftTimeLimitExceeded
from django.core.cache import cache
from django.utils import timezone
from factories import make_project, make_span

from overbae.core import llms
from overbae.core.model_registry import ROLE_CHAINS, Role
from overbae.innermind import (
    admission,
    evals,
    judge_health,
    replay,
    signals,
    span_processor,
    trajectory,
)
from overbae.innermind.evals import Judge, is_unavailable
from overbae.models import EvalMatrix, LiveScoringWork, Span
from overbae.services.codebase import task_type
from overbae.tasks import innermind

pytestmark = pytest.mark.covers("llm-gateway")

# Each xdist worker gets its own Redis database, so concurrent tests never share keys.
_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw")
REDIS_URL = (
    f"{os.environ.get('TEST_CACHE_REDIS', 'redis://localhost:6379')}/{15 - int(_WORKER or 0)}"
)
EXPECTATION = "The answer addresses the question."
JUDGE_DOWN = "overmind:judge:unavailable"


def _capability():
    return SimpleNamespace(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        description="Route support requests to the appropriate team.",
        decision_logic="Choose the team from the request's subject.",
        policy_markdown="Return exactly one team.",
        model="",
        input_schema={},
        output_fields={},
        improvement_metadata={"system_prompt": "Classify the request into one support queue."},
    )


def _span(project, **fields):
    return make_span(
        project,
        trace_id=uuid.uuid4().hex,
        attributes={
            "input": "Where is my order?",
            "output": "I like turtles.",
            "overmind.expectations": [EXPECTATION],
        },
        **fields,
    )


def _models(fake_llm) -> set[str]:
    return {request.model for request in fake_llm.requests}


@pytest.fixture
def remote_chains(monkeypatch):
    monkeypatch.setitem(ROLE_CHAINS, Role.FAST, ("gpt-6-luna",))
    monkeypatch.setitem(ROLE_CHAINS, Role.EXPLAIN, ("gpt-6-luna",))


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
    client.delete(JUDGE_DOWN)
    yield REDIS_URL
    client.delete(JUDGE_DOWN)


@pytest.fixture
def sampled(settings):
    settings.TRACE_SCORING_SAMPLE_RATE = 1


def test_live_judging_uses_only_the_local_judge_and_never_explains(fake_llm):
    fake_llm.verdict = lambda state: "no"
    span = Span(span_id="s", trace_id="t", attributes={"input": "q", "output": "a"})
    row = EvalMatrix(name="r", type=EvalMatrix.Type.LLM, source=EXPECTATION)

    metric = async_to_sync(Judge("project").score)(row, span, [span])

    assert (metric.verdict, metric.score, metric.reason) == ("no", 0.0, "")
    assert _models(fake_llm) == {"local/judge"}
    assert fake_llm.requests[0].body["temperature"] == 0
    assert fake_llm.requests[0].schema_name == "Verdict"
    assert "response_format" in fake_llm.requests[0].body
    assert fake_llm.requests[0].body["response_format"]["type"] == "json_schema"


def test_person_started_work_uses_the_fast_and_explain_models_only(fake_llm, remote_chains):
    fake_llm.verdict = lambda state: "no"
    fake_llm.on(
        lambda r: r.system.startswith("Classify the task an AI capability"),
        '{"task_type":"classification"}',
    )
    span = Span(span_id="s", trace_id="t", attributes={"input": "q", "output": "a"})
    row = EvalMatrix(name="r", type=EvalMatrix.Type.LLM, source=EXPECTATION)

    metric = async_to_sync(Judge("project", Role.FAST).score)(row, span, [span])
    assert metric.score == 0.0
    assert metric.reason
    assert task_type.classify_capability_task(_capability()) == "classification"
    tools = SimpleNamespace(tool_definitions=lambda: [], execute=lambda *a, **k: None)
    replay.run_capability(
        input_messages=[{"role": "user", "content": "hello"}],
        tool_provider=tools,
        model="gpt-6-luna",
    )

    assert _models(fake_llm) == {"openai/gpt-6-luna"}
    assert len(fake_llm.judgments()) == 1
    assert len(fake_llm.requests) == 4


@pytest.mark.django_db(transaction=True)
def test_a_live_span_is_judged_through_the_local_judge_only(fake_llm, remote_chains, sampled):
    span = _span(make_project())
    scored = async_to_sync(span_processor.score_project_spans)(span.project_id, [span])

    assert len(scored) == 1
    assert _models(fake_llm) == {"local/judge"}
    span.refresh_from_db()
    assert span.eval_attributes["score"] == 1.0


@pytest.mark.django_db(transaction=True)
def test_a_judge_held_by_another_caller_leaves_the_span_unclaimed(
    fake_llm, redis_cache, sampled, monkeypatch
):
    monkeypatch.setenv("LLM_LOCK_WAIT_MAX", "0.3")
    span = _span(make_project())
    client = redis.Redis.from_url(redis_cache)
    client.set("overmind:llm:single-stream:local-judge", "another-caller", px=60_000)
    try:
        async_to_sync(span_processor.score_project_spans)(span.project_id, [span])
        span.refresh_from_db()
        assert "score" not in span.eval_attributes
        assert span.eval_attributes.get("attempts", 0) == 0
        assert span.eval_claimed_at is None
        assert fake_llm.requests == []

        client.delete("overmind:llm:single-stream:local-judge")
        async_to_sync(span_processor.score_project_spans)(span.project_id, [span])
        span.refresh_from_db()
        assert span.eval_attributes["score"] == 1.0
        assert client.get("overmind:llm:single-stream:local-judge") is None
    finally:
        client.delete("overmind:llm:single-stream:local-judge")


@pytest.mark.django_db(transaction=True)
def test_an_answer_without_a_verdict_is_an_attempt_and_the_third_is_scored_with_errors(
    fake_llm, sampled
):
    span = _span(make_project())
    fake_llm.on(lambda r: r.schema_name == "Verdict", "no verdict here")

    for expected in (1, 2):
        async_to_sync(span_processor.score_project_spans)(span.project_id, [span])
        span.refresh_from_db()
        assert "score" not in span.eval_attributes
        assert span.eval_attributes["attempts"] == expected
        assert span.eval_claimed_at is None

    async_to_sync(span_processor.score_project_spans)(span.project_id, [span])
    span.refresh_from_db()
    assert span.eval_attributes["score"] is None
    assert {m["error"] for m in span.eval_attributes["metrics"].values()} == {
        "scoring failed after 3 attempts"
    }


def test_slow_and_broken_judges_are_unavailable_but_a_bad_request_is_not():
    request = SimpleNamespace(method="POST", url="http://gateway")

    def status(code):
        response = SimpleNamespace(status_code=code, request=request, headers={})
        return openai.APIStatusError("boom", response=response, body=None)

    def wrapped(cause):
        error = RuntimeError("Error calling LLM")
        error.__cause__ = cause
        return error

    assert is_unavailable(wrapped(llms.CompletionDeadlineExceededError("slow")))
    assert is_unavailable(wrapped(llms.LLMBusyError("busy")))
    assert is_unavailable(wrapped(status(502)))
    assert is_unavailable(wrapped(status(429)))
    assert not is_unavailable(wrapped(status(400)))
    assert evals.JUDGE_STATE_LIMIT == 24_000


def test_a_single_stream_call_waits_for_the_stream_and_gives_up(fake_llm, redis_cache, monkeypatch):
    monkeypatch.setenv("LLM_LOCK_WAIT_MAX", "0.3")
    cache.clear()
    client = redis.Redis.from_url(redis_cache)
    client.set("overmind:llm:single-stream:flint", "another-caller", px=60_000)
    try:
        with pytest.raises(llms.LLMBusyError):
            llms.call_llm("hello", model="flint", retry_deadline=0)
        assert fake_llm.requests == []

        # The judge's wait cap does not apply to an agent call: it waits out its own deadline.
        monkeypatch.setenv("LLM_LOCK_WAIT_MAX", "0.05")
        started = time.monotonic()
        with pytest.raises(llms.LLMBusyError):
            llms.call_llm("hello", model="flint", retry_deadline=0.6)
        assert time.monotonic() - started >= 0.5

        client.delete("overmind:llm:single-stream:flint")
        llms.call_llm("hello", model="flint", retry_deadline=0)
        assert _models(fake_llm) == {"recursivesi/flint"}
        assert client.get("overmind:llm:single-stream:flint") is None
    finally:
        client.delete("overmind:llm:single-stream:flint")


@pytest.mark.django_db
def test_the_task_sweep_enqueues_the_newest_spans_first():
    project = make_project()
    old = timezone.now() - timedelta(hours=3)
    spans = [make_span(project, trace_id=f"t{i}") for i in range(3)]
    for age, span in enumerate(spans):
        Span.objects.filter(pk=span.pk).update(received_at=old - timedelta(minutes=age))

    innermind.sweep_unscored_spans()

    queued = LiveScoringWork.objects.order_by("id").values_list("key", flat=True)
    assert list(queued) == [span.span_id for span in spans]


@pytest.mark.django_db(transaction=True)
def test_a_single_stream_judge_gets_one_call_at_a_time(fake_llm, sampled):
    in_flight, peak = [0], [0]
    guard = threading.Lock()

    def verdict(_state):
        with guard:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.05)
        with guard:
            in_flight[0] -= 1
        return "yes"

    fake_llm.verdict = verdict
    project = make_project()
    spans = [_span(project) for _ in range(4)]

    scored = async_to_sync(span_processor.score_project_spans)(project.pk, spans)

    assert len(scored) == 4
    assert peak[0] == 1


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("status", [500, 503, 429, 401, 403, 404])
def test_a_judge_that_is_down_leaves_the_span_unclaimed_and_uncounted(fake_llm, sampled, status):
    span = _span(make_project())
    fake_llm.fail(lambda r: r.schema_name == "Verdict", status)

    for _ in range(4):
        async_to_sync(span_processor.score_project_spans)(span.project_id, [span])
        span.refresh_from_db()
        assert "score" not in span.eval_attributes
        assert span.eval_attributes.get("attempts", 0) == 0
        assert span.eval_claimed_at is None

    fake_llm.forget()
    async_to_sync(span_processor.score_project_spans)(span.project_id, [span])
    span.refresh_from_db()
    assert span.eval_attributes["score"] == 1.0


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("status", [400, 422])
def test_a_request_the_judge_refuses_is_an_attempt(fake_llm, sampled, status):
    span = _span(make_project())
    fake_llm.fail(lambda r: r.schema_name == "Verdict", status)

    async_to_sync(span_processor.score_project_spans)(span.project_id, [span])

    span.refresh_from_db()
    assert span.eval_attributes["attempts"] == 1
    assert span.eval_claimed_at is None


def test_the_in_process_wait_for_the_judge_is_bounded(monkeypatch):
    monkeypatch.setenv("LLM_LOCK_WAIT_MAX", "0.2")
    holding, release = threading.Event(), threading.Event()

    def hold():
        holding.set()
        release.wait(5)

    holder = threading.Thread(target=evals.throttled_single_stream, args=(hold,))
    holder.start()
    assert holding.wait(5)
    try:
        started = time.monotonic()
        with pytest.raises(evals.JudgeUnavailableError):
            evals.throttled_single_stream(lambda: "never runs")
        assert time.monotonic() - started < 3
    finally:
        release.set()
        holder.join()


@pytest.mark.django_db(transaction=True)
def test_a_pass_that_runs_out_of_time_keeps_its_rows_and_releases_the_span(fake_llm, sampled):
    project = make_project()
    second = "The order is named."
    span = make_span(
        project,
        trace_id=uuid.uuid4().hex,
        attributes={
            "input": "Where is my order?",
            "output": "Order 42 is on its way.",
            "overmind.expectations": [EXPECTATION, second],
        },
    )

    def verdict(state):
        if state["expectation"] == second:
            raise SoftTimeLimitExceeded()
        return "yes"

    fake_llm.verdict = verdict
    with pytest.raises(SoftTimeLimitExceeded):
        async_to_sync(span_processor.score_project_spans)(span.project_id, [span])

    span.refresh_from_db()
    assert "score" not in span.eval_attributes
    assert span.eval_attributes.get("attempts", 0) == 0
    assert span.eval_claimed_at is None
    assert [row["verdict"] for row in span.eval_attributes["partial"].values()] == ["yes"]

    fake_llm.verdict = None
    async_to_sync(span_processor.score_project_spans)(span.project_id, [span])
    span.refresh_from_db()
    assert span.eval_attributes["score"] == 1.0
    # The first row is not judged again; only the second is.
    assert len(fake_llm.judgments()) == 2


def _trace(project):
    trace_id = uuid.uuid4().hex
    root = make_span(
        project, trace_id=trace_id, eval_attributes={"score": 0.9, "intent": "Resolve the order"}
    )
    make_span(
        project,
        trace_id=trace_id,
        parent_span_id=root.span_id,
        span_type="tool_call",
        eval_attributes={"score": 0.9},
        attributes={"tool.name": "lookup", "input": "42", "output": "shipped"},
    )
    return root


@pytest.mark.django_db(transaction=True)
def test_a_down_judge_releases_the_trajectory_uncounted(fake_llm, sampled):
    root = _trace(make_project())
    fake_llm.fail(lambda r: r.schema_name == "Verdict", 503)

    with pytest.raises(evals.JudgeUnavailableError):
        async_to_sync(trajectory.score_trace)(root.project_id, root.trace_id)

    root.refresh_from_db()
    assert "trajectory" not in root.eval_attributes
    assert "trajectory_claimed_at" not in root.eval_attributes
    assert root.eval_attributes.get("trajectory_attempts", 0) == 0

    fake_llm.forget()
    assert async_to_sync(trajectory.score_trace)(root.project_id, root.trace_id) is True
    root.refresh_from_db()
    assert "trajectory" in root.eval_attributes
    assert "trajectory_claimed_at" not in root.eval_attributes


@pytest.mark.django_db(transaction=True)
def test_two_passes_never_judge_one_trajectory(fake_llm, sampled):
    root = _trace(make_project())
    Span.objects.filter(pk=root.pk).update(
        eval_attributes={
            **root.eval_attributes,
            "trajectory_claimed_at": timezone.now().isoformat(),
        }
    )

    assert async_to_sync(trajectory.score_trace)(root.project_id, root.trace_id) is False
    assert fake_llm.requests == []


@pytest.mark.django_db(transaction=True)
def test_plan_drift_judges_five_steps_a_run_and_saves_each(fake_llm):
    project = make_project()
    root = make_span(project, trace_id="drift", eval_attributes={"score": 1.0})
    steps = [
        make_span(
            project,
            trace_id="drift",
            parent_span_id=root.span_id,
            eval_attributes={"score": 0.9, "intent": "Resolve the order"},
            attributes={"input": "q", "output": "a"},
        )
        for _ in range(7)
    ]

    async_to_sync(signals.PlanDriftSignal(str(project.pk)).check)([root, *steps])

    judged = Span.objects.filter(trace_id="drift", eval_attributes__has_key="drift").count()
    assert judged == 5
    assert len(fake_llm.judgments()) == 5


@pytest.mark.django_db(transaction=True)
def test_a_judge_that_is_off_holds_the_span_unscored_and_unattempted(
    fake_llm, sampled, monkeypatch
):
    monkeypatch.setitem(ROLE_CHAINS, Role.JUDGE, ())
    span = _span(make_project())

    async_to_sync(span_processor.score_project_spans)(span.project_id, [span])

    span.refresh_from_db()
    assert "score" not in span.eval_attributes
    assert span.eval_attributes.get("attempts", 0) == 0
    assert span.eval_claimed_at is None
    assert fake_llm.requests == []


@pytest.mark.django_db(transaction=True)
def test_a_judge_that_is_off_still_writes_the_unsampled_marker(fake_llm, settings, monkeypatch):
    monkeypatch.setitem(ROLE_CHAINS, Role.JUDGE, ())
    settings.TRACE_SCORING_SAMPLE_RATE = 0
    span = _span(make_project())

    async_to_sync(span_processor.score_project_spans)(span.project_id, [span])

    span.refresh_from_db()
    assert span.eval_attributes["unsampled"] is True
    assert span.eval_claimed_at is None


def _dispatch():
    published = []
    admission.dispatch(lambda work_id, token: published.append(work_id))
    return [LiveScoringWork.objects.get(pk=work_id).key for work_id in published]


@pytest.mark.django_db
def test_span_work_is_dispatched_newest_first_one_at_a_time_for_a_single_stream_judge():
    project = make_project()
    now = timezone.now()
    spans = {age: make_span(project, trace_id=f"t{age}", span_type="llm_call") for age in (3, 1, 2)}
    for age, span in spans.items():
        Span.objects.filter(pk=span.pk).update(received_at=now - timedelta(hours=age))
    admission.enqueue(project.pk, "span", [spans[age].span_id for age in (3, 1, 2)])

    assert _dispatch() == [spans[1].span_id]
    LiveScoringWork.objects.filter(state="queued").update(state="done")
    assert _dispatch() == [spans[2].span_id]


@pytest.mark.django_db
def test_span_work_is_dispatched_newest_first_to_a_judge_that_can_take_two(monkeypatch):
    monkeypatch.setitem(ROLE_CHAINS, Role.JUDGE, ("gpt-6-luna",))
    project = make_project()
    now = timezone.now()
    spans = [make_span(project, trace_id=f"t{age}") for age in (3, 1, 2)]
    for age, span in zip((3, 1, 2), spans, strict=True):
        Span.objects.filter(pk=span.pk).update(received_at=now - timedelta(hours=age))
    admission.enqueue(project.pk, "span", [span.span_id for span in spans])

    assert _dispatch() == [spans[1].span_id, spans[2].span_id]


@pytest.mark.django_db
def test_no_judge_work_is_admitted_while_the_judge_is_backed_off(redis_cache):
    project = make_project()
    span = make_span(project, trace_id="backed-off")
    admission.enqueue(project.pk, "span", [span.span_id])
    judge_health.mark_unavailable()

    assert _dispatch() == []
    assert redis.Redis.from_url(redis_cache).pttl(JUDGE_DOWN) > 0

    redis.Redis.from_url(redis_cache).delete(JUDGE_DOWN)
    assert _dispatch() == [span.span_id]


@pytest.mark.django_db(transaction=True)
def test_a_down_judge_backs_judging_off(fake_llm, sampled, redis_cache):
    span = _span(make_project())
    fake_llm.fail(lambda r: r.schema_name == "Verdict", 503)

    async_to_sync(span_processor.score_project_spans)(span.project_id, [span])

    assert redis.Redis.from_url(redis_cache).pttl(JUDGE_DOWN) > 0


@pytest.mark.django_db(transaction=True)
def test_a_slow_judge_backs_judging_off_even_when_it_answers(
    fake_llm, sampled, redis_cache, monkeypatch
):
    monkeypatch.setenv("JUDGE_SLOW_SECONDS", "0.05")

    def slow(_state):
        time.sleep(0.15)
        return "yes"

    fake_llm.verdict = slow
    span = _span(make_project())

    async_to_sync(span_processor.score_project_spans)(span.project_id, [span])

    span.refresh_from_db()
    assert span.eval_attributes["score"] == 1.0
    assert redis.Redis.from_url(redis_cache).pttl(JUDGE_DOWN) > 0


@pytest.mark.django_db(transaction=True)
def test_a_spent_daily_limit_sends_nothing_keeps_the_span_offerable_and_resets_next_day(
    fake_llm, sampled, redis_cache, monkeypatch
):
    monkeypatch.setenv("JUDGE_DAILY_LIMIT", "1")
    day = [datetime(2026, 11, 22, 10, 0, tzinfo=UTC)]
    monkeypatch.setattr(judge_health, "now", lambda: day[0])
    client = redis.Redis.from_url(redis_cache)
    client.delete("overmind:judge:calls:2026-11-22", "overmind:judge:calls:2026-11-23")
    project = make_project()
    first, second = _span(project), _span(project)

    async_to_sync(span_processor.score_project_spans)(project.pk, [first])
    assert len(fake_llm.judgments()) == 1

    async_to_sync(span_processor.score_project_spans)(project.pk, [second])
    second.refresh_from_db()
    assert len(fake_llm.judgments()) == 1
    assert "score" not in second.eval_attributes
    assert second.eval_attributes.get("attempts", 0) == 0
    assert second.eval_claimed_at is None
    assert judge_health.backed_off()
    assert 0 < client.pttl(JUDGE_DOWN) <= 14 * 3600 * 1000

    judge_health.clear()
    day[0] = datetime(2026, 11, 23, 0, 5, tzinfo=UTC)
    async_to_sync(span_processor.score_project_spans)(project.pk, [second])
    second.refresh_from_db()
    assert second.eval_attributes["score"] == 1.0
    client.delete("overmind:judge:calls:2026-11-22", "overmind:judge:calls:2026-11-23")


def test_an_invalid_daily_limit_fails_loudly(monkeypatch):
    for bad in ("0", "-5", "many"):
        monkeypatch.setenv("JUDGE_DAILY_LIMIT", bad)
        with pytest.raises(ValueError, match="JUDGE_DAILY_LIMIT"):
            judge_health.daily_limit()
