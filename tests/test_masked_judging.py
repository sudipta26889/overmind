from __future__ import annotations

import os
import re
import unicodedata
import uuid
from datetime import timedelta

import pytest
import redis
from asgiref.sync import async_to_sync
from django.utils import timezone
from factories import make_project, make_span
from fakes import masking as fake_masking

from overbae.core import llms
from overbae.core.model_registry import ROLE_CHAINS, Role
from overbae.innermind import admission, evals, judge_health, signals, span_processor, trajectory
from overbae.models import LiveScoringWork, Project, SignalSubscription, Span
from overbae.services.masking import MaskedPayload
from overbae.services.masking.send import call_masked
from overbae.tasks import innermind

pytestmark = pytest.mark.covers("masking-egress")

# Each xdist worker gets its own Redis database, so concurrent tests never share keys.
_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw")
REDIS_URL = f"redis://localhost:6379/{15 - int(_WORKER or 0)}"
CLOUD = "openai/gpt-6-luna"
LOCAL = "local/judge"
EXPECTATION = "The answer addresses the question."

# Made up, and planted in every trace below. None may appear in any request.
PLANTED = {
    "name": "Ravi Kumar",
    "phone": "9876543210",
    "email": "ravi.kumar@example.com",
    "amount": "₹18,000",
    "date": "5th March 2026",
    "condition": "diabetes",
    "medication": "metformin",
}


@pytest.fixture(autouse=True)
def _remote(monkeypatch, settings, fake_llm, tmp_path):
    monkeypatch.setitem(ROLE_CHAINS, Role.CLOUD_JUDGE, ("gpt-6-luna",))
    settings.TRACE_SCORING_SAMPLE_RATE = 1
    fake_masking.enable_remote(settings, fake_llm, tmp_path, names=[PLANTED["name"]])


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


def approved(project=None):
    project = project or make_project()
    Project.objects.filter(pk=project.pk).update(cloud_judging=True)
    return project


def story() -> str:
    return (
        f"{PLANTED['name']} says: call {PLANTED['phone']} or {PLANTED['email']}; the EMI of "
        f"{PLANTED['amount']} is due on {PLANTED['date']}. He has {PLANTED['condition']} and "
        f"takes {PLANTED['medication']}."
    )


def span_of(project, **fields):
    return make_span(
        project,
        trace_id=uuid.uuid4().hex,
        attributes={
            "input": story(),
            "output": "Thanks, noted. " + story(),
            "overmind.expectations": [EXPECTATION],
        },
        **fields,
    )


def trace_of(project):
    trace_id = uuid.uuid4().hex
    root = make_span(
        project,
        trace_id=trace_id,
        eval_attributes={"score": 0.9, "intent": f"Help {PLANTED['name']} with the order"},
        attributes={"output": story()},
    )
    make_span(
        project,
        trace_id=trace_id,
        parent_span_id=root.span_id,
        span_type="tool_call",
        eval_attributes={"score": 0.9, "intent": f"Find the order for {PLANTED['name']}"},
        attributes={
            "tool.name": "lookup",
            "input": '{"phone": "' + PLANTED["phone"] + '", "patient": "' + PLANTED["name"] + '"}',
            "output": story(),
        },
    )
    return root


def flat(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\s\-,.]", "", text)


def assert_nothing_planted_was_sent(fake_llm):
    assert fake_llm.requests, "no request was made"
    for request in fake_llm.requests:
        sent = flat(request.text + request.system)
        for kind, value in PLANTED.items():
            assert flat(value) not in sent, f"{kind} reached {request.model}"


def counter(client, reason: str) -> int:
    day = judge_health.now().date().isoformat()
    return int(client.get(f"overmind:masking:fallback:{reason}:{day}") or 0)


def score_span(project, span=None):
    span = span or span_of(project)
    async_to_sync(span_processor.score_project_spans)(project.pk, [span])
    span.refresh_from_db()
    return span


@pytest.mark.django_db(transaction=True)
def test_a_span_is_judged_and_explained_on_a_masked_copy_and_the_reason_is_restored(fake_llm):
    fake_llm.verdict = lambda state: "no"
    fake_llm.on(
        lambda r: r.schema_name != "Verdict",
        "<PERSON_1> asked about <AMOUNT_1> and the answer ignores it.",
    )
    span = score_span(approved())

    assert span.eval_attributes["score"] == 0.0
    reason = next(iter(span.eval_attributes["metrics"].values()))["reason"]
    assert reason == f"{PLANTED['name']} asked about {PLANTED['amount']} and the answer ignores it."
    assert [request.model for request in fake_llm.requests] == [CLOUD, CLOUD]
    assert_nothing_planted_was_sent(fake_llm)


@pytest.mark.django_db(transaction=True)
def test_a_trajectory_is_judged_on_a_masked_copy_with_tool_arguments_masked(fake_llm):
    project = approved()
    root = trace_of(project)

    assert async_to_sync(trajectory.score_trace)(project.pk, root.trace_id)

    assert {request.model for request in fake_llm.requests} == {CLOUD}
    assert_nothing_planted_was_sent(fake_llm)


@pytest.mark.django_db(transaction=True)
def test_plan_drift_is_judged_on_a_masked_copy(fake_llm):
    project = approved()
    root = trace_of(project)
    SignalSubscription.objects.create(project=project, signal_type="plan_drift", enabled=True)

    async_to_sync(signals.SignalRunner(project.pk).run)()

    step = Span.objects.get(trace_id=root.trace_id, parent_span_id=root.span_id)
    assert "drift" in step.eval_attributes
    assert {request.model for request in fake_llm.requests} == {CLOUD}
    assert_nothing_planted_was_sent(fake_llm)


@pytest.mark.django_db(transaction=True)
def test_every_remote_request_carries_the_remote_key_and_the_audit_user(fake_llm):
    project = approved()
    score_span(project)

    for request in fake_llm.requests:
        assert request.headers["authorization"] == f"Bearer {fake_masking.CLOUD_KEY}"
        assert request.body["user"] == str(project.pk)
    assert all(
        "user" not in request.body for request in fake_llm.requests if request.model == LOCAL
    )


def test_the_default_key_cannot_reach_the_remote_model_and_only_a_masked_payload_may_be_sent(
    fake_llm,
):
    with pytest.raises(RuntimeError):
        llms.call_llm("hello", model="gpt-6-luna")
    assert [request.model for request in fake_llm.requests] == [CLOUD]
    with pytest.raises(TypeError):
        call_masked("a raw string", model="gpt-6-luna")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MaskedPayload(text="raw", state_json="", mapping={})


@pytest.mark.django_db(transaction=True)
def test_a_remote_request_made_with_the_default_key_is_refused_by_the_gateway(fake_llm, settings):
    project = approved()
    fake_llm.cloud_key = "someone-elses-key"

    span = score_span(project)

    # The remote key is not the one the gateway knows: a 403 is "judge unavailable", never an attempt.
    assert "score" not in span.eval_attributes
    assert span.eval_attributes.get("attempts", 0) == 0
    assert all(request.model != LOCAL for request in fake_llm.requests)


@pytest.mark.covers("masking-fail-closed")
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    ("reason", "arrange"),
    [
        (
            "model_not_loaded",
            lambda s: (setattr(s, "MASKING_NER_BACKEND", ""), setattr(s, "MASKING_MODEL_DIR", "")),
        ),
        ("timeout", lambda s: setattr(s, "MASKING_TIMEOUT_SECONDS", 0)),
        ("too_large", lambda s: setattr(s, "MASKING_MAX_BYTES", 80)),
        ("cloud_key_missing", lambda s: setattr(s, "LLM_CLOUD_API_KEY_FILE", "/nonexistent")),
        # Docker creates a directory where a single-file bind mount points at a missing file.
        ("cloud_key_missing", lambda s: setattr(s, "LLM_CLOUD_API_KEY_FILE", "/usr")),
    ],
)
def test_a_masking_failure_sends_nothing_marks_the_span_and_the_local_judge_takes_it(
    fake_llm, redis_cache, settings, reason, arrange
):
    arrange(settings)
    span = score_span(approved())

    assert [r for r in fake_llm.requests if r.model == CLOUD] == []
    assert "score" not in span.eval_attributes
    assert span.eval_attributes.get("attempts", 0) == 0
    assert span.eval_attributes["masking_local"] == reason
    assert span.eval_claimed_at is None
    assert counter(redis_cache, reason) == 1

    # Offered again, the marked span is judged on this machine, with the same text unmasked.
    span = score_span(approved(), span)
    assert span.eval_attributes["score"] == 1.0
    assert {request.model for request in fake_llm.requests} == {LOCAL}


@pytest.mark.covers("masking-residual-fallback")
@pytest.mark.django_db(transaction=True)
def test_a_gateway_guardrail_refusal_marks_the_span_counts_a_leak_and_goes_local(
    fake_llm, redis_cache
):
    fake_llm.fail(lambda r: r.model == CLOUD, 400, "Request blocked by guardrail presidio-pii")
    project = approved()

    span = score_span(project)

    assert span.eval_attributes["masking_local"] == "guardrail"
    assert span.eval_attributes.get("attempts", 0) == 0
    assert counter(redis_cache, "guardrail") == 1
    day = judge_health.now().date().isoformat()
    assert int(redis_cache.get(f"overmind:masking:leak:{day}")) == 1
    assert not judge_health.backed_off(cloud=True)

    assert score_span(project, span).eval_attributes["score"] == 1.0


@pytest.mark.covers("masking-residual-fallback")
@pytest.mark.django_db(transaction=True)
def test_a_trajectory_that_cannot_be_masked_marks_its_root_and_goes_local(
    fake_llm, redis_cache, settings
):
    settings.MASKING_TIMEOUT_SECONDS = 0
    project = approved()
    root = trace_of(project)

    with pytest.raises(evals.MaskedFallbackError):
        async_to_sync(trajectory.score_trace)(project.pk, root.trace_id)

    root.refresh_from_db()
    assert root.eval_attributes["masking_local"] == "timeout"
    assert "trajectory" not in root.eval_attributes
    assert root.eval_attributes.get("trajectory_attempts", 0) == 0
    assert "trajectory_claimed_at" not in root.eval_attributes
    assert [r for r in fake_llm.requests if r.model == CLOUD] == []

    assert async_to_sync(trajectory.score_trace)(project.pk, root.trace_id)
    assert {request.model for request in fake_llm.requests} == {LOCAL}


@pytest.mark.covers("masking-residual-fallback")
@pytest.mark.django_db(transaction=True)
def test_a_plan_drift_step_that_cannot_be_masked_is_marked_and_judged_locally_next(
    fake_llm, redis_cache, settings
):
    settings.MASKING_TIMEOUT_SECONDS = 0
    project = approved()
    root = trace_of(project)
    SignalSubscription.objects.create(project=project, signal_type="plan_drift", enabled=True)

    async_to_sync(signals.SignalRunner(project.pk).run)()
    step = Span.objects.get(trace_id=root.trace_id, parent_span_id=root.span_id)
    assert step.eval_attributes["masking_local"] == "timeout"
    assert "drift" not in step.eval_attributes
    assert step.eval_attributes.get("drift_attempts", 0) == 0

    settings.MASKING_TIMEOUT_SECONDS = 5
    async_to_sync(signals.SignalRunner(project.pk).run)()
    step.refresh_from_db()
    assert "drift" in step.eval_attributes
    assert {request.model for request in fake_llm.requests} == {LOCAL}


@pytest.mark.covers("masking-residual-fallback")
@pytest.mark.django_db(transaction=True)
def test_an_explanation_the_gateway_refuses_is_counted_and_leaves_the_reason_empty(
    fake_llm, redis_cache
):
    fake_llm.verdict = lambda state: "no"
    fake_llm.fail(lambda r: r.schema_name != "Verdict", 400, "blocked by guardrail presidio")
    span = score_span(approved())

    assert span.eval_attributes["score"] == 0.0
    assert next(iter(span.eval_attributes["metrics"].values()))["reason"] == ""
    assert "masking_local" not in span.eval_attributes
    assert counter(redis_cache, "guardrail") == 1


@pytest.mark.covers("masking-route")
@pytest.mark.django_db(transaction=True)
def test_a_marked_span_stays_on_the_local_judge_even_when_its_project_is_approved(fake_llm):
    project = approved()
    span = span_of(project, eval_attributes={"masking_local": "timeout"})

    span = score_span(project, span)

    assert span.eval_attributes["score"] == 1.0
    assert {request.model for request in fake_llm.requests} == {LOCAL}


@pytest.mark.covers("masking-route")
@pytest.mark.django_db
def test_admission_sends_approved_work_to_the_masking_task_and_marked_work_to_the_local_one():
    project = approved()
    old = timezone.now() - timedelta(hours=1)
    plain, marked = span_of(project), span_of(project, eval_attributes={"masking_local": "x"})
    for span in (plain, marked):
        Span.objects.filter(pk=span.pk).update(received_at=old)
    admission.enqueue(project.pk, "span", [plain.span_id, marked.span_id])
    sent = {"local": [], "masked": []}

    def dispatch():
        admission.dispatch(
            lambda work_id, token: sent["local"].append(work_id),
            lambda work_id, token: sent["masked"].append(work_id),
        )
        LiveScoringWork.objects.filter(state="queued").update(state="done")

    for _ in range(2):
        dispatch()

    keys = {
        name: [LiveScoringWork.objects.get(pk=i).key for i in ids] for name, ids in sent.items()
    }
    assert keys == {"local": [marked.span_id], "masked": [plain.span_id]}


@pytest.mark.covers("masking-route")
def test_the_masking_task_runs_on_its_own_queue(settings):
    routes = settings.CELERY_TASK_ROUTES
    assert routes[innermind.execute_masked_scoring.name] == {"queue": "masking"}
    assert routes[innermind.execute_live_scoring.name] == {"queue": "io_traces"}
    assert innermind.execute_masked_scoring.soft_time_limit == 600
    assert innermind.execute_masked_scoring.time_limit == 660
