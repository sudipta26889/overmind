from __future__ import annotations

import io
import os
import threading
import uuid
from datetime import timedelta

import pytest
import redis
from asgiref.sync import async_to_sync
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection, transaction
from django.utils import timezone
from factories import auth_client, make_project, make_span, make_user
from fakes import masking as fake_masking

from overbae.core.model_registry import ROLE_CHAINS, Role, check_cloud_judge, check_local_judge
from overbae.innermind import admission, evals, judge_health, signals, span_processor, trajectory
from overbae.innermind.evals import Judge, JudgeUnavailableError, judge_route, route_for
from overbae.models import (
    LiveScoringWork,
    Project,
    ProjectCloudJudgingChange,
    SignalSubscription,
    Span,
)

pytestmark = pytest.mark.covers("cloud-judging-routing")

# Each xdist worker gets its own Redis database, so concurrent tests never share keys.
_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw")
REDIS_URL = (
    f"{os.environ.get('TEST_CACHE_REDIS', 'redis://localhost:6379')}/{15 - int(_WORKER or 0)}"
)
CLOUD = "overmind/cloud-judge-masked"
LOCAL = "local/judge"
EXPECTATION = "The answer addresses the question."


@pytest.fixture(autouse=True)
def _cloud_judge(monkeypatch, settings, fake_llm, tmp_path, redis_cache):
    monkeypatch.setitem(ROLE_CHAINS, Role.CLOUD_JUDGE, ("cloud-judge-masked",))
    fake_masking.enable_remote(settings, fake_llm, tmp_path)
    # The local judge samples almost nothing; a certified project is judged in full anyway.
    settings.TRACE_SCORING_SAMPLE_RATE = 1


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
    keys = (judge_health.KEY, judge_health.CLOUD_KEY)
    client.delete(*keys)
    yield client
    client.delete(*keys)


def calls_today(client) -> int:
    return int(client.get(f"overmind:judge:calls:{judge_health.now().date().isoformat()}") or 0)


def certify(project):
    Project.objects.filter(pk=project.pk).update(cloud_judging=True)


def span_of(project, **fields):
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


def trace_of(project):
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


def models(fake_llm) -> list[str]:
    return [request.model for request in fake_llm.requests]


def score_span(project):
    span = span_of(project)
    async_to_sync(span_processor.score_project_spans)(project.pk, [span])
    span.refresh_from_db()
    return span


def score_trajectory(project):
    root = trace_of(project)
    assert async_to_sync(trajectory.score_trace)(project.pk, root.trace_id)


def score_plan_drift(project):
    root = make_span(project, trace_id="drift", eval_attributes={"score": 1.0})
    step = make_span(
        project,
        trace_id="drift",
        parent_span_id=root.span_id,
        eval_attributes={"score": 0.9, "intent": "Resolve the order"},
        attributes={"input": "q", "output": "a"},
    )
    SignalSubscription.objects.create(project=project, signal_type="plan_drift", enabled=True)
    async_to_sync(signals.SignalRunner(project.pk).run)()
    step.refresh_from_db()
    assert "drift" in step.eval_attributes


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("path", [score_span, score_trajectory, score_plan_drift])
def test_a_project_without_approval_is_never_judged_remotely(fake_llm, path):
    # A "no" verdict is what asks a remote judge for an explanation; here none may be sent.
    fake_llm.verdict = lambda state: "no"

    path(make_project())

    assert models(fake_llm) and set(models(fake_llm)) == {LOCAL}
    assert len(fake_llm.requests) == len(fake_llm.judgments())
    assert all("user" not in request.body for request in fake_llm.requests)


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("path", [score_trajectory, score_plan_drift])
def test_a_certified_trajectory_and_plan_drift_are_judged_remotely_with_the_audit_user(
    fake_llm, path
):
    project = make_project()
    certify(project)

    path(project)

    assert models(fake_llm) and set(models(fake_llm)) == {CLOUD}
    assert {request.body["user"] for request in fake_llm.requests} == {str(project.pk)}


@pytest.mark.django_db(transaction=True)
def test_a_span_scored_for_a_project_that_has_no_row_is_judged_locally(fake_llm):
    """A span cannot exist without its project row (a foreign key), so a missing row is
    reached through the project id the work item carries; a trajectory or signal run for a
    missing project finds no spans at all and is covered by ``route_for`` below."""
    project = make_project()
    certify(project)
    span = span_of(project)

    async_to_sync(span_processor.score_project_spans)(uuid.uuid4(), [span])

    assert models(fake_llm) and set(models(fake_llm)) == {LOCAL}


@pytest.mark.django_db
def test_any_doubt_about_the_project_keeps_the_work_local():
    project = make_project()
    assert route_for(project.pk).local
    assert route_for(uuid.uuid4()).local
    assert route_for("not-a-uuid").local
    certify(project)
    remote = route_for(project.pk)
    assert (remote.local, remote.model, remote.sample_rate, remote.explain) == (
        False,
        "cloud-judge-masked",
        1.0,
        True,
    )


@pytest.mark.django_db
def test_a_certified_project_stays_local_while_no_remote_judge_is_configured(monkeypatch):
    project = make_project()
    certify(project)
    monkeypatch.setitem(ROLE_CHAINS, Role.CLOUD_JUDGE, ())

    assert route_for(project.pk).local


@pytest.mark.django_db(transaction=True)
def test_a_remote_route_whose_project_is_gone_sends_nothing(fake_llm):
    remote = evals.JudgeRoute(
        model="cloud-judge-masked", sample_rate=1.0, explain=True, local=False
    )

    with pytest.raises(JudgeUnavailableError):
        async_to_sync(Judge(str(uuid.uuid4()), route=remote).judge)({"output": "a"}, EXPECTATION)

    assert fake_llm.requests == []


@pytest.mark.django_db(transaction=True)
def test_a_certified_project_is_judged_in_full_on_the_remote_model_with_an_explanation(
    fake_llm, settings, redis_cache
):
    settings.TRACE_SCORING_SAMPLE_RATE = 0
    fake_llm.verdict = lambda state: "no"
    fake_llm.on(lambda r: r.schema_name != "Verdict", "It never addresses the question.")
    project = make_project()
    certify(project)
    calls_before = calls_today(redis_cache)

    span = score_span(project)

    assert span.eval_attributes["score"] == 0.0
    metric = next(iter(span.eval_attributes["metrics"].values()))
    assert metric["reason"] == "It never addresses the question."
    assert models(fake_llm) == [CLOUD, CLOUD]
    assert [request.body["user"] for request in fake_llm.requests] == [str(project.pk)] * 2
    assert calls_today(redis_cache) == calls_before


@pytest.mark.django_db(transaction=True)
def test_a_revocation_between_a_traces_spans_and_its_trajectory_ends_remote_calls(fake_llm):
    project = make_project()
    certify(project)
    trace_id = uuid.uuid4().hex
    root = make_span(
        project,
        trace_id=trace_id,
        attributes={
            "input": "Where is my order?",
            "output": "I like turtles.",
            "overmind.expectations": [EXPECTATION],
            "intent": "Resolve the order",
        },
    )
    child = make_span(project, trace_id=trace_id, parent_span_id=root.span_id)
    async_to_sync(span_processor.score_project_spans)(project.pk, [root, child])
    assert set(models(fake_llm)) == {CLOUD}
    sent = len(fake_llm.requests)

    Project.objects.filter(pk=project.pk).update(cloud_judging=False)
    assert async_to_sync(trajectory.score_trace)(project.pk, trace_id)

    after = models(fake_llm)[sent:]
    assert after and set(after) == {LOCAL}


@pytest.mark.django_db(transaction=True)
def test_a_flip_during_an_item_sends_no_remote_request_and_releases_the_span(fake_llm):
    project = make_project()
    certify(project)
    span = span_of(project)
    remote = async_to_sync(judge_route)(project.pk)
    Project.objects.filter(pk=project.pk).update(cloud_judging=False)

    with pytest.raises(JudgeUnavailableError):
        async_to_sync(Judge(str(project.pk), route=remote).judge)({"output": "a"}, EXPECTATION)

    assert fake_llm.requests == []
    async_to_sync(span_processor.score_project_spans)(project.pk, [span])
    span.refresh_from_db()
    assert span.eval_attributes["score"] == 1.0
    assert set(models(fake_llm)) == {LOCAL}


@pytest.mark.django_db(transaction=True)
def test_a_remote_5xx_backs_off_the_remote_judge_and_never_the_local_one(fake_llm, redis_cache):
    project = make_project()
    certify(project)
    fake_llm.fail(lambda r: r.model == CLOUD, 503)
    span = span_of(project)

    async_to_sync(span_processor.score_project_spans)(project.pk, [span])

    span.refresh_from_db()
    assert "score" not in span.eval_attributes
    assert span.eval_attributes.get("attempts", 0) == 0
    assert span.eval_claimed_at is None
    assert judge_health.backed_off(cloud=True)
    assert not judge_health.backed_off()
    assert 0 < redis_cache.pttl(judge_health.CLOUD_KEY) <= 60_000

    other = make_project()
    assert score_span(other).eval_attributes["score"] == 1.0


@pytest.mark.django_db
def test_admission_holds_a_local_project_not_a_remote_one_while_the_local_judge_is_busy(
    redis_cache,
):
    local, remote = make_project(), make_project()
    certify(remote)
    old = timezone.now() - timedelta(hours=1)
    for project in (local, remote):
        span = make_span(project, trace_id=uuid.uuid4().hex)
        Span.objects.filter(pk=span.pk).update(received_at=old)
        admission.enqueue(project.pk, "span", [span.span_id])
    judge_health.mark_unavailable()

    published = []
    admission.dispatch(lambda work_id, token: published.append(work_id))

    admitted = {LiveScoringWork.objects.get(pk=work_id).project_id for work_id in published}
    assert admitted == {remote.pk}


@pytest.mark.django_db
def test_admission_does_not_serialise_a_remote_project_behind_the_local_judge():
    first, second, remote = make_project(), make_project(), make_project()
    certify(remote)
    for project in (first, second, remote):
        admission.enqueue(
            project.pk, "span", [make_span(project, trace_id=uuid.uuid4().hex).span_id]
        )

    published = []
    admission.dispatch(lambda work_id, token: published.append(work_id))

    admitted = {LiveScoringWork.objects.get(pk=work_id).project_id for work_id in published}
    assert remote.pk in admitted
    assert len(admitted & {first.pk, second.pk}) == 1


def test_the_remote_judge_may_not_be_a_local_model():
    check_cloud_judge(("cloud-judge-masked",))
    check_cloud_judge(())
    for local in ("local-judge", "flint"):
        with pytest.raises(ValueError, match="local model"):
            check_cloud_judge((local,))


def command(project, *args, slug=None):
    out = io.StringIO()
    extra = ["--slug", slug] if slug else []
    call_command(
        "set_project_cloud_judging",
        "--project-id",
        str(project if isinstance(project, str) else project.pk),
        *extra,
        *args,
        stdout=out,
    )
    return out.getvalue()


def revoke(project, value="off"):
    return command(project, "--cloud-judging", value, "--by", "Guardian", "--reason", "Audit 12")


@pytest.mark.django_db
def test_the_command_changes_the_flag_once_prints_the_project_and_logs_it():
    project = make_project(name="Studio")

    first = revoke(project, "on")
    assert first.startswith("changed") and str(project.pk) in first
    assert "Studio" in first and project.slug in first
    assert revoke(project, "on").startswith("unchanged")

    project.refresh_from_db()
    assert project.cloud_judging is True
    change = ProjectCloudJudgingChange.objects.get()
    assert (change.project_id, change.project_id_copy, change.project_slug) == (
        project.pk,
        project.pk,
        project.slug,
    )
    assert (change.old_cloud_judging, change.new_cloud_judging) == (False, True)
    assert (change.changed_by, change.reason) == ("Guardian", "Audit 12")

    revoke(project, "off")
    assert list(ProjectCloudJudgingChange.objects.values_list("new_cloud_judging", flat=True)) == [
        True,
        False,
    ]


@pytest.mark.django_db
def test_the_command_refuses_an_unknown_project_a_bad_id_or_a_slug_that_differs():
    project = make_project()
    with pytest.raises(CommandError, match="No project"):
        revoke(str(uuid.uuid4()), "on")
    with pytest.raises(CommandError, match="not a project id"):
        revoke("not-a-uuid", "on")
    with pytest.raises(CommandError, match="slug"):
        command(project, "--cloud-judging", "on", "--by", "G", "--reason", "x", slug="other")

    project.refresh_from_db()
    assert project.cloud_judging is False
    assert not ProjectCloudJudgingChange.objects.exists()


@pytest.mark.django_db
def test_the_change_log_survives_the_deletion_of_its_project():
    project = make_project()
    revoke(project, "on")

    Project.objects.filter(pk=project.pk).delete()

    change = ProjectCloudJudgingChange.objects.get()
    assert change.project_id is None
    assert (change.project_id_copy, change.project_slug) == (project.pk, project.slug)


@pytest.mark.django_db
def test_a_stale_save_cannot_undo_a_revocation():
    project = make_project()
    certify(project)
    stale = Project.objects.get(pk=project.pk)
    stale.cloud_judging = True

    revoke(project, "off")
    stale.name = "Renamed"
    stale.save()

    project.refresh_from_db()
    assert project.cloud_judging is False
    assert project.name == "Renamed"


@pytest.mark.django_db
def test_the_api_shows_the_flag_and_never_writes_it():
    user = make_user()
    project = make_project(member=user)
    client = auth_client(user)

    shown = client.get(f"/api/projects/{project.pk}/")
    assert shown.status_code == 200
    assert shown.json()["cloud_judging"] is False

    changed = client.patch(f"/api/projects/{project.pk}/", {"cloud_judging": True}, format="json")
    assert changed.status_code == 200
    project.refresh_from_db()
    assert project.cloud_judging is False
    assert not ProjectCloudJudgingChange.objects.exists()


def test_the_local_judge_must_be_local():
    check_local_judge(())
    check_local_judge(("local-judge",))
    check_local_judge(("flint",))
    with pytest.raises(ValueError, match="remote model"):
        check_local_judge(("cloud-judge-masked",))


@pytest.mark.django_db(transaction=True)
def test_a_remote_judge_makes_one_attempt_so_the_flag_is_read_before_each(fake_llm):
    project = make_project()
    certify(project)
    fake_llm.fail(lambda r: r.model == CLOUD, 503)

    span = score_span(project)

    assert models(fake_llm) == [CLOUD]
    assert "score" not in span.eval_attributes
    assert span.eval_attributes.get("attempts", 0) == 0


@pytest.mark.django_db(transaction=True)
def test_a_remote_explanation_makes_one_attempt(fake_llm):
    project = make_project()
    certify(project)
    fake_llm.verdict = lambda state: "no"
    fake_llm.fail(lambda r: r.schema_name != "Verdict", 503)

    span = score_span(project)

    assert models(fake_llm) == [CLOUD, CLOUD]
    assert span.eval_attributes["score"] == 0.0


@pytest.mark.django_db(transaction=True)
def test_a_database_error_in_the_route_does_not_abort_the_callers_transaction():
    project = make_project()
    locked, release = threading.Event(), threading.Event()

    def hold():
        try:
            with transaction.atomic():
                with connection.cursor() as cursor:
                    cursor.execute("LOCK TABLE overbae_project IN ACCESS EXCLUSIVE MODE")
                locked.set()
                release.wait(10)
        finally:
            connection.close()

    holder = threading.Thread(target=hold)
    holder.start()
    assert locked.wait(10)
    try:
        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL lock_timeout = '100ms'")
            assert route_for(project.pk).local
            # The outer transaction is still usable.
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                assert cursor.fetchone() == (1,)
    finally:
        release.set()
        holder.join()


@pytest.mark.django_db(transaction=True)
def test_the_remote_daily_limit_re_offers_certified_items_without_an_attempt(
    fake_llm, redis_cache, monkeypatch
):
    monkeypatch.setenv("CLOUD_JUDGE_DAILY_LIMIT", "2")
    day = judge_health.now().date().isoformat()
    redis_cache.delete(f"overmind:judge:cloud_calls:{day}")
    fake_llm.verdict = lambda state: "no"
    project = make_project()
    certify(project)

    first = score_span(project)
    assert first.eval_attributes["score"] == 0.0
    assert models(fake_llm) == [CLOUD, CLOUD]

    second = score_span(project)
    assert models(fake_llm) == [CLOUD, CLOUD]
    assert "score" not in second.eval_attributes
    assert second.eval_attributes.get("attempts", 0) == 0
    assert second.eval_claimed_at is None
    assert judge_health.backed_off(cloud=True) and not judge_health.backed_off()
    assert 0 < redis_cache.pttl(judge_health.CLOUD_KEY) <= 24 * 3600 * 1000
    assert int(redis_cache.get(f"overmind:judge:cloud_calls:{day}")) == 3
    redis_cache.delete(f"overmind:judge:cloud_calls:{day}")


def test_an_invalid_remote_daily_limit_fails_loudly(monkeypatch):
    for bad in ("0", "-1", "lots"):
        monkeypatch.setenv("CLOUD_JUDGE_DAILY_LIMIT", bad)
        with pytest.raises(ValueError, match="CLOUD_JUDGE_DAILY_LIMIT"):
            judge_health.cloud_daily_limit()


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    "message",
    [
        "Unknown parameter: 'user'",
        "extra inputs are not permitted",
        "bad user field",
        "Unrecognized request argument supplied: user",
    ],
)
def test_a_gateway_that_refuses_the_audit_field_re_offers_the_item_uncounted(
    fake_llm, redis_cache, message
):
    project = make_project()
    certify(project)
    fake_llm.fail(lambda r: r.model == CLOUD, 400, message)

    span = score_span(project)

    assert "score" not in span.eval_attributes
    assert span.eval_attributes.get("attempts", 0) == 0
    assert span.eval_claimed_at is None
    assert judge_health.backed_off(cloud=True)


@pytest.mark.django_db(transaction=True)
def test_any_other_bad_request_to_the_remote_judge_is_an_attempt(fake_llm, redis_cache):
    project = make_project()
    certify(project)
    fake_llm.fail(
        lambda r: r.model == CLOUD, 400, "maximum context length exceeded in the user message"
    )

    span = score_span(project)

    assert span.eval_attributes["attempts"] == 1
    assert not judge_health.backed_off(cloud=True)
