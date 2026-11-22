from __future__ import annotations

import io
import uuid
from datetime import timedelta

import pytest
import redis
from asgiref.sync import async_to_sync
from django.core.management import call_command
from django.core.management.base import CommandError
from django.utils import timezone
from factories import auth_client, make_project, make_span, make_user

from overbae.core.model_registry import ROLE_CHAINS, Role, check_cloud_judge
from overbae.innermind import admission, evals, judge_health, signals, span_processor, trajectory
from overbae.innermind.evals import Judge, JudgeUnavailableError, judge_route, route_for
from overbae.models import LiveScoringWork, Project, ProjectPersonalDataChange, Span

pytestmark = pytest.mark.covers("llm-gateway")

REDIS_URL = "redis://localhost:6379/15"
CLOUD = "openai/gpt-6-luna"
LOCAL = "local/judge"
EXPECTATION = "The answer addresses the question."


@pytest.fixture(autouse=True)
def _cloud_judge(monkeypatch, settings):
    monkeypatch.setitem(ROLE_CHAINS, Role.CLOUD_JUDGE, ("gpt-6-luna",))
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
    Project.objects.filter(pk=project.pk).update(personal_data=False)


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
    runner = signals.SignalRunner(project.pk)
    runner.route = async_to_sync(judge_route)(project.pk)
    signal = signals.PlanDriftSignal(str(project.pk))
    signal.route = runner.route
    async_to_sync(signal.check)([root, step])


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("path", [score_span, score_trajectory, score_plan_drift])
def test_a_project_with_personal_data_is_never_judged_remotely(fake_llm, path):
    path(make_project())

    assert models(fake_llm) and set(models(fake_llm)) == {LOCAL}
    assert all("user" not in request.body for request in fake_llm.requests)


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
        "gpt-6-luna",
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
    remote = evals.JudgeRoute(model="gpt-6-luna", sample_rate=1.0, explain=True, local=False)

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
def test_a_project_certified_back_to_personal_data_stops_remote_calls_on_the_next_item(
    fake_llm,
):
    project = make_project()
    certify(project)
    root = trace_of(project)
    span_processor_route = async_to_sync(judge_route)(project.pk)
    assert not span_processor_route.local
    Project.objects.filter(pk=project.pk).update(personal_data=True)

    assert async_to_sync(trajectory.score_trace)(project.pk, root.trace_id)

    assert models(fake_llm) and set(models(fake_llm)) == {LOCAL}


@pytest.mark.django_db(transaction=True)
def test_a_flip_during_an_item_sends_no_remote_request_and_releases_the_span(fake_llm):
    project = make_project()
    certify(project)
    span = span_of(project)
    remote = async_to_sync(judge_route)(project.pk)
    Project.objects.filter(pk=project.pk).update(personal_data=True)

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
    check_cloud_judge(("gpt-6-luna",))
    check_cloud_judge(())
    for local in ("local-judge", "flint"):
        with pytest.raises(ValueError, match="local model"):
            check_cloud_judge((local,))


def run_command(*args):
    out = io.StringIO()
    call_command("set_project_personal_data", *args, stdout=out)
    return out.getvalue()


@pytest.mark.django_db
def test_the_command_changes_the_flag_once_and_logs_it():
    project = make_project()
    args = [project.slug, "--personal-data", "no", "--by", "Guardian", "--reason", "Audit 12"]

    assert run_command(*args).startswith("changed")
    assert run_command(*args).startswith("unchanged")

    project.refresh_from_db()
    assert project.personal_data is False
    change = ProjectPersonalDataChange.objects.get()
    assert (change.project_id, change.old_value, change.new_value) == (project.pk, True, False)
    assert (change.changed_by, change.reason) == ("Guardian", "Audit 12")

    run_command(project.slug, "--personal-data", "yes", "--by", "Guardian", "--reason", "Revoked")
    assert list(ProjectPersonalDataChange.objects.values_list("new_value", flat=True)) == [
        False,
        True,
    ]


@pytest.mark.django_db
def test_the_command_refuses_an_unknown_project_and_writes_nothing():
    with pytest.raises(CommandError, match="No project"):
        run_command("nope", "--personal-data", "no", "--by", "Guardian", "--reason", "x")

    assert not ProjectPersonalDataChange.objects.exists()


@pytest.mark.django_db
def test_the_api_shows_the_flag_and_never_writes_it():
    user = make_user()
    project = make_project(member=user)
    client = auth_client(user)

    shown = client.get(f"/api/projects/{project.pk}/")
    assert shown.status_code == 200
    assert shown.json()["personal_data"] is True

    changed = client.patch(f"/api/projects/{project.pk}/", {"personal_data": False}, format="json")
    assert changed.status_code == 200
    project.refresh_from_db()
    assert project.personal_data is True
    assert not ProjectPersonalDataChange.objects.exists()
