import hashlib
from unittest.mock import patch

import pytest
from conftest import frozen_dataset

from overbae.models import NativeEvaluationPlan, Project
from overbae.services import native_evaluation
from overbae.tasks.native_evaluation import advance_plan

pytestmark = [pytest.mark.django_db, pytest.mark.covers("recovery")]

BASE = {"key": "base", "name": "Base", "kind": "foundation", "model": "Qwen/Qwen3.5-4B"}
OTHER = {"key": "other", "name": "Other", "kind": "foundation", "model": "Qwen/Qwen3.5-2B"}


def decision_rows(count, *, state="e"):
    return [
        {
            "input": {
                "decision": {
                    "state": f"{state}{i}",
                    "question": "Pick",
                    "kind": "choice",
                    "options": ["a", "b"],
                }
            },
            "expected_output": {"probabilities": [1, 0]},
            "benchmark": "fixture",
            "group": str(i),
        }
        for i in range(count)
    ]


def comparison(project, cell, participants, *, key="compare", calibration=None, **extra):
    return native_evaluation.create_plan(
        project,
        name="Compare",
        request_key=key,
        final_cell=cell,
        calibration_cell=calibration,
        participants=participants,
        baseline=participants[0]["key"],
        bootstrap_samples=5,
        **extra,
    )


def launched(plan):
    with patch.object(advance_plan, "delay"):
        native_evaluation.launch(plan)
    return plan


def run(plan, steps=24):
    for _ in range(steps):
        native_evaluation.advance(plan.pk)
    plan.refresh_from_db()
    return plan


@pytest.fixture
def workspace(decisions, tmp_path, settings):
    settings.MEDIA_ROOT = tmp_path
    project = Project.objects.create(name="Comparison", slug="comparison")
    cell = frozen_dataset(project, decision_rows(3), contract="eval").active_cell
    return project, cell


def test_a_standalone_comparison_is_idempotent_and_needs_no_training_or_calibration(workspace):
    project, cell = workspace
    plan = comparison(project, cell, [BASE, OTHER])
    assert comparison(project, cell, [BASE, OTHER]).pk == plan.pk
    assert NativeEvaluationPlan.objects.filter(project=project).count() == 1

    run(launched(plan))

    assert plan.state == "completed", plan.error
    assert plan.job_id is None and plan.calibration_cell_id is None
    assert plan.results["calibration"] is None
    assert not native_evaluation.directory(plan, "calibration").exists()
    report = plan.results["comparisons"]["other"]["raw"]["benchmarks"]["fixture"]
    assert report["expected"] == report["paired_decisions"] == 3


def test_reuse_is_explicit_and_refuses_changed_conditions(workspace, fake_modal):
    project, cell = workspace
    source = run(launched(comparison(project, cell, [BASE], key="source")))
    assert source.state == "completed", source.error
    spawned = len(fake_modal.spawns())

    target = comparison(project, cell, [BASE], key="target")
    native_evaluation.reuse_predictions(
        target, source=source, participant="base", source_participant="base"
    )
    run(launched(target))

    assert target.state == "completed", target.error
    assert target.calls["final_base"]["receipt"]["reused_from"] == str(source.pk)
    assert len(fake_modal.spawns()) == spawned

    incompatible = comparison(
        project, cell, [BASE], key="incompatible", inference={"context_length": 1024}
    )
    with pytest.raises(ValueError, match="identical sources"):
        native_evaluation.reuse_predictions(
            incompatible, source=source, participant="base", source_participant="base"
        )
    assert not incompatible.config.get("reuse")

    foreign = Project.objects.create(name="Other tenant", slug="reuse-other-tenant")
    NativeEvaluationPlan.objects.filter(pk=incompatible.pk).update(project=foreign)
    incompatible.refresh_from_db()
    with pytest.raises(ValueError, match="same project"):
        native_evaluation.reuse_predictions(
            incompatible, source=source, participant="base", source_participant="base"
        )


def serve(decisions, name, handler):
    """Replaces a worker function with ``handler(original, evaluation_id, *args, **kwargs)``."""
    app = decisions.release["app"]
    original, _ = decisions.cloud.handlers[(app, name)]
    decisions.cloud.deploy(
        app, name, lambda evaluation_id, *a, **k: handler(original, evaluation_id, *a, **k)
    )


def trimmed(decisions, original, evaluation_id, *args, keep, start=0, **kwargs):
    """The worker's prediction file, cut to ``keep`` rows from ``start``, with a receipt that
    matches it."""
    result = original(evaluation_id, *args, **kwargs)
    name = "base" if kwargs.get("base_only") else "candidate"
    path = f"/decision-evaluations/{evaluation_id}/{name}.jsonl"
    lines = decisions.cloud.files[path].splitlines(keepends=True)[start : start + keep]
    decisions.cloud.files[path] = b"".join(lines)
    return {
        **result,
        "predictions_sha256": hashlib.sha256(b"".join(lines)).hexdigest(),
        "decisions": len(lines),
    }


def test_an_unreachable_observer_retries_without_blocking_an_independent_participant(
    workspace, decisions, fake_modal
):
    project, cell = workspace

    def offline(original, evaluation_id, *args, **kwargs):
        if evaluation_id.endswith("-final"):
            raise ConnectionError("observer offline")
        return original(evaluation_id, *args, **kwargs)

    serve(decisions, "prepare", offline)
    plan = run(launched(comparison(project, cell, [BASE, OTHER])), steps=8)

    assert plan.state == "running"
    assert plan.calls["final_prepare"]["state"] == "retrying"
    assert plan.calls["final_prepare_other"]["state"] == "completed"
    assert "final_base" not in plan.calls

    spawned = len(fake_modal.spawns())
    plan.state = "paused"
    plan.save()
    native_evaluation.advance(plan.pk, stage="final_prepare")
    plan.refresh_from_db()
    assert plan.calls["final_prepare"]["state"] == "retrying"
    assert len(fake_modal.spawns()) == spawned


def test_a_baseline_missing_predictions_pairs_only_the_rows_it_has(workspace, decisions):
    project, cell = workspace

    def short(original, evaluation_id, *args, **kwargs):
        keep = 2 if evaluation_id.endswith("-final") else 3
        return trimmed(decisions, original, evaluation_id, *args, keep=keep, **kwargs)

    serve(decisions, "predict", short)
    plan = run(launched(comparison(project, cell, [BASE, OTHER])))

    assert plan.state == "completed", plan.error
    coverage = plan.results["completion"]["coverage"]["final"]
    assert coverage["base"]["scored"] == 2
    assert coverage["other"]["scored"] == 3
    report = plan.results["comparisons"]["other"]["raw"]["benchmarks"]["fixture"]
    assert report["paired_decisions"] == 2


def test_an_incomplete_calibration_does_not_block_final_raw_scores(decisions, tmp_path, settings):
    settings.MEDIA_ROOT = tmp_path
    project = Project.objects.create(name="Calibration", slug="calibration")
    final = frozen_dataset(project, decision_rows(3), contract="eval").active_cell
    calibration = frozen_dataset(project, decision_rows(3, state="c"), contract="eval").active_cell

    def empty_calibration(original, evaluation_id, *args, **kwargs):
        if "-calibration" in evaluation_id:
            return trimmed(decisions, original, evaluation_id, *args, keep=0, **kwargs)
        return original(evaluation_id, *args, **kwargs)

    serve(decisions, "predict", empty_calibration)
    plan = run(launched(comparison(project, final, [BASE, OTHER], calibration=calibration)), 40)

    assert plan.state == "completed", plan.error
    assert (
        plan.results["comparisons"]["other"]["raw"]["benchmarks"]["fixture"]["paired_decisions"]
        == 3
    )


def test_a_missing_first_prediction_does_not_skip_the_later_chunks(decisions, tmp_path, settings):
    settings.MEDIA_ROOT = tmp_path
    project = Project.objects.create(name="Chunks", slug="chunks")
    cell = frozen_dataset(project, decision_rows(130), contract="eval").active_cell

    def first_missing(original, evaluation_id, *args, **kwargs):
        return trimmed(decisions, original, evaluation_id, *args, keep=129, start=1, **kwargs)

    serve(decisions, "predict", first_missing)
    plan = run(launched(comparison(project, cell, [BASE, OTHER])))

    assert plan.state == "completed", plan.error
    coverage = plan.results["completion"]["coverage"]["final"]
    assert coverage["base"]["scored"] == coverage["other"]["scored"] == 129


def test_unfinished_external_work_is_ended_by_the_migration_and_older_runs_are_guarded(workspace):
    from importlib import import_module

    from django.apps import apps

    from overbae.models import DecisionPerformanceRun
    from overbae.services import decision_performance

    project, cell = workspace
    external = {"key": "cloud", "name": "Cloud", "kind": "external", "model": "gone"}
    stale = NativeEvaluationPlan.objects.create(
        project=project,
        name="Cloud comparison",
        state="running",
        final_cell=cell,
        request_key="cloud",
        config={"participants": [external], "baseline": "cloud"},
    )
    done = NativeEvaluationPlan.objects.create(
        project=project,
        name="Finished cloud comparison",
        state="completed",
        final_cell=cell,
        request_key="cloud-done",
        config={"participants": [external], "baseline": "cloud"},
    )
    current = comparison(project, cell, [BASE])
    legacy = DecisionPerformanceRun.objects.create(
        project=project,
        evaluation=stale,
        name="Cloud latency",
        request_key="cloud-latency",
        state="running",
        workload={"settings": {}, "qualification": {"cloud": {"role": "final"}}},
    )

    import_module("overbae.migrations.0057_end_external_participant_work").end_external_work(
        apps, None
    )

    for plan, state in ((stale, "failed"), (done, "completed"), (current, "draft")):
        plan.refresh_from_db()
        assert plan.state == state
    legacy.refresh_from_db()
    assert legacy.state == "failed"

    older = DecisionPerformanceRun.objects.create(
        project=project,
        evaluation=current,
        name="Older run",
        request_key="older",
        state="queued",
        workload={"settings": {}, "qualification": {"base": {"role": "final"}}},
    )
    decision_performance.advance(older.pk)
    older.refresh_from_db()
    assert older.state == "failed"
    assert older.error == "UnsupportedLegacyRun"
