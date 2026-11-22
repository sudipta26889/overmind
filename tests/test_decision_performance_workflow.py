import json
from unittest.mock import Mock, patch

import pytest
from conftest import frozen_dataset

from modal_shared.decision_inference import input_digest
from overbae.models import DecisionPerformanceRequest, Project
from overbae.services import decision_performance, native_evaluation
from overbae.services.decision_performance import validate_response
from overbae.services.native_evaluation import create_plan

pytestmark = [pytest.mark.django_db, pytest.mark.covers("work-once")]


def fixture():
    project = Project.objects.create(name="Performance", slug="performance")
    decision = {
        "state": "Evidence",
        "question": "Pick",
        "kind": "choice",
        "options": ["a", "b", "c"],
    }
    dataset = frozen_dataset(
        project,
        [{"input": {"decision": decision}, "expected_output": {"probabilities": [1, 0, 0]}}],
        contract="eval",
    )
    participant = {"key": "base", "name": "Base", "kind": "foundation", "model": "Qwen/Qwen3.5-4B"}
    with patch(
        "overbae.services.native_evaluation.runtime",
        return_value={"app": "offline", "environment": "test"},
    ):
        plan = create_plan(
            project,
            name="Performance",
            request_key="performance",
            participants=[participant],
            baseline="base",
            final_cell=dataset.active_cell,
        )
    return plan, participant, decision


def test_native_performance_rejects_changed_model_and_runtime():

    participant = {"kind": "trained", "quality_identity": "sealed", "quality_runtime": "runtime"}
    with pytest.raises(ValueError, match="identity"):
        validate_response(
            participant,
            [],
            {"predictions": [], "model_identity": "other", "runtime_fingerprint": "runtime"},
        )
    with pytest.raises(ValueError, match="runtime"):
        validate_response(
            participant,
            [],
            {"predictions": [], "model_identity": "sealed", "runtime_fingerprint": "other"},
        )


@pytest.mark.parametrize("failed", [False, True])
def test_native_measurement_collects_saved_calls_and_reports_observer_latency_unknown(failed):

    plan, _, decision = fixture()
    native_evaluation.seal_suite(plan, "final")
    plan.config["participants"] = [
        {"key": "base", "kind": "foundation", "model": "Qwen/Qwen3.5-4B"}
    ]
    plan.results = {
        "comparisons": {"base": {"raw": {"candidate": {"model_identity": "base-seal"}}}}
    }
    plan.calls = {
        "final_base": {"state": "completed", "receipt": {"runtime_fingerprint": "runtime"}}
    }
    (native_evaluation.directory(plan, "final") / "base.jsonl").write_text(
        json.dumps({"model_identity": "base-seal"}) + "\n"
    )
    plan.state = "completed"
    plan.save()
    run = decision_performance.create(
        plan,
        name="Native recovery",
        request_key="native-recovery",
        workload={
            "sample_size": 1,
            "repetitions": 1,
            "concurrency": 1,
            "seed": 1,
            "questions_per_request": 1,
        },
    )
    DecisionPerformanceRequest.objects.create(
        run=run, participant="base", position=0, state="submission_unknown", call_id="fc-existing"
    )
    reply = {
        "predictions": [{"input_sha256": input_digest(decision), "probabilities": [1, 0, 0]}],
        "model_identity": "base-seal",
        "runtime_fingerprint": "runtime",
        "worker_instance": "worker-1",
    }
    call = Mock(object_id="fc-existing")
    call.get.side_effect = [
        TimeoutError("observer timeout"),
        RuntimeError("worker failed") if failed else reply,
    ]
    worker = Mock()
    worker.predict.spawn.side_effect = AssertionError("never repeat a submitted call")
    with (
        patch.object(decision_performance, "endpoint", return_value=worker),
        patch.object(
            decision_performance.modal.FunctionCall, "from_id", return_value=call
        ) as lookup,
    ):
        decision_performance.advance(run.pk)
        run.refresh_from_db()
        assert run.state == "queued"
        decision_performance.advance(run.pk)
        decision_performance.advance(run.pk)
    assert lookup.call_count == 2
    worker.predict.spawn.assert_not_called()
    run.refresh_from_db()
    assert run.state == "completed"
    measurement = run.results["participants"]["base"]
    assert measurement["valid_decisions"] == (0 if failed else 1)
    assert measurement["unmeasured_latency_requests"] == (0 if failed else 1)
    assert measurement["latency_ms"]["p50"] is None
    assert measurement["marginal_recorded_usd_per_valid_decision"] is None
    assert measurement["amortized_recorded_component_usd_per_decision"] is None


def test_performance_acknowledges_before_scanning_frozen_workload():
    plan, _, _ = fixture()
    native_evaluation.seal_suite(plan, "final")
    plan.results = {
        "comparisons": {"base": {"raw": {"candidate": {"model_identity": "base-seal"}}}}
    }
    plan.calls = {
        "final_base": {"state": "completed", "receipt": {"runtime_fingerprint": "runtime"}}
    }
    (native_evaluation.directory(plan, "final") / "base.jsonl").write_text(
        json.dumps({"model_identity": "base-seal"}) + "\n"
    )
    plan.state = "completed"
    plan.save()
    with patch.object(
        native_evaluation, "seal_suite", side_effect=AssertionError("do not scan in the request")
    ):
        run = decision_performance.create(
            plan,
            name="Background workload",
            request_key="background",
            workload={
                "sample_size": 1,
                "repetitions": 1,
                "concurrency": 1,
                "seed": 2,
                "questions_per_request": 1,
            },
        )
    assert run.state == "queued"
    assert "inputs" not in run.workload
