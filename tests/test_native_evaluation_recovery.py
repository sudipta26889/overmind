import hashlib
import json
import math
from datetime import timedelta
from unittest.mock import patch

import pytest
from conftest import frozen_dataset
from django.utils import timezone

from overbae.models import Project
from overbae.services import (
    native_evaluation,
)
from overbae.tasks.native_evaluation import advance_plan

pytestmark = [pytest.mark.django_db, pytest.mark.covers("recovery")]


@pytest.fixture
def evaluation():
    project = Project.objects.create(name="Recovery", slug="automatic-recovery")
    cell = frozen_dataset(
        project,
        [
            {
                "input": {
                    "decision": {
                        "state": f"Evidence {i}",
                        "question": "Choose",
                        "kind": "choice",
                        "options": ["a", "b"],
                    }
                },
                "expected_output": {"probabilities": [1, 0]},
            }
            for i in range(2)
        ],
        contract="eval",
    ).active_cell
    plan = native_evaluation.create_plan(
        project,
        name="Recover evaluation",
        request_key="recover",
        final_cell=cell,
        participants=[
            {"key": "base", "name": "Base", "kind": "foundation", "model": "Qwen/Qwen3.5-4B"}
        ],
        baseline="base",
        bootstrap_samples=2,
        inference={"concurrency": 1},
    )
    with patch.object(advance_plan, "delay"):
        native_evaluation.launch(plan)
    native_evaluation.advance(plan.pk, stage="verify_inputs")
    plan.refresh_from_db()
    return plan


def retry_due(plan, stage):
    plan.refresh_from_db()
    plan.calls[stage]["retry_at"] = (timezone.now() - timedelta(seconds=1)).isoformat()
    plan.save()


def test_native_collection_recovers_same_call_without_new_submission(evaluation, fake_modal):
    plan = evaluation
    plan.config["participants"] = [
        {"key": "base", "name": "Base", "kind": "foundation", "model": "fixture"}
    ]
    plan.config["baseline"] = "base"
    plan.config["runtime"] = {"app": "fixture", "environment": "test"}
    plan.calls["final_prepare"] = {"state": "running", "id": "fc-retained"}
    plan.save()
    remote = fake_modal.adopt("fc-retained", unreachable=ConnectionError("observer offline"))
    native_evaluation.advance(plan.pk, stage="final_prepare")
    plan.refresh_from_db()
    assert plan.state == "running"
    assert plan.calls["final_prepare"]["state"] == "retrying"
    retry_due(plan, "final_prepare")
    remote.unreachable = None
    native_evaluation.advance(plan.pk, stage="final_prepare")
    plan.refresh_from_db()
    assert plan.calls["final_prepare"]["state"] == "running"
    assert plan.calls["final_prepare"]["id"] == "fc-retained"
    assert fake_modal.spawns() == []


@pytest.mark.parametrize("progress_when", ["none", "before_failure", "after_failure"])
def test_exhausted_native_collection_scores_only_verified_progress(
    evaluation, fake_modal, progress_when
):
    plan = evaluation
    plan.config["participants"] = [
        {"key": "base", "name": "Base", "kind": "foundation", "model": "fixture"}
    ]
    plan.config["baseline"] = "base"
    plan.config["runtime"] = {"app": "fixture", "environment": "test", "training": "pinned"}
    plan.calls["final_prepare"] = {
        "state": "completed",
        "receipt": {
            "artifact_identity": "fixture",
            "input_sha256": "inputs",
            "ready_decisions": 2,
        },
    }
    plan.calls["final_base"] = {"state": "running", "id": "fc-retained"}
    plan.save()
    remote = fake_modal.adopt("fc-retained")
    if progress_when != "none":
        row = next(
            native_evaluation.records(native_evaluation.directory(plan, "final") / "inputs.jsonl")
        )
        body = (
            json.dumps(
                {
                    "key": row["key"],
                    "input_sha256": row["input_sha256"],
                    "kind": "choice",
                    "model_identity": "fixture",
                    "probabilities": [0.9, 0.1],
                    "log_probabilities": [math.log(0.9), math.log(0.1)],
                }
            ).encode()
            + b"\n"
        )
        root = f"/decision-evaluations/{plan.pk}-final/progress/fc-retained"
        fake_modal.files[root + "/0.jsonl"] = body
        fake_modal.files[root + "/manifest.json"] = json.dumps(
            {
                "call_id": remote.object_id,
                "artifact_identity": "fixture",
                "input_sha256": "inputs",
                "runtime_fingerprint": "pinned",
                "expected": 2,
                "completed": 1,
                "chunks": [{"sha256": hashlib.sha256(body).hexdigest(), "rows": 1}],
            }
        ).encode()
        if progress_when == "before_failure":
            native_evaluation.advance(plan.pk, stage="final_base")
    remote.unreachable = ConnectionError("observer offline")
    for _ in range(3):
        native_evaluation.advance(plan.pk, stage="final_base")
        retry_due(plan, "final_base")
    native_evaluation.advance(plan.pk, stage="score")
    plan.refresh_from_db()
    assert plan.state == "completed", plan.error
    coverage = plan.results["completion"]["coverage"]["final"]["base"]
    assert coverage["scored"] == int(progress_when != "none")
    assert coverage["missing_predictions"] == 2 - int(progress_when != "none")
    assert plan.results["completion"]["issues"][0]["reason"] == "Automatic recovery exhausted."
    assert fake_modal.spawns() == []


def test_exhausted_native_preparation_does_not_submit_dependent_work(evaluation, fake_modal):
    plan = evaluation
    plan.config["participants"] = [
        {"key": "base", "name": "Base", "kind": "foundation", "model": "fixture"}
    ]
    plan.config["baseline"] = "base"
    plan.calls["final_prepare"] = {"state": "running", "id": "fc-retained"}
    plan.save()
    fake_modal.adopt("fc-retained", unreachable=ConnectionError("offline"))
    for _ in range(3):
        native_evaluation.advance(plan.pk, stage="final_prepare")
        retry_due(plan, "final_prepare")
    native_evaluation.advance(plan.pk, stage="final_base")
    native_evaluation.advance(plan.pk, stage="score")
    plan.refresh_from_db()
    assert plan.state == "completed", plan.error
    assert plan.results["completion"]["coverage"]["final"]["base"]["scored"] == 0
    assert fake_modal.spawns() == []


def test_invalid_local_input_stays_failed(evaluation):
    source = native_evaluation.directory(evaluation, "final") / "inputs.jsonl"
    source.write_text("changed inputs")
    evaluation.calls.pop("verify_inputs")
    evaluation.save()
    native_evaluation.advance(evaluation.pk, stage="verify_inputs")
    evaluation.refresh_from_db()
    assert evaluation.state == "failed"
    assert evaluation.calls["verify_inputs"]["state"] == "failed"


def test_preparation_recovery_preserves_the_explicit_launch_boundary(evaluation):
    plan = evaluation
    plan.state = "preparing"
    plan.calls = {
        "verify_inputs": {
            "state": "retrying",
            "retry_at": (timezone.now() + timedelta(seconds=30)).isoformat(),
        }
    }
    native_evaluation.settle(plan)
    plan.save()
    assert plan.state == "preparing"
    assert native_evaluation.ready_stages(plan) == []
    retry_due(plan, "verify_inputs")
    native_evaluation.advance(plan.pk, stage="verify_inputs")
    plan.refresh_from_db()
    assert plan.state == "prepared"
    assert native_evaluation.ready_stages(plan) == []
    assert not plan.requests.exists()
