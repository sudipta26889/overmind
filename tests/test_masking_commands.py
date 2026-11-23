from __future__ import annotations

import io
import json
import os

import pytest
import redis
from django.core.management import CommandError, call_command
from fakes import masking as fake_masking

from overbae.celery import app
from overbae.core.model_registry import ROLE_CHAINS, Role
from overbae.tasks import masking_commands

pytestmark = [pytest.mark.covers("masking-detectors"), pytest.mark.django_db]

_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw")
REDIS_URL = (
    f"{os.environ.get('TEST_CACHE_REDIS', 'redis://localhost:6379')}/{15 - int(_WORKER or 0)}"
)
NAME = "Ravi Kumar"


@pytest.fixture(autouse=True)
def _worker(monkeypatch, settings, fake_llm, tmp_path):
    """The masking worker, run in-process: Celery's eager mode stands in for the broker."""
    monkeypatch.setitem(ROLE_CHAINS, Role.CLOUD_JUDGE, ("cloud-judge-masked",))
    fake_masking.enable_remote(settings, fake_llm, tmp_path, names=[NAME])
    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
            "KEY_PREFIX": "overmind",
        }
    }
    client = redis.Redis.from_url(REDIS_URL)
    client.flushdb()
    eager = app.conf.task_always_eager
    app.conf.task_always_eager = True
    yield
    app.conf.task_always_eager = eager
    client.flushdb()


def fixture_row(serial: int, domain: str = "health") -> dict:
    return {
        "id": f"en-{domain}-{serial:03d}",
        "lang": "en",
        "domain": domain,
        "state": {
            "intent": "Summarise the visit",
            "expectation": f"The plan names the medicine ({serial}).",
            "input": f"Patient {NAME} on 9876543210 has diabetes.",
            "output": "Continue metformin.",
        },
        "planted": [
            {"type": "PERSON", "value": NAME, "path": "/input"},
            {"type": "PHONE", "value": "9876543210", "path": "/input"},
        ],
    }


def fixtures(tmp_path, rows):
    directory = tmp_path / "fx"
    directory.mkdir()
    (directory / "en.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / "._en.jsonl").write_bytes(b"\x00\x05 not json")
    return directory


def run(*args):
    out, err = io.StringIO(), io.StringIO()
    code = 0
    try:
        call_command("masking_run", *args, stdout=out, stderr=err)
    except SystemExit as stop:
        code = stop.code
    return out.getvalue(), err.getvalue(), code


def test_the_report_is_enqueued_run_by_the_worker_and_printed(tmp_path):
    directory = fixtures(tmp_path, [fixture_row(i) for i in range(2)])

    out, _, code = run("report", "--", "--fixtures", str(directory))

    assert "fixtures 2" in out
    assert "recall / precision by type and language" in out
    assert "latency p50" in out
    assert code == 1  # one language only, so most bars have nothing planted: the report fails


def test_kappa_is_enqueued_prints_its_line_and_writes_the_verdict_files_here(fake_llm, tmp_path):
    directory = fixtures(tmp_path, [fixture_row(i) for i in range(2)])
    out_file = tmp_path / "mine" / "verdicts.jsonl"

    out, _, code = run(
        "kappa", "--", "--fixtures", str(directory), "--out", str(out_file), "--mode", "same-model"
    )

    assert "mode: same-model" in out
    assert "pairs 2, skipped 0" in out
    assert "kappa" in out
    assert code == 1  # under the 60-pair floor
    rows = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert [set(r) for r in rows] == [{"id", "masked_verdict", "local_verdict"}] * 2
    assert (tmp_path / "mine" / "verdicts.jsonl.skipped").is_file()
    assert NAME not in out_file.read_text()  # ids and verdicts only
    cloud = [r for r in fake_llm.requests if r.model == "overmind/cloud-judge-masked"]
    assert cloud and all("9876543210" not in r.text for r in cloud)


def test_no_mask_report_runs_from_verdicts_sent_with_the_request(tmp_path):
    directory = fixtures(tmp_path, [fixture_row(i) for i in range(2)])
    verdicts = tmp_path / "v.jsonl"
    verdicts.write_text(
        "".join(
            json.dumps(
                {"id": f"en-health-{i:03d}", "masked_verdict": "yes", "local_verdict": "yes"}
            )
            + "\n"
            for i in range(2)
        )
    )

    out, _, code = run(
        "report", "--", "--fixtures", str(directory), "--verdicts", str(verdicts), "--no-mask"
    )

    assert "kappa 1.00 on 2 pairs (bar 0.6): fail" in out
    assert code == 1


def test_it_refuses_where_there_is_no_masking_worker_or_no_fixtures(settings, tmp_path):
    directory = fixtures(tmp_path, [fixture_row(0)])
    settings.MASKING_QUEUE_ENABLED = False
    with pytest.raises(CommandError, match="one-off container"):
        call_command("masking_run", "report", "--", "--fixtures", str(directory))
    settings.MASKING_QUEUE_ENABLED = True
    with pytest.raises(CommandError, match="--fixtures"):
        call_command("masking_run", "report", "--")


def test_the_task_runs_only_the_two_masking_commands_and_only_on_a_masking_worker(settings):
    with pytest.raises(ValueError, match="not a masking command"):
        masking_commands.run_command(command="migrate", args=[], fixtures={}, verdicts=None)
    settings.MASKING_ENABLED = False
    with pytest.raises(RuntimeError, match="masking worker"):
        masking_commands.run_command(command="masking_report", args=[], fixtures={}, verdicts=None)


def test_the_task_is_routed_to_the_masking_queue():
    from django.conf import settings

    route = settings.CELERY_TASK_ROUTES["overbae.tasks.masking_commands.run_command"]
    assert route == {"queue": "masking"}
