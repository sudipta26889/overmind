from __future__ import annotations

import io
import json
import os
import re
import unicodedata

import pytest
import redis
from django.core.management import CommandError, call_command
from fakes import masking as fake_masking

from overbae.core.model_registry import ROLE_CHAINS, Role
from overbae.services.masking import report

pytestmark = [pytest.mark.covers("masking-egress"), pytest.mark.django_db]

_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw")
REDIS_URL = (
    f"{os.environ.get('TEST_CACHE_REDIS', 'redis://localhost:6379')}/{15 - int(_WORKER or 0)}"
)
CLOUD = "overmind/cloud-judge-masked"
LOCAL = "local/judge"
PLANTED = ["Ravi Kumar", "9876543210", "ravi.kumar@example.com"]


@pytest.fixture(autouse=True)
def _remote(monkeypatch, settings, fake_llm, tmp_path):
    monkeypatch.setitem(ROLE_CHAINS, Role.CLOUD_JUDGE, ("cloud-judge-masked",))
    fake_masking.enable_remote(settings, fake_llm, tmp_path, names=["Ravi Kumar"])
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


def fixture_row(serial: int, domain: str = "health") -> dict:
    return {
        "id": f"en-{domain}-{serial:03d}",
        "lang": "en",
        "domain": domain,
        "state": {
            "intent": "Summarise the visit",
            "expectation": "The plan names the medicine.",
            "input": f"Patient {PLANTED[0]} on {PLANTED[1]}, mail {PLANTED[2]}, has diabetes.",
            "output": "Continue metformin and return in a month.",
        },
        "planted": [{"type": "PERSON", "value": PLANTED[0], "path": "/input"}],
    }


def write(tmp_path, rows):
    directory = tmp_path / "fx"
    directory.mkdir()
    (directory / "en.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return directory


def run(directory, out, *extra):
    stdout = io.StringIO()
    call_command(
        "masking_kappa", "--fixtures", str(directory), "--out", str(out), *extra, stdout=stdout
    )
    return stdout.getvalue()


def flat(text: str) -> str:
    return re.sub(r"[\s\-,.]", "", unicodedata.normalize("NFKC", text).casefold())


def test_each_fixture_is_judged_masked_by_the_cloud_and_unmasked_by_the_local_judge(
    fake_llm, tmp_path
):
    directory = write(tmp_path, [fixture_row(i) for i in range(3)] + [fixture_row(9, "general")])
    out = tmp_path / "verdicts.jsonl"

    printed = run(directory, out)

    assert "pairs 3, skipped 0, cloud calls 3, local calls 3" in printed
    cloud = [r for r in fake_llm.requests if r.model == CLOUD]
    local = [r for r in fake_llm.requests if r.model == LOCAL]
    assert (len(cloud), len(local)) == (3, 3)
    for request in cloud:
        for value in PLANTED:
            assert flat(value) not in flat(request.text + request.system)
        assert request.headers["authorization"] == f"Bearer {fake_masking.CLOUD_KEY}"
    assert PLANTED[0] in local[0].text  # the local judge sees the real state

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [set(r) for r in rows] == [{"id", "masked_verdict", "local_verdict"}] * 3
    assert all(PLANTED[0] not in line for line in out.read_text().splitlines())
    # The file is exactly what masking_report --verdicts reads.
    outcomes = [report.Outcome(fixture_row(i), 0.0) for i in range(3)]
    value, pairs = report.verdict_kappa(outcomes, {r["id"]: r for r in rows})
    assert pairs == 3
    assert value is not None


def test_a_fixture_masking_cannot_handle_is_skipped_with_its_reason_and_never_sent(
    fake_llm, tmp_path, settings
):
    settings.MASKING_MAX_BYTES = 90
    directory = write(tmp_path, [fixture_row(0)])
    out = tmp_path / "verdicts.jsonl"

    printed = run(directory, out)

    assert "pairs 0, skipped 1, cloud calls 0, local calls 0" in printed
    assert fake_llm.requests == []
    assert out.read_text() == ""
    skipped = json.loads((tmp_path / "verdicts.jsonl.skipped").read_text())
    assert skipped == {"id": "en-health-000", "reason": "masking: too_large"}


def test_it_refuses_without_the_masking_worker_a_remote_judge_or_too_large_a_limit(
    tmp_path, settings, monkeypatch
):
    directory = write(tmp_path, [fixture_row(0)])
    out = tmp_path / "v.jsonl"

    with pytest.raises(CommandError, match="150"):
        run(directory, out, "--limit", "151")
    settings.MASKING_ENABLED = False
    with pytest.raises(CommandError, match="masking worker"):
        run(directory, out)
    settings.MASKING_ENABLED = True
    monkeypatch.setitem(ROLE_CHAINS, Role.CLOUD_JUDGE, ())
    with pytest.raises(CommandError, match="LLM_CLOUD_JUDGE_CHAIN"):
        run(directory, out)


def test_the_limit_picks_fixtures_spread_across_the_set(fake_llm, tmp_path):
    directory = write(tmp_path, [fixture_row(i) for i in range(10)])

    run(directory, tmp_path / "v.jsonl", "--limit", "2")

    ids = [json.loads(line)["id"] for line in (tmp_path / "v.jsonl").read_text().splitlines()]
    assert ids == ["en-health-000", "en-health-005"]


def test_the_reference_model_is_named_and_must_be_local(fake_llm, tmp_path):
    directory = write(tmp_path, [fixture_row(0)])
    out = tmp_path / "v.jsonl"

    printed = run(directory, out, "--reference-model", "flint")

    assert "reference model: flint" in printed
    assert [r.model for r in fake_llm.requests if r.model != CLOUD] == ["recursivesi/flint"]
    assert "reference model: local-judge" in run(directory, out)

    with pytest.raises(CommandError, match="local"):
        run(directory, out, "--reference-model", "cloud-judge-masked")
    with pytest.raises(CommandError, match="local"):
        run(directory, out, "--reference-model", "gpt-6-luna")
