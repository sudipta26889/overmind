from __future__ import annotations

import contextlib
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


@pytest.fixture
def redis_client():
    return redis.Redis.from_url(REDIS_URL)


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
    return run_with_code(directory, out, *extra)[0]


def run_with_code(directory, out, *extra):
    """The printed text and the exit code (1 when kappa fails or has too few pairs)."""
    stdout = io.StringIO()
    if "--mode" not in extra:
        extra = (*extra, "--mode", "reference")  # the earlier tests exercise the reference mode
    try:
        call_command(
            "masking_kappa", "--fixtures", str(directory), "--out", str(out), *extra, stdout=stdout
        )
    except SystemExit as stop:
        return stdout.getvalue(), stop.code
    return stdout.getvalue(), 0


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


def test_the_reference_half_is_not_live_scoring_and_a_local_backoff_does_not_skip_fixtures(
    fake_llm, tmp_path, monkeypatch, redis_client
):
    from overbae.innermind import judge_health

    monkeypatch.setenv("JUDGE_DAILY_LIMIT", "1")
    judge_health.mark_unavailable()  # held live items set this all day in production
    directory = write(tmp_path, [fixture_row(i) for i in range(3)])

    printed = run(directory, tmp_path / "v.jsonl", "--reference-model", "flint")

    assert "pairs 3, skipped 0, cloud calls 3, local calls 3" in printed
    assert [r.model for r in fake_llm.requests if r.model != CLOUD] == ["recursivesi/flint"] * 3
    day = judge_health.now().date().isoformat()
    assert int(redis_client.get(f"overmind:judge:calls:{day}") or 0) == 0  # no local cap spent


def test_an_answer_without_a_verdict_from_the_reference_is_skipped_as_no_verdict(
    fake_llm, tmp_path
):
    fake_llm.on(lambda r: r.model == "recursivesi/flint", "I cannot decide.")
    directory = write(tmp_path, [fixture_row(0)])

    printed = run(directory, tmp_path / "v.jsonl", "--reference-model", "flint")

    assert "pairs 0, skipped 1" in printed
    skipped = json.loads((tmp_path / "v.jsonl.skipped").read_text())
    assert skipped["reason"] == "local: no verdict"


def test_kappa_is_printed_and_the_exit_is_one_below_the_bar_or_under_sixty_pairs(
    fake_llm, tmp_path
):
    few, code = run_with_code(write(tmp_path, [fixture_row(0)]), tmp_path / "a.jsonl")
    assert code == 1
    assert "kappa 1.00 on 1 pairs (bar 0.6): fail" in few

    many_dir = tmp_path / "many"
    many_dir.mkdir()
    (many_dir / "en.jsonl").write_text(
        "".join(json.dumps(fixture_row(i)) + "\n" for i in range(60))
    )
    printed, code = run_with_code(many_dir, tmp_path / "b.jsonl", "--limit", "60")
    assert code == 0
    assert "kappa 1.00 on 60 pairs (bar 0.6): ok" in printed


def test_same_model_mode_is_the_default_and_judges_full_masking_against_ids_only(
    fake_llm, tmp_path
):
    directory = write(tmp_path, [fixture_row(0)])
    stdout = io.StringIO()
    with contextlib.suppress(SystemExit):  # one pair is below the 60-pair floor
        call_command(
            "masking_kappa",
            "--fixtures",
            str(directory),
            "--out",
            str(tmp_path / "v.jsonl"),
            stdout=stdout,
        )

    printed = stdout.getvalue()
    assert f"mode: same-model, model: {CLOUD}" in printed
    assert "pairs 1, skipped 0, cloud calls 2, local calls 0" in printed
    assert {r.model for r in fake_llm.requests} == {CLOUD}
    assert len(fake_llm.requests) == 2
    full, ids_only = sorted(fake_llm.requests, key=lambda r: PLANTED[0] in r.text)

    # Full masking hides the name, the condition and every identifier.
    for value in (*PLANTED, "diabetes"):
        assert flat(value) not in flat(full.text)
    # Ids-only keeps the name and the condition as written, and hides every identifier.
    assert PLANTED[0] in ids_only.text
    assert "diabetes" in ids_only.text
    for identifier in PLANTED[1:]:
        assert flat(identifier) not in flat(ids_only.text)
    assert all(
        r.headers["authorization"] == f"Bearer {fake_masking.CLOUD_KEY}" for r in fake_llm.requests
    )
    row = json.loads((tmp_path / "v.jsonl").read_text())
    assert set(row) == {"id", "masked_verdict", "local_verdict"}


def test_a_content_blocked_refusal_on_the_ids_only_copy_is_skipped_with_the_pattern(
    fake_llm, tmp_path
):
    fake_llm.fail(
        lambda r: r.model == CLOUD and PLANTED[0] in r.text,
        400,
        "Content blocked: Aadhaar pattern detected",
    )
    directory = write(tmp_path, [fixture_row(0), fixture_row(1)])

    printed, _ = run_with_code(directory, tmp_path / "v.jsonl", "--mode", "same-model")

    assert "pairs 0, skipped 2" in printed
    reasons = {
        json.loads(line)["reason"]
        for line in (tmp_path / "v.jsonl.skipped").read_text().splitlines()
    }
    assert reasons == {"ids-only missed: Aadhaar"}


def test_the_ids_only_mask_replaces_identifiers_and_leaves_names_and_health_terms():
    from overbae.services.masking.masker import mask_ids_only

    state = {
        "input": f"{PLANTED[0]} has diabetes, phone {PLANTED[1]}, mail {PLANTED[2]}, owes ₹4500"
    }

    payload = mask_ids_only(state)

    assert PLANTED[0] in payload.text
    assert "diabetes" in payload.text
    assert "₹4500" in payload.text
    assert PLANTED[1] not in payload.text
    assert PLANTED[2] not in payload.text
    assert set(payload.mapping.values()) == {PLANTED[1], PLANTED[2]}


def test_only_the_made_up_fixtures_are_judged(fake_llm, tmp_path):
    row = fixture_row(0)
    del row["planted"]
    directory = write(tmp_path, [row])

    with pytest.raises(RuntimeError, match="made-up fixtures"):
        run(directory, tmp_path / "v.jsonl")
    assert fake_llm.requests == []
