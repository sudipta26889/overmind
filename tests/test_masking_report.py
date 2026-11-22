from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from django.core.management import call_command
from fakes import masking as fake_ner

from overbae.services.masking import report

pytestmark = pytest.mark.covers("masking-detectors")

FIXTURES = Path(__file__).parent / "fixtures" / "masking"


def resolve(document, pointer: str):
    """A JSON pointer that descends into JSON-string fields."""
    node = document
    for part in [p for p in pointer.split("/") if p]:
        if isinstance(node, str):
            node = json.loads(node)
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def test_the_held_out_fixtures_use_no_term_the_lexicon_holds():
    from overbae.services.masking import lexicon

    known = lexicon.terms()
    fixtures = report.load_fixtures(FIXTURES / "unseen")

    assert len(fixtures) == 30
    for fixture in fixtures:
        for planted in fixture["planted"]:
            if planted["type"] in {"CONDITION", "MEDICATION", "TEST", "RELATION", "ORG", "ADDRESS"}:
                assert lexicon.normalise(planted["value"]).casefold() not in known, planted


def test_the_shipped_fixtures_are_well_formed_and_every_planted_value_is_in_its_state():
    fixtures = report.load_fixtures(FIXTURES) + report.load_fixtures(FIXTURES / "unseen")

    assert len(fixtures) >= 330
    assert sum(f["domain"] in ("health", "finance") for f in fixtures) >= 60
    assert {f["lang"] for f in fixtures} == set(report.LANGUAGES)
    assert len({f["id"] for f in fixtures}) == len(fixtures)
    for fixture in fixtures:
        assert fixture["planted"], fixture["id"]
        for planted in fixture["planted"]:
            where = resolve(fixture["state"], planted["path"])
            assert planted["value"] in str(where), (fixture["id"], planted)


def outcome(final: str, *, lang="en", domain="general", planted=None, fallback="", id_="x"):
    fixture = {
        "id": id_,
        "lang": lang,
        "domain": domain,
        "planted": planted or [],
        "state": {},
    }
    return report.Outcome(fixture, 0.1, fallback=fallback, final=final)


def test_a_planted_value_that_survives_is_a_leak_even_with_separators_and_case_changes():
    planted = [{"type": "PHONE", "value": "98765 43210", "path": "/input"}]
    leaked = outcome("call 98765-43210 now", planted=planted)
    clean = outcome("call <PHONE_1> now", planted=planted)

    result = report.evaluate([leaked, clean])

    assert result["recall_counts"]["PHONE/en"] == [1, 2]
    assert result["leaks"] == [("x", "PHONE", "en", "98765 43210")]


def test_a_fixture_that_fails_closed_sends_nothing_and_is_left_out_of_recall():
    planted = [{"type": "PERSON", "value": "Ravi", "path": "/input"}]

    result = report.evaluate([outcome("", planted=planted, fallback="timeout")])

    assert result["recall_counts"] == {}
    assert result["fallbacks"] == {"timeout": 1}
    assert result["fallback_rate"] == 1.0
    assert result["leaks"] == []


def test_a_leak_in_a_tool_argument_fails_the_report_whatever_the_recall():
    planted = [
        {
            "type": "PERSON",
            "value": "Ravi",
            "path": "/input/1/tool_calls/0/function/arguments/customer",
        }
    ]
    rows = [outcome("hi Ravi", planted=planted)] + [
        outcome("hi <PERSON_1>", planted=[{"type": "PERSON", "value": "Ravi", "path": "/input"}])
        for _ in range(999)
    ]

    result = report.evaluate(rows)

    assert result["bars"]["names/en"]["ok"] is True
    assert "tool_argument_leaks" in report.failing(result)


def test_kappa_compares_the_masked_and_the_local_verdicts_on_health_and_finance_only():
    rows = [
        outcome("", domain="health", id_="a"),
        outcome("", domain="finance", id_="b"),
        outcome("", domain="health", id_="c"),
        outcome("", domain="general", id_="d"),
    ]
    verdicts = {
        "a": {"masked_verdict": "yes", "local_verdict": "yes"},
        "b": {"masked_verdict": "no", "local_verdict": "no"},
        "c": {"masked_verdict": "yes", "local_verdict": "no"},
        "d": {"masked_verdict": "no", "local_verdict": "yes"},
    }

    value, pairs = report.verdict_kappa(rows, verdicts)

    assert pairs == 3
    assert round(value, 2) == 0.4
    assert report.kappa([("yes", "yes"), ("no", "no")]) == 1.0
    assert report.verdict_kappa(rows, {}) == (None, 0)


@pytest.mark.parametrize("planted_name_leaks", [False, True])
def test_the_command_exits_with_one_when_a_bar_fails(settings, tmp_path, planted_name_leaks):
    settings.MASKING_NER_BACKEND = fake_ner.SPEC
    fake_ner.NAMES[:] = [] if planted_name_leaks else ["Ravi Kumar"]
    row = {
        "id": "en-1",
        "lang": "en",
        "domain": "general",
        "state": {"output": "Ravi Kumar asked about the delivery of the parcel on 9876543210"},
        "planted": [
            {"type": "PERSON", "value": "Ravi Kumar", "path": "/output"},
            {"type": "PHONE", "value": "9876543210", "path": "/output"},
        ],
    }
    (tmp_path / "en.jsonl").write_text(json.dumps(row) + "\n")
    out = io.StringIO()

    try:
        call_command("masking_report", "--fixtures", str(tmp_path), stdout=out)
        exit_code = 0
    except SystemExit as stop:
        exit_code = stop.code

    text = out.getvalue()
    assert "recall / precision by type and language" in text
    assert "latency p50" in text
    # With one fixture most bars have nothing planted and cannot pass, so the command fails;
    # what changes is whether the name is listed as a leak.
    assert exit_code == 1
    assert ("leak en-1 PERSON" in text) is planted_name_leaks
    fake_ner.NAMES[:] = []
