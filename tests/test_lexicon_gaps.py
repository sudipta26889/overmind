from __future__ import annotations

import io
import json
import os

import pytest
import redis
from django.core.management import CommandError, call_command
from factories import make_project, make_span
from fakes import masking as fake_ner

from overbae.core.model_registry import ROLE_CHAINS, Role
from overbae.innermind import judge_health, lexicon_gaps
from overbae.models import LexiconGapReport
from overbae.tasks import innermind

pytestmark = [pytest.mark.covers("lexicon-gaps"), pytest.mark.django_db]

_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw")
REDIS_URL = (
    f"{os.environ.get('TEST_CACHE_REDIS', 'redis://localhost:6379')}/{15 - int(_WORKER or 0)}"
)
TEXT = "The patient was given an unfamiliar syrup after the visit and the fee was refunded."


@pytest.fixture(autouse=True)
def _setup(settings, redis_cache):
    settings.MASKING_NER_BACKEND = fake_ner.SPEC
    fake_ner.NAMES[:] = ["Ravi Kumar"]
    yield
    fake_ner.NAMES[:] = []


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


def spans(count=3):
    project = make_project()
    for i in range(count):
        make_span(
            project,
            trace_id=f"{i:032x}",
            attributes={"input": TEXT + f" case {i}", "output": "Noted and filed."},
        )


def reply(fake_llm, terms):
    fake_llm.on(lambda r: True, json.dumps({"terms": terms}))


def only_local(fake_llm):
    assert fake_llm.requests
    assert {r.model for r in fake_llm.requests} == {"local/judge"}


def test_the_pass_asks_the_local_judge_only_and_keeps_new_terms_with_counts(fake_llm):
    spans()
    reply(fake_llm, [{"term": "Zincovit Forte", "type": "MEDICATION", "lang": "en"}])

    gaps = lexicon_gaps.collect(3, 14)

    only_local(fake_llm)
    assert gaps.skipped == ""
    assert gaps.calls == 3
    assert gaps.terms() == [
        {"term": "zincovit forte", "type": "MEDICATION", "lang": "en", "count": 3}
    ]


@pytest.mark.parametrize(
    "term",
    [
        "diabetes",  # the lexicon already matches it
        "Ravi Kumar",  # a person
        "9876543210",  # an identifier
        "ravi@example.com",
        "a term with far too many separate words in it",
        "x" * 60,
    ],
)
def test_a_term_the_lexicon_matches_or_that_is_personal_or_too_long_is_dropped(fake_llm, term):
    spans(1)
    reply(fake_llm, [{"term": term, "type": "CONDITION", "lang": "en"}])

    assert lexicon_gaps.collect(1, 14).terms() == []


def test_a_remote_judge_model_is_refused_before_any_text_leaves(fake_llm, monkeypatch):
    spans(1)
    monkeypatch.setitem(ROLE_CHAINS, Role.JUDGE, ("cloud-judge-masked",))
    reply(fake_llm, [])

    gaps = lexicon_gaps.collect(1, 14)

    assert gaps.skipped == "off"
    assert fake_llm.requests == []


def test_the_daily_cap_stops_the_pass_and_is_shared_with_live_scoring(
    fake_llm, monkeypatch, redis_cache
):
    spans(5)
    monkeypatch.setenv("JUDGE_DAILY_LIMIT", "2")
    reply(fake_llm, [])

    gaps = lexicon_gaps.collect(5, 14)

    assert gaps.skipped == "cap"
    assert len(fake_llm.requests) == 2
    assert judge_health.backed_off()


def test_a_backed_off_judge_is_not_called(fake_llm):
    spans(2)
    judge_health.mark_unavailable()
    reply(fake_llm, [])

    gaps = lexicon_gaps.collect(2, 14)

    assert gaps.skipped == "backed_off"
    assert fake_llm.requests == []


def test_an_unavailable_judge_backs_the_pass_off_instead_of_failing_it(fake_llm):
    spans(3)
    fake_llm.fail(lambda r: True, status=503)

    gaps = lexicon_gaps.collect(3, 14)

    assert gaps.skipped == "backed_off"
    assert judge_health.backed_off()


def test_the_weekly_task_stores_counts_only_and_skips_above_eighty_percent_of_the_cap(
    fake_llm, monkeypatch, redis_cache
):
    spans(2)
    monkeypatch.setenv("JUDGE_DAILY_LIMIT", "10")
    reply(fake_llm, [{"term": "Zincovit Forte", "type": "MEDICATION", "lang": "en"}])

    ran = innermind.report_lexicon_gaps()

    row = LexiconGapReport.objects.get()
    assert ran == {"sampled": 2, "calls": 2, "skipped": ""}
    assert row.terms == [{"term": "zincovit forte", "type": "MEDICATION", "lang": "en", "count": 2}]
    assert TEXT not in json.dumps(row.terms)

    redis_cache.set(f"overmind:judge:calls:{judge_health.now().date().isoformat()}", 9)
    before = len(fake_llm.requests)
    skipped = innermind.report_lexicon_gaps()

    assert skipped["skipped"] == "cap"
    assert len(fake_llm.requests) == before
    assert LexiconGapReport.objects.count() == 2


def test_the_command_prints_terms_and_never_the_trace_text(fake_llm, tmp_path):
    spans(2)
    reply(fake_llm, [{"term": "Zincovit Forte", "type": "MEDICATION", "lang": "en"}])
    out, err = io.StringIO(), io.StringIO()

    call_command(
        "lexicon_gaps", "--sample", "2", "--out", str(tmp_path / "gaps.tsv"), stdout=out, stderr=err
    )

    assert "zincovit forte\tMEDICATION\ten\t2" in out.getvalue()
    assert "syrup" not in out.getvalue() + (tmp_path / "gaps.tsv").read_text()
    assert "sampled 2, judge calls 2" in err.getvalue()
    only_local(fake_llm)


def test_the_pass_refuses_to_run_without_the_name_model(fake_llm, settings):
    spans(1)
    settings.MASKING_NER_BACKEND = ""
    settings.MASKING_ENABLED = False
    reply(fake_llm, [{"term": "Zincovit Forte", "type": "MEDICATION", "lang": "en"}])

    assert lexicon_gaps.collect(1, 14).skipped == "no_ner"
    assert fake_llm.requests == []

    ran = innermind.report_lexicon_gaps()
    assert ran["skipped"] == "no_ner"
    assert LexiconGapReport.objects.get().skipped == "no_ner"

    with pytest.raises(CommandError, match="name model"):
        call_command("lexicon_gaps", "--sample", "1", stdout=io.StringIO(), stderr=io.StringIO())
    assert fake_llm.requests == []


def test_the_weekly_task_runs_on_the_masking_queue():
    from django.conf import settings

    route = settings.CELERY_TASK_ROUTES["overbae.tasks.innermind.report_lexicon_gaps"]
    assert route == {"queue": "masking"}


def test_with_the_local_judge_off_the_pass_and_the_weekly_task_skip_without_a_request(
    fake_llm, monkeypatch
):
    spans(1)
    monkeypatch.setitem(ROLE_CHAINS, Role.JUDGE, ())
    reply(fake_llm, [])

    assert lexicon_gaps.collect(1, 14).skipped == "judge_off"
    assert innermind.report_lexicon_gaps()["skipped"] == "judge_off"
    assert fake_llm.requests == []
