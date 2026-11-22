import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from importlib import import_module

import pytest
from django.apps import apps
from factories import make_project, make_span

from overbae.innermind.span_processor import in_scoring_sample
from overbae.models import LiveScoringWork, Span
from overbae.tasks import innermind

pytestmark = [pytest.mark.django_db, pytest.mark.covers("recovery")]

MARKER = {"score": None, "metrics": {}, "unsampled": True}
PAUSED = datetime(2026, 11, 19, 16, 0, tzinfo=UTC)
RATE = 0.5


def trace_ids(*, sampled, count=2):
    found = []
    while len(found) < count:
        candidate = uuid.uuid4().hex
        bucket_sampled = (
            int.from_bytes(hashlib.sha256(candidate.encode()).digest()[:4], "big") < RATE * 2**32
        )
        if bucket_sampled is sampled:
            found.append(candidate)
    return found


def marked(project, trace_id, received, **extra):
    span = make_span(project, trace_id=trace_id, eval_attributes={**MARKER, **extra})
    Span.objects.filter(pk=span.pk).update(
        received_at=received, eval_claimed_at=received if extra.get("claimed") else None
    )
    return span


def migrate(monkeypatch):
    monkeypatch.setenv("TRACE_SCORING_SAMPLE_RATE", str(RATE))
    import_module("overbae.migrations.0058_rescore_paused_traces").rescore_paused_traces(apps, None)


def test_paused_traces_in_the_sample_are_offered_again_and_the_rest_are_left(monkeypatch, settings):
    settings.TRACE_SCORING_SAMPLE_RATE = RATE
    project = make_project()
    inside, outside = trace_ids(sampled=True)[0], trace_ids(sampled=False)[0]
    during = PAUSED + timedelta(hours=3)
    in_sample_span = marked(project, inside, during)
    out_of_sample = marked(project, outside, during)
    before = marked(project, trace_ids(sampled=True)[0], PAUSED - timedelta(hours=1))
    released = make_span(
        project, trace_id=trace_ids(sampled=True)[0], eval_attributes={"score": 0.8, "metrics": {}}
    )
    Span.objects.filter(pk=released.pk).update(received_at=during)
    root = make_span(
        project,
        trace_id=inside,
        eval_attributes={
            "score": 0.9,
            "trajectory": {"score": None, "metrics": {}, "unsampled": True},
        },
    )
    Span.objects.filter(pk=root.pk).update(received_at=during)

    migrate(monkeypatch)

    for span in (in_sample_span, out_of_sample, before, released, root):
        span.refresh_from_db()
    assert in_scoring_sample(inside) and not in_scoring_sample(outside)
    assert in_sample_span.eval_attributes == {}
    assert out_of_sample.eval_attributes == MARKER
    assert before.eval_attributes == MARKER
    assert released.eval_attributes == {"score": 0.8, "metrics": {}}
    assert root.eval_attributes == {"score": 0.9}

    assert innermind.sweep_unscored_spans() == {"spans": 1}
    assert list(LiveScoringWork.objects.values_list("key", flat=True)) == [in_sample_span.span_id]
