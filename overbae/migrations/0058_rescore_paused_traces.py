import hashlib
import os
from datetime import UTC, datetime

from django.db import migrations
from django.db.models import Q

# Live scoring ran with TRACE_SCORING_SAMPLE_RATE=0 from here until local judging shipped, so
# every span that arrived marked itself final-without-judging.
PAUSED_FROM = datetime(2026, 11, 19, 16, 0, tzinfo=UTC)
BATCH = 1_000


def in_sample(trace_id, rate):
    # Copied from span_processor.in_scoring_sample: a migration must not follow later changes.
    if rate >= 1:
        return True
    bucket = int.from_bytes(hashlib.sha256(trace_id.encode()).digest()[:4], "big")
    return bucket < rate * 2**32


def rescore_paused_traces(apps, schema_editor):
    Span = apps.get_model("overbae", "Span")
    rate = float(os.environ.get("TRACE_SCORING_SAMPLE_RATE") or "1")
    marked = Q(eval_attributes__unsampled=True) | Q(eval_attributes__trajectory__unsampled=True)
    spans = (
        Span.objects.filter(marked, received_at__gte=PAUSED_FROM)
        .only("pk", "trace_id", "eval_attributes")
        .order_by("pk")
    )
    batch = []

    def flush():
        Span.objects.bulk_update(batch, ["eval_attributes", "eval_claimed_at"], batch_size=BATCH)
        batch.clear()

    for span in spans.iterator(chunk_size=BATCH):
        if not in_sample(span.trace_id, rate):
            continue
        attrs = dict(span.eval_attributes)
        if attrs.get("unsampled") is True:
            for key in ("unsampled", "score", "metrics"):
                attrs.pop(key, None)
        if (attrs.get("trajectory") or {}).get("unsampled") is True:
            attrs.pop("trajectory")
        span.eval_attributes = attrs
        span.eval_claimed_at = None
        batch.append(span)
        if len(batch) >= BATCH:
            flush()
    if batch:
        flush()


class Migration(migrations.Migration):
    dependencies = [
        ("overbae", "0057_end_external_participant_work"),
    ]

    operations = [
        migrations.RunPython(rescore_paused_traces, migrations.RunPython.noop),
    ]
