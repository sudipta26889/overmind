from django.core.management.base import BaseCommand
from django.db.models import Count

from overbae.core.llms import cache_redis
from overbae.innermind import judge_health
from overbae.models import LiveScoringWork, Project, Span


class Command(BaseCommand):
    help = (
        "Read-only: live scoring queue counts by state and kind, the work and spans waiting on "
        "the local route, and today's remote and masking counters. Aggregate queries only."
    )

    def handle(self, *args, **options):
        write = self.stdout.write
        write("live scoring work (state kind count)")
        rows = (
            LiveScoringWork.objects.values("state", "kind")
            .annotate(total=Count("pk"))
            .order_by("state", "kind")
        )
        for row in rows:
            write(f"  {row['state']:<8} {row['kind']:<10} {row['total']}")
        remote = Project.objects.filter(cloud_judging=True).values("pk")
        held_work = (
            LiveScoringWork.objects.filter(state="waiting").exclude(project_id__in=remote).count()
        )
        held_spans = (
            Span.objects.exclude(eval_attributes__has_key="score")
            .exclude(project_id__in=remote)
            .count()
        )
        write(f"held on the local route: {held_work} waiting items, {held_spans} unscored spans")
        write(f"projects approved for remote judging: {remote.count()}")
        today = judge_health.now().date().isoformat()
        client = cache_redis()
        if client is None:
            write("counters: no cache Redis")
            return
        for label, key in (
            ("local calls", f"overmind:judge:calls:{today}"),
            ("cloud calls", f"overmind:judge:cloud_calls:{today}"),
            ("leaks", f"overmind:masking:leak:{today}"),
        ):
            write(f"{label} today: {int(client.get(key) or 0)}")
        for key in sorted(client.scan_iter(match=f"overmind:masking:fallback:*:{today}")):
            reason = key.decode().split(":")[3]
            write(f"fallback {reason} today: {int(client.get(key) or 0)}")
        write(
            f"back-off: local {judge_health.backed_off()}, cloud {judge_health.backed_off(cloud=True)}"
        )
