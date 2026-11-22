from datetime import UTC, datetime
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from overbae.innermind import lexicon_gaps


class Command(BaseCommand):
    help = (
        "List terms the masking lexicon lacks, found in recent spans by the local judge only. "
        "Prints term, type, language and count; never trace text. It does not edit the lexicon."
    )

    def add_arguments(self, parser):
        parser.add_argument("--sample", type=int, default=200)
        parser.add_argument("--days", type=int, default=14)
        parser.add_argument(
            "--out", help="Also write the list here (default: scratch/lexicon-gaps/)."
        )

    def handle(self, *args, sample, days, out, **options):
        gaps = lexicon_gaps.collect(sample, days)
        lines = ["\t".join(("term", "type", "lang", "count"))] + [
            "\t".join((t["term"], t["type"], t["lang"], str(t["count"]))) for t in gaps.terms()
        ]
        target = (
            Path(out)
            if out
            else (
                Path(settings.BASE_DIR)
                / "scratch"
                / "lexicon-gaps"
                / f"{datetime.now(UTC).date().isoformat()}.tsv"
            )
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n")
        self.stdout.write("\n".join(lines))
        note = f"sampled {gaps.sampled}, judge calls {gaps.calls}, terms {len(gaps.terms())}"
        self.stderr.write(note + (f", stopped: {gaps.skipped}" if gaps.skipped else ""))
        self.stderr.write(f"written to {target}")
