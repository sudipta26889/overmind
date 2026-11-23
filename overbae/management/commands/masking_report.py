import json
import sys
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from overbae.services.masking import report

DEFAULT_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "masking"


class Command(BaseCommand):
    help = (
        "Mask every fixture and report recall and precision per type, language and domain, the "
        "leaks, and the latency. Exits 1 if a bar fails. Run it in the masking image: it loads the "
        "real name model."
    )

    def add_arguments(self, parser):
        parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
        parser.add_argument("--format", choices=["text", "json"], default="text")
        parser.add_argument(
            "--verdicts",
            help="JSON lines {id, masked_verdict, local_verdict} from the two judges, for kappa.",
        )

    def handle(self, *args, fixtures, format, verdicts, **options):
        directory = Path(fixtures)
        if not directory.is_dir():
            raise CommandError(f"{directory} is not a directory.")
        loaded = report.load_fixtures(directory)
        if not loaded:
            raise CommandError(f"No fixtures in {directory}.")
        pairs = {}
        if verdicts:
            for line in Path(verdicts).read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    pairs[row["id"]] = row
        result = report.evaluate(report.run(loaded), pairs)
        failed = report.failing(result)
        if format == "json":
            self.stdout.write(
                json.dumps({**result, "failing": failed}, ensure_ascii=False, indent=1)
            )
        else:
            self.stdout.write(report.format_text(result))
            self.stdout.write("")
            self.stdout.write("FAILED: " + ", ".join(failed) if failed else "all bars pass")
        if failed:
            sys.exit(1)
