import json
import sys
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from overbae.innermind import masking_kappa
from overbae.services.masking import report

DEFAULT_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "masking"


class Command(BaseCommand):
    help = (
        "Mask every fixture and report recall and precision per type, language and domain, the "
        "leaks, and the latency. Exits 1 if a bar fails. Run it in the masking image: it loads the "
        "real name model. Run it as a one-off container from the masking image (docker run "
        "--memory 8g), not inside the 4 GB worker, which would load the models twice and be "
        "OOM-killed; --no-mask needs no model."
    )

    def add_arguments(self, parser):
        parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
        parser.add_argument("--format", choices=["text", "json"], default="text")
        parser.add_argument(
            "--verdicts",
            help="JSON lines {id, masked_verdict, local_verdict} from the two judges, for kappa.",
        )
        parser.add_argument(
            "--kept-bar",
            type=float,
            default=report.BARS["kept"],
            help="Share of non-sensitive words that must stay unmasked, per language: 0.85 on "
            "the seen fixtures (the default); 0.75 is the gate for the unseen and holdout sets.",
        )
        parser.add_argument(
            "--no-mask",
            action="store_true",
            help="Load no model and mask nothing: print kappa from --verdicts and the fixtures' "
            "domains only.",
        )

    def handle(self, *args, fixtures, format, verdicts, no_mask, kept_bar, **options):
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
        if no_mask:
            if not pairs:
                raise CommandError("--no-mask needs --verdicts.")
            line, ok = masking_kappa.summary(
                list(pairs.values()), {f["id"]: f["domain"] for f in loaded}
            )
            self.stdout.write(line)
            if not ok:
                sys.exit(1)
            return
        result = report.evaluate(report.run(loaded), pairs, kept_bar=kept_bar)
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
