import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from overbae.innermind import masking_kappa
from overbae.services.masking import report


class Command(BaseCommand):
    help = (
        "Judge each fixture twice: its masked copy by the remote judge, its unmasked state by the "
        "local judge. Writes {id, masked_verdict, local_verdict} lines for masking_report "
        "--verdicts. Runs on the masking worker only; spends real calls."
    )

    def add_arguments(self, parser):
        parser.add_argument("--fixtures", required=True)
        parser.add_argument("--domains", default=",".join(masking_kappa.DOMAINS))
        parser.add_argument("--limit", type=int, default=masking_kappa.DEFAULT_LIMIT)
        parser.add_argument("--out", required=True)
        parser.add_argument(
            "--reference-model",
            default=masking_kappa.DEFAULT_REFERENCE,
            help="Local model for the unmasked half (default: local-judge).",
        )

    def handle(self, *args, fixtures, domains, limit, out, reference_model, **options):
        if limit > masking_kappa.HARD_LIMIT:
            raise CommandError(f"--limit above {masking_kappa.HARD_LIMIT} is refused.")
        if limit < 1:
            raise CommandError("--limit must be at least 1.")
        try:
            masking_kappa.require_masking_worker()
            masking_kappa.check_reference(reference_model)
        except RuntimeError as exc:
            raise CommandError(str(exc)) from exc
        directory = Path(fixtures)
        if not directory.is_dir():
            raise CommandError(f"{directory} is not a directory.")
        wanted = tuple(d.strip() for d in domains.split(",") if d.strip())
        chosen = masking_kappa.pick(report.load_fixtures(directory), wanted, limit)
        if not chosen:
            raise CommandError("No fixtures match the domains.")
        self.stdout.write(f"reference model: {reference_model}")
        outcome = masking_kappa.run(chosen, reference_model)
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(row) + "\n" for row in outcome.pairs))
        skipped = target.with_name(target.name + ".skipped")
        skipped.write_text("".join(json.dumps(row) + "\n" for row in outcome.skipped))
        self.stdout.write(
            f"fixtures {len(chosen)}, pairs {len(outcome.pairs)}, skipped {len(outcome.skipped)}, "
            f"cloud calls {outcome.cloud_calls}, local calls {outcome.local_calls}"
        )
        self.stdout.write(f"verdicts: {target}   skipped: {skipped}")
