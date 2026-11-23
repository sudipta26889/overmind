import json
import sys
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from overbae.core.model_registry import MODELS_BY_NAME, ROLE_CHAINS, Role
from overbae.innermind import masking_kappa
from overbae.services.masking import report


class Command(BaseCommand):
    help = (
        "Run it as a one-off container from the masking image (docker run --memory 8g, the key "
        "directory mounted, MASKING_ENABLED=1), never inside the 4 GB worker: it loads the "
        "models a second time and is OOM-killed. Judge each fixture twice: its masked copy by the remote judge, its unmasked state by the "
        "local judge. Writes {id, masked_verdict, local_verdict} lines for masking_report "
        "--verdicts. Runs on the masking worker only; spends real calls."
    )

    def add_arguments(self, parser):
        parser.add_argument("--fixtures", required=True)
        parser.add_argument("--domains", default=",".join(masking_kappa.DOMAINS))
        parser.add_argument("--limit", type=int, default=masking_kappa.DEFAULT_LIMIT)
        parser.add_argument("--out", required=True)
        parser.add_argument(
            "--mode",
            choices=masking_kappa.MODES,
            default=masking_kappa.DEFAULT_MODE,
            help="same-model: the cloud judge on full masking against ids-only masking (the "
            "primary measure); stability: the same judge on the same ids-only text twice (the "
            "run-to-run floor); reference: the cloud judge on masked text against a local model "
            "on the unmasked text. Every mode prints agreement, the 3x3 table and kappa.",
        )
        parser.add_argument(
            "--reference-model",
            default=masking_kappa.DEFAULT_REFERENCE,
            help="Local model for the unmasked half (default: local-judge).",
        )

    def handle(self, *args, fixtures, domains, limit, out, reference_model, mode, **options):
        if limit > masking_kappa.HARD_LIMIT:
            raise CommandError(f"--limit above {masking_kappa.HARD_LIMIT} is refused.")
        if limit < 1:
            raise CommandError("--limit must be at least 1.")
        try:
            masking_kappa.require_masking_worker()
            if mode == "reference":
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
        cloud = MODELS_BY_NAME[ROLE_CHAINS[Role.CLOUD_JUDGE][0]].slug
        self.stdout.write(f"mode: {mode}, model: {cloud}")
        if mode == "reference":
            self.stdout.write(f"reference model: {reference_model}")
        outcome = masking_kappa.run(chosen, reference_model, mode)
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
        line, ok = masking_kappa.summary(outcome.pairs, {f["id"]: f["domain"] for f in chosen})
        self.stdout.write(line)
        if not ok:
            sys.exit(1)
