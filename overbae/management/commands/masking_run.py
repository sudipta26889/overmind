import sys
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from overbae.tasks import masking_commands

PATH_OPTIONS = {"--fixtures": "@fixtures", "--verdicts": "@verdicts", "--out": "@out"}


def split_paths(args: list[str]) -> tuple[list[str], dict[str, str]]:
    """The arguments with each local path replaced by a placeholder the worker fills in, and
    the paths that were replaced."""
    argv: list[str] = []
    paths: dict[str, str] = {}
    items = iter(args)
    for item in items:
        name, equals, inline = item.partition("=")
        if name in PATH_OPTIONS:
            value = inline if equals else next(items, "")
            argv += [name, PATH_OPTIONS[name]]
            paths[PATH_OPTIONS[name]] = value
        else:
            argv.append(item)
    return argv, paths


class Command(BaseCommand):
    help = (
        "Run masking_report or masking_kappa inside the running masking worker, whose models are "
        "already loaded. Usage: masking_run report|kappa -- [that command's own options] (the -- is needed). The "
        "fixtures (and --verdicts) are read here and sent; the result is printed here, and "
        "kappa's --out files are written here. Running either command directly in the worker "
        "loads the models a second time and is OOM-killed; the fallback is a one-off container "
        "from the masking image (docker run --memory 8g), not the worker."
    )

    def add_arguments(self, parser):
        parser.add_argument("which", choices=["report", "kappa"])
        parser.add_argument("--timeout", type=float, default=1500)
        parser.add_argument(
            "options", nargs="*", help="Passed to the command (use -- before them)."
        )

    def handle(self, *args, which, timeout, options, **kwargs):
        if not settings.MASKING_QUEUE_ENABLED:
            raise CommandError(
                "No masking worker runs in this deployment. Use the one-off container form."
            )
        argv, paths = split_paths([o for o in options if o != "--"])
        fixtures_dir = Path(paths.get("@fixtures") or "\0")
        if not paths.get("@fixtures") or not fixtures_dir.is_dir():
            raise CommandError("--fixtures DIR is required and must be a directory.")
        fixtures = {
            path.name: path.read_text()
            for path in sorted(fixtures_dir.glob("*.jsonl"))
            if not path.name.startswith(".")
        }
        if not fixtures:
            raise CommandError(f"No fixtures in {fixtures_dir}.")
        verdicts = Path(paths["@verdicts"]).read_text() if "@verdicts" in paths else None
        result = masking_commands.run_command.apply_async(
            kwargs={
                "command": f"masking_{which}",
                "args": argv,
                "fixtures": fixtures,
                "verdicts": verdicts,
            }
        ).get(timeout=timeout)
        if "@out" in paths:
            target = Path(paths["@out"])
            target.parent.mkdir(parents=True, exist_ok=True)
            for name, text in result["files"].items():
                suffix = ".skipped" if name.endswith(".skipped") else ""
                Path(str(target) + suffix).write_text(text)
        self.stdout.write(result["stdout"].rstrip("\n"))
        if result["stderr"]:
            self.stderr.write(result["stderr"].rstrip("\n"))
        if result["exit"]:
            sys.exit(result["exit"])
