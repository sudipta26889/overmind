"""Runs ``masking_report`` and ``masking_kappa`` inside the masking worker.

The worker already holds both models, so the work happens there instead of in a second process
that would load them again and be OOM-killed. The payload is made-up fixture text and the
result holds only counts, bars, leak lines and (for kappa) ids and verdicts: no surrogate map
ever crosses the broker.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError

ALLOWED = {"masking_report", "masking_kappa"}
# Where the client's paths land inside the worker's scratch directory.
PLACEHOLDERS = {"@fixtures": "fixtures", "@verdicts": "verdicts.jsonl", "@out": "out.jsonl"}
MAX_PAYLOAD_BYTES = 5_000_000


@shared_task(
    name="overbae.tasks.masking_commands.run_command",
    soft_time_limit=1500,
    time_limit=1560,
)
def run_command(*, command: str, args: list[str], fixtures: dict[str, str], verdicts: str | None):
    if command not in ALLOWED:
        raise ValueError(f"{command} is not a masking command")
    if not settings.MASKING_ENABLED:
        raise RuntimeError("This worker has no masking models; run it on the masking worker.")
    if sum(map(len, fixtures.values())) + len(verdicts or "") > MAX_PAYLOAD_BYTES:
        raise ValueError("The payload is larger than the broker should carry.")
    out, err = io.StringIO(), io.StringIO()
    code = 0
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        (root / "fixtures").mkdir()
        for name, text in fixtures.items():
            (root / "fixtures" / Path(name).name).write_text(text)
        if verdicts is not None:
            (root / "verdicts.jsonl").write_text(verdicts)
        argv = [str(root / PLACEHOLDERS[a]) if a in PLACEHOLDERS else a for a in args]
        try:
            call_command(command, *argv, stdout=out, stderr=err)
        except SystemExit as stop:
            code = int(stop.code or 0)
        except CommandError as exc:
            err.write(f"{exc}\n")
            code = 2
        for name in ("out.jsonl", "out.jsonl.skipped"):
            if (root / name).is_file():
                files[name] = (root / name).read_text()
    return {"exit": code, "stdout": out.getvalue(), "stderr": err.getvalue(), "files": files}
