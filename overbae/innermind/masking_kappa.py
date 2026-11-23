"""Measures whether masking changes the judge's verdicts: the same fixture is judged once on
its masked copy by the remote judge and once unmasked by the local judge, and the two verdicts
are written side by side for ``masking_report --verdicts``.

It runs on the masking worker, through the normal paths: the remote one masks, checks the
residual, spends the remote cap and makes one attempt; the local one spends the local cap and
takes the single stream. The output holds ids and verdicts, never text.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from django.conf import settings

from overbae.core import async_loop
from overbae.core.model_registry import ROLE_CHAINS, Role
from overbae.innermind import judge_health
from overbae.innermind.evals import (
    JUDGE_MAX_TOKENS,
    Judge,
    JudgeAnswerError,
    JudgeUnavailableError,
    MaskedFallbackError,
    Verdict,
    fit,
    local_route,
    refuses_guardrail,
)
from overbae.services.masking.errors import MaskingError
from overbae.services.masking.masker import mask_ids_only
from overbae.services.masking.send import call_masked

DEFAULT_LIMIT = 100
DEFAULT_REFERENCE = "local-judge"
MODES = ("same-model", "reference")
DEFAULT_MODE = "same-model"
HARD_LIMIT = 150
DOMAINS = ("health", "finance")
AUDIT_ID = "masking-kappa"


class RemoteOnlyJudge(Judge):
    """The normal remote path, for a fixture that belongs to no project: the route says
    remote, so the project's flag is not read; everything after that is unchanged."""

    async def remote_model(self) -> str | None:
        chain = ROLE_CHAINS[Role.CLOUD_JUDGE]
        return chain[0] if chain else None


@dataclass
class Outcome:
    pairs: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    cloud_calls: int = 0
    local_calls: int = 0


def pick(fixtures: list[dict], domains: tuple[str, ...], limit: int) -> list[dict]:
    """Evenly spaced fixtures of the wanted domains, so a limit still spans every language."""
    wanted = [f for f in fixtures if f["domain"] in domains]
    if len(wanted) <= limit:
        return wanted
    step = len(wanted) / limit
    return [wanted[int(i * step)] for i in range(limit)]


def remote_route():
    from overbae.innermind.evals import JudgeRoute  # noqa: PLC0415

    chain = ROLE_CHAINS[Role.CLOUD_JUDGE]
    return JudgeRoute(model=chain[0], sample_rate=1.0, explain=False, local=False)


async def judge_pair(fixture: dict, outcome: Outcome, reference: str) -> None:
    state, expectation = fixture["state"], fixture["state"].get("expectation", "")
    remote = RemoteOnlyJudge(AUDIT_ID, route=remote_route())
    try:
        masked = await remote.judge(dict(state), expectation)
        outcome.cloud_calls += 1
    except MaskedFallbackError as exc:
        return outcome.skipped.append({"id": fixture["id"], "reason": f"masking: {exc.reason}"})
    except (JudgeUnavailableError, JudgeAnswerError) as exc:
        return outcome.skipped.append({"id": fixture["id"], "reason": f"cloud: {exc}"[:120]})
    # Person-started work, not live scoring: it neither draws on the local daily cap nor sets or
    # obeys the local back-off, and it takes the single-stream lock only if the model is one.
    local = Judge(AUDIT_ID, role=Role.FAST, route=local_route())
    local.model_override = reference
    try:
        unmasked = await local.judge(dict(state), expectation)
    except (JudgeUnavailableError, JudgeAnswerError) as exc:
        return outcome.skipped.append({"id": fixture["id"], "reason": f"local: {exc}"[:120]})
    if unmasked.error or not unmasked.verdict:
        return outcome.skipped.append({"id": fixture["id"], "reason": "local: no verdict"})
    outcome.local_calls += 1
    outcome.pairs.append(
        {"id": fixture["id"], "masked_verdict": masked.verdict, "local_verdict": unmasked.verdict}
    )


_BLOCKED = re.compile(r"Content blocked:\s*(.+?)\s*pattern detected", re.IGNORECASE)


async def judge_same_model(fixture: dict, outcome: Outcome) -> None:
    """The same remote judge twice: on the fully masked copy (``masked_verdict``) and on the
    ids-only copy (stored as ``local_verdict`` so ``masking_report --verdicts`` reads it)."""
    state, expectation = fixture["state"], fixture["state"].get("expectation", "")
    full = RemoteOnlyJudge(AUDIT_ID, route=remote_route())
    try:
        masked = await full.judge(dict(state), expectation)
        outcome.cloud_calls += 1
    except MaskedFallbackError as exc:
        return outcome.skipped.append({"id": fixture["id"], "reason": f"masking: {exc.reason}"})
    except (JudgeUnavailableError, JudgeAnswerError) as exc:
        return outcome.skipped.append({"id": fixture["id"], "reason": f"cloud: {exc}"[:120]})
    try:
        payload = await asyncio.to_thread(mask_ids_only, state, fit=fit)
    except MaskingError as exc:
        return outcome.skipped.append({"id": fixture["id"], "reason": f"ids-only: {exc.reason}"})
    if not judge_health.spend_cloud_call():
        return outcome.skipped.append({"id": fixture["id"], "reason": "cloud: daily cap spent"})
    try:
        text, _ = await asyncio.to_thread(
            call_masked,
            payload,
            model=ROLE_CHAINS[Role.CLOUD_JUDGE][0],
            system_prompt=Judge.instructions,
            response_format=Verdict,
            request_kwargs={"temperature": 0, "user": AUDIT_ID},
            max_tokens=JUDGE_MAX_TOKENS,
        )
        ids_only = Judge.parse(text)
        outcome.cloud_calls += 1
    except Exception as exc:  # noqa: BLE001 — each failure is one skipped fixture, not a failed run
        if refuses_guardrail(exc):
            found = _BLOCKED.search(str(exc))
            reason = f"ids-only missed: {found.group(1) if found else 'identifier'}"
        elif isinstance(exc, JudgeAnswerError):
            reason = "cloud: no verdict"
        else:
            reason = f"cloud: {type(exc).__name__}"
        return outcome.skipped.append({"id": fixture["id"], "reason": reason})
    outcome.pairs.append(
        {"id": fixture["id"], "masked_verdict": masked.verdict, "local_verdict": ids_only}
    )


def run(
    fixtures: list[dict], reference: str = DEFAULT_REFERENCE, mode: str = DEFAULT_MODE
) -> Outcome:
    outcome = Outcome()
    for fixture in fixtures:
        if "planted" not in fixture or "state" not in fixture:
            raise RuntimeError("masking_kappa judges only the made-up fixtures it is given")
        if judge_health.backed_off(cloud=True):
            outcome.skipped.append({"id": fixture["id"], "reason": "cloud judge backed off"})
            continue
        if mode == "same-model":
            async_loop.run(judge_same_model(fixture, outcome))
        else:
            async_loop.run(judge_pair(fixture, outcome, reference))
    return outcome


def require_masking_worker() -> None:
    if not settings.MASKING_ENABLED:
        raise RuntimeError("masking_kappa runs on the masking worker (MASKING_ENABLED=1)")
    if not ROLE_CHAINS[Role.CLOUD_JUDGE]:
        raise RuntimeError("masking_kappa needs LLM_CLOUD_JUDGE_CHAIN")


def check_reference(name: str) -> None:
    """The unmasked half sees real fixture text, so it only ever goes to a local-class model
    (this machine's, or a single-stream one), never to the remote judge or any cloud model."""
    from overbae.core.model_registry import MODELS_BY_NAME, Vendor  # noqa: PLC0415

    spec = MODELS_BY_NAME.get(name)
    if spec is None:
        raise RuntimeError(f"Unknown reference model: {name}")
    if not (spec.single_stream or spec.vendor is Vendor.LOCAL):
        raise RuntimeError(f"The reference model must be a local one, not {name}")


KAPPA_BAR = 0.6
MIN_PAIRS = 60


def kappa_line(pairs: list[dict], domains: dict[str, str]) -> tuple[str, bool]:
    """Kappa between the masked and the reference verdicts over the health and finance pairs,
    and whether it clears the bar with enough pairs."""
    from overbae.services.masking import report  # noqa: PLC0415

    chosen = [
        (p["masked_verdict"], p["local_verdict"])
        for p in pairs
        if domains.get(p["id"]) in ("health", "finance")
    ]
    value = report.kappa(chosen)
    ok = value is not None and value >= KAPPA_BAR and len(chosen) >= MIN_PAIRS
    shown = "n/a" if value is None else f"{value:.2f}"
    return f"kappa {shown} on {len(chosen)} pairs (bar {KAPPA_BAR}): {'ok' if ok else 'fail'}", ok
