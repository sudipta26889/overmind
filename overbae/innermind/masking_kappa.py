"""Measures whether masking changes the judge's verdicts: the same fixture is judged once on
its masked copy by the remote judge and once unmasked by the local judge, and the two verdicts
are written side by side for ``masking_report --verdicts``.

It runs on the masking worker, through the normal paths: the remote one masks, checks the
residual, spends the remote cap and makes one attempt; the local one spends the local cap and
takes the single stream. The output holds ids and verdicts, never text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from django.conf import settings

from overbae.core import async_loop
from overbae.core.model_registry import ROLE_CHAINS, Role
from overbae.innermind import judge_health
from overbae.innermind.evals import (
    Judge,
    JudgeAnswerError,
    JudgeUnavailableError,
    MaskedFallbackError,
    local_route,
)

DEFAULT_LIMIT = 100
DEFAULT_REFERENCE = "local-judge"
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
    local = Judge(AUDIT_ID, route=local_route())
    local.model_override = reference
    try:
        unmasked = await local.judge(dict(state), expectation)
        outcome.local_calls += 1
    except (JudgeUnavailableError, JudgeAnswerError) as exc:
        return outcome.skipped.append({"id": fixture["id"], "reason": f"local: {exc}"[:120]})
    outcome.pairs.append(
        {"id": fixture["id"], "masked_verdict": masked.verdict, "local_verdict": unmasked.verdict}
    )


def run(fixtures: list[dict], reference: str = DEFAULT_REFERENCE) -> Outcome:
    outcome = Outcome()
    for fixture in fixtures:
        if judge_health.backed_off() or judge_health.backed_off(cloud=True):
            outcome.skipped.append({"id": fixture["id"], "reason": "judge backed off"})
            continue
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
