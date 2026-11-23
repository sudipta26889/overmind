"""Evidence for the masker: planted-value recall and precision over a fixture set.

A fixture is ``{id, lang, domain, state, planted: [{type, value, path}]}``. Every entity in a
fixture is planted. A leak is a planted value still present in the final string after NFKC and
casefold, with spaces, hyphens and digit separators removed. A fixture that fails closed sends
nothing, so it is counted as a fallback and left out of recall.
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from overbae.services.masking import MaskingError, mask_state, patterns

LANGUAGES = ("en", "hi", "bn", "hinglish", "banglish")
DOMAINS = ("general", "health", "finance")
IDENTIFIER_TYPES = ("PHONE", "EMAIL", "ACCOUNT", "ID", "URL")
VALUE_TYPES = ("AMOUNT", "DATE", "VALUE")
LEXICON_TYPES = ("CONDITION", "MEDICATION", "TEST", "ORG", "RELATION")
BARS = {"identifiers": 0.99, "names": 0.90, "values": 0.99, "lexicon": 0.95, "kappa": 0.6}
_SEPARATORS = re.compile(r"[\s\-,./]")


def flat(text: str) -> str:
    return _SEPARATORS.sub("", unicodedata.normalize("NFKC", text).casefold())


_SHORT = 3


def present(value: str, final: str) -> bool:
    """Whether ``value`` survives in ``final``. A value of three characters or fewer (ma, মা)
    counts only as a whole word; it would otherwise match inside any longer word."""
    if len(flat(value)) > _SHORT:
        return flat(value) in flat(final)
    word = rf"(?<!{patterns.WORD}){re.escape(unicodedata.normalize('NFKC', value).casefold())}(?!{patterns.WORD})"
    return re.search(word, unicodedata.normalize("NFKC", final).casefold()) is not None


@dataclass
class Outcome:
    fixture: dict
    seconds: float
    fallback: str = ""
    final: str = ""
    surrogates: dict[str, str] = field(default_factory=dict)


def load_fixtures(directory: Path) -> list[dict]:
    fixtures = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        if path.name.startswith("."):
            continue  # macOS ``._*`` side files and other dotfiles are not fixtures
        for line in path.read_text().splitlines():
            if line.strip():
                fixtures.append(json.loads(line))
    return fixtures


def run(fixtures: list[dict]) -> list[Outcome]:
    outcomes = []
    for fixture in fixtures:
        started = time.perf_counter()
        try:
            payload = mask_state(fixture["state"])
        except MaskingError as exc:
            outcomes.append(Outcome(fixture, time.perf_counter() - started, fallback=exc.reason))
            continue
        outcomes.append(
            Outcome(
                fixture,
                time.perf_counter() - started,
                final=payload.text,
                surrogates=dict(payload.mapping),
            )
        )
    return outcomes


def ratio(hit: int, total: int) -> float | None:
    return hit / total if total else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa between two raters' verdicts."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[k] * second[k] for k in first) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def verdict_kappa(outcomes: list[Outcome], verdicts: dict[str, dict]) -> tuple[float | None, int]:
    pairs = [
        (verdicts[o.fixture["id"]]["masked_verdict"], verdicts[o.fixture["id"]]["local_verdict"])
        for o in outcomes
        if o.fixture["domain"] in ("health", "finance") and o.fixture["id"] in verdicts
    ]
    return kappa(pairs), len(pairs)


def evaluate(outcomes: list[Outcome], verdicts: dict[str, dict] | None = None) -> dict:
    caught: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    by_domain: dict[str, dict] = {d: Counter() for d in DOMAINS}
    leaks, tool_leaks = [], []
    precision: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    fallbacks: Counter[str] = Counter()
    for outcome in outcomes:
        fixture = outcome.fixture
        domain = by_domain.setdefault(fixture["domain"], Counter())
        domain["fixtures"] += 1
        if outcome.fallback:
            fallbacks[outcome.fallback] += 1
            domain["fallback"] += 1
            domain[f"fallback:{outcome.fallback}"] += 1
            continue
        planted_flat = [flat(p["value"]) for p in fixture["planted"]]
        for planted in fixture["planted"]:
            leaked = present(planted["value"], outcome.final)
            cell = caught[(planted["type"], fixture["lang"])]
            cell[1] += 1
            cell[0] += not leaked
            domain["planted"] += 1
            domain["leaked"] += leaked
            if leaked:
                line = (fixture["id"], planted["type"], fixture["lang"], planted["value"])
                leaks.append(line)
                if "/arguments/" in planted["path"]:
                    tool_leaks.append(line)
        for token, original in outcome.surrogates.items():
            kind = token.strip("<>").rsplit("_", 1)[0]
            original_flat = flat(original)
            true_positive = any(
                v and (v in original_flat or original_flat in v) for v in planted_flat
            )
            cell = precision[(kind, fixture["lang"])]
            cell[1] += 1
            cell[0] += true_positive
    seconds = [o.seconds for o in outcomes]
    total = len(outcomes)
    overall = _bars(caught)
    kappa_value, paired = verdict_kappa(outcomes, verdicts or {})
    return {
        "fixtures": total,
        "fallback_rate": ratio(sum(fallbacks.values()), total),
        "fallbacks": dict(fallbacks),
        "recall": {f"{t}/{lang}": ratio(*cell) for (t, lang), cell in sorted(caught.items())},
        "recall_counts": {f"{t}/{lang}": cell for (t, lang), cell in sorted(caught.items())},
        "precision": {f"{t}/{lang}": ratio(*cell) for (t, lang), cell in sorted(precision.items())},
        "domains": {d: dict(c) for d, c in by_domain.items()},
        "latency": {"p50": percentile(seconds, 0.5), "p95": percentile(seconds, 0.95)},
        "leaks": leaks,
        "tool_argument_leaks": tool_leaks,
        "bars": overall,
        "kappa": {"value": kappa_value, "pairs": paired, "bar": BARS["kappa"]},
    }


def _group(caught, types, lang=None) -> float | None:
    hit = total = 0
    for (kind, language), (ok, n) in caught.items():
        if kind in types and (lang is None or language == lang):
            hit, total = hit + ok, total + n
    return ratio(hit, total)


def _bars(caught) -> dict[str, dict]:
    """Each bar as ``{value, bar, ok}``; ``None`` (nothing planted) never passes."""
    bars: dict[str, dict] = {}

    def add(name: str, value: float | None, bar: float):
        bars[name] = {"value": value, "bar": bar, "ok": value is not None and value >= bar}

    add("identifiers", _group(caught, IDENTIFIER_TYPES), BARS["identifiers"])
    for kind in VALUE_TYPES:
        add(f"{kind.lower()}s", _group(caught, (kind,)), BARS["values"])
    for lang in LANGUAGES:
        add(f"names/{lang}", _group(caught, ("PERSON",), lang), BARS["names"])
        add(f"lexicon/{lang}", _group(caught, LEXICON_TYPES, lang), BARS["lexicon"])
    return bars


def failing(report: dict) -> list[str]:
    names = [name for name, bar in report["bars"].items() if not bar["ok"]]
    if report["tool_argument_leaks"]:
        names.append("tool_argument_leaks")
    return names


def format_text(report: dict) -> str:
    def pct(value):
        return "  n/a" if value is None else f"{value:5.1%}"

    lines = [f"fixtures {report['fixtures']}   fallback {pct(report['fallback_rate'])}"]
    for reason, count in sorted(report["fallbacks"].items()):
        lines.append(f"  fallback {reason}: {count}")
    kinds = sorted({key.split("/")[0] for key in {**report["recall"], **report["precision"]}})
    lines += ["", "recall / precision by type and language"]
    lines.append(f"{'type':<11}" + "".join(f"{lang:>16}" for lang in LANGUAGES))
    for kind in kinds:
        row = f"{kind:<11}"
        for lang in LANGUAGES:
            key = f"{kind}/{lang}"
            row += f"{pct(report['recall'].get(key))} /{pct(report['precision'].get(key))}".rjust(
                16
            )
        lines.append(row)
    lines += [
        "",
        f"{'domain':<10}{'fixtures':>9}{'fallback':>10}{'planted':>9}{'leaked':>8}{'residual':>10}",
    ]
    for domain, c in report["domains"].items():
        lines.append(
            f"{domain:<10}{c.get('fixtures', 0):>9}{c.get('fallback', 0):>10}"
            f"{c.get('planted', 0):>9}{c.get('leaked', 0):>8}{c.get('fallback:residual', 0):>10}"
        )
    latency = report["latency"]
    lines += [
        "",
        f"latency p50 {latency['p50'] or 0:.2f}s  p95 {latency['p95'] or 0:.2f}s (reported, not gated)",
        "",
        "bars",
    ]
    for name, bar in report["bars"].items():
        lines.append(
            f"  {'ok  ' if bar['ok'] else 'FAIL'} {name:<18}{pct(bar['value'])}  (bar {bar['bar']:.2f})"
        )
    lines.append(
        f"  {'ok  ' if not report['tool_argument_leaks'] else 'FAIL'} tool-argument leaks  {len(report['tool_argument_leaks'])}"
    )
    k = report["kappa"]
    if k["value"] is None:
        lines.append(
            "  n/a  kappa              not measured (needs --verdicts from the two judges)"
        )
    else:
        lines.append(
            f"  {'ok  ' if k['value'] >= k['bar'] else 'LOW '} kappa              {k['value']:.2f} over {k['pairs']} fixtures (bar {k['bar']:.1f}; GuardianAI decides)"
        )
    lines += ["", f"leaks {len(report['leaks'])}"]
    lines += [f"  leak {i} {t} {lang} {value!r}" for i, t, lang, value in report["leaks"]]
    return "\n".join(lines)
