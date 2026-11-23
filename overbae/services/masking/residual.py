"""The residual check, on the exact string that would be sent.

It runs the regex detectors, the six-digit rule, the key rule and the lexicon again, and checks
every surrogate is one this call issued, so a cut or an encoding that exposes a value is caught.
It is not an independent second detector: it cannot find a name the model missed, and free prose
about a person's health can pass both layers. That limit is stated, not hidden.
"""

from __future__ import annotations

import json
import re
from typing import Any

from overbae.services.masking import lexicon, patterns
from overbae.services.masking.errors import MaskingError
from overbae.services.masking.payload import SURROGATE

# What an ids-only check may report: the structured identifiers and a surrogate this call did
# not issue. Names, health terms and the rest are allowed to stay in that mode.
IDENTIFIER_KINDS = frozenset({"EMAIL", "PHONE", "ACCOUNT", "ID", "surrogate"})
_MASKED = re.compile(r"^(?:<[A-Z]+(?:_[A-Z]+)*_\d+>|<ATTACHMENT>)$")


def _walk(value: Any, key: str | None, siblings: Any, kinds: set[str], mapping: dict) -> None:
    from overbae.services.masking.masker import (
        key_type as _key_type,  # noqa: PLC0415 — masker imports us
    )

    def key_type(name, siblings):
        # A key that is itself a surrogate (``<PHONE_1>``) is already masked; its word is not a rule.
        return None if _MASKED.match(name) else _key_type(name, siblings)

    if isinstance(value, dict):
        for k, v in value.items():
            _text(str(k), kinds, mapping)
            _walk(v, str(k), value, kinds, mapping)
    elif isinstance(value, list):
        for item in value:
            _walk(item, key, siblings, kinds, mapping)
    elif isinstance(value, str):
        if key and key_type(key, siblings if isinstance(siblings, dict) else None):
            if value.strip() and not _MASKED.match(value.strip()):
                kinds.add("key")
            return
        _text(value, kinds, mapping)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if key and key_type(key, siblings if isinstance(siblings, dict) else None):
            kinds.add("key")
        else:
            _text(str(value), kinds, mapping)


def _text(text: str, kinds: set[str], mapping: dict) -> None:
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            _walk(json.loads(stripped), None, None, kinds, mapping)
            return
        except ValueError:
            pass
    for hit in patterns.detect(text):
        kinds.add(hit.type)
    for hit in lexicon.detect(text):
        kinds.add(hit.type)


def findings(
    final: str,
    mapping: dict[str, str],
    *,
    structured: bool = True,
    only: frozenset[str] | None = None,
) -> list[str]:
    """The kinds of things still visible in ``final``; empty when it is clean."""
    kinds: set[str] = set()
    unknown = {m.group(0) for m in SURROGATE.finditer(final)} - set(mapping)
    if unknown:
        kinds.add("surrogate")
    if structured:
        try:
            _walk(json.loads(final), None, None, kinds, mapping)
        except ValueError:
            _text(final, kinds, mapping)
    else:
        _text(final, kinds, mapping)
    return sorted(kinds if only is None else kinds & only)


def require_clean(
    final: str,
    mapping: dict[str, str],
    *,
    structured: bool = True,
    only: frozenset[str] | None = None,
) -> None:
    kinds = findings(final, mapping, structured=structured, only=only)
    if kinds:
        error = MaskingError("residual")
        error.kinds = kinds
        raise error
