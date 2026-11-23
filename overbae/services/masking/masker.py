"""Masks a judge state: every identifier, amount, date, measured value and health or finance
term becomes a ``<TYPE_n>`` surrogate, numbered by first appearance within one call.

The map lives in the ``MaskedPayload`` and nowhere else. Anything the masker is unsure about
raises ``MaskingError``; nothing is ever sent unmasked.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from django.conf import settings

from overbae.services.masking import lexicon, ner, patterns, residual
from overbae.services.masking.errors import MaskingError
from overbae.services.masking.patterns import Hit
from overbae.services.masking.payload import MaskedPayload, issue

_TEXT_PARTS = {"text", "input_text", "output_text"}
_ATTACHMENT_PARTS = {
    "image_url",
    "image",
    "input_image",
    "file",
    "input_file",
    "audio",
    "input_audio",
}
_DATA_URL = re.compile(r"^\s*data:[\w/+.\-]+(?:;[\w=.\-]+)*;base64,", re.IGNORECASE)
_BASE64 = re.compile(r"^[A-Za-z0-9+/\s]{256,}={0,2}\s*$")
_EXISTING = re.compile(r"<((?:[A-Z]+(?:_[A-Z]+)*_\d+)|ATTACHMENT)>")

# A key whose value is always masked, whatever it says. ``name`` is exempt for tool and
# function metadata, whose values are the judge's evidence of what was called.
_KEY_TYPES = {
    "name": "PERSON",
    "patient": "PERSON",
    "phone": "PHONE",
    "mobile": "PHONE",
    "email": "EMAIL",
    "address": "ADDRESS",
    "dob": "DATE",
    "amount": "AMOUNT",
    "account": "ACCOUNT",
    "policy": "ACCOUNT",
    "upi": "ACCOUNT",
    "ifsc": "ACCOUNT",
    "pan": "ID",
    "aadhaar": "ID",
    "aadhar": "ID",
}
_LONG_KEYS = {k for k in _KEY_TYPES if len(k) >= 5}
_NAME_EXEMPT = {
    "tool",
    "function",
    "model",
    "capability",
    "agent",
    "span",
    "trace",
    "class",
    "method",
    "event",
    "metric",
    "column",
}
_TOOL_SIBLINGS = {"arguments", "parameters"}
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def key_tokens(key: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", _CAMEL.sub("_", key).lower()) if t]


def key_type(key: str, siblings: Mapping[str, Any] | None = None) -> str | None:
    """The surrogate type for a sensitive key, or None."""
    tokens = key_tokens(key)
    found = next((_KEY_TYPES[t] for t in tokens if t in _KEY_TYPES), None)
    if found is None:
        flat = "".join(tokens)
        found = next((_KEY_TYPES[k] for k in _LONG_KEYS if k in flat), None)
    exempt = _NAME_EXEMPT & set(tokens) or (siblings and _TOOL_SIBLINGS & set(siblings))
    if found == "PERSON" and "patient" not in tokens and exempt:
        return None
    return found


def looks_like_attachment(text: str) -> bool:
    if _DATA_URL.match(text):
        return True
    if _BASE64.match(text):
        try:
            base64.b64decode("".join(text.split()), validate=True)
        except (binascii.Error, ValueError):
            return False
        return True
    return False


def resolve(hits: list[Hit]) -> list[Hit]:
    """Longest hit first, then by priority; a hit that overlaps one already kept is dropped."""
    kept: list[Hit] = []
    for hit in sorted(hits, key=lambda h: (-(h.end - h.start), -h.priority, h.start)):
        if not any(hit.start < other.end and other.start < hit.end for other in kept):
            kept.append(hit)
    return sorted(kept, key=lambda h: h.start)


# The identifiers a gateway pattern filter refuses: what an ids-only mask still replaces.
IDENTIFIER_TYPES = frozenset({"EMAIL", "PHONE", "ACCOUNT", "ID"})


class Masker:
    # ids_only is for the masking_kappa comparison alone: only structured identifiers are
    # replaced and no key forces a type. Production masking never sets it.
    ids_only = False

    def __init__(self, *, deadline: float):
        self.mapping: dict[str, str] = {}
        self._index: dict[tuple[str, str], str] = {}
        self._counts: Counter[str] = Counter()
        self._deadline = deadline
        self._ner: ner.Ner | None = None

    def surrogate(self, type_: str, original: str) -> str:
        key = (type_, lexicon.normalise(original).casefold().strip())
        if key not in self._index:
            self._counts[type_] += 1
            token = f"<{type_}_{self._counts[type_]}>"
            self._index[key] = token
            self.mapping[token] = original
        return self._index[key]

    def _check_time(self) -> None:
        if time.monotonic() >= self._deadline:
            raise MaskingError("timeout")

    def _detect(self, text: str) -> list[Hit]:
        if self.ids_only:
            found = patterns.structured(patterns.neutral(text))
            return [hit for hit in found if hit.type in IDENTIFIER_TYPES]
        hits = patterns.detect(text) + lexicon.detect(text)
        if ner.needs_ner(text):
            if self._ner is None:
                self._ner = ner.backend()
            try:
                hits += self._ner.detect(text)
            except MaskingError:
                raise
            except Exception as exc:
                raise MaskingError("ner_error") from exc
        return hits

    def text(self, text: str, *, force: str | None = None) -> str:
        self._check_time()
        if force:
            return self.surrogate(force, text) if text.strip() else text
        if looks_like_attachment(text):
            return "<ATTACHMENT>"
        decoded = self._json(text)
        if decoded is not None:
            return decoded
        text = _EXISTING.sub(lambda m: f"<ESC:{m.group(1)}>", lexicon.normalise(text))
        out, cursor = [], 0
        for hit in resolve(self._detect(text)):
            out.append(text[cursor : hit.start])
            out.append(self.surrogate(hit.type, text[hit.start : hit.end]))
            cursor = hit.end
        out.append(text[cursor:])
        return "".join(out)

    def _json(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped or stripped[0] not in "{[":
            return None
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return None
        return json.dumps(self.value(parsed), ensure_ascii=False)

    def value(self, value: Any, key: str | None = None, siblings: Mapping | None = None) -> Any:
        self._check_time()
        forced = key_type(key, siblings) if key and not self.ids_only else None
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, str):
            return self.text(value, force=forced)
        if isinstance(value, (int, float)):
            if forced:
                return self.surrogate(forced, str(value))
            masked = self.text(str(value))
            return masked if masked != str(value) else value
        if isinstance(value, list):
            if key == "content":
                return [self.part(item) for item in value]
            return [self.value(item, key) for item in value]
        if isinstance(value, dict):
            if value.get("type") in _ATTACHMENT_PARTS:
                return "<ATTACHMENT>"
            # A key is text too: {"Rahul Sharma": ...} or {"9876543210": ...} is a value in key
            # position. The value is still read under the original key's meaning.
            return {self.text(str(k)): self.value(v, str(k), value) for k, v in value.items()}
        raise MaskingError("non_text_part")

    def part(self, item: Any) -> Any:
        """One part of a message's content: text is masked, a file or image becomes an
        attachment, and any other kind of part is refused."""
        if isinstance(item, str):
            return self.text(item)
        if isinstance(item, dict):
            kind = item.get("type")
            if kind in _ATTACHMENT_PARTS:
                return "<ATTACHMENT>"
            if kind in _TEXT_PARTS:
                return {k: self.value(v, str(k), item) for k, v in item.items()}
        raise MaskingError("non_text_part")


def _limits() -> tuple[float, int]:
    return (
        float(getattr(settings, "MASKING_TIMEOUT_SECONDS", 5)),
        int(getattr(settings, "MASKING_MAX_BYTES", 24_000)),
    )


def mask_state(
    state: Mapping[str, Any],
    *,
    fit: Callable[[dict], dict] | None = None,
    after_mask: Callable[[dict], dict] | None = None,
) -> MaskedPayload:
    """Masks the uncut state, then applies ``after_mask`` and ``fit`` to the masked copy, and
    checks the exact string that would be sent. The result is the only thing a remote call takes."""
    timeout, max_bytes = _limits()
    try:
        size = len(json.dumps(state, ensure_ascii=False, default=str).encode())
    except (TypeError, ValueError) as exc:
        raise MaskingError("non_text_part") from exc
    if size > max_bytes:
        raise MaskingError("too_large")
    masker = Masker(deadline=time.monotonic() + timeout)
    masked = masker.value(dict(state))
    try:
        if after_mask is not None:
            masked = after_mask(masked)
        if fit is not None:
            masked = fit(masked)
    except MaskingError:
        raise
    except Exception as exc:
        # A cut that cannot read the masked shape sends nothing; the item goes local.
        raise MaskingError("cut") from exc
    final = json.dumps(masked, ensure_ascii=False)
    residual.require_clean(final, masker.mapping)
    return issue(final, final, masker.mapping)


def derive(payload: MaskedPayload, prefix: str) -> MaskedPayload:
    """A prompt that wraps an already-masked state in fixed wording, with the same surrogate
    map, so an explanation never rebuilds raw text. The wording is checked too."""
    residual.require_clean(prefix, payload.mapping, structured=False)
    return issue(prefix + payload.state_json, payload.state_json, payload.mapping)


class _IdsOnlyMasker(Masker):
    ids_only = True


def mask_ids_only(
    state: Mapping[str, Any], *, fit: Callable[[dict], dict] | None = None
) -> MaskedPayload:
    """Replaces only structured identifiers; names, health terms, amounts and dates stay. Used
    by ``masking_kappa`` alone, to measure what full masking costs the judge; the residual
    check still insists no identifier is left."""
    timeout, max_bytes = _limits()
    try:
        size = len(json.dumps(state, ensure_ascii=False, default=str).encode())
    except (TypeError, ValueError) as exc:
        raise MaskingError("non_text_part") from exc
    if size > max_bytes:
        raise MaskingError("too_large")
    masker = _IdsOnlyMasker(deadline=time.monotonic() + timeout)
    masked = masker.value(dict(state))
    try:
        if fit is not None:
            masked = fit(masked)
    except Exception as exc:
        raise MaskingError("cut") from exc
    final = json.dumps(masked, ensure_ascii=False)
    residual.require_clean(final, masker.mapping, only=residual.IDENTIFIER_KINDS)
    return issue(final, final, masker.mapping)
