"""The name detector. Chitragupta is the only model; it is loaded once per worker process.

``MASKING_NER_BACKEND`` (``package.module:Class``) names a replacement for tests; production
leaves it empty. A backend that is not loaded raises ``MaskingError``: masking fails closed.
"""

from __future__ import annotations

import importlib
import re
from functools import cache
from pathlib import Path
from typing import Protocol

from django.conf import settings

from overbae.services.masking.errors import MaskingError
from overbae.services.masking.patterns import Hit


class Ner(Protocol):
    def detect(self, text: str) -> list[Hit]: ...


class Union:
    """Every detector in turn; their hits are merged by the masker's overlap rule."""

    def __init__(self, *members: Ner):
        self.members = members

    def detect(self, text: str) -> list[Hit]:
        return [hit for member in self.members for hit in member.detect(text)]


@cache
def _load(spec: str, model_dir: str, context_dir: str = "") -> Ner:
    if spec:
        module, _, name = spec.partition(":")
        return getattr(importlib.import_module(module), name)()
    if not model_dir or not Path(model_dir).is_dir():
        raise MaskingError("model_not_loaded")
    try:
        from overbae.services.masking import chitragupta
    except ImportError as exc:
        raise MaskingError("model_unavailable") from exc
    names = chitragupta.load(model_dir, getattr(settings, "MASKING_MODEL_REVISION", ""))
    if not context_dir:
        return names
    try:
        from overbae.services.masking import gliner_ner
    except ImportError as exc:
        raise MaskingError("model_unavailable") from exc
    return Union(
        names,
        gliner_ner.load(
            context_dir,
            getattr(settings, "MASKING_CONTEXT_REVISION", ""),
            float(getattr(settings, "MASKING_CONTEXT_THRESHOLD", 0.15)),
        ),
    )


def backend() -> Ner:
    return _load(
        getattr(settings, "MASKING_NER_BACKEND", ""),
        getattr(settings, "MASKING_MODEL_DIR", ""),
        getattr(settings, "MASKING_CONTEXT_DIR", ""),
    )


_IDENTIFIER = re.compile(r"^[a-z0-9_.:/\-]{1,24}$")


def needs_ner(text: str) -> bool:
    """Any text with letters is read by the model, and any Devanagari or Bengali text, except a
    short lowercase token (a tool name, a status). A short sentence can hold a name just as a
    long one can, so length alone is not a reason to skip the model."""
    if any("\u0900" <= ch <= "\u09ff" for ch in text):
        return True
    if sum(ch.isalpha() for ch in text) < 2:
        return False
    return not _IDENTIFIER.match(text)
