"""Function and common words that are never sensitive (``lexicon/stop.txt``).

A span from a model that is made only of these words is not masked, and a roman lexicon variant
that is one of them is not loaded: that is how "er", "jonno" and "premium" stop becoming
RELATION, ADDRESS or ORG.
"""

from __future__ import annotations

import re
import unicodedata
from functools import cache
from pathlib import Path

from overbae.services.masking.patterns import WORD

_TOKEN = re.compile(f"{WORD}+")


def normal(word: str) -> str:
    return unicodedata.normalize("NFKC", word).casefold().strip()


@cache
def words() -> frozenset[str]:
    lines = (Path(__file__).parent / "lexicon" / "stop.txt").read_text().splitlines()
    return frozenset(normal(line) for line in lines if line.strip() and not line.startswith("#"))


def is_stop(word: str) -> bool:
    return normal(word) in words()


def only_stop_words(text: str) -> bool:
    """True when ``text`` has letters and every word in it is a stop-list word."""
    tokens = [t for t in _TOKEN.findall(text) if any(c.isalpha() for c in t)]
    return bool(tokens) and all(is_stop(t) for t in tokens)
