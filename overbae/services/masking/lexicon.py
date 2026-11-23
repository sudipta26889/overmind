"""The curated lexicon: conditions, medications, tests, lenders and relationship words.

Files are ``lexicon/{en,hi,bn}.tsv``: ``term<TAB>TYPE[<TAB>roman variant|roman variant]``. The
optional third column is the transliteration table: Hindi and Bengali terms as people type them
in Latin script (Hinglish, Banglish). Matching is on whole words, after NFKC and casefold, over
every file at once: a Hindi term turns up in English text too.
"""

from __future__ import annotations

import re
import unicodedata
from functools import cache
from pathlib import Path

from overbae.services.masking.patterns import (
    NOT_WORD_AFTER,
    NOT_WORD_BEFORE,
    VALUE_AFTER_TEST,
    Hit,
    neutral,
)

DIRECTORY = Path(__file__).parent / "lexicon"
LANGUAGES = ("en", "hi", "bn", "places")
# The surrogate type each lexicon type becomes.
SURROGATE = {
    "CONDITION": "CONDITION",
    "MEDICATION": "MEDICATION",
    "TEST": "TEST",
    "FINANCE_ORG": "ORG",
    "RELATION": "RELATION",
    "LOCATION": "LOCATION",
}


def normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def read(language: str) -> list[tuple[str, str, list[str]]]:
    rows = []
    for number, line in enumerate((DIRECTORY / f"{language}.tsv").read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if len(cells) < 2 or cells[1] not in SURROGATE:
            raise ValueError(f"{language}.tsv line {number}: expected term<TAB>TYPE")
        romans = [r.strip() for r in cells[2].split("|")] if len(cells) > 2 and cells[2] else []
        rows.append((cells[0].strip(), cells[1], [r for r in romans if r]))
    return rows


@cache
def terms() -> dict[str, str]:
    """Every term and roman variant, casefolded and normalised, mapped to its surrogate type."""
    found: dict[str, str] = {}
    for language in LANGUAGES:
        for term, kind, romans in read(language):
            for form in (term, *romans):
                found.setdefault(normalise(form).casefold(), SURROGATE[kind])
    return found


@cache
def matcher() -> re.Pattern:
    ordered = sorted(terms(), key=len, reverse=True)
    body = "|".join(re.escape(term) for term in ordered)
    return re.compile(f"{NOT_WORD_BEFORE}(?:{body}){NOT_WORD_AFTER}", re.IGNORECASE)


def detect(text: str) -> list[Hit]:
    """Lexicon hits in ``text``, which must already be NFKC-normalised."""
    table = terms()
    found = []
    flat = neutral(text)
    for match in matcher().finditer(flat):
        kind = table.get(normalise(match.group()).casefold())
        if kind:
            found.append(Hit(match.start(), match.end(), kind))
        if kind == "TEST":
            # A reading follows the test it belongs to: "ECG shows 72", "फेरिटिन में 18".
            reading = VALUE_AFTER_TEST.match(flat, match.end())
            if reading:
                found.append(Hit(*reading.span(1), "VALUE"))
    return found
