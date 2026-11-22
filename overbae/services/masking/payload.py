"""``MaskedPayload``: the only thing a remote call accepts.

It is built only by this package (``issue`` needs a token no other module holds), so a caller
cannot hand a raw string to the remote path by constructing one. The surrogate map stays in
this process and out of ``repr``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN = object()
SURROGATE = re.compile(r"<([A-Z]+(?:_[A-Z]+)*_\d+)>")
ESCAPED = re.compile(r"<ESC:([A-Z]+(?:_[A-Z]+)*(?:_\d+)?)>")


@dataclass(frozen=True)
class MaskedPayload:
    text: str
    state_json: str
    mapping: dict[str, str] = field(repr=False, compare=False)
    token: object = field(repr=False, compare=False, default=None)

    def __post_init__(self):
        if self.token is not _TOKEN:
            raise TypeError("A MaskedPayload is built only by the masker.")

    def restore(self, text: str) -> str:
        """Puts the original values back into ``text``, locally, for what a remote model wrote."""
        restored = SURROGATE.sub(lambda m: self.mapping.get(m.group(0), m.group(0)), text)
        return ESCAPED.sub(lambda m: f"<{m.group(1)}>", restored)


def issue(text: str, state_json: str, mapping: dict[str, str]) -> MaskedPayload:
    return MaskedPayload(text=text, state_json=state_json, mapping=mapping, token=_TOKEN)
