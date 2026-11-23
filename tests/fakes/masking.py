"""A stand-in for the Chitragupta model, which is a heavyweight dependency we do not own.

It finds the names a test planted, so every other masking layer runs for real. Set
``NAMES`` to the planted names; ``settings.MASKING_NER_BACKEND`` points at ``NameFinder``.
"""

from __future__ import annotations

import re

from overbae.services.masking.patterns import Hit

NAMES: list[str] = []
SPEC = "fakes.masking:NameFinder"


class NameFinder:
    def detect(self, text: str) -> list[Hit]:
        hits = []
        for name in NAMES:
            for match in re.finditer(re.escape(name), text, re.IGNORECASE):
                hits.append(Hit(match.start(), match.end(), "PERSON"))
        return hits


class Broken:
    def detect(self, text: str) -> list[Hit]:
        raise RuntimeError("model crashed")


CLOUD_KEY = "cloud-secret"


def enable_remote(settings, fake_llm, tmp_path, names=()):
    """What GuardianAI's split gives the masking worker: the name model, the remote key in a
    file, and a gateway that lets only that key call the remote model."""
    settings.MASKING_NER_BACKEND = SPEC
    settings.MASKING_ENABLED = True
    settings.MASKING_QUEUE_ENABLED = True
    settings.MASKING_TIMEOUT_SECONDS = 5
    settings.MASKING_MAX_BYTES = 24_000
    NAMES[:] = list(names)
    key = tmp_path / "cloud.key"
    key.write_text(CLOUD_KEY + "\n")
    settings.LLM_CLOUD_API_KEY_FILE = str(key)
    fake_llm.cloud_key = CLOUD_KEY
