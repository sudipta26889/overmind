"""The context detector: a zero-shot GLiNER that reads a sentence for medical, place, relation
and lender words no list holds. It runs beside Chitragupta; the masker takes the union.

Model: ``knowledgator/gliner-x-base`` (Apache-2.0, mT5 encoder, multilingual), pinned by revision
and fetched at image build. ``trust_remote_code`` is never set; the directory holds weights and
config only. ``masking.quantise`` saves it as int8 (``model.int8.pt``) at build time.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from overbae.services.masking.errors import MaskingError
from overbae.services.masking.patterns import Hit

logger = logging.getLogger(__name__)

QUANTISED = "model.int8.pt"
LABELS = {
    "medical condition": "CONDITION",
    "symptom": "CONDITION",
    "medication": "MEDICATION",
    "medical test": "TEST",
    "street address": "ADDRESS",
    "neighbourhood or locality": "ADDRESS",
    "city": "ADDRESS",
    "family relation": "RELATION",
    "bank or lender": "ORG",
}
# The model reads about 384 words; longer text goes in pieces so nothing is cut off unread.
CHUNK_WORDS = 150
_WORD = re.compile(r"\S+")
_instance: Detector | None = None
_lock = threading.Lock()


def chunks(text: str, size: int = CHUNK_WORDS) -> list[tuple[int, int]]:
    words = [m.span() for m in _WORD.finditer(text)]
    return [
        (words[i][0], words[min(i + size, len(words)) - 1][1]) for i in range(0, len(words), size)
    ]


class Detector:
    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold

    def detect(self, text: str) -> list[Hit]:
        hits: list[Hit] = []
        for start, end in chunks(text):
            for found in self.model.predict_entities(
                text[start:end], list(LABELS), threshold=self.threshold
            ):
                kind = LABELS.get(found["label"])
                if kind and found["end"] > found["start"]:
                    hits.append(Hit(start + found["start"], start + found["end"], kind))
        return hits


def load(directory: str, revision: str, threshold: float) -> Detector:
    global _instance
    with _lock:
        if _instance is not None:
            return _instance
        marker = Path(directory) / ".revision"
        if not revision or not marker.is_file() or marker.read_text().strip() != revision:
            raise MaskingError("model_revision")
        try:
            import torch  # noqa: PLC0415 — the masking extra is not installed in other images

            saved = Path(directory) / QUANTISED
            if saved.is_file():
                model = torch.load(saved, weights_only=False, map_location="cpu", mmap=True)
            else:
                from gliner import GLiNER  # noqa: PLC0415

                model = GLiNER.from_pretrained(directory, local_files_only=True)
            model.eval()
            logger.info(
                "Masking context model loaded (%s)", "saved int8" if saved.is_file() else "fp32"
            )
            _instance = Detector(model, threshold)
        except MaskingError:
            raise
        except Exception as exc:
            raise MaskingError("model_unavailable") from exc
        return _instance
