"""The name model: Chitragupta, an XLM-RoBERTa token tagger for Indian PII, run on CPU.

This is a loader, not a copy of the upstream ``chitragupta.py``: it uses the standard
``transformers`` classes with ``trust_remote_code`` off, so no code comes from the model
directory. The rule layer (``rules.py``, ``ids.py``) is vendored from the pinned revision.
The model directory must hold the revision named in ``MASKING_MODEL_REVISION`` (written to
``.revision`` when the image is built); any other directory is refused.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from overbae.services.masking.chitragupta.rules import postprocess
from overbae.services.masking.errors import MaskingError
from overbae.services.masking.patterns import Hit

# Where the surrogate types come from: the model's label, as the masker names it.
LABELS = {
    "PERSON": "PERSON",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "ADDRESS": "ADDRESS",
    "PINCODE": "ID",
    "DOB": "DATE",
    "AADHAAR": "ID",
    "PAN": "ID",
    "VOTER_ID": "ID",
    "PASSPORT": "ID",
    "DRIVING_LICENCE": "ID",
    "VEHICLE_REG": "ID",
    "GSTIN": "ID",
    "IP_ADDRESS": "ID",
    "UPI_ID": "ACCOUNT",
    "BANK_ACCOUNT": "ACCOUNT",
    "IFSC": "ACCOUNT",
    "CARD_NUMBER": "ACCOUNT",
}
logger = logging.getLogger(__name__)
QUANTISED = "model.int8.pt"
MAX_LEN = 510
WINDOW = 256
STRIDE = 96

_instance: Tagger | None = None
_lock = threading.Lock()


def _decode(offsets, tags):
    """Token tags to character spans (label, start, end)."""
    spans, cur = [], None
    for (a, b), tag in zip(offsets, tags, strict=True):
        if a == b:
            continue
        if tag == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        prefix, label = tag.split("-", 1)
        # A bare sentencepiece "▁" overlaps the next character; an overlapping B- of the same
        # label continues the span instead of opening a one-character duplicate.
        if prefix == "B" and cur is not None and cur[0] == label and a < cur[2]:
            cur[2] = max(cur[2], b)
        elif prefix == "B" or cur is None or cur[0] != label:
            if cur:
                spans.append(cur)
            cur = [label, a, b]
        else:
            cur[2] = b
    if cur:
        spans.append(cur)
    return [tuple(span) for span in spans]


def _strip(text, span):
    label, a, b = span
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return label, a, b


def quantise(model):
    """int8 weights for every linear layer and every embedding table."""
    import torch  # noqa: PLC0415
    from torch.ao.quantization import float_qparams_weight_only_qconfig  # noqa: PLC0415

    return torch.quantization.quantize_dynamic(
        model,
        {
            torch.nn.Linear: torch.ao.quantization.default_dynamic_qconfig,
            torch.nn.Embedding: float_qparams_weight_only_qconfig,
        },
    )


class Tagger:
    def __init__(self, directory: str, *, quantize: bool = True):
        import torch  # noqa: PLC0415 — the masking extra is not installed in other images
        from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=False)
        saved = Path(directory) / QUANTISED
        if saved.is_file():
            # Built into the image by ``masking.quantise``: int8 weights go straight into
            # memory and fp32 never exists in this process. The file is our own build output.
            model = torch.load(saved, weights_only=False, map_location="cpu", mmap=True).eval()
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                directory, trust_remote_code=False, use_safetensors=True
            ).eval()
            if quantize:
                model = quantise(model)
        logger.info(
            "Masking name model: %d int8 layers (%s)",
            sum("quantized" in type(m).__module__ for m in model.modules()),
            "saved weights" if saved.is_file() else "quantised at load",
        )
        self.model = model
        self.id2label = self.model.config.id2label

    def _windows(self, n: int):
        if n <= MAX_LEN:
            return [(0, n)]
        step = WINDOW - STRIDE
        starts = [*range(0, n - WINDOW, step), n - WINDOW]
        return [(s, s + WINDOW) for s in starts]

    def spans(self, text: str):
        torch = self._torch
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            split_special_tokens=True,
        )
        ids = enc["input_ids"]
        best = [(-1, "O")] * len(ids)
        cls, sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        with torch.no_grad():
            for a, b in self._windows(len(ids)):
                x = torch.tensor([[cls, *ids[a:b], sep]])
                logits = self.model(input_ids=x, attention_mask=torch.ones_like(x)).logits
                for j, p in zip(range(a, b), logits.argmax(-1)[0, 1:-1].tolist(), strict=True):
                    edge = min(j - a, b - 1 - j)
                    if edge > best[j][0]:
                        best[j] = (edge, self.id2label[p])
        found = {_strip(text, s) for s in _decode(enc["offset_mapping"], [t for _, t in best])}
        return postprocess(text, [s for s in found if s[1] < s[2]])

    def detect(self, text: str) -> list[Hit]:
        return [
            Hit(a, b, LABELS[label])
            for label, a, b in self.spans(text)
            if label in LABELS and a < b
        ]


def load(directory: str, revision: str) -> Tagger:
    """The one tagger of this process. It is loaded once, in the worker's ``worker_init``,
    before the pool forks; a directory that is not the pinned revision is refused."""
    global _instance
    with _lock:
        if _instance is not None:
            return _instance
        marker = Path(directory) / ".revision"
        if not revision or not marker.is_file() or marker.read_text().strip() != revision:
            raise MaskingError("model_revision")
        try:
            _instance = Tagger(directory)
        except MaskingError:
            raise
        except Exception as exc:
            raise MaskingError("model_unavailable") from exc
        return _instance
