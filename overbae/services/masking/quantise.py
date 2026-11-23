"""Build-time tool: turn a downloaded model into the small int8 file the worker loads.

    python -m overbae.services.masking.quantise chitragupta SRC DST
    python -m overbae.services.masking.quantise gliner SRC DST

SRC is the pinned snapshot, DST the directory that goes into the image: config and tokenizer
files, ``.revision``, the licence files and ``model.int8.pt``. The fp32 weights stay behind.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

KEEP = {
    "config.json",
    "gliner_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "sentencepiece.bpe.model",
    "LICENSE",
    "NOTICE",
    ".revision",
}


def copy_small(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in KEEP:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)


def chitragupta(src: Path, dst: Path) -> None:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForTokenClassification  # noqa: PLC0415

    from overbae.services.masking import chitragupta as module  # noqa: PLC0415

    model = AutoModelForTokenClassification.from_pretrained(src, use_safetensors=True).eval()
    copy_small(src, dst)
    torch.save(module.quantise(model), dst / module.QUANTISED)


def gliner(src: Path, dst: Path) -> None:
    import json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from gliner import GLiNER  # noqa: PLC0415

    from overbae.services.masking import chitragupta as module  # noqa: PLC0415

    copy_small(src, dst)
    # The default word splitter (stanza) downloads data at run time; whitespace needs nothing.
    config = json.loads((src / "gliner_config.json").read_text())
    config["words_splitter_type"] = "whitespace"
    (dst / "gliner_config.json").write_text(json.dumps(config))
    with tempfile.TemporaryDirectory() as work:
        for entry in src.iterdir():
            (Path(work) / entry.name).symlink_to(entry.resolve())
        (Path(work) / "gliner_config.json").unlink()
        shutil.copy2(dst / "gliner_config.json", Path(work) / "gliner_config.json")
        model = GLiNER.from_pretrained(work, local_files_only=True)
    model.eval()
    torch.save(module.quantise(model), dst / "model.int8.pt")


if __name__ == "__main__":
    kind, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {"chitragupta": chitragupta, "gliner": gliner}[kind](src, dst)
