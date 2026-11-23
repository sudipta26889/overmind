"""The remote call. It takes a ``MaskedPayload`` and nothing else, on the remote key.

The remote key sits in a file only the masking worker is given. The gateway allows that key
the remote judge model and nothing else, and refuses the default key that model, so a call that
bypasses this module is refused there.
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from pydantic import BaseModel

from overbae.core.llms import call_llm
from overbae.services.masking.errors import MaskingError
from overbae.services.masking.payload import MaskedPayload


def cloud_key() -> str:
    path = getattr(settings, "LLM_CLOUD_API_KEY_FILE", "")
    try:
        key = Path(path).read_text().strip() if path else ""
    except OSError:
        key = ""
    if not key:
        raise MaskingError("cloud_key_missing")
    return key


def call_masked(
    payload: MaskedPayload,
    *,
    model: str,
    system_prompt: str | None = None,
    response_format: type[BaseModel] | None = None,
    request_kwargs: dict | None = None,
    max_tokens: int = 5000,
) -> tuple[str, dict]:
    """One attempt, no fallback models: every remote request follows a fresh read of the
    project's flag. ``payload.text`` is exactly what is sent."""
    if not isinstance(payload, MaskedPayload):
        raise TypeError("Only a masked payload may be sent to a remote model.")
    return call_llm(
        payload.text,
        system_prompt=system_prompt,
        model=model,
        response_format=response_format,
        request_kwargs=request_kwargs,
        max_tokens=max_tokens,
        max_attempts=1,
        api_key=cloud_key(),
    )
