"""The hours the local judge may start a call, in India time.

``LOCAL_JUDGE_HOURS_IST`` is ``"<start>-<end>"``: the start hour is inclusive and the end hour
exclusive, in Asia/Kolkata. A window may wrap midnight (``"23-2"``). An empty value means the
judge is open at every hour. This module imports nothing from Django so settings can validate
the value at start-up.
"""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
_WINDOW = re.compile(r"^(\d{1,2})-(\d{1,2})$")


def parse(value: str) -> tuple[int, int] | None:
    """The (start, end) hours, or None for no window. Anything else malformed raises."""
    value = (value or "").strip()
    if not value:
        return None
    match = _WINDOW.match(value)
    if not match:
        raise ValueError(f"LOCAL_JUDGE_HOURS_IST={value!r} must look like '1-8' or '23-2'")
    start, end = int(match.group(1)), int(match.group(2))
    if not (0 <= start <= 23 and 0 <= end <= 24):
        raise ValueError(f"LOCAL_JUDGE_HOURS_IST={value!r}: hours run 0-23 (end may be 24)")
    if start == end:
        raise ValueError(f"LOCAL_JUDGE_HOURS_IST={value!r}: start and end must differ")
    return start, end


def is_open(window: tuple[int, int] | None, moment: datetime) -> bool:
    if window is None:
        return True
    start, end = window
    hour = moment.astimezone(IST).hour
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end
