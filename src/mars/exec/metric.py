from __future__ import annotations
import re
from typing import Optional, Tuple


def parse_metric(stdout_text: str, final_metric_regex: str) -> Tuple[Optional[float], bool]:
    """Parse metric from stdout using a regex with one capture group."""
    pattern = re.compile(final_metric_regex, flags=re.MULTILINE)
    m = pattern.search(stdout_text or "")
    if not m:
        return None, False
    try:
        return float(m.group(1)), True
    except Exception:
        return None, False


def summarize_error(stderr_text: str, max_chars: int = 2000) -> str:
    s = (stderr_text or "").strip()
    if len(s) > max_chars:
        return s[:max_chars] + "\n... (truncated) ..."
    return s
