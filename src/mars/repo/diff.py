from __future__ import annotations
import subprocess
from pathlib import Path


def _decode_stdout(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def diff_summary(repo_a: str | Path, repo_b: str | Path, max_lines: int = 400) -> str:
    """Return a short diff between two repos (best-effort).
    Uses `git diff --no-index` so repos don't need to be git repos.
    """
    p = subprocess.run(
        ["git", "diff", "--no-index", "--", str(repo_a), str(repo_b)],
        capture_output=True,
    )
    out = _decode_stdout(p.stdout)
    lines = out.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (truncated) ..."]
    return "\n".join(lines)
