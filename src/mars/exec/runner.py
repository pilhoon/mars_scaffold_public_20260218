from __future__ import annotations
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


def run_cmd(cmd: str, cwd: str | Path, timeout_sec: int, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str, float]:
    """Run a command with timeout. Returns (exit_code, stdout, stderr, elapsed_sec)."""
    start = time.time()
    p = subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    elapsed = time.time() - start
    return p.returncode, p.stdout, p.stderr, elapsed
