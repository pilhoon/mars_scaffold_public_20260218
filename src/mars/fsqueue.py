from __future__ import annotations
import errno
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FSQueuePaths:
    root: Path
    pending: Path
    running: Path
    done: Path
    failed: Path


def ensure_queue_dirs(base_dir: str | Path, queue_name: str) -> FSQueuePaths:
    root = Path(base_dir) / "spool" / queue_name
    pending = root / "pending"
    running = root / "running"
    done = root / "done"
    failed = root / "failed"
    for p in (pending, running, done, failed):
        p.mkdir(parents=True, exist_ok=True)
    return FSQueuePaths(root=root, pending=pending, running=running, done=done, failed=failed)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic on POSIX


def enqueue(paths: FSQueuePaths, job: Dict[str, Any]) -> str:
    job_id = job.get("job_id") or str(uuid.uuid4())
    job = dict(job)
    job["job_id"] = job_id
    job_path = paths.pending / f"{job_id}.json"
    _atomic_write_json(job_path, job)
    return job_id


def claim(paths: FSQueuePaths, worker_id: str) -> Optional[Path]:
    """Claim the oldest pending job by atomic rename pending->running."""
    jobs = sorted(paths.pending.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for p in jobs:
        # First, atomically claim within the same directory. This remains atomic
        # even in environments where cross-directory rename is unsupported.
        claimed = p.with_name(f"{p.stem}.{worker_id}.claim")
        try:
            os.replace(p, claimed)
        except FileNotFoundError:
            continue
        except PermissionError:
            continue

        target = paths.running / p.name
        try:
            os.replace(claimed, target)
            return target
        except OSError as e:
            if e.errno != errno.EXDEV:
                # Best effort rollback so another worker can retry.
                try:
                    os.replace(claimed, p)
                except Exception:
                    pass
                continue

            # Cross-device move fallback: copy into running and remove claimed.
            try:
                shutil.copy2(claimed, target)
                os.unlink(claimed)
                return target
            except FileNotFoundError:
                continue
        except FileNotFoundError:
            continue
    return None


def complete(paths: FSQueuePaths, running_job_path: Path, result: Dict[str, Any], ok: bool) -> Path:
    job_id = running_job_path.stem
    out_dir = paths.done if ok else paths.failed
    out_path = out_dir / f"{job_id}.result.json"
    _atomic_write_json(out_path, result)
    # Preserve claimed job payload for traceability while clearing running state.
    job_trace_path = out_dir / f"{job_id}.job.json"
    try:
        os.replace(running_job_path, job_trace_path)
    except FileNotFoundError:
        pass
    except OSError:
        # Fallback for uncommon cross-device or transient rename failures.
        shutil.copy2(running_job_path, job_trace_path)
        try:
            os.unlink(running_job_path)
        except FileNotFoundError:
            pass
    return out_path


def read_job(job_path: Path) -> Dict[str, Any]:
    return json.loads(job_path.read_text(encoding="utf-8"))


def read_result(result_path: Path) -> Dict[str, Any]:
    return json.loads(result_path.read_text(encoding="utf-8"))


def wait_for_result(paths: FSQueuePaths, job_id: str, poll_sec: float = 0.2, timeout_sec: float = 3600.0) -> Dict[str, Any]:
    """Controller-side helper: wait until job_id.result.json appears in done/failed."""
    deadline = time.time() + timeout_sec
    done_path = paths.done / f"{job_id}.result.json"
    fail_path = paths.failed / f"{job_id}.result.json"
    while time.time() < deadline:
        if done_path.exists():
            return read_result(done_path)
        if fail_path.exists():
            return read_result(fail_path)
        time.sleep(poll_sec)
    raise TimeoutError(f"Timed out waiting for result job_id={job_id}")
