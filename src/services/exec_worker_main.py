from __future__ import annotations
import argparse
import logging
import re
import sqlite3
import time
from pathlib import Path

from mars.fsqueue import ensure_queue_dirs, claim, read_job, complete
from mars.exec.runner import run_cmd
from mars.exec.metric import parse_metric, summarize_error
from mars.artifacts import ensure_run_dirs, write_text


def _count_jobs_json(path: Path) -> int:
    try:
        return sum(1 for p in path.iterdir() if p.is_file() and p.suffix == ".json")
    except FileNotFoundError:
        return 0


def _llm_profile_count_from_controller_log(run_dir: Path) -> int | None:
    log_path = run_dir / "logs" / "controller.log"
    if not log_path.exists():
        return None
    marker = "llm_profiles="
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            # Read a tail chunk only; latest controller start line is near the end.
            read_from = max(0, size - 1_000_000)
            f.seek(read_from, 0)
            chunk = f.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    for line in reversed(chunk.splitlines()):
        idx = line.find(marker)
        if idx < 0:
            continue
        raw = line[idx + len(marker):].strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if parts:
            return len(parts)
        return None
    return None


def _llm_next_ready_wait_sec(run_dir: Path, now_unix: float) -> float | None:
    db_path = run_dir / "state" / "mars.sqlite"
    if not db_path.exists():
        return None
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
        rows = conn.execute(
            "SELECT key, value FROM best_state WHERE key LIKE 'llm_profile.%.next_ready_unix'"
        ).fetchall()
        count_row = conn.execute(
            "SELECT value FROM best_state WHERE key='llm_profile.count'"
        ).fetchone()

        profile_count = 0
        if count_row and count_row[0] is not None:
            try:
                profile_count = max(0, int(str(count_row[0]).strip()))
            except Exception:
                profile_count = 0
        if profile_count <= 0:
            guessed_count = _llm_profile_count_from_controller_log(run_dir)
            if guessed_count is not None:
                profile_count = max(0, int(guessed_count))

        values_by_index: dict[int, float] = {}
        for key, value in rows:
            m = re.fullmatch(r"llm_profile\.(\d+)\.next_ready_unix", str(key))
            if m is None:
                continue
            try:
                idx = int(m.group(1))
                next_ready = float(value)
            except Exception:
                continue
            values_by_index[idx] = max(0.0, next_ready)

        if profile_count <= 0:
            if not values_by_index:
                return None
            profile_count = max(values_by_index.keys()) + 1

        waits: list[float] = []
        # Missing profile keys are treated as ready-now (0.0).
        for idx in range(profile_count):
            next_ready = float(values_by_index.get(idx, 0.0))
            waits.append(max(0.0, next_ready - now_unix))

        if not waits:
            return None
        return min(waits)
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()


def _estimate_exec_idle_wait(
    run_dir: Path,
    exec_q: object,
    llm_q: object,
    now_unix: float,
) -> tuple[str, str, int, int, int, str]:
    exec_pending = _count_jobs_json(exec_q.pending)
    llm_pending = _count_jobs_json(llm_q.pending)
    llm_running = _count_jobs_json(llm_q.running)
    llm_ready_wait = _llm_next_ready_wait_sec(run_dir, now_unix)

    if exec_pending > 0:
        return "0.0", "exec_pending", exec_pending, llm_pending, llm_running, (
            f"{llm_ready_wait:.1f}" if llm_ready_wait is not None else "unknown"
        )
    if llm_ready_wait is not None and llm_ready_wait > 0:
        return f"{llm_ready_wait:.1f}", "llm_profile_cooldown", exec_pending, llm_pending, llm_running, f"{llm_ready_wait:.1f}"
    if llm_pending > 0 or llm_running > 0:
        return "unknown", "llm_inflight", exec_pending, llm_pending, llm_running, (
            f"{llm_ready_wait:.1f}" if llm_ready_wait is not None else "unknown"
        )
    return "unknown", "no_upstream_jobs", exec_pending, llm_pending, llm_running, (
        f"{llm_ready_wait:.1f}" if llm_ready_wait is not None else "unknown"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("exec_worker")

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--worker-id", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ensure_run_dirs(run_dir)
    q = ensure_queue_dirs(run_dir, "exec")
    llm_q = ensure_queue_dirs(run_dir, "llm")
    logger.info("[exec-worker:%s] start run_dir=%s", args.worker_id, run_dir)
    idle_heartbeat_sec = 60.0
    last_idle_log = 0.0

    while True:
        job_path = claim(q, worker_id=args.worker_id)
        if job_path is None:
            now = time.time()
            if now - last_idle_log >= idle_heartbeat_sec:
                est_wait_sec, reason, exec_pending, llm_pending, llm_running, llm_next_ready_in_sec = _estimate_exec_idle_wait(
                    run_dir,
                    q,
                    llm_q,
                    now,
                )
                logger.info(
                    "[exec-worker:%s] idle waiting for exec jobs est_wait_sec=%s reason=%s exec_pending=%s llm_pending=%s llm_running=%s llm_next_ready_in_sec=%s",
                    args.worker_id,
                    est_wait_sec,
                    reason,
                    exec_pending,
                    llm_pending,
                    llm_running,
                    llm_next_ready_in_sec,
                )
                last_idle_log = now
            time.sleep(0.2)
            continue

        job = read_job(job_path)
        job_id = str(job.get("job_id", ""))
        meta = job.get("meta", {}) or {}
        iter_index = meta.get("iter_index")
        stage = meta.get("stage")
        payload = job.get("payload", {}) or {}
        repo_path = payload.get("repo_path")
        cmd = payload.get("cmd", "python runfile.py")
        timeout_sec = int(payload.get("timeout_sec", 14400))
        regex = payload.get("final_metric_regex", r"^Final Validation Metric:\s*([0-9.eE+-]+)\s*$")
        logger.info(
            "[exec-worker:%s] claimed job_id=%s iter=%s stage=%s timeout_sec=%s cmd=%s repo=%s",
            args.worker_id,
            job_id,
            iter_index,
            stage,
            timeout_sec,
            cmd,
            repo_path,
        )

        try:
            t0 = time.time()
            logger.info("[exec-worker:%s] run start job_id=%s", args.worker_id, job_id)
            rc, out, err, elapsed = run_cmd(cmd, cwd=repo_path, timeout_sec=timeout_sec)
            # Persist logs
            stdout_path = write_text(run_dir, f"tree/exec_logs/{job['job_id']}.stdout.txt", out)
            stderr_path = write_text(run_dir, f"tree/exec_logs/{job['job_id']}.stderr.txt", err)

            metric, found = parse_metric(out, regex)
            logger.info(
                "[exec-worker:%s] run done job_id=%s exit_code=%s metric_found=%s metric=%s elapsed_sec=%.2f wall_elapsed_sec=%.2f",
                args.worker_id,
                job_id,
                rc,
                found,
                metric,
                elapsed,
                time.time() - t0,
            )

            result = {
                "job_id": job["job_id"],
                "ok": (rc == 0),
                "exit_code": rc,
                "exec_time_sec": elapsed,
                "metric_value": metric,
                "metric_found": found,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "error_summary": summarize_error(err) if rc != 0 else None,
            }
            complete(q, job_path, result, ok=True)  # ok=True means job execution completed (not that the program succeeded)
            logger.info("[exec-worker:%s] complete job_id=%s result_state=done", args.worker_id, job_id)
        except Exception as e:
            result = {
                "job_id": job.get("job_id"),
                "ok": False,
                "error": repr(e),
                "exit_code": -1,
                "exec_time_sec": 0.0,
                "metric_value": None,
                "metric_found": False,
                "stdout_path": "",
                "stderr_path": "",
            }
            complete(q, job_path, result, ok=False)
            logger.exception("[exec-worker:%s] exception job_id=%s error=%r", args.worker_id, job_id, e)


if __name__ == "__main__":
    main()
