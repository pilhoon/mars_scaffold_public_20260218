from __future__ import annotations
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from mars.config import LLMConfig
from mars.store import connect, init_schema, record_llm_call, reserve_rate_limit_slot, set_kv


LOGGER = logging.getLogger(__name__)
_SCHEMA_READY_RUN_DIRS: set[str] = set()
_RATE_LIMIT_ENV_WARNED = False


@dataclass(frozen=True)
class LLMResponse:
    ok: bool
    text: str
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class LLMCallTrace:
    run_dir: str | Path | None = None
    process: str = ""
    worker_id: str = ""
    iter_index: int | None = None
    stage: str = ""
    job_id: str = ""
    job_type: str = ""


def _build_env(cfg: LLMConfig) -> dict[str, str]:
    env = dict(os.environ)
    cli_name = Path(cfg.cli).name.lower()
    if cli_name == "codex":
        codex_home = env.get("CODEX_HOME")
        if not codex_home:
            codex_home = str(Path.home() / ".codex")
        Path(codex_home).mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = codex_home
    return env


def _record_llm_call_trace(trace: LLMCallTrace | None) -> int | None:
    if trace is None or trace.run_dir is None:
        return None

    conn = None
    run_dir_key = str(Path(trace.run_dir))
    try:
        conn = connect(trace.run_dir)
        if run_dir_key not in _SCHEMA_READY_RUN_DIRS:
            init_schema(conn)
            _SCHEMA_READY_RUN_DIRS.add(run_dir_key)
        seq = record_llm_call(
            conn,
            process=trace.process,
            worker_id=trace.worker_id,
            iter_index=trace.iter_index,
            stage=trace.stage,
            job_id=trace.job_id,
            job_type=trace.job_type,
        )
        set_kv(conn, "llm_call_total_all_processes", str(seq))
        return seq
    except Exception as e:
        LOGGER.warning("llm call tracking failed: %r", e)
        return None
    finally:
        if conn is not None:
            conn.close()


def _enforce_llm_rate_limit(trace: LLMCallTrace | None, cfg: LLMConfig) -> float:
    if trace is None or trace.run_dir is None:
        return 0.0

    global _RATE_LIMIT_ENV_WARNED
    interval_sec = max(0.0, float(getattr(cfg, "min_call_interval_sec", 0.0)))
    env_interval = os.environ.get("LLM_MIN_CALL_INTERVAL_SEC")
    if env_interval is not None and env_interval != "":
        try:
            interval_sec = max(0.0, float(env_interval))
        except ValueError:
            if not _RATE_LIMIT_ENV_WARNED:
                LOGGER.warning(
                    "invalid LLM_MIN_CALL_INTERVAL_SEC=%r; using config value %.2f",
                    env_interval,
                    interval_sec,
                )
                _RATE_LIMIT_ENV_WARNED = True
    if interval_sec <= 0.0:
        return 0.0

    conn = None
    run_dir_key = str(Path(trace.run_dir))
    try:
        conn = connect(trace.run_dir)
        if run_dir_key not in _SCHEMA_READY_RUN_DIRS:
            init_schema(conn)
            _SCHEMA_READY_RUN_DIRS.add(run_dir_key)
        _, wait_sec = reserve_rate_limit_slot(
            conn,
            limiter_name="llm_global",
            interval_sec=interval_sec,
        )
    finally:
        if conn is not None:
            conn.close()

    if wait_sec > 0.0:
        LOGGER.info(
            "llm rate-limit waiting wait_sec=%.2f interval_sec=%.2f process=%s worker_id=%s iter=%s stage=%s job_id=%s job_type=%s",
            wait_sec,
            interval_sec,
            trace.process,
            trace.worker_id,
            trace.iter_index,
            trace.stage,
            trace.job_id,
            trace.job_type,
        )
        time.sleep(wait_sec)
    return wait_sec


def call_llm(
    prompt: str,
    cfg: LLMConfig,
    trace: LLMCallTrace | None = None,
    *,
    cwd: str | Path | None = None,
) -> LLMResponse:
    """Call LLM CLI once.

    Examples:
    - argv mode:
        $ gemini -m gemini-3-pro-preview -p "prompt"
    - stdin mode:
        $ cat prompt.txt | gemini -m gemini-3-pro-preview -p -

    Notes:
    - Very large prompts may exceed argv limits; use cfg.mode="stdin" for those CLIs.
    - We never use shell=True for safety.
    """
    cmd = [cfg.cli, *cfg.args]
    env = _build_env(cfg)
    run_cwd = str(cwd) if cwd is not None else None
    wait_sec = _enforce_llm_rate_limit(trace, cfg)
    call_seq = _record_llm_call_trace(trace)
    if call_seq is not None:
        LOGGER.info(
            "llm call seq=%s total=%s process=%s worker_id=%s iter=%s stage=%s job_id=%s job_type=%s rate_limit_wait_sec=%.2f",
            call_seq,
            call_seq,
            trace.process if trace else "",
            trace.worker_id if trace else "",
            trace.iter_index if trace else None,
            trace.stage if trace else "",
            trace.job_id if trace else "",
            trace.job_type if trace else "",
            wait_sec,
        )

    try:
        if cfg.mode == "stdin":
            p = subprocess.run(
                cmd,
                input=prompt,
                env=env,
                cwd=run_cwd,
                text=True,
                capture_output=True,
                timeout=cfg.timeout_sec,
            )
        else:
            p = subprocess.run(
                [*cmd, prompt],
                env=env,
                cwd=run_cwd,
                text=True,
                capture_output=True,
                timeout=cfg.timeout_sec,
            )
        ok = (p.returncode == 0)
        if call_seq is not None:
            LOGGER.info(
                "llm call done seq=%s ok=%s returncode=%s stdout_len=%s stderr_len=%s",
                call_seq,
                ok,
                p.returncode,
                len(p.stdout),
                len(p.stderr),
            )
        # For many CLIs, the model response is in stdout.
        return LLMResponse(ok=ok, text=p.stdout, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        err = e.stderr if isinstance(e.stderr, str) else ""
        timeout_msg = f"LLM CLI timed out after {cfg.timeout_sec} sec."
        err = (err + "\n" + timeout_msg).strip() if err else timeout_msg
        if call_seq is not None:
            LOGGER.info("llm call done seq=%s ok=False returncode=124 timeout=True", call_seq)
        return LLMResponse(ok=False, text=out, returncode=124, stdout=out, stderr=err)
    except FileNotFoundError as e:
        if call_seq is not None:
            LOGGER.info("llm call done seq=%s ok=False returncode=127 cli_missing=True", call_seq)
        return LLMResponse(
            ok=False,
            text="",
            returncode=127,
            stdout="",
            stderr=f"LLM CLI not found: {cfg.cli}. ({e})",
        )
    except Exception as e:
        if call_seq is not None:
            LOGGER.info("llm call done seq=%s ok=False returncode=1 error=%r", call_seq, e)
        return LLMResponse(
            ok=False,
            text="",
            returncode=1,
            stdout="",
            stderr=f"LLM CLI failed unexpectedly: {e!r}",
        )
