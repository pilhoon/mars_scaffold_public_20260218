from __future__ import annotations
import argparse
from dataclasses import replace
import logging
import re
import time
from pathlib import Path
from typing import Any

from mars.config import LLMConfig, load_global_config
from mars.fsqueue import ensure_queue_dirs, claim, read_job, complete
from mars.llm.client import LLMCallTrace, call_llm
from mars.llm.prompts import (
    EXECUTION_REVIEW_TEMPLATE,
    IDEA_IMPROVE_TEMPLATE,
    IDEA_INITIAL_TEMPLATE,
    MODULAR_DECOMPOSE_TEMPLATE,
    PATCH_DEBUG_TEMPLATE,
    PATCH_DRAFT_TEMPLATE,
    PATCH_IMPROVE_TEMPLATE,
    PATCH_MODULE_TEMPLATE,
    PATCH_ORCHESTRATE_TEMPLATE,
    TASK_PREP_TEMPLATE,
    render_prompt,
)
from mars.artifacts import ensure_run_dirs, write_text

_ERROR_HINT_RE = re.compile(
    r"(error|exception|traceback|usage limit|argument list too long|timed out|unauthorized|invalid)",
    re.IGNORECASE,
)


def _resolve_llm_cwd(run_dir: Path, payload: dict[str, Any]) -> Path:
    raw = str(payload.get("llm_cwd", "")).strip()
    base = run_dir.resolve()
    if not raw:
        return base

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (run_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.is_dir():
        return base
    try:
        candidate.relative_to(base)
    except ValueError:
        return base
    return candidate


def _resolve_job_llm_config(base_cfg: LLMConfig, payload: dict[str, Any]) -> tuple[LLMConfig, str, int | None]:
    profile_name = str(payload.get("llm_profile_name", "default")).strip() or "default"
    profile_index_raw = payload.get("llm_profile_index")
    profile_index: int | None = None
    if isinstance(profile_index_raw, int):
        profile_index = profile_index_raw

    updates: dict[str, Any] = {}

    cli_raw = str(payload.get("llm_cli", "")).strip()
    if cli_raw:
        updates["cli"] = cli_raw

    args_raw = payload.get("llm_args")
    if isinstance(args_raw, list) and all(isinstance(item, str) for item in args_raw):
        updates["args"] = list(args_raw)

    mode_raw = str(payload.get("llm_mode", "")).strip().lower()
    if mode_raw in ("stdin", "argv"):
        updates["mode"] = mode_raw

    if not updates:
        return base_cfg, profile_name, profile_index
    return replace(base_cfg, **updates), profile_name, profile_index


def _summarize_error_text(*texts: str, max_chars: int = 240) -> str:
    for raw in texts:
        lines = [ln.strip() for ln in str(raw).splitlines() if ln.strip()]
        if not lines:
            continue
        chosen = lines[0]
        for ln in lines:
            if _ERROR_HINT_RE.search(ln):
                chosen = ln
                break
        compact = " ".join(chosen.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3] + "..."
    return "(no error text)"


def _log_text_block(
    logger: logging.Logger,
    *,
    worker_id: str,
    label: str,
    job_id: str,
    text: str,
) -> None:
    logger.info(
        "[llm-worker:%s] %s begin job_id=%s chars=%s",
        worker_id,
        label,
        job_id,
        len(text),
    )
    if text:
        logger.info(
            "[llm-worker:%s] %s body job_id=%s\n%s",
            worker_id,
            label,
            job_id,
            text,
        )
    else:
        logger.info(
            "[llm-worker:%s] %s body job_id=%s <empty>",
            worker_id,
            label,
            job_id,
        )
    logger.info(
        "[llm-worker:%s] %s end job_id=%s",
        worker_id,
        label,
        job_id,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("llm_worker")

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--default-conf", default=str(Path(__file__).resolve().parents[2] / "conf" / "default.yaml"))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ensure_run_dirs(run_dir)
    cfg = load_global_config(args.default_conf)
    logger.info(
        "[llm-worker:%s] start run_dir=%s cli=%s timeout_sec=%s min_call_interval_sec=%s",
        args.worker_id,
        run_dir,
        cfg.llm.cli,
        cfg.llm.timeout_sec,
        cfg.llm.min_call_interval_sec,
    )

    q = ensure_queue_dirs(run_dir, "llm")
    idle_heartbeat_sec = 30.0
    last_idle_log = 0.0

    while True:
        job_path = claim(q, worker_id=args.worker_id)
        if job_path is None:
            now = time.time()
            if now - last_idle_log >= idle_heartbeat_sec:
                logger.info("[llm-worker:%s] idle waiting for llm jobs", args.worker_id)
                last_idle_log = now
            time.sleep(0.2)
            continue

        job = read_job(job_path)
        job_id = str(job.get("job_id", ""))
        jtype = str(job.get("type", ""))
        payload = job.get("payload", {}) or {}
        meta = job.get("meta", {}) or {}
        iter_index = meta.get("iter_index")
        stage = meta.get("stage")
        attempt = meta.get("attempt")
        llm_call_seq = meta.get("llm_call_seq")
        llm_profile_name = str(meta.get("llm_profile_name") or payload.get("llm_profile_name") or "default")
        llm_profile_index = meta.get("llm_profile_index", payload.get("llm_profile_index"))
        llm_model = str(meta.get("llm_model") or payload.get("llm_model") or "")
        llm_cwd = _resolve_llm_cwd(run_dir, payload)
        job_llm_cfg, llm_profile_name, llm_profile_index = _resolve_job_llm_config(cfg.llm, payload)
        logger.info(
            "[llm-worker:%s] claimed job_id=%s type=%s iter=%s stage=%s attempt=%s llm_call_seq=%s llm_profile=%s llm_profile_index=%s llm_model=%s llm_cli=%s llm_mode=%s llm_cwd=%s",
            args.worker_id,
            job_id,
            jtype,
            iter_index,
            stage,
            attempt,
            llm_call_seq,
            llm_profile_name,
            llm_profile_index,
            llm_model,
            job_llm_cfg.cli,
            job_llm_cfg.mode,
            llm_cwd,
        )

        try:
            if jtype == "IDEA_PROPOSE":
                idea_mode = str(payload.get("mode", "initial")).lower()
                template = IDEA_INITIAL_TEMPLATE if idea_mode == "initial" else IDEA_IMPROVE_TEMPLATE
                prompt = render_prompt(
                    template,
                    instruction=payload.get("instruction", ""),
                    task_prep=payload.get("task_prep", "(none)"),
                    experiment_context=payload.get("experiment_context", "(none)"),
                    previous_ideas=payload.get("previous_ideas", ""),
                    lessons=payload.get("lessons", "(none)"),
                )
            elif jtype == "EXEC_REVIEW":
                prompt = render_prompt(
                    EXECUTION_REVIEW_TEMPLATE,
                    instruction=payload.get("instruction", ""),
                    experiment_context=payload.get("experiment_context", "(none)"),
                    code=payload.get("code", "(not provided)"),
                    term_out=payload.get("term_out", "(not provided)"),
                )
            elif jtype == "TASK_PREPARE":
                prompt = render_prompt(
                    TASK_PREP_TEMPLATE,
                    instruction=payload.get("instruction", ""),
                    task_workdir=payload.get("task_workdir", "(not provided)"),
                    file_list=payload.get("file_list", "(not provided)"),
                    local_context=payload.get("local_context", "(not provided)"),
                )
            elif jtype == "MODULAR_DECOMPOSE":
                prompt = render_prompt(
                    MODULAR_DECOMPOSE_TEMPLATE,
                    instruction=payload.get("instruction", ""),
                    idea=payload.get("idea", "(not provided)"),
                    task_prep=payload.get("task_prep", "(none)"),
                    experiment_context=payload.get("experiment_context", "(none)"),
                )
            elif jtype == "GENERATE_PATCH":
                patch_mode = str(payload.get("mode", "improve")).lower()
                if patch_mode == "draft":
                    template = PATCH_DRAFT_TEMPLATE
                elif patch_mode == "debug":
                    template = PATCH_DEBUG_TEMPLATE
                elif patch_mode == "module":
                    template = PATCH_MODULE_TEMPLATE
                elif patch_mode == "orchestrate":
                    template = PATCH_ORCHESTRATE_TEMPLATE
                else:
                    template = PATCH_IMPROVE_TEMPLATE
                prompt = render_prompt(
                    template,
                    instruction=payload.get("instruction", ""),
                    file_list=payload.get("file_list", "(not provided)"),
                    lessons=payload.get("lessons", "(none)"),
                    task_prep=payload.get("task_prep", "(none)"),
                    experiment_context=payload.get("experiment_context", "(none)"),
                    idea=payload.get("idea", "(not provided)"),
                    module_plan=payload.get("module_plan", "(not provided)"),
                    module_spec=payload.get("module_spec", "(not provided)"),
                    error_summary=payload.get("error_summary", "(not provided)"),
                )
            else:
                raise ValueError(f"unsupported llm job type: {jtype}")

            prompt_path = write_text(run_dir, f"artifacts/prompts/{job['job_id']}.txt", prompt)
            logger.info(
                "[llm-worker:%s] prompt ready job_id=%s chars=%s path=%s",
                args.worker_id,
                job_id,
                len(prompt),
                prompt_path,
            )
            _log_text_block(
                logger,
                worker_id=args.worker_id,
                label="prompt",
                job_id=job_id,
                text=prompt,
            )
            t0 = time.time()
            logger.info(
                "[llm-worker:%s] llm call start llm_call_seq=%s job_id=%s type=%s iter=%s stage=%s llm_profile=%s llm_profile_index=%s llm_model=%s llm_cli=%s llm_mode=%s",
                args.worker_id,
                llm_call_seq,
                job_id,
                jtype,
                iter_index,
                stage,
                llm_profile_name,
                llm_profile_index,
                llm_model,
                job_llm_cfg.cli,
                job_llm_cfg.mode,
            )
            resp = call_llm(
                prompt,
                job_llm_cfg,
                trace=LLMCallTrace(
                    run_dir=run_dir,
                    process="llm-worker",
                    worker_id=args.worker_id,
                    iter_index=int(iter_index) if isinstance(iter_index, int) else None,
                    stage=str(stage or ""),
                    job_id=job_id,
                    job_type=jtype,
                ),
                cwd=llm_cwd,
            )
            elapsed = time.time() - t0
            logger.info(
                "[llm-worker:%s] llm call done llm_call_seq=%s job_id=%s ok=%s returncode=%s elapsed_sec=%.2f text_len=%s llm_profile=%s llm_profile_index=%s llm_model=%s",
                args.worker_id,
                llm_call_seq,
                job_id,
                resp.ok,
                resp.returncode,
                elapsed,
                len(resp.text),
                llm_profile_name,
                llm_profile_index,
                llm_model,
            )
            if not resp.ok:
                logger.warning(
                    "[llm-worker:%s] llm call failed llm_call_seq=%s job_id=%s type=%s iter=%s stage=%s returncode=%s llm_profile=%s llm_profile_index=%s llm_model=%s error_summary=%s",
                    args.worker_id,
                    llm_call_seq,
                    job_id,
                    jtype,
                    iter_index,
                    stage,
                    resp.returncode,
                    llm_profile_name,
                    llm_profile_index,
                    llm_model,
                    _summarize_error_text(resp.stderr, resp.text),
                )
            raw_path = write_text(run_dir, f"artifacts/llm_raw/{job['job_id']}.txt", resp.text)
            _log_text_block(
                logger,
                worker_id=args.worker_id,
                label="response",
                job_id=job_id,
                text=resp.text,
            )
            _log_text_block(
                logger,
                worker_id=args.worker_id,
                label="response_stderr",
                job_id=job_id,
                text=resp.stderr,
            )

            result = {
                "job_id": job["job_id"],
                "ok": resp.ok,
                "returncode": resp.returncode,
                "text": resp.text,
                "stdout": resp.stdout,
                "stderr": resp.stderr,
                "prompt_path": prompt_path,
                "raw_path": raw_path,
                "llm_profile_name": llm_profile_name,
                "llm_profile_index": llm_profile_index,
                "llm_model": llm_model,
                "llm_cli": job_llm_cfg.cli,
                "llm_mode": job_llm_cfg.mode,
            }
            complete(q, job_path, result, ok=resp.ok)
            logger.info(
                "[llm-worker:%s] complete job_id=%s result_state=%s",
                args.worker_id,
                job_id,
                "done" if resp.ok else "failed",
            )
        except Exception as e:
            result = {
                "job_id": job.get("job_id"),
                "ok": False,
                "error": repr(e),
            }
            complete(q, job_path, result, ok=False)
            logger.exception("[llm-worker:%s] exception job_id=%s error=%r", args.worker_id, job_id, e)


if __name__ == "__main__":
    main()
