from __future__ import annotations
import re
from pathlib import Path

from mars.config import LLMConfig
from mars.llm.client import LLMCallTrace, call_llm
from mars.llm.prompts import DEBUG_LESSON_TEMPLATE, SOLUTION_LESSON_TEMPLATE, render_prompt
from mars.memory.lessons import make_lesson
from mars.types import ExecResult, Lesson


def _strip_code_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[^\n]*\n", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n```$", "", s)
    return s.strip()


def _parse_labeled_sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for raw in _strip_code_fence(text).splitlines():
        line = raw.strip()
        m = re.match(r"^(Title|Summary|Empirical Findings|Key Lesson|Explanation|Detection):\s*(.*)$", line, flags=re.IGNORECASE)
        if m:
            if current_key is not None:
                out[current_key] = "\n".join(current_lines).strip()
            current_key = m.group(1).lower()
            current_lines = [m.group(2).strip()] if m.group(2).strip() else []
        elif current_key is not None:
            current_lines.append(raw.rstrip())
    if current_key is not None:
        out[current_key] = "\n".join(current_lines).strip()
    return out


def _solution_fallback(new_summary: str, diff_summary: str, exec_result: ExecResult, source_node_id: int) -> Lesson:
    title = "Record experiment outcome"
    body = f"New summary: {new_summary}\nMetric: {exec_result.metric_value} time={exec_result.exec_time_sec}\nDiff:\n{diff_summary}"
    return make_lesson(kind="solution", title=title, body=body, tags=["fallback"], source_node_id=source_node_id)


def _debug_fallback(error_summary: str, fix_diff: str, source_node_id: int) -> Lesson:
    title = "Fix runtime error"
    body = f"Error: {error_summary}\nFix diff:\n{fix_diff}"
    return make_lesson(kind="debug", title=title, body=body, tags=["fallback"], source_node_id=source_node_id)


def distill_solution_lesson(
    best_summary: str | None,
    new_summary: str,
    diff_summary: str,
    exec_result: ExecResult,
    source_node_id: int,
    llm_cfg: LLMConfig | None = None,
    run_dir: str | Path | None = None,
    iter_index: int | None = None,
) -> Lesson:
    if llm_cfg is None:
        return _solution_fallback(new_summary, diff_summary, exec_result, source_node_id)

    prompt = render_prompt(
        SOLUTION_LESSON_TEMPLATE,
        best_summary=best_summary or "(none)",
        new_summary=new_summary,
        diff_summary=diff_summary,
        metric=exec_result.metric_value,
        exec_time_sec=exec_result.exec_time_sec,
        valid_metric=exec_result.metric_found,
    )
    try:
        resp = call_llm(
            prompt,
            llm_cfg,
            trace=LLMCallTrace(
                run_dir=run_dir,
                process="controller",
                stage="distill_solution_lesson",
                iter_index=iter_index,
                job_type="MEMORY_DISTILL_SOLUTION",
            ),
        )
        if not resp.ok:
            return _solution_fallback(new_summary, diff_summary, exec_result, source_node_id)
        fields = _parse_labeled_sections(resp.text)
        title = fields.get("title") or "Solution lesson"
        summary = fields.get("summary") or new_summary
        empirical = fields.get("empirical findings") or f"metric={exec_result.metric_value}, time={exec_result.exec_time_sec:.3f}s"
        key_lesson = fields.get("key lesson") or "Prefer targeted ablations and verify metric validity."
        body = f"Summary:\n{summary}\n\nEmpirical Findings:\n{empirical}\n\nKey Lesson:\n{key_lesson}"
        return make_lesson(kind="solution", title=title, body=body, tags=["llm"], source_node_id=source_node_id)
    except Exception:
        return _solution_fallback(new_summary, diff_summary, exec_result, source_node_id)


def distill_debug_lesson(
    error_summary: str,
    fix_diff: str,
    source_node_id: int,
    llm_cfg: LLMConfig | None = None,
    debug_outcome: str = "unknown",
    run_dir: str | Path | None = None,
    iter_index: int | None = None,
) -> Lesson:
    if llm_cfg is None:
        return _debug_fallback(error_summary, fix_diff, source_node_id)

    prompt = render_prompt(
        DEBUG_LESSON_TEMPLATE,
        error_summary=error_summary,
        fix_diff=fix_diff,
        debug_outcome=debug_outcome,
    )
    try:
        resp = call_llm(
            prompt,
            llm_cfg,
            trace=LLMCallTrace(
                run_dir=run_dir,
                process="controller",
                stage="distill_debug_lesson",
                iter_index=iter_index,
                job_type="MEMORY_DISTILL_DEBUG",
            ),
        )
        if not resp.ok:
            return _debug_fallback(error_summary, fix_diff, source_node_id)
        fields = _parse_labeled_sections(resp.text)
        title = fields.get("title") or "Debug lesson"
        explanation = fields.get("explanation") or error_summary
        detection = fields.get("detection") or "Look for recurring traceback signatures and failing call sites."
        body = f"Explanation:\n{explanation}\n\nDetection:\n{detection}\n\nFix Diff:\n{fix_diff}"
        return make_lesson(kind="debug", title=title, body=body, tags=["llm"], source_node_id=source_node_id)
    except Exception:
        return _debug_fallback(error_summary, fix_diff, source_node_id)
