from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from mars.config import LLMConfig
from mars.llm.client import LLMCallTrace, call_llm
from mars.llm.prompts import LESSON_DEDUP_TEMPLATE, render_prompt
from mars.types import Lesson


def _strip_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[^\n]*\n", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n```$", "", s)
    return s.strip()


def _parse_json_object(text: str) -> dict:
    s = _strip_fence(text)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}


def is_duplicate_lesson_llm(
    new_lesson: Lesson,
    existing_lessons: Iterable[Lesson],
    llm_cfg: LLMConfig | None,
    *,
    run_dir: str | Path | None = None,
    iter_index: int | None = None,
) -> bool:
    existing = list(existing_lessons)
    if llm_cfg is None or not existing:
        return False

    existing_text = "\n\n".join(
        f"[{L.lesson_id}] {L.title}\n{L.body}" for L in existing
    )[:12000]
    new_text = f"{new_lesson.title}\n{new_lesson.body}"[:4000]
    prompt = render_prompt(
        LESSON_DEDUP_TEMPLATE,
        existing_lessons=existing_text or "(none)",
        new_lesson=new_text,
    )

    try:
        resp = call_llm(
            prompt,
            llm_cfg,
            trace=LLMCallTrace(
                run_dir=run_dir,
                process="controller",
                stage="lesson_dedup_review",
                iter_index=iter_index,
                job_type="MEMORY_DEDUP_REVIEW",
            ),
        )
        if not resp.ok:
            return False
        obj = _parse_json_object(resp.text)
        return bool(obj.get("duplicate", False))
    except Exception:
        return False
