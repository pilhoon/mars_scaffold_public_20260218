from __future__ import annotations
from typing import Any, List

from mars.types import Lesson


def render_prompt(template: str, **kwargs: Any) -> str:
    """Simple template renderer (format-map).
    TODO: replace with more robust templating if needed.
    """
    return template.format(**kwargs)


IDEA_INITIAL_TEMPLATE = """You are an Initial Idea Generation Agent.
==== Task ====
{instruction}
==== Task Preparation Summary ====
{task_prep}
==== Iteration Experiment Context ====
{experiment_context}
==== Previous Ideas ====
{previous_ideas}
==== Solution Lessons ====
{lessons}

Propose a lightweight baseline approach.
Requirements:
- The proposal must be novel vs previous ideas.
- Prioritize speed and simplicity over peak accuracy.
- No code. Natural language only.
Output format:
- Model
- Data
- Training
- Evaluation
"""


IDEA_IMPROVE_TEMPLATE = """You are an Idea Improvement Agent.
==== Task ====
{instruction}
==== Task Preparation Summary ====
{task_prep}
==== Iteration Experiment Context ====
{experiment_context}
==== Previous Ideas ====
{previous_ideas}
==== Solution Lessons ====
{lessons}

Propose a structurally improved strategy:
- Keep components proven useful by lessons.
- Introduce non-trivial structural change (not only hyperparameters).
- Keep computational budget feasible.
- Cite applied lesson ids inline as "Cite <lesson_id>".
No code. Natural language only.
Output format:
- Model
- Data
- Training
- Evaluation
"""


MODULAR_DECOMPOSE_TEMPLATE = """You are a modular decomposition agent.
==== Idea ====
{idea}
==== Task ====
{instruction}
==== Task Preparation Summary ====
{task_prep}
==== Iteration Experiment Context ====
{experiment_context}

Design a repository-level module plan (no code) and return JSON only.
Schema:
{{
  "modules": [
    {{
      "path": "relative/path.py",
      "purpose": "short sentence",
      "interfaces": ["func_or_class_1", "func_or_class_2"],
      "unit_test_focus": "short sentence"
    }}
  ],
  "orchestration_notes": ["step for runfile.py wiring"]
}}
Rules:
- Keep modules composable and implementation-ready.
- Do not include runfile.py in modules; keep that for orchestration_notes.
- Prefer 2-8 modules.
"""


PATCH_DRAFT_TEMPLATE = """You are a coding agent implementing a new draft solution.
==== Task ====
{instruction}
==== Task Preparation Summary ====
{task_prep}
==== Iteration Experiment Context ====
{experiment_context}
==== Draft Idea ====
{idea}
==== Module Plan ====
{module_plan}
==== Repo file listing ====
{file_list}
==== Lessons ====
{lessons}

Generate a focused unified diff patch that implements the draft idea.
Constraints:
- Output ONLY unified diff (git apply compatible).
- Keep changes localized and executable.
- Preserve required metric print format if already present.
"""


PATCH_IMPROVE_TEMPLATE = """You are a coding agent improving an existing solution.
==== Task ====
{instruction}
==== Task Preparation Summary ====
{task_prep}
==== Iteration Experiment Context ====
{experiment_context}
==== Current Goal ====
Improve validation metric while keeping runtime efficient.
==== Repo file listing ====
{file_list}
==== Lessons ====
{lessons}

Modify the existing repository with targeted ablation-style changes.
Constraints:
- Output ONLY unified diff (git apply compatible).
- Do not rewrite everything; improve specific bottlenecks.
- Preserve required metric print format if already present.
"""


PATCH_DEBUG_TEMPLATE = """You are a debugging agent.
==== Task ====
{instruction}
==== Task Preparation Summary ====
{task_prep}
==== Iteration Experiment Context ====
{experiment_context}
==== Error Summary ====
{error_summary}
==== Repo file listing ====
{file_list}
==== Debug Lessons ====
{lessons}

Fix the runtime error with minimal targeted changes.
Constraints:
- Output ONLY unified diff (git apply compatible).
- Fix root cause; do not suppress errors with broad try/except.
- Preserve core training/evaluation logic.
"""


PATCH_MODULE_TEMPLATE = """You are implementing one module from a decomposition plan.
==== Task ====
{instruction}
==== Task Preparation Summary ====
{task_prep}
==== Iteration Experiment Context ====
{experiment_context}
==== Draft Idea ====
{idea}
==== Module Plan ====
{module_plan}
==== Target Module Spec ====
{module_spec}
==== Repo file listing ====
{file_list}
==== Lessons ====
{lessons}

Generate ONLY a unified diff patch focused on this module spec.
Keep changes minimal and composable with later module/orchestration patches.
"""


PATCH_ORCHESTRATE_TEMPLATE = """You are finalizing orchestration after module implementations.
==== Task ====
{instruction}
==== Task Preparation Summary ====
{task_prep}
==== Iteration Experiment Context ====
{experiment_context}
==== Draft Idea ====
{idea}
==== Module Plan ====
{module_plan}
==== Repo file listing ====
{file_list}
==== Lessons ====
{lessons}

Generate ONLY a unified diff patch to wire modules together (typically runfile.py and glue logic).
Preserve required final metric print format.
"""


TASK_PREP_TEMPLATE = """You are a Task Preparation agent.
==== Task ====
{instruction}
==== Workdir Hint ====
{task_workdir}
==== Initial Repo File Listing ====
{file_list}
==== Local Metadata / EDA Signals ====
{local_context}

Produce a concise preparation summary with these sections:
1) Metric and optimization direction assumptions.
2) Metadata/data handling plan (including validation split checks).
3) EDA checklist for first run.
4) Candidate model families and lightweight baseline order.
5) Risk checklist (leakage, metric mismatch, runtime bottlenecks).
"""


SOLUTION_LESSON_TEMPLATE = """You distill a reusable solution lesson.
==== Current Best Summary ====
{best_summary}
==== New Solution Summary ====
{new_summary}
==== Diff Summary ====
{diff_summary}
==== Execution Result ====
- metric: {metric}
- exec_time_sec: {exec_time_sec}
- valid_metric: {valid_metric}

Return exactly:
Title: <one line>
Summary: <one paragraph>
Empirical Findings: <one paragraph>
Key Lesson: <one paragraph rule of thumb>
"""


DEBUG_LESSON_TEMPLATE = """You distill a reusable debugging lesson.
==== Error Summary ====
{error_summary}
==== Fix Diff ====
{fix_diff}
==== Debug Outcome ====
{debug_outcome}

Return exactly:
Title: <one line>
Explanation: <one paragraph>
Detection: <one paragraph>
"""


LESSON_DEDUP_TEMPLATE = """Determine whether the New Lesson duplicates Existing Lessons.
==== Existing Lessons ====
{existing_lessons}
==== New Lesson ====
{new_lesson}

Return JSON only:
{{
  "reasoning": "<brief reason>",
  "duplicate": true_or_false
}}
"""


EXECUTION_REVIEW_TEMPLATE = """You evaluate whether the reported final validation metric is valid.
==== Task ====
{instruction}
==== Iteration Experiment Context ====
{experiment_context}
==== Code (runfile.py excerpt) ====
{code}
==== Execution Output ====
{term_out}

Return JSON only:
{{
  "summary": "<brief empirical findings>",
  "metric": <number_or_null>,
  "valid_metric": true_or_false
}}
Mark valid_metric=false if metric is missing, malformed, computed on wrong split, or appears leaked/invalid.
"""


def lessons_to_text(lessons: List[Lesson]) -> str:
    if not lessons:
        return "(none)"
    parts = []
    for L in lessons:
        parts.append(f"[{L.lesson_id}] {L.title}\n{L.body}\n")
    return "\n".join(parts)
