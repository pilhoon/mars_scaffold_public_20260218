from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Dict, List


class NodeStatus(str, Enum):
    PENDING = "pending"
    CODEGEN_DONE = "codegen_done"
    PATCHED = "patched"
    RUNNING = "running"
    DONE = "done"
    BUGGY = "buggy"


class ActionType(str, Enum):
    DRAFT = "draft"
    IMPROVE = "improve"
    DEBUG = "debug"


class JobKind(str, Enum):
    LLM = "llm"
    EXEC = "exec"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    workdir: str
    wallclock_sec: int
    per_run_timeout_sec: int
    metric_name: str
    higher_is_better: bool
    entrypoint: str
    final_metric_regex: str
    repo_template_path: str


@dataclass(frozen=True)
class NodeRecord:
    node_id: int
    parent_id: Optional[int]
    action: ActionType
    status: NodeStatus
    repo_path: str
    metric: Optional[float]
    runtime_sec: Optional[float]
    reward: Optional[float]
    visits: int
    value_sum: float


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    kind: JobKind
    type: str
    payload: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    exit_code: int
    exec_time_sec: float
    metric_value: Optional[float]
    metric_found: bool
    stdout_path: str
    stderr_path: str
    error_summary: Optional[str] = None


@dataclass(frozen=True)
class Lesson:
    lesson_id: Optional[int]
    kind: str  # "solution" | "debug"
    title: str
    body: str
    tags: List[str]
    source_node_id: int
    created_at: str
    fingerprint: str
