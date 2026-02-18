from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from mars.types import TaskSpec


def _default_codex_args() -> List[str]:
    return [
        "-m",
        "gpt-5.3-codex-spark",
        "-c",
        "model_reasoning_effort=high",
        "exec",
        "--ephemeral",
        "--sandbox",
        "danger-full-access",
        "--skip-git-repo-check",
    ]


@dataclass(frozen=True)
class LLMConfig:
    cli: str = "codex"
    args: List[str] = field(default_factory=_default_codex_args)
    profiles: List[Dict[str, Any]] = field(default_factory=list)
    profile_selection_mode: str = "sticky"  # "sticky" or "round_robin"
    mode: str = "stdin"         # "argv" or "stdin"
    timeout_sec: int = 1800
    min_call_interval_sec: float = 10.0
    auth_retry_backoff_sec: float = 60.0
    usage_limit_wait_sec: float = 1800.0


@dataclass(frozen=True)
class MCTSConfig:
    c_uct: float = 1.4
    w_time_penalty: float = -0.07
    max_root_drafts: int = 1
    root_reactivate_after_valid_no_improve: int = 5
    max_debug_attempts: int = 10
    max_improve_children: int = 2
    identical_metric_streak_to_force_debug: int = 3


@dataclass(frozen=True)
class MemoryConfig:
    keep_recent_k: int = 30


@dataclass(frozen=True)
class ExecConfig:
    per_run_timeout_sec: int = 14400


@dataclass(frozen=True)
class GlobalConfig:
    llm: LLMConfig
    mcts: MCTSConfig
    memory: MemoryConfig
    exec: ExecConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_global_config(default_yaml: str | Path, override_yaml: str | Path | None = None) -> GlobalConfig:
    base = load_yaml(default_yaml)
    if override_yaml is not None:
        ov = load_yaml(override_yaml)
        base = deep_merge(base, ov)

    llm = base.get("llm", {}) or {}
    mcts = base.get("mcts", {}) or {}
    mem = base.get("memory", {}) or {}
    ex = base.get("exec", {}) or {}

    return GlobalConfig(
        llm=LLMConfig(**llm),
        mcts=MCTSConfig(**mcts),
        memory=MemoryConfig(**mem),
        exec=ExecConfig(**ex),
    )


def load_task_spec(task_yaml: str | Path) -> TaskSpec:
    d = load_yaml(task_yaml)

    budget = d.get("budget", {}) or {}
    obj = d.get("objective", {}) or {}
    exe = d.get("execution", {}) or {}
    repo = d.get("repo_template", {}) or {}

    return TaskSpec(
        task_id=str(d["task_id"]),
        instruction=str(d.get("instruction", "")),
        workdir=str(d.get("workdir", ".")),
        wallclock_sec=int(budget.get("wallclock_sec", 86400)),
        per_run_timeout_sec=int(budget.get("per_run_timeout_sec", 14400)),
        metric_name=str(obj.get("metric_name", "metric")),
        higher_is_better=bool(obj.get("higher_is_better", True)),
        entrypoint=str(exe.get("entrypoint", "python runfile.py")),
        final_metric_regex=str(exe.get("final_metric_regex", r"^Final Validation Metric:\s*([0-9.eE+-]+)\s*$")),
        repo_template_path=str(repo.get("path", "./templates/demo_repo_template")),
    )


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict b into dict a (a wins unless b overrides)."""
    out: Dict[str, Any] = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
