from __future__ import annotations
from datetime import datetime, timedelta
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mars.artifacts import ensure_run_dirs, write_json, write_text
from mars.config import GlobalConfig, load_global_config, load_task_spec
from mars.exec.metric import summarize_error
from mars.fsqueue import ensure_queue_dirs, enqueue, wait_for_result
from mars.llm.prompts import lessons_to_text
from mars.mcts.policy import decide_action, expansion_limit
from mars.mcts.reward import MetricHistory, efficiency_guided_reward
from mars.mcts.uct import uct_score
from mars.memory.distill import distill_debug_lesson, distill_solution_lesson
from mars.memory.retrieve import get_recent_lessons
from mars.memory.review import is_duplicate_lesson_llm
from mars.notify import send_telegram
from mars.repo.diff import diff_summary
from mars.repo.patch import apply_unified_diff, extract_unified_diff
from mars.repo.snapshot import materialize_repo
from mars.taskprep import collect_task_preparation_context, task_prep_context_text
from mars.store import (
    add_lesson,
    connect,
    create_node,
    get_kv,
    get_node,
    inc_visit_and_value,
    init_schema,
    lesson_exists_by_fingerprint,
    list_children,
    set_kv,
    update_node_result,
    update_node_status,
)
from mars.types import ActionType, ExecResult, NodeStatus


@dataclass
class ControllerContext:
    run_dir: Path
    worker_id: str
    task_yaml: Path
    default_conf_yaml: Path


@dataclass(frozen=True)
class LLMProfileSpec:
    name: str
    cli: str
    args: list[str]
    mode: str


_ITER_RE = re.compile(r"^iter_(\d+)\.json$")
_AUTH_401_PATTERNS = (
    "401 unauthorized",
    "missing bearer or basic authentication",
    "invalid_api_key",
)
_USAGE_LIMIT_PATTERNS = (
    "you've hit your usage limit",
    "you have hit your usage limit",
    "hit your usage limit",
    "exhausted your capacity on this model",
    "quota will reset after",
    "terminalquotaerror",
    "resource_exhausted",
    "rate limit exceeded",
    "too many requests",
)
_USAGE_LIMIT_RETRY_AT_RE = re.compile(
    r"try again at\s+([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?\s*(?:[aApP][mM])?)",
    re.IGNORECASE,
)
_ERROR_HINT_RE = re.compile(
    r"(error|exception|traceback|usage limit|argument list too long|timed out|unauthorized|invalid)",
    re.IGNORECASE,
)
_PROFILE_SELECTION_STICKY = "sticky"
_PROFILE_SELECTION_ROUND_ROBIN = "round_robin"
LOGGER = logging.getLogger(__name__)


def _increment_best_state_counter(conn: Any | None, key: str) -> int:
    if conn is None:
        return 0
    raw = get_kv(conn, key, default="0") or "0"
    try:
        current = int(raw)
    except Exception:
        current = 0
    current += 1
    set_kv(conn, key, str(current))
    return current


def _repo_file_listing(repo_path: str | Path, max_files: int = 500) -> str:
    root = Path(repo_path)
    files: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        files.append(rel)
    if not files:
        return "(empty repo)"
    if len(files) > max_files:
        omitted = len(files) - max_files
        files = files[:max_files] + [f"... ({omitted} files omitted)"]
    return "\n".join(files)


def _iter_file_index(path: Path) -> int | None:
    m = _ITER_RE.match(path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _next_iter_index(run_dir: Path) -> int:
    tree = Path(run_dir) / "tree"
    if not tree.exists():
        return 1
    last = 0
    for p in tree.glob("iter_*.json"):
        idx = _iter_file_index(p)
        if idx is not None and idx > last:
            last = idx
    return last + 1


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _llm_failure_text(res: dict[str, Any]) -> str:
    chunks = [
        str(res.get("error", "")),
        str(res.get("stderr", "")),
        str(res.get("text", "")),
    ]
    return "\n".join(chunks)


def _single_line(text: str, max_chars: int = 240) -> str:
    compact = " ".join(str(text).split())
    if not compact:
        return ""
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _best_error_line(text: str) -> str:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return ""
    for line in lines:
        if _ERROR_HINT_RE.search(line):
            return line
    return lines[0]


def _llm_error_summary(res: dict[str, Any], max_chars: int = 240) -> str:
    # Prefer structured error field, then meaningful stderr line, then text.
    raw_error = str(res.get("error", "")).strip()
    if raw_error:
        return _single_line(raw_error, max_chars=max_chars)
    stderr_line = _best_error_line(str(res.get("stderr", "")))
    if stderr_line:
        return _single_line(stderr_line, max_chars=max_chars)
    text_line = _best_error_line(str(res.get("text", "")))
    if text_line:
        return _single_line(text_line, max_chars=max_chars)
    return "(no error text)"


def _is_auth_401_failure(res: dict[str, Any]) -> bool:
    if bool(res.get("ok", False)):
        return False
    haystack = _llm_failure_text(res).lower()
    return any(token in haystack for token in _AUTH_401_PATTERNS)


def _is_usage_limit_failure(res: dict[str, Any]) -> bool:
    if bool(res.get("ok", False)):
        return False
    haystack = _llm_failure_text(res).lower()
    return any(token in haystack for token in _USAGE_LIMIT_PATTERNS)


def _usage_limit_wait_sec_from_response(
    res: dict[str, Any],
    default_wait_sec: float,
    *,
    now_ts: float | None = None,
) -> tuple[float, str, str | None]:
    """Resolve usage-limit sleep time from response text or fallback to config."""
    fallback = max(1.0, float(default_wait_sec))
    match = _USAGE_LIMIT_RETRY_AT_RE.search(_llm_failure_text(res))
    if not match:
        return fallback, "default_config", None

    retry_hint = " ".join(match.group(1).split())
    token = retry_hint.upper()
    now_value = time.time() if now_ts is None else float(now_ts)
    now_dt = datetime.fromtimestamp(now_value)

    for fmt in ("%I:%M %p", "%H:%M", "%I:%M:%S %p", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(token, fmt)
        except ValueError:
            continue
        retry_dt = now_dt.replace(
            hour=parsed.hour,
            minute=parsed.minute,
            second=parsed.second,
            microsecond=0,
        )
        if retry_dt.timestamp() <= now_value:
            retry_dt += timedelta(days=1)
        wait_sec = max(1.0, retry_dt.timestamp() - now_value)
        return wait_sec, "parsed_retry_at", retry_hint

    return fallback, "default_config_parse_failed", retry_hint


def _normalize_llm_profiles(cfg: GlobalConfig) -> list[LLMProfileSpec]:
    fallback = LLMProfileSpec(
        name="default",
        cli=str(cfg.llm.cli),
        args=list(cfg.llm.args),
        mode=str(cfg.llm.mode),
    )
    raw_profiles = getattr(cfg.llm, "profiles", None)
    if not isinstance(raw_profiles, list) or not raw_profiles:
        return [fallback]

    out: list[LLMProfileSpec] = []
    for idx, raw in enumerate(raw_profiles):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", f"profile-{idx + 1}")).strip() or f"profile-{idx + 1}"
        cli_raw = str(raw.get("cli", fallback.cli)).strip()
        cli = cli_raw or fallback.cli
        has_args = "args" in raw
        args_raw = raw.get("args")
        if has_args and isinstance(args_raw, list) and all(isinstance(item, str) for item in args_raw):
            args = list(args_raw)
        elif has_args:
            args = list(fallback.args)
        else:
            # Do not inherit codex flags into different CLIs by default.
            cli_name = Path(cli).name.strip().lower()
            if cli == fallback.cli:
                args = list(fallback.args)
            elif cli_name == "gemini":
                # Stream prompt via stdin to avoid argv length limits.
                args = ["-m", "gemini-3-pro-preview", "-p", "-"]
            else:
                args = []

        has_mode = "mode" in raw
        mode_raw = str(raw.get("mode", "")).strip().lower() if has_mode else ""
        if has_mode:
            mode = mode_raw if mode_raw in ("stdin", "argv") else fallback.mode
        else:
            # Gemini CLI can read prompt from stdin via: gemini -m gemini-3-pro-preview -p -
            mode = "stdin" if cli.lower() == "gemini" else fallback.mode
        out.append(LLMProfileSpec(name=name, cli=cli, args=args, mode=mode))

    return out or [fallback]


def _llm_profile_model(profile: LLMProfileSpec) -> str:
    for idx, token in enumerate(profile.args):
        if token == "-m" and idx + 1 < len(profile.args):
            model = str(profile.args[idx + 1]).strip()
            if model:
                return model
        if token.startswith("--model="):
            model = token.split("=", 1)[1].strip()
            if model:
                return model
    cli_name = Path(profile.cli).name.strip().lower()
    if cli_name:
        return cli_name
    return "unknown"


def _llm_profile_ready_key(profile_index: int) -> str:
    return f"llm_profile.{int(profile_index)}.next_ready_unix"


def _load_llm_profile_ready_state(conn: Any | None, profile_count: int) -> dict[int, float]:
    state = {idx: 0.0 for idx in range(max(0, int(profile_count)))}
    if conn is None:
        return state
    for idx in state.keys():
        raw = get_kv(conn, _llm_profile_ready_key(idx), default="0") or "0"
        try:
            state[idx] = max(0.0, float(raw))
        except Exception:
            state[idx] = 0.0
    return state


def _save_llm_profile_ready_state(conn: Any | None, profile_index: int, next_ready_unix: float) -> None:
    if conn is None:
        return
    set_kv(conn, _llm_profile_ready_key(profile_index), f"{max(0.0, float(next_ready_unix)):.6f}")


def _normalize_llm_profile_selection_mode(raw: str | None) -> str:
    mode = str(raw or "").strip().lower()
    if mode == _PROFILE_SELECTION_ROUND_ROBIN:
        return _PROFILE_SELECTION_ROUND_ROBIN
    return _PROFILE_SELECTION_STICKY


def _load_llm_profile_active_index(conn: Any | None, profile_count: int) -> int:
    if conn is None:
        return 0
    raw = get_kv(conn, "llm_profile.active_index", default="0") or "0"
    try:
        idx = int(raw)
    except Exception:
        idx = 0
    if profile_count <= 0:
        return 0
    return min(max(0, idx), profile_count - 1)


def _save_llm_profile_active_index(conn: Any | None, index: int) -> None:
    if conn is None:
        return
    set_kv(conn, "llm_profile.active_index", str(max(0, int(index))))


def _load_llm_profile_rr_cursor(conn: Any | None, profile_count: int) -> int:
    if conn is None:
        return 0
    raw = get_kv(conn, "llm_profile.rr_cursor", default="0") or "0"
    try:
        idx = int(raw)
    except Exception:
        idx = 0
    if profile_count <= 0:
        return 0
    return min(max(0, idx), profile_count - 1)


def _save_llm_profile_rr_cursor(conn: Any | None, cursor: int) -> None:
    if conn is None:
        return
    set_kv(conn, "llm_profile.rr_cursor", str(max(0, int(cursor))))


def _resolve_llm_profile_selection_mode(conn: Any | None, cfg: GlobalConfig) -> str:
    default_mode = _normalize_llm_profile_selection_mode(cfg.llm.profile_selection_mode)
    if conn is None:
        return default_mode
    raw = get_kv(conn, "llm_profile.selection_mode")
    if raw is None:
        return default_mode
    candidate = str(raw).strip().lower()
    if candidate in (_PROFILE_SELECTION_STICKY, _PROFILE_SELECTION_ROUND_ROBIN):
        return candidate
    return default_mode


def _ensure_llm_profile_state_keys(conn: Any | None, cfg: GlobalConfig) -> None:
    if conn is None:
        return
    profiles = _normalize_llm_profiles(cfg)
    selection_default = _normalize_llm_profile_selection_mode(cfg.llm.profile_selection_mode)
    signature = json.dumps(
        [
            {
                "name": p.name,
                "cli": p.cli,
                "args": p.args,
                "mode": p.mode,
            }
            for p in profiles
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    previous_signature = get_kv(conn, "llm_profile.signature")
    signature_changed = (previous_signature != signature)
    set_kv(conn, "llm_profile.signature", signature)
    if get_kv(conn, "llm_profile.selection_mode") is None:
        set_kv(conn, "llm_profile.selection_mode", selection_default)
    set_kv(conn, "llm_profile.count", str(len(profiles)))
    for idx in range(len(profiles)):
        key = _llm_profile_ready_key(idx)
        if signature_changed:
            # Index-based cooldown keys are invalid when profile ordering changes.
            set_kv(conn, key, "0")
        elif get_kv(conn, key) is None:
            set_kv(conn, key, "0")
    if signature_changed:
        active = 0
        rr_cursor = 0
    else:
        active = _load_llm_profile_active_index(conn, len(profiles))
        rr_cursor = _load_llm_profile_rr_cursor(conn, len(profiles))
    _save_llm_profile_active_index(conn, active)
    _save_llm_profile_rr_cursor(conn, rr_cursor)


def _select_ready_profile_index(
    *,
    profile_count: int,
    next_ready_unix: dict[int, float],
    now_unix: float,
    selection_mode: str,
    active_index: int,
    rr_cursor: int,
) -> int | None:
    if profile_count <= 0:
        return None
    mode = _normalize_llm_profile_selection_mode(selection_mode)
    if mode == _PROFILE_SELECTION_ROUND_ROBIN:
        start = min(max(0, int(rr_cursor)), profile_count - 1)
        for offset in range(profile_count):
            idx = (start + offset) % profile_count
            if float(next_ready_unix.get(idx, 0.0)) <= float(now_unix):
                return idx
        return None

    current = min(max(0, int(active_index)), profile_count - 1)
    if float(next_ready_unix.get(current, 0.0)) <= float(now_unix):
        return current
    for idx in range(current + 1, profile_count):
        if float(next_ready_unix.get(idx, 0.0)) <= float(now_unix):
            return idx
    for idx in range(0, current):
        if float(next_ready_unix.get(idx, 0.0)) <= float(now_unix):
            return idx
    return None


def _soonest_profile_wait(
    *,
    next_ready_unix: dict[int, float],
    now_unix: float,
) -> tuple[int, float]:
    soonest_index = min(next_ready_unix.keys(), key=lambda idx: float(next_ready_unix.get(idx, 0.0)))
    wait_sec = max(1.0, float(next_ready_unix.get(soonest_index, 0.0)) - float(now_unix))
    return soonest_index, wait_sec


def _llm_profiles_summary(cfg: GlobalConfig) -> str:
    profiles = _normalize_llm_profiles(cfg)
    return ", ".join(
        f"{p.name}:{_llm_profile_model(p)}:{p.mode}"
        for p in profiles
    )


def _fmt_unix_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _notify_llm_usage_limit(
    *,
    run_dir: Path,
    iter_index: int | None,
    stage: str,
    profile_name: str,
    profile_index: int,
    model: str,
    wait_sec: float,
    wait_source: str,
    retry_at_hint: str | None,
    next_ready_unix: float,
) -> None:
    wait_until_text = _fmt_unix_ts(next_ready_unix)
    msg = "\n".join(
        [
            "[MARS] LLM usage limit",
            f"run={run_dir.name}",
            f"iter={iter_index} stage={stage}",
            f"profile={profile_name} index={profile_index} model={model}",
            f"wait_sec={wait_sec:.1f} source={wait_source}",
            f"retry_at_hint={retry_at_hint or '-'}",
            f"wait_until={wait_until_text}",
        ]
    )
    send_telegram(
        msg,
        dedupe_key=f"llm-usage-limit:{run_dir}:{profile_index}:{int(next_ready_unix)}",
        min_interval_sec=5.0,
    )


def _notify_all_profiles_usage_limited(
    *,
    run_dir: Path,
    iter_index: int | None,
    stage: str,
    wait_sec: float,
    wait_until_unix: float,
    selected_profile_index: int,
    selected_profile_name: str,
    profiles: list[LLMProfileSpec],
    profile_ready_state: dict[int, float],
    now_unix: float,
) -> None:
    lines: list[str] = []
    for idx, profile in enumerate(profiles):
        next_ready = float(profile_ready_state.get(idx, 0.0))
        remain = max(0.0, next_ready - now_unix)
        lines.append(
            f"- idx={idx} name={profile.name} model={_llm_profile_model(profile)} wait_sec={remain:.1f} next_ready={_fmt_unix_ts(next_ready)}"
        )
    msg = "\n".join(
        [
            "[MARS] All LLM profiles are usage-limited",
            f"run={run_dir.name}",
            f"iter={iter_index} stage={stage}",
            f"selected_wait_profile={selected_profile_name} index={selected_profile_index}",
            f"sleep_sec={wait_sec:.1f} wait_until={_fmt_unix_ts(wait_until_unix)}",
            "profiles:",
            *lines,
        ]
    )
    send_telegram(
        msg,
        dedupe_key=f"llm-usage-limit-all:{run_dir}:{int(wait_until_unix)}",
        min_interval_sec=5.0,
    )


def _sync_auth_retry_state(conn: Any | None, state: dict[str, int]) -> None:
    if conn is None:
        return
    set_kv(conn, "llm_auth_401_consecutive", str(int(state.get("consecutive", 0))))
    set_kv(conn, "llm_auth_401_total", str(int(state.get("total", 0))))


def _sync_usage_limit_state(conn: Any | None, state: dict[str, int]) -> None:
    if conn is None:
        return
    set_kv(conn, "llm_usage_limit_consecutive", str(int(state.get("consecutive", 0))))
    set_kv(conn, "llm_usage_limit_total", str(int(state.get("total", 0))))


def _update_identical_metric_guard(
    *,
    conn: Any,
    run_dir: Path,
    iter_index: int,
    metric_value: float,
    threshold: int,
) -> tuple[int, bool, str]:
    """Track identical metric streak and request forced DEBUG when threshold is reached."""
    threshold = max(1, int(threshold))
    prev_raw = get_kv(conn, "last_metric_value", default=None)
    prev_metric: float | None = None
    if prev_raw not in (None, ""):
        try:
            prev_metric = float(prev_raw)
        except Exception:
            prev_metric = None

    streak = int(get_kv(conn, "identical_metric_streak") or "0")
    if prev_metric is not None and metric_value == prev_metric:
        streak += 1
    else:
        streak = 1

    set_kv(conn, "last_metric_value", repr(metric_value))
    set_kv(conn, "identical_metric_streak", str(streak))

    if streak < threshold:
        return streak, False, ""

    reason = (
        f"Identical metric repeated {streak} consecutive times "
        f"(metric={metric_value}); forcing DEBUG for experiment-design diagnosis."
    )
    set_kv(conn, "force_debug_next_iter", "1")
    set_kv(conn, "force_debug_reason", reason)
    _append_jsonl(
        Path(run_dir) / "artifacts" / "guards" / "identical_metric_guard.jsonl",
        {
            "event": "identical_metric_guard_triggered",
            "iter_index": iter_index,
            "metric_value": metric_value,
            "streak": streak,
            "threshold": threshold,
            "reason": reason,
            "timestamp_unix": time.time(),
        },
    )
    return streak, True, reason


def _build_iteration_experiment_context(
    *,
    conn: Any,
    run_dir: Path,
    iter_index: int,
    parent_id: int,
    action: ActionType,
    selection_path: list[int],
    max_recent_iters: int = 6,
) -> str:
    lines: list[str] = []
    lines.append("== Iteration Snapshot ==")
    lines.append(f"iter_index: {iter_index}")
    lines.append(f"selected_parent_id: {parent_id}")
    lines.append(f"selected_action: {action.value}")
    lines.append(f"selection_path: {selection_path}")

    best_id_raw = get_kv(conn, "best_node_id", default=None)
    if best_id_raw:
        try:
            best_node = get_node(conn, int(best_id_raw))
            lines.append(f"best_node_id: {best_node.node_id}")
            lines.append(f"best_metric: {best_node.metric}")
            lines.append(f"best_status: {best_node.status.value}")
        except Exception:
            lines.append(f"best_node_id: {best_id_raw}")
    lines.append(f"valid_since_best: {get_kv(conn, 'valid_since_best', default='0')}")
    lines.append(f"identical_metric_streak: {get_kv(conn, 'identical_metric_streak', default='0')}")
    lines.append(f"metric_history_min: {get_kv(conn, 'metric_history_min', default='(none)')}")
    lines.append(f"metric_history_max: {get_kv(conn, 'metric_history_max', default='(none)')}")
    lines.append(f"llm_auth_401_total: {get_kv(conn, 'llm_auth_401_total', default='0')}")
    lines.append(f"llm_usage_limit_total: {get_kv(conn, 'llm_usage_limit_total', default='0')}")

    status_parts: list[str] = []
    try:
        for status, cnt in conn.execute("SELECT status, COUNT(*) FROM nodes GROUP BY status").fetchall():
            status_parts.append(f"{status}={cnt}")
    except Exception:
        pass
    if status_parts:
        lines.append("node_status_counts: " + ", ".join(sorted(status_parts)))

    spool_parts: list[str] = []
    for queue_name in ("llm", "exec"):
        for state in ("pending", "running", "done", "failed"):
            p = Path(run_dir) / "spool" / queue_name / state
            try:
                c = sum(1 for _ in p.glob("*.json")) if state in ("pending", "running") else sum(
                    1 for _ in p.glob("*.result.json")
                )
            except Exception:
                c = 0
            spool_parts.append(f"{queue_name}/{state}={c}")
    lines.append("spool_counts: " + ", ".join(spool_parts))

    lines.append("recent_iterations:")
    tree_dir = Path(run_dir) / "tree"
    iter_files = sorted(
        tree_dir.glob("iter_*.json"),
        key=lambda p: _iter_file_index(p) or -1,
    )
    total_iters = 0
    total_patch_applied = 0
    total_metric_found = 0
    total_exit0 = 0
    total_exec_fail = 0
    for p in iter_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        total_iters += 1
        if bool(obj.get("patch_applied")):
            total_patch_applied += 1
        ex = obj.get("exec", {})
        if isinstance(ex, dict):
            if bool(ex.get("metric_found")):
                total_metric_found += 1
            exit_code_raw = ex.get("exit_code", 1)
            try:
                exit_code = int(exit_code_raw)
            except Exception:
                exit_code = 1
            if exit_code == 0:
                total_exit0 += 1
            if not bool(ex.get("ok", False)):
                total_exec_fail += 1
    lines.append(
        "iteration_stats_total: "
        f"iters={total_iters} patch_applied={total_patch_applied} "
        f"metric_found={total_metric_found} exit0={total_exit0} exec_fail={total_exec_fail}"
    )
    if not iter_files:
        lines.append("- (none)")
    else:
        for p in iter_files[-max_recent_iters:]:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                lines.append(f"- {p.name}: unreadable")
                continue
            ex = obj.get("exec", {}) if isinstance(obj, dict) else {}
            err = str(ex.get("error_summary", "") or "").strip().replace("\n", " ")
            if len(err) > 140:
                err = err[:137] + "..."
            lines.append(
                "- "
                f"{p.name}: action={obj.get('action')} patch_applied={obj.get('patch_applied')} "
                f"metric_found={ex.get('metric_found')} metric={ex.get('metric_value')} "
                f"exit={ex.get('exit_code')} err={err or '(none)'}"
            )

    return "\n".join(lines)[:12000]


def _load_json_list(conn: Any, key: str) -> list[str]:
    raw = get_kv(conn, key, default="")
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        return []
    except Exception:
        return []


def _save_json_list(conn: Any, key: str, items: list[str]) -> None:
    set_kv(conn, key, json.dumps(items, ensure_ascii=False))


def _save_lesson_if_new(
    conn: Any,
    lesson: Any,
    cfg: GlobalConfig,
    *,
    run_dir: Path | None = None,
    iter_index: int | None = None,
) -> bool:
    if lesson_exists_by_fingerprint(conn, lesson.fingerprint):
        return False
    existing = get_recent_lessons(conn, kind=lesson.kind, k=cfg.memory.keep_recent_k)
    if is_duplicate_lesson_llm(
        lesson,
        existing,
        llm_cfg=cfg.llm,
        run_dir=run_dir,
        iter_index=iter_index,
    ):
        return False
    add_lesson(conn, lesson)
    return True


def _objective_metric(raw_metric: float, higher_is_better: bool) -> float:
    return raw_metric if higher_is_better else -raw_metric


def _is_improved(new_metric: float, best_metric: float | None, higher_is_better: bool) -> bool:
    if best_metric is None:
        return True
    if higher_is_better:
        return new_metric > best_metric
    return new_metric < best_metric


def _is_fully_expanded(conn: Any, node_id: int, root_id: int, cfg: GlobalConfig, valid_since_best: int) -> bool:
    node = get_node(conn, node_id)
    children = list_children(conn, node_id)

    if node_id == root_id:
        # Root can generate drafts up to configured cap, then re-activate after n_s non-improving valid nodes.
        if len(children) < max(1, int(cfg.mcts.max_root_drafts)):
            return False
        if valid_since_best >= max(1, int(cfg.mcts.root_reactivate_after_valid_no_improve)):
            return False
        return True

    # Paper rule: buggy nodes are always fully expanded; debugging is handled as in-iteration loop.
    if node.status == NodeStatus.BUGGY:
        return True

    limit = expansion_limit(node, is_root=False, cfg=cfg.mcts)
    return len(children) >= limit


def _select_parent_for_expansion(conn: Any, root_id: int, cfg: GlobalConfig, valid_since_best: int) -> tuple[int, list[int]]:
    """UCT selection with root re-activation when traversal hits fully expanded leaves."""
    current_id = root_id
    path = [root_id]

    while True:
        if not _is_fully_expanded(conn, current_id, root_id, cfg, valid_since_best=valid_since_best):
            return current_id, path

        children = list_children(conn, current_id)
        if not children:
            # Leaf is fully expanded -> reactivate root for new draft branch.
            return root_id, [root_id]

        current = get_node(conn, current_id)
        next_child: int | None = None
        next_score: float | None = None
        for child_id in children:
            child = get_node(conn, child_id)
            score = uct_score(
                parent_visits=max(1, int(current.visits)),
                child_visits=int(child.visits),
                child_value_sum=float(child.value_sum),
                c_uct=cfg.mcts.c_uct,
            )
            if next_score is None or score > next_score:
                next_child = child_id
                next_score = score

        if next_child is None:
            return root_id, [root_id]

        current_id = next_child
        path.append(current_id)


def _backup_path(conn: Any, path_node_ids: list[int], reward: float) -> None:
    for node_id in path_node_ids:
        inc_visit_and_value(conn, node_id, reward)


def _remaining_budget_sec(run_started_at: float, wallclock_sec: int) -> float:
    return float(wallclock_sec) - max(0.0, time.time() - run_started_at)


def _run_exec_once(
    exec_q: Any,
    task: Any,
    repo_path: str,
    timeout_sec: int | None = None,
    *,
    iter_index: int | None = None,
    stage: str = "exec_run",
) -> tuple[str, dict[str, Any], ExecResult]:
    effective_timeout = int(timeout_sec if timeout_sec is not None else task.per_run_timeout_sec)
    effective_timeout = max(1, effective_timeout)
    exec_job = {
        "kind": "exec",
        "type": "RUN",
        "payload": {
            "repo_path": repo_path,
            "cmd": task.entrypoint,
            "timeout_sec": effective_timeout,
            "final_metric_regex": task.final_metric_regex,
        },
        "meta": {
            "iter_index": iter_index,
            "stage": stage,
        },
    }
    LOGGER.info(
        "exec enqueue iter=%s stage=%s timeout_sec=%s repo=%s",
        iter_index,
        stage,
        effective_timeout,
        repo_path,
    )
    exec_job_id = enqueue(exec_q, exec_job)
    LOGGER.info(
        "exec waiting iter=%s stage=%s job_id=%s",
        iter_index,
        stage,
        exec_job_id,
    )
    exec_res_raw = wait_for_result(exec_q, exec_job_id, timeout_sec=effective_timeout + 60)
    exec_res = ExecResult(
        ok=bool(exec_res_raw.get("ok", False)),
        exit_code=int(exec_res_raw.get("exit_code", 1)),
        exec_time_sec=float(exec_res_raw.get("exec_time_sec", 0.0)),
        metric_value=exec_res_raw.get("metric_value", None),
        metric_found=bool(exec_res_raw.get("metric_found", False)),
        stdout_path=str(exec_res_raw.get("stdout_path", "")),
        stderr_path=str(exec_res_raw.get("stderr_path", "")),
        error_summary=exec_res_raw.get("error_summary", None),
    )
    LOGGER.info(
        "exec done iter=%s stage=%s job_id=%s exit_code=%s metric_found=%s metric=%s elapsed_sec=%.2f",
        iter_index,
        stage,
        exec_job_id,
        exec_res.exit_code,
        exec_res.metric_found,
        exec_res.metric_value,
        exec_res.exec_time_sec,
    )
    return exec_job_id, exec_res_raw, exec_res


def _request_llm(
    llm_q: Any,
    run_dir: Path,
    cfg: GlobalConfig,
    job_type: str,
    payload: dict[str, Any],
    *,
    llm_cwd: str | None = None,
    conn: Any | None = None,
    iter_index: int | None = None,
    stage: str = "",
    auth_retry_state: dict[str, int] | None = None,
) -> tuple[str, dict[str, Any]]:
    backoff_sec = max(1.0, float(cfg.llm.auth_retry_backoff_sec))
    usage_wait_sec = max(1.0, float(cfg.llm.usage_limit_wait_sec))
    llm_profiles = _normalize_llm_profiles(cfg)
    profile_ready_state = _load_llm_profile_ready_state(conn, len(llm_profiles))
    profile_active_index = _load_llm_profile_active_index(conn, len(llm_profiles))
    profile_rr_cursor = _load_llm_profile_rr_cursor(conn, len(llm_profiles))
    local_state = auth_retry_state if auth_retry_state is not None else {"consecutive": 0, "total": 0}
    usage_state = {"consecutive": 0, "total": 0}
    events_path = Path(run_dir) / "artifacts" / "llm_auth_401_events.jsonl"
    usage_events_path = Path(run_dir) / "artifacts" / "llm_usage_limit_events.jsonl"
    attempt_number = 0

    while True:
        stage_label = stage or job_type
        now_unix = time.time()
        selection_mode = _resolve_llm_profile_selection_mode(conn, cfg)
        profile_index = _select_ready_profile_index(
            profile_count=len(llm_profiles),
            next_ready_unix=profile_ready_state,
            now_unix=now_unix,
            selection_mode=selection_mode,
            active_index=profile_active_index,
            rr_cursor=profile_rr_cursor,
        )
        if profile_index is None:
            wait_profile_index, wait_sec = _soonest_profile_wait(
                next_ready_unix=profile_ready_state,
                now_unix=now_unix,
            )
            wait_profile = llm_profiles[wait_profile_index]
            wait_until = float(profile_ready_state.get(wait_profile_index, now_unix))
            wait_event = {
                "event": "llm_usage_limit_all_profiles_wait",
                "iter_index": iter_index,
                "stage": stage,
                "job_type": job_type,
                "attempt_number": attempt_number,
                "wait_sec": wait_sec,
                "wait_until_unix": wait_until,
                "selected_profile_name": wait_profile.name,
                "selected_profile_index": wait_profile_index,
                "timestamp_unix": now_unix,
            }
            _append_jsonl(usage_events_path, wait_event)
            LOGGER.warning(
                "llm usage limit all profiles waiting; sleeping iter=%s stage=%s wait_sec=%s wait_until_unix=%s selected_profile=%s selected_profile_index=%s",
                iter_index,
                stage_label,
                wait_sec,
                wait_until,
                wait_profile.name,
                wait_profile_index,
            )
            _notify_all_profiles_usage_limited(
                run_dir=run_dir,
                iter_index=iter_index,
                stage=stage_label,
                wait_sec=wait_sec,
                wait_until_unix=wait_until,
                selected_profile_index=wait_profile_index,
                selected_profile_name=wait_profile.name,
                profiles=llm_profiles,
                profile_ready_state=profile_ready_state,
                now_unix=now_unix,
            )
            time.sleep(wait_sec)
            continue

        selected_profile = llm_profiles[profile_index]
        selected_model = _llm_profile_model(selected_profile)
        if profile_index != profile_active_index:
            profile_active_index = profile_index
            _save_llm_profile_active_index(conn, profile_active_index)
        if selection_mode == _PROFILE_SELECTION_ROUND_ROBIN:
            profile_rr_cursor = (profile_index + 1) % max(1, len(llm_profiles))
            _save_llm_profile_rr_cursor(conn, profile_rr_cursor)
        else:
            profile_rr_cursor = profile_index
            _save_llm_profile_rr_cursor(conn, profile_rr_cursor)

        attempt_number += 1
        llm_call_seq = _increment_best_state_counter(conn, "llm_call_total")
        job_payload = dict(payload)
        if llm_cwd:
            job_payload["llm_cwd"] = str(llm_cwd)
        job_payload["llm_profile_name"] = selected_profile.name
        job_payload["llm_profile_index"] = profile_index
        job_payload["llm_model"] = selected_model
        job_payload["llm_cli"] = selected_profile.cli
        job_payload["llm_args"] = list(selected_profile.args)
        job_payload["llm_mode"] = selected_profile.mode
        job = {
            "kind": "llm",
            "type": job_type,
            "payload": job_payload,
            "meta": {
                "iter_index": iter_index,
                "stage": stage_label,
                "attempt": attempt_number,
                "llm_call_seq": llm_call_seq,
                "llm_profile_name": selected_profile.name,
                "llm_profile_index": profile_index,
                "llm_model": selected_model,
            },
        }
        LOGGER.info(
            "llm enqueue iter=%s stage=%s type=%s attempt=%s llm_call_seq=%s profile=%s profile_index=%s model=%s",
            iter_index,
            stage_label,
            job_type,
            attempt_number,
            llm_call_seq,
            selected_profile.name,
            profile_index,
            selected_model,
        )
        job_id = enqueue(llm_q, job)
        LOGGER.info(
            "llm waiting iter=%s stage=%s job_id=%s profile=%s profile_index=%s model=%s",
            iter_index,
            stage_label,
            job_id,
            selected_profile.name,
            profile_index,
            selected_model,
        )
        res = wait_for_result(llm_q, job_id, timeout_sec=cfg.llm.timeout_sec + 60)
        write_json(run_dir, f"artifacts/llm_raw/{job_id}.json", res)
        text_len = len(str(res.get("text", "")))
        LOGGER.info(
            "llm result iter=%s stage=%s job_id=%s ok=%s returncode=%s text_len=%s profile=%s profile_index=%s model=%s",
            iter_index,
            stage_label,
            job_id,
            bool(res.get("ok", False)),
            res.get("returncode"),
            text_len,
            selected_profile.name,
            profile_index,
            selected_model,
        )
        LOGGER.info(
            "llm artifacts iter=%s stage=%s job_id=%s prompt_path=%s raw_path=%s stderr_len=%s profile=%s profile_index=%s model=%s",
            iter_index,
            stage_label,
            job_id,
            res.get("prompt_path"),
            res.get("raw_path"),
            len(str(res.get("stderr", ""))),
            selected_profile.name,
            profile_index,
            selected_model,
        )
        if not bool(res.get("ok", False)):
            LOGGER.warning(
                "llm failure iter=%s stage=%s job_id=%s returncode=%s profile=%s profile_index=%s model=%s error_summary=%s",
                iter_index,
                stage_label,
                job_id,
                res.get("returncode"),
                selected_profile.name,
                profile_index,
                selected_model,
                _llm_error_summary(res),
            )

        if bool(res.get("ok", False)):
            local_state["consecutive"] = 0
            _sync_auth_retry_state(conn, local_state)
            usage_state["consecutive"] = 0
            _sync_usage_limit_state(conn, usage_state)
            if float(profile_ready_state.get(profile_index, 0.0)) > 0.0:
                profile_ready_state[profile_index] = 0.0
                _save_llm_profile_ready_state(conn, profile_index, 0.0)
            return job_id, res

        if not _is_auth_401_failure(res):
            _sync_auth_retry_state(conn, local_state)
            is_usage_limit = _is_usage_limit_failure(res)
            if is_usage_limit:
                usage_state["consecutive"] = int(usage_state.get("consecutive", 0)) + 1
                usage_state["total"] = int(usage_state.get("total", 0)) + 1
                usage_sleep_sec, usage_sleep_source, usage_retry_hint = _usage_limit_wait_sec_from_response(
                    res,
                    usage_wait_sec,
                )
            else:
                usage_state["consecutive"] = 0
                usage_sleep_sec = usage_wait_sec
                usage_sleep_source = "default_config"
                usage_retry_hint = None
            _sync_usage_limit_state(conn, usage_state)
            usage_next_ready_unix = time.time() + usage_sleep_sec
            profile_ready_state[profile_index] = max(
                float(profile_ready_state.get(profile_index, 0.0)),
                usage_next_ready_unix,
            )
            _save_llm_profile_ready_state(conn, profile_index, profile_ready_state[profile_index])
            usage_event = {
                "event": "llm_usage_limit" if is_usage_limit else "llm_failure_retry",
                "iter_index": iter_index,
                "stage": stage,
                "job_type": job_type,
                "job_id": job_id,
                "attempt_number": attempt_number,
                "sleep_sec": usage_sleep_sec,
                "sleep_source": usage_sleep_source,
                "retry_at_hint": usage_retry_hint,
                "profile_name": selected_profile.name,
                "profile_index": profile_index,
                "model": selected_model,
                "profile_next_ready_unix": profile_ready_state[profile_index],
                "consecutive": int(usage_state.get("consecutive", 0)),
                "total": int(usage_state.get("total", 0)),
                "timestamp_unix": time.time(),
            }
            _append_jsonl(usage_events_path, usage_event)
            if is_usage_limit:
                LOGGER.warning(
                    "llm usage limit; iter=%s stage=%s attempt=%s profile=%s profile_index=%s model=%s wait_sec=%s source=%s retry_at_hint=%s next_ready_unix=%s",
                    iter_index,
                    stage or job_type,
                    attempt_number,
                    selected_profile.name,
                    profile_index,
                    selected_model,
                    usage_sleep_sec,
                    usage_sleep_source,
                    usage_retry_hint,
                    profile_ready_state[profile_index],
                )
                _notify_llm_usage_limit(
                    run_dir=run_dir,
                    iter_index=iter_index,
                    stage=stage_label,
                    profile_name=selected_profile.name,
                    profile_index=profile_index,
                    model=selected_model,
                    wait_sec=usage_sleep_sec,
                    wait_source=usage_sleep_source,
                    retry_at_hint=usage_retry_hint,
                    next_ready_unix=profile_ready_state[profile_index],
                )
            else:
                LOGGER.warning(
                    "llm failure retry; iter=%s stage=%s attempt=%s profile=%s profile_index=%s model=%s wait_sec=%s source=%s error_summary=%s",
                    iter_index,
                    stage or job_type,
                    attempt_number,
                    selected_profile.name,
                    profile_index,
                    selected_model,
                    usage_sleep_sec,
                    usage_sleep_source,
                    _llm_error_summary(res),
                )
            continue

        local_state["consecutive"] = int(local_state.get("consecutive", 0)) + 1
        local_state["total"] = int(local_state.get("total", 0)) + 1
        _sync_auth_retry_state(conn, local_state)
        event = {
            "event": "llm_auth_401",
            "iter_index": iter_index,
            "stage": stage,
            "job_type": job_type,
            "job_id": job_id,
            "attempt_number": attempt_number,
            "retry_mode": "infinite",
            "sleep_sec": backoff_sec,
            "profile_name": selected_profile.name,
            "profile_index": profile_index,
            "model": selected_model,
            "consecutive": int(local_state.get("consecutive", 0)),
            "total": int(local_state.get("total", 0)),
            "timestamp_unix": time.time(),
        }
        _append_jsonl(events_path, event)
        LOGGER.warning(
            "llm auth 401; sleeping iter=%s stage=%s attempt=%s sleep_sec=%s profile=%s profile_index=%s model=%s",
            iter_index,
            stage or job_type,
            attempt_number,
            backoff_sec,
            selected_profile.name,
            profile_index,
            selected_model,
        )
        time.sleep(backoff_sec)


def _request_llm_with_auth_stop(
    *,
    llm_q: Any,
    run_dir: Path,
    cfg: GlobalConfig,
    conn: Any,
    iter_index: int | None,
    stage: str,
    auth_retry_state: dict[str, int],
    job_type: str,
    payload: dict[str, Any],
    llm_cwd: str | None = None,
) -> tuple[str, dict[str, Any]]:
    return _request_llm(
        llm_q,
        run_dir=run_dir,
        cfg=cfg,
        job_type=job_type,
        payload=payload,
        llm_cwd=llm_cwd,
        conn=conn,
        iter_index=iter_index,
        stage=stage,
        auth_retry_state=auth_retry_state,
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3]
    s = s.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            obj = json.loads(s[start:end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _parse_json_value(text: str) -> Any:
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3]
    s = s.strip()
    candidates: list[str] = [s]

    obj_start = s.find("{")
    obj_end = s.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidates.append(s[obj_start:obj_end + 1])

    arr_start = s.find("[")
    arr_end = s.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidates.append(s[arr_start:arr_end + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def _safe_read(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def _extract_module_specs(module_plan_text: str, max_modules: int = 8) -> list[str]:
    specs: list[str] = []
    seen_paths: set[str] = set()

    def add_spec(path: str, purpose: str = "", interfaces: list[str] | None = None) -> None:
        p = (path or "").strip()
        if not p or ".py" not in p or "runfile.py" in p.lower():
            return
        key = p.lower()
        if key in seen_paths:
            return
        desc = p
        if purpose:
            desc = f"{desc} | purpose: {purpose.strip()[:140]}"
        if interfaces:
            iface = ", ".join(str(x).strip() for x in interfaces if str(x).strip())
            if iface:
                desc = f"{desc} | interfaces: {iface[:180]}"
        specs.append(desc[:360])
        seen_paths.add(key)

    parsed = _parse_json_value(module_plan_text)
    modules: list[Any] = []
    if isinstance(parsed, dict):
        raw_modules = parsed.get("modules", [])
        if isinstance(raw_modules, list):
            modules = raw_modules
    elif isinstance(parsed, list):
        modules = parsed

    for m in modules:
        if isinstance(m, str):
            add_spec(m)
        elif isinstance(m, dict):
            path = str(m.get("path", "")).strip()
            purpose = str(m.get("purpose", "")).strip()
            interfaces = m.get("interfaces", [])
            interfaces_list = interfaces if isinstance(interfaces, list) else []
            add_spec(path, purpose=purpose, interfaces=interfaces_list)
        if len(specs) >= max_modules:
            return specs[:max_modules]

    # Fallback for free-form text plans.
    for raw in (module_plan_text or "").splitlines():
        line = raw.strip().lstrip("-").strip()
        if not line:
            continue
        m = re.search(r"([A-Za-z0-9_./-]+\.py)\b", line)
        if not m:
            continue
        add_spec(m.group(1))
        if len(specs) >= max_modules:
            break
    return specs[:max_modules]


def _attempt_apply_patch(
    *,
    ctx: ControllerContext,
    conn: Any,
    child_id: int,
    child_repo: str,
    patch_job_id: str,
    patch_res: dict[str, Any],
) -> tuple[bool, str, str | None]:
    if not bool(patch_res.get("ok", False)):
        return False, "", None
    update_node_status(conn, child_id, NodeStatus.CODEGEN_DONE)
    patch_text = extract_unified_diff(str(patch_res.get("text", "")))
    write_text(ctx.run_dir, f"artifacts/patches/{patch_job_id}.diff", patch_text)
    if not patch_text:
        return False, "", None
    try:
        apply_unified_diff(child_repo, patch_text)
        update_node_status(conn, child_id, NodeStatus.PATCHED)
        return True, patch_text, None
    except Exception as e:
        return False, patch_text, summarize_error(repr(e))


def _ensure_task_preparation(
    *,
    conn: Any,
    llm_q: Any,
    ctx: ControllerContext,
    cfg: GlobalConfig,
    task: Any,
    root_repo: str,
    auth_retry_state: dict[str, int] | None = None,
) -> tuple[str | None, str]:
    local_ctx_obj = collect_task_preparation_context(task=task, repo_path=root_repo)
    write_json(ctx.run_dir, "artifacts/task_prep/local_context.json", local_ctx_obj)
    local_ctx_text = task_prep_context_text(local_ctx_obj, max_chars=12000)
    write_text(ctx.run_dir, "artifacts/task_prep/local_context.txt", local_ctx_text)
    set_kv(conn, "task_prep_local_context", local_ctx_text)

    existing_summary = get_kv(conn, "task_prep_summary", default="") or ""
    existing_job_id = get_kv(conn, "task_prep_job_id", default=None)
    if existing_summary:
        return existing_job_id, existing_summary

    prep_job_id, prep_res = _request_llm(
        llm_q,
        run_dir=ctx.run_dir,
        cfg=cfg,
        job_type="TASK_PREPARE",
        llm_cwd=root_repo,
        conn=conn,
        iter_index=None,
        stage="task_prepare",
        auth_retry_state=auth_retry_state,
        payload={
            "instruction": task.instruction,
            "task_workdir": task.workdir,
            "file_list": _repo_file_listing(root_repo),
            "local_context": local_ctx_text,
        },
    )
    llm_text = str(prep_res.get("text", "")).strip() if bool(prep_res.get("ok", False)) else ""
    if llm_text:
        prep_summary = (llm_text + "\n\n==== Local Task-Prep Context ====\n" + local_ctx_text)[:12000]
    else:
        prep_summary = local_ctx_text
    set_kv(conn, "task_prep_job_id", prep_job_id)
    set_kv(conn, "task_prep_summary", prep_summary)
    return prep_job_id, prep_summary


def init_run(ctx: ControllerContext) -> tuple[GlobalConfig, Any]:
    ensure_run_dirs(ctx.run_dir)

    cfg = load_global_config(ctx.default_conf_yaml)
    task = load_task_spec(ctx.task_yaml)

    # Create queues
    ensure_queue_dirs(ctx.run_dir, "llm")
    ensure_queue_dirs(ctx.run_dir, "exec")

    conn = connect(ctx.run_dir)
    init_schema(conn)

    # Root node (node_id=1 in DB order). Use an empty repo_path placeholder until materialized.
    existing_root = get_kv(conn, "root_node_id")
    if existing_root is None:
        root_repo = materialize_repo(ctx.run_dir, node_id=0, template_path=task.repo_template_path)
        root_id = create_node(conn, parent_id=None, action=ActionType.DRAFT, status=NodeStatus.DONE, repo_path=root_repo)
        set_kv(conn, "root_node_id", str(root_id))
        set_kv(conn, "best_node_id", str(root_id))
        set_kv(conn, "metric_history_min", "")
        set_kv(conn, "metric_history_max", "")
        set_kv(conn, "valid_since_best", "0")
        set_kv(conn, "idea_history_json", "[]")
    else:
        if get_kv(conn, "valid_since_best") is None:
            set_kv(conn, "valid_since_best", "0")
        if get_kv(conn, "idea_history_json") is None:
            set_kv(conn, "idea_history_json", "[]")
    return cfg, task


def run_controller(ctx: ControllerContext, max_iters: int = 3) -> None:
    cfg, task = init_run(ctx)
    conn = connect(ctx.run_dir)

    llm_q = ensure_queue_dirs(ctx.run_dir, "llm")
    exec_q = ensure_queue_dirs(ctx.run_dir, "exec")

    root_id = int(get_kv(conn, "root_node_id") or "1")
    best_id = int(get_kv(conn, "best_node_id") or str(root_id))
    valid_since_best = int(get_kv(conn, "valid_since_best") or "0")
    force_debug_next_iter = (get_kv(conn, "force_debug_next_iter") or "0") == "1"
    force_debug_reason = get_kv(conn, "force_debug_reason", default="") or ""
    identical_metric_streak = int(get_kv(conn, "identical_metric_streak") or "0")
    auth_retry_state = {
        "consecutive": int(get_kv(conn, "llm_auth_401_consecutive") or "0"),
        "total": int(get_kv(conn, "llm_auth_401_total") or "0"),
    }
    idea_history = _load_json_list(conn, "idea_history_json")
    root_repo = get_node(conn, root_id).repo_path
    start_iter = _next_iter_index(ctx.run_dir)
    if start_iter > 1:
        LOGGER.info("resume detected: start_iter=%s max_iters=%s", start_iter, max_iters)
    if start_iter > max_iters:
        LOGGER.info("done: existing iterations already reached max_iters=%s", max_iters)
        return

    # Persist profile count/default ready-keys for worker-side wait estimation.
    _ensure_llm_profile_state_keys(conn, cfg)

    task_prep_job_id, task_prep_summary = _ensure_task_preparation(
        conn=conn,
        llm_q=llm_q,
        ctx=ctx,
        cfg=cfg,
        task=task,
        root_repo=root_repo,
        auth_retry_state=auth_retry_state,
    )
    run_started_raw = get_kv(conn, "run_started_at")
    if run_started_raw is None:
        run_started_at = time.time()
        set_kv(conn, "run_started_at", f"{run_started_at:.6f}")
    else:
        try:
            run_started_at = float(run_started_raw)
        except Exception:
            run_started_at = time.time()
            set_kv(conn, "run_started_at", f"{run_started_at:.6f}")

    # Metric history for reward normalization (Eq 3)
    mh = MetricHistory()
    mh_min = get_kv(conn, "metric_history_min")
    mh_max = get_kv(conn, "metric_history_max")
    if mh_min:
        mh.m_min = float(mh_min)
    if mh_max:
        mh.m_max = float(mh_max)

    LOGGER.info(
        "start run_dir=%s max_iters=%s wallclock_sec=%s per_run_timeout_sec=%s metric=%s llm_profiles=%s llm_profile_selection_mode=%s",
        ctx.run_dir,
        max_iters,
        task.wallclock_sec,
        task.per_run_timeout_sec,
        task.metric_name,
        _llm_profiles_summary(cfg),
        _resolve_llm_profile_selection_mode(conn, cfg),
    )

    for it in range(start_iter, max_iters + 1):
        budget_left_before_iter = _remaining_budget_sec(run_started_at, task.wallclock_sec)
        if budget_left_before_iter <= 0:
            LOGGER.info("stop: wallclock budget exhausted at iter=%s", it)
            break
        LOGGER.info(
            "iter start iter=%s/%s budget_left_sec=%.1f best_node_id=%s valid_since_best=%s",
            it,
            max_iters,
            budget_left_before_iter,
            best_id,
            valid_since_best,
        )

        # UCT selection: descend from root until reaching an expandable node.
        parent_id, selection_path = _select_parent_for_expansion(
            conn,
            root_id=root_id,
            cfg=cfg,
            valid_since_best=valid_since_best,
        )
        parent = get_node(conn, parent_id)

        is_root = (parent_id == root_id)
        action = decide_action(parent, is_root=is_root, cfg=cfg.mcts)
        LOGGER.info(
            "iter selection iter=%s parent_id=%s action=%s path=%s",
            it,
            parent_id,
            action.value,
            selection_path,
        )
        forced_debug_this_iter = False
        forced_debug_error_summary = ""
        if force_debug_next_iter:
            action = ActionType.DEBUG
            forced_debug_this_iter = True
            forced_debug_error_summary = force_debug_reason or (
                "Identical metric repeated and experiment-design guard requested DEBUG."
            )
            force_debug_next_iter = False
            force_debug_reason = ""
            set_kv(conn, "force_debug_next_iter", "0")
            set_kv(conn, "force_debug_reason", "")
            LOGGER.warning(
                "guard forcing DEBUG due to identical metric streak iter=%s reason=%s",
                it,
                forced_debug_error_summary,
            )

        def _iteration_context() -> str:
            return _build_iteration_experiment_context(
                conn=conn,
                run_dir=ctx.run_dir,
                iter_index=it,
                parent_id=parent_id,
                action=action,
                selection_path=selection_path,
            )

        solution_lessons = get_recent_lessons(conn, kind="solution", k=cfg.memory.keep_recent_k)
        debug_lessons = get_recent_lessons(conn, kind="debug", k=cfg.memory.keep_recent_k)

        idea_job_id: str | None = None
        idea_text = ""
        module_plan_job_id: str | None = None
        module_plan_text = ""
        if action == ActionType.DRAFT:
            previous_ideas = "\n\n".join(f"[{idx+1}] {idea}" for idx, idea in enumerate(idea_history[-20:])) or "(none)"
            previous_ideas = previous_ideas[:12000]
            idea_mode = "initial" if not solution_lessons else "improve"
            idea_job_id, idea_res = _request_llm_with_auth_stop(
                llm_q=llm_q,
                run_dir=ctx.run_dir,
                cfg=cfg,
                conn=conn,
                iter_index=it,
                stage="idea_propose",
                auth_retry_state=auth_retry_state,
                job_type="IDEA_PROPOSE",
                llm_cwd=parent.repo_path,
                payload={
                    "mode": idea_mode,
                    "instruction": task.instruction,
                    "task_prep": task_prep_summary,
                    "previous_ideas": previous_ideas,
                    "lessons": lessons_to_text(solution_lessons),
                    "experiment_context": _iteration_context(),
                },
            )
            LOGGER.info(
                "iter idea iter=%s job_id=%s ok=%s",
                it,
                idea_job_id,
                bool(idea_res.get("ok", False)),
            )
            if bool(idea_res.get("ok", False)):
                idea_candidate = str(idea_res.get("text", "")).strip()
                if idea_candidate:
                    idea_text = idea_candidate[:4000]
                    if not idea_history or idea_history[-1] != idea_text:
                        idea_history.append(idea_text)
                    if len(idea_history) > 100:
                        idea_history = idea_history[-100:]
                    _save_json_list(conn, "idea_history_json", idea_history)
            if idea_text:
                module_plan_job_id, module_plan_res = _request_llm_with_auth_stop(
                    llm_q=llm_q,
                    run_dir=ctx.run_dir,
                    cfg=cfg,
                    conn=conn,
                    iter_index=it,
                    stage="modular_decompose",
                    auth_retry_state=auth_retry_state,
                    job_type="MODULAR_DECOMPOSE",
                    llm_cwd=parent.repo_path,
                    payload={
                        "instruction": task.instruction,
                        "task_prep": task_prep_summary,
                        "idea": idea_text,
                        "experiment_context": _iteration_context(),
                    },
                )
                LOGGER.info(
                    "iter module_plan iter=%s job_id=%s ok=%s",
                    it,
                    module_plan_job_id,
                    bool(module_plan_res.get("ok", False)),
                )
                if bool(module_plan_res.get("ok", False)):
                    module_plan_text = str(module_plan_res.get("text", "")).strip()[:8000]

        # Create child from parent snapshot so each iteration accumulates changes.
        child_repo = materialize_repo(ctx.run_dir, node_id=it, template_path=parent.repo_path)
        child_id = create_node(conn, parent_id=parent_id, action=action, status=NodeStatus.PENDING, repo_path=child_repo)
        update_node_status(conn, child_id, NodeStatus.PENDING)
        LOGGER.info("iter child iter=%s child_id=%s repo=%s", it, child_id, child_repo)

        llm_job_ids: list[str] = []
        if module_plan_job_id:
            llm_job_ids.append(module_plan_job_id)
        patch_text = ""
        patch_applied = False
        patch_apply_error: str | None = None

        if action == ActionType.DRAFT and module_plan_text:
            module_specs = _extract_module_specs(module_plan_text)
            if module_specs:
                for module_spec in module_specs:
                    repo_listing = _repo_file_listing(child_repo)
                    module_job_id, module_res = _request_llm_with_auth_stop(
                        llm_q=llm_q,
                        run_dir=ctx.run_dir,
                        cfg=cfg,
                        conn=conn,
                        iter_index=it,
                        stage="generate_patch_module",
                        auth_retry_state=auth_retry_state,
                        job_type="GENERATE_PATCH",
                        llm_cwd=child_repo,
                        payload={
                            "mode": "module",
                            "instruction": task.instruction,
                            "task_prep": task_prep_summary,
                            "file_list": repo_listing,
                            "lessons": lessons_to_text(solution_lessons + debug_lessons),
                            "idea": idea_text or "(none)",
                            "module_plan": module_plan_text or "(none)",
                            "module_spec": module_spec,
                            "error_summary": "",
                            "experiment_context": _iteration_context(),
                        },
                    )
                    LOGGER.info(
                        "iter patch_module iter=%s child_id=%s module_job_id=%s ok=%s",
                        it,
                        child_id,
                        module_job_id,
                        bool(module_res.get("ok", False)),
                    )
                    llm_job_ids.append(module_job_id)
                    applied, module_patch_text, module_patch_error = _attempt_apply_patch(
                        ctx=ctx,
                        conn=conn,
                        child_id=child_id,
                        child_repo=child_repo,
                        patch_job_id=module_job_id,
                        patch_res=module_res,
                    )
                    if module_patch_text:
                        patch_text = module_patch_text
                    patch_applied = patch_applied or applied
                    if module_patch_error:
                        patch_apply_error = module_patch_error
                        break

                if patch_apply_error is None:
                    repo_listing = _repo_file_listing(child_repo)
                    orchestrate_job_id, orchestrate_res = _request_llm_with_auth_stop(
                        llm_q=llm_q,
                        run_dir=ctx.run_dir,
                        cfg=cfg,
                        conn=conn,
                        iter_index=it,
                        stage="generate_patch_orchestrate",
                        auth_retry_state=auth_retry_state,
                        job_type="GENERATE_PATCH",
                        llm_cwd=child_repo,
                        payload={
                            "mode": "orchestrate",
                            "instruction": task.instruction,
                            "task_prep": task_prep_summary,
                            "file_list": repo_listing,
                            "lessons": lessons_to_text(solution_lessons + debug_lessons),
                            "idea": idea_text or "(none)",
                            "module_plan": module_plan_text or "(none)",
                            "module_spec": "(orchestration)",
                            "error_summary": "",
                            "experiment_context": _iteration_context(),
                        },
                    )
                    LOGGER.info(
                        "iter patch_orchestrate iter=%s child_id=%s job_id=%s ok=%s",
                        it,
                        child_id,
                        orchestrate_job_id,
                        bool(orchestrate_res.get("ok", False)),
                    )
                    llm_job_ids.append(orchestrate_job_id)
                    applied, orch_patch_text, orch_patch_error = _attempt_apply_patch(
                        ctx=ctx,
                        conn=conn,
                        child_id=child_id,
                        child_repo=child_repo,
                        patch_job_id=orchestrate_job_id,
                        patch_res=orchestrate_res,
                    )
                    if orch_patch_text:
                        patch_text = orch_patch_text
                    patch_applied = patch_applied or applied
                    if orch_patch_error:
                        patch_apply_error = orch_patch_error
            else:
                # If no module specs were parsed, fall back to draft patching.
                repo_listing = _repo_file_listing(child_repo)
                fallback_job_id, fallback_res = _request_llm_with_auth_stop(
                    llm_q=llm_q,
                    run_dir=ctx.run_dir,
                    cfg=cfg,
                    conn=conn,
                    iter_index=it,
                    stage="generate_patch_draft_fallback",
                    auth_retry_state=auth_retry_state,
                    job_type="GENERATE_PATCH",
                    llm_cwd=child_repo,
                    payload={
                        "mode": "draft",
                        "instruction": task.instruction,
                        "task_prep": task_prep_summary,
                        "file_list": repo_listing,
                        "lessons": lessons_to_text(solution_lessons + debug_lessons),
                        "idea": idea_text or "(none)",
                        "module_plan": module_plan_text or "(none)",
                        "module_spec": "(none)",
                        "error_summary": "",
                        "experiment_context": _iteration_context(),
                    },
                )
                LOGGER.info(
                    "iter patch_fallback iter=%s child_id=%s job_id=%s ok=%s",
                    it,
                    child_id,
                    fallback_job_id,
                    bool(fallback_res.get("ok", False)),
                )
                llm_job_ids.append(fallback_job_id)
                applied, fallback_patch_text, fallback_patch_error = _attempt_apply_patch(
                    ctx=ctx,
                    conn=conn,
                    child_id=child_id,
                    child_repo=child_repo,
                    patch_job_id=fallback_job_id,
                    patch_res=fallback_res,
                )
                if fallback_patch_text:
                    patch_text = fallback_patch_text
                patch_applied = patch_applied or applied
                if fallback_patch_error:
                    patch_apply_error = fallback_patch_error
        else:
            patch_mode = "debug" if action == ActionType.DEBUG else "improve"
            repo_listing = _repo_file_listing(child_repo)
            initial_patch_job_id, initial_patch_res = _request_llm_with_auth_stop(
                llm_q=llm_q,
                run_dir=ctx.run_dir,
                cfg=cfg,
                conn=conn,
                iter_index=it,
                stage=f"generate_patch_{patch_mode}",
                auth_retry_state=auth_retry_state,
                job_type="GENERATE_PATCH",
                llm_cwd=child_repo,
                payload={
                    "mode": patch_mode,
                    "instruction": task.instruction,
                    "task_prep": task_prep_summary,
                    "file_list": repo_listing,
                    "lessons": lessons_to_text(solution_lessons + debug_lessons),
                    "idea": idea_text or "(none)",
                    "module_plan": module_plan_text or "(none)",
                    "module_spec": "(none)",
                    "error_summary": forced_debug_error_summary if patch_mode == "debug" else "",
                    "experiment_context": _iteration_context(),
                },
            )
            LOGGER.info(
                "iter patch_initial iter=%s child_id=%s mode=%s job_id=%s ok=%s",
                it,
                child_id,
                patch_mode,
                initial_patch_job_id,
                bool(initial_patch_res.get("ok", False)),
            )
            llm_job_ids.append(initial_patch_job_id)
            applied, initial_patch_text, initial_patch_error = _attempt_apply_patch(
                ctx=ctx,
                conn=conn,
                child_id=child_id,
                child_repo=child_repo,
                patch_job_id=initial_patch_job_id,
                patch_res=initial_patch_res,
            )
            if initial_patch_text:
                patch_text = initial_patch_text
            patch_applied = patch_applied or applied
            if initial_patch_error:
                patch_apply_error = initial_patch_error

        exec_job_id = ""
        budget_exhausted_during_iter = False
        if patch_apply_error is None and not patch_applied:
            patch_apply_error = "No patch was generated/applied; skip execution of unchanged repository."
        if patch_apply_error:
            exec_res = ExecResult(
                ok=False,
                exit_code=-1,
                exec_time_sec=0.0,
                metric_value=None,
                metric_found=False,
                stdout_path="",
                stderr_path="",
                error_summary=f"Patch apply failed: {patch_apply_error}",
            )
            exec_res_raw = {
                "ok": False,
                "exit_code": -1,
                "exec_time_sec": 0.0,
                "metric_value": None,
                "metric_found": False,
                "stdout_path": "",
                "stderr_path": "",
                "error_summary": exec_res.error_summary,
            }
        else:
            remaining_for_exec = _remaining_budget_sec(run_started_at, task.wallclock_sec)
            if remaining_for_exec <= 0:
                budget_exhausted_during_iter = True
                exec_res = ExecResult(
                    ok=False,
                    exit_code=-1,
                    exec_time_sec=0.0,
                    metric_value=None,
                    metric_found=False,
                    stdout_path="",
                    stderr_path="",
                    error_summary="Wallclock budget exhausted before execution.",
                )
                exec_res_raw = {
                    "ok": False,
                    "exit_code": -1,
                    "exec_time_sec": 0.0,
                    "metric_value": None,
                    "metric_found": False,
                    "stdout_path": "",
                    "stderr_path": "",
                    "error_summary": exec_res.error_summary,
                }
            else:
                exec_timeout = int(min(float(task.per_run_timeout_sec), max(1.0, remaining_for_exec)))
                exec_job_id, exec_res_raw, exec_res = _run_exec_once(
                    exec_q,
                    task,
                    child_repo,
                    timeout_sec=exec_timeout,
                    iter_index=it,
                    stage="exec_run",
                )

        debug_attempt_logs: list[dict[str, Any]] = []
        for debug_idx in range(1, int(cfg.mcts.max_debug_attempts) + 1):
            if budget_exhausted_during_iter:
                break
            if exec_res.exit_code == 0 and exec_res.metric_found:
                break

            remaining_for_debug = _remaining_budget_sec(run_started_at, task.wallclock_sec)
            if remaining_for_debug <= 0:
                debug_attempt_logs.append(
                    {"attempt": debug_idx, "applied": False, "reason": "wallclock_budget_exhausted"}
                )
                budget_exhausted_during_iter = True
                break

            debug_lessons = get_recent_lessons(conn, kind="debug", k=cfg.memory.keep_recent_k)
            error_summary = exec_res.error_summary or f"exit_code={exec_res.exit_code}, metric_found={exec_res.metric_found}"
            debug_job_id, debug_res = _request_llm_with_auth_stop(
                llm_q=llm_q,
                run_dir=ctx.run_dir,
                cfg=cfg,
                conn=conn,
                iter_index=it,
                stage="generate_patch_debug_retry",
                auth_retry_state=auth_retry_state,
                job_type="GENERATE_PATCH",
                llm_cwd=child_repo,
                payload={
                    "mode": "debug",
                    "instruction": task.instruction,
                    "task_prep": task_prep_summary,
                    "file_list": _repo_file_listing(child_repo),
                    "lessons": lessons_to_text(debug_lessons),
                    "idea": idea_text or "(none)",
                    "module_plan": module_plan_text or "(none)",
                    "module_spec": "(debug-fix)",
                    "error_summary": error_summary,
                    "experiment_context": _iteration_context(),
                },
            )
            LOGGER.info(
                "iter debug_patch iter=%s child_id=%s attempt=%s job_id=%s ok=%s",
                it,
                child_id,
                debug_idx,
                debug_job_id,
                bool(debug_res.get("ok", False)),
            )
            llm_job_ids.append(debug_job_id)
            debug_patch = extract_unified_diff(str(debug_res.get("text", ""))) if bool(debug_res.get("ok", False)) else ""
            write_text(ctx.run_dir, f"artifacts/patches/{debug_job_id}.diff", debug_patch)
            if not debug_patch:
                debug_attempt_logs.append(
                    {"attempt": debug_idx, "job_id": debug_job_id, "applied": False, "reason": "empty_or_failed_patch"}
                )
                break

            try:
                apply_unified_diff(child_repo, debug_patch)
                patch_applied = True
                patch_text = debug_patch
            except Exception as e:
                exec_res = ExecResult(
                    ok=False,
                    exit_code=-1,
                    exec_time_sec=0.0,
                    metric_value=None,
                    metric_found=False,
                    stdout_path="",
                    stderr_path="",
                    error_summary=f"Patch apply failed during debug: {summarize_error(repr(e))}",
                )
                exec_res_raw = {
                    "ok": False,
                    "exit_code": -1,
                    "exec_time_sec": 0.0,
                    "metric_value": None,
                    "metric_found": False,
                    "stdout_path": "",
                    "stderr_path": "",
                    "error_summary": exec_res.error_summary,
                }
                debug_attempt_logs.append(
                    {"attempt": debug_idx, "job_id": debug_job_id, "applied": False, "reason": exec_res.error_summary}
                )
                continue

            debug_timeout = int(min(float(task.per_run_timeout_sec), max(1.0, remaining_for_debug)))
            exec_job_id, exec_res_raw, exec_res = _run_exec_once(
                exec_q,
                task,
                child_repo,
                timeout_sec=debug_timeout,
                iter_index=it,
                stage=f"exec_debug_retry_{debug_idx}",
            )
            debug_attempt_logs.append(
                {
                    "attempt": debug_idx,
                    "job_id": debug_job_id,
                    "applied": True,
                    "exit_code": exec_res.exit_code,
                    "metric_found": exec_res.metric_found,
                }
            )

        # Execute-and-review: validate metric correctness with an LLM reviewer.
        runfile_text = _safe_read(str(Path(child_repo) / "runfile.py"))
        stdout_text = _safe_read(exec_res.stdout_path)
        stderr_text = _safe_read(exec_res.stderr_path)
        term_out = (stdout_text + "\n\n[STDERR]\n" + stderr_text).strip()
        review_job_id, review_res = _request_llm_with_auth_stop(
            llm_q=llm_q,
            run_dir=ctx.run_dir,
            cfg=cfg,
            conn=conn,
            iter_index=it,
            stage="exec_review",
            auth_retry_state=auth_retry_state,
            job_type="EXEC_REVIEW",
            llm_cwd=child_repo,
            payload={
                "instruction": task.instruction,
                "code": runfile_text[:12000],
                "term_out": term_out[:12000],
                "experiment_context": _iteration_context(),
            },
        )
        LOGGER.info(
            "iter exec_review iter=%s child_id=%s job_id=%s ok=%s",
            it,
            child_id,
            review_job_id,
            bool(review_res.get("ok", False)),
        )
        llm_job_ids.append(review_job_id)
        review_obj = _parse_json_object(str(review_res.get("text", ""))) if bool(review_res.get("ok", False)) else {}
        if review_obj:
            exec_res_raw["review"] = review_obj
            valid_metric = review_obj.get("valid_metric")
            if isinstance(valid_metric, bool) and not valid_metric:
                exec_res = ExecResult(
                    ok=exec_res.ok,
                    exit_code=exec_res.exit_code,
                    exec_time_sec=exec_res.exec_time_sec,
                    metric_value=None,
                    metric_found=False,
                    stdout_path=exec_res.stdout_path,
                    stderr_path=exec_res.stderr_path,
                    error_summary=str(review_obj.get("summary", "Metric reviewer marked output invalid.")),
                )
            elif exec_res.metric_value is None and isinstance(review_obj.get("metric"), (int, float)):
                exec_res = ExecResult(
                    ok=exec_res.ok,
                    exit_code=exec_res.exit_code,
                    exec_time_sec=exec_res.exec_time_sec,
                    metric_value=float(review_obj["metric"]),
                    metric_found=True,
                    stdout_path=exec_res.stdout_path,
                    stderr_path=exec_res.stderr_path,
                    error_summary=exec_res.error_summary,
                )

        # Compute reward if metric valid
        reward = 0.0
        if exec_res.metric_found and exec_res.metric_value is not None:
            raw_metric = float(exec_res.metric_value)
            identical_metric_streak, guard_triggered, guard_reason = _update_identical_metric_guard(
                conn=conn,
                run_dir=ctx.run_dir,
                iter_index=it,
                metric_value=raw_metric,
                threshold=int(cfg.mcts.identical_metric_streak_to_force_debug),
            )
            if guard_triggered:
                force_debug_next_iter = True
                force_debug_reason = guard_reason
                LOGGER.warning(
                    "guard identical metric streak reached threshold iter=%s streak=%s metric=%s",
                    it,
                    identical_metric_streak,
                    raw_metric,
                )
            objective_metric = _objective_metric(raw_metric, task.higher_is_better)
            mh.update(objective_metric)
            set_kv(conn, "metric_history_min", "" if mh.m_min is None else str(mh.m_min))
            set_kv(conn, "metric_history_max", "" if mh.m_max is None else str(mh.m_max))

            norm = mh.normalize(objective_metric)
            reward = efficiency_guided_reward(norm, exec_res.exec_time_sec, float(task.per_run_timeout_sec), cfg.mcts.w_time_penalty)

            update_node_result(conn, child_id, metric=raw_metric, runtime_sec=exec_res.exec_time_sec, reward=reward, status=NodeStatus.DONE)
            _backup_path(conn, selection_path + [child_id], reward)
            best_node_before = get_node(conn, best_id)
            solution_lesson = distill_solution_lesson(
                best_summary=f"node={best_node_before.node_id} metric={best_node_before.metric} runtime={best_node_before.runtime_sec}",
                new_summary=f"action={action.value} metric={raw_metric} runtime={exec_res.exec_time_sec:.3f}s",
                diff_summary=diff_summary(parent.repo_path, child_repo, max_lines=250),
                exec_result=exec_res,
                source_node_id=child_id,
                llm_cfg=cfg.llm,
                run_dir=ctx.run_dir,
                iter_index=it,
            )
            _save_lesson_if_new(
                conn,
                solution_lesson,
                cfg=cfg,
                run_dir=ctx.run_dir,
                iter_index=it,
            )

            # Update best (simple)
            best_metric = best_node_before.metric
            improved = _is_improved(raw_metric, best_metric, task.higher_is_better)
            if improved:
                best_id = child_id
                set_kv(conn, "best_node_id", str(best_id))
                valid_since_best = 0
            else:
                valid_since_best += 1
        else:
            # Mark buggy
            identical_metric_streak = 0
            set_kv(conn, "identical_metric_streak", "0")
            failure_reward = 0.0
            update_node_result(
                conn,
                child_id,
                metric=None,
                runtime_sec=exec_res.exec_time_sec,
                reward=failure_reward,
                status=NodeStatus.BUGGY,
            )
            _backup_path(conn, selection_path + [child_id], failure_reward)
            debug_lesson = distill_debug_lesson(
                error_summary=exec_res.error_summary or "Execution failed without stderr.",
                fix_diff=patch_text or diff_summary(parent.repo_path, child_repo, max_lines=200),
                source_node_id=child_id,
                llm_cfg=cfg.llm,
                debug_outcome="failed",
                run_dir=ctx.run_dir,
                iter_index=it,
            )
            _save_lesson_if_new(
                conn,
                debug_lesson,
                cfg=cfg,
                run_dir=ctx.run_dir,
                iter_index=it,
            )

        set_kv(conn, "valid_since_best", str(valid_since_best))

        write_json(
            ctx.run_dir,
            f"tree/iter_{it:04d}.json",
            {
                "parent_id": parent_id,
                "child_id": child_id,
                "action": action.value,
                "task_prep_job_id": task_prep_job_id,
                "idea_job_id": idea_job_id,
                "idea": idea_text,
                "module_plan_job_id": module_plan_job_id,
                "module_plan": module_plan_text,
                "llm_job_ids": llm_job_ids,
                "exec_job_id": exec_job_id,
                "selection_path": selection_path,
                "patch_applied": patch_applied,
                "backup_reward": reward if exec_res.metric_found and exec_res.metric_value is not None else 0.0,
                "valid_since_best": valid_since_best,
                "identical_metric_streak": identical_metric_streak,
                "forced_debug_this_iter": forced_debug_this_iter,
                "force_debug_next_iter": force_debug_next_iter,
                "force_debug_reason": force_debug_reason,
                "budget_left_before_iter_sec": budget_left_before_iter,
                "budget_left_after_iter_sec": _remaining_budget_sec(run_started_at, task.wallclock_sec),
                "budget_exhausted": budget_exhausted_during_iter,
                "debug_attempts": debug_attempt_logs,
                "exec": exec_res_raw,
            },
        )
        LOGGER.info(
            "iter done iter=%s child_id=%s status=%s metric_found=%s metric=%s best_node_id=%s valid_since_best=%s",
            it,
            child_id,
            get_node(conn, child_id).status.value,
            exec_res.metric_found,
            exec_res.metric_value,
            best_id,
            valid_since_best,
        )
        time.sleep(0.2)

    # Final
    set_kv(conn, "best_node_id", str(best_id))
    LOGGER.info("done. best_node_id=%s", best_id)
