from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from typing import Any

from mars.types import TaskSpec


DATA_FILE_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".txt"}
DATA_HINT_TOKENS = ("data", "dataset", "train", "valid", "val", "test", "sample")


def _iter_repo_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        files.append(p)
    return files


def _score_data_candidate(relpath: str, suffix: str) -> int:
    lower = relpath.lower()
    score = 0
    if suffix in DATA_FILE_EXTS:
        score += 2
    if any(tok in lower for tok in DATA_HINT_TOKENS):
        score += 1
    return score


def _safe_count_lines(path: Path, hard_limit: int = 200_000) -> int | None:
    n = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                n += 1
                if n >= hard_limit:
                    return None
        return n
    except Exception:
        return None


def _profile_csv_like(path: Path, root: Path, delimiter: str) -> dict[str, Any]:
    cols: list[str] = []
    sample_rows: list[list[str]] = []
    note = ""
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, [])
            cols = [str(x).strip() for x in header][:60]
            for _, row in zip(range(3), reader):
                sample_rows.append([str(x)[:80] for x in row[:12]])
    except Exception as e:
        note = f"failed_to_parse: {type(e).__name__}"

    line_count = _safe_count_lines(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "type": "tsv" if delimiter == "\t" else "csv",
        "size_bytes": path.stat().st_size,
        "line_count": line_count,
        "columns": cols,
        "sample_rows": sample_rows,
        "note": note,
    }


def _profile_jsonl(path: Path, root: Path) -> dict[str, Any]:
    keys: set[str] = set()
    samples: list[dict[str, Any]] = []
    note = ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _, line in zip(range(5), f):
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if isinstance(obj, dict):
                    keys.update(str(k) for k in obj.keys())
                    samples.append({str(k): obj[k] for k in list(obj.keys())[:8]})
                else:
                    samples.append({"_value": obj})
    except Exception as e:
        note = f"failed_to_parse: {type(e).__name__}"

    line_count = _safe_count_lines(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "type": "jsonl",
        "size_bytes": path.stat().st_size,
        "line_count": line_count,
        "keys": sorted(keys)[:80],
        "sample_rows": samples,
        "note": note,
    }


def _profile_json(path: Path, root: Path) -> dict[str, Any]:
    note = ""
    top_type = "unknown"
    keys: list[str] = []
    sample: Any = None
    try:
        if path.stat().st_size <= 1_000_000:
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, dict):
                top_type = "dict"
                keys = [str(k) for k in list(obj.keys())[:80]]
                sample = {str(k): obj[k] for k in list(obj.keys())[:8]}
            elif isinstance(obj, list):
                top_type = "list"
                sample = obj[:2]
            else:
                top_type = type(obj).__name__
                sample = obj
        else:
            note = "skipped_parse_large_file"
    except Exception as e:
        note = f"failed_to_parse: {type(e).__name__}"

    return {
        "path": path.relative_to(root).as_posix(),
        "type": "json",
        "size_bytes": path.stat().st_size,
        "top_type": top_type,
        "keys": keys,
        "sample": sample,
        "note": note,
    }


def _profile_generic(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "type": path.suffix.lower().lstrip(".") or "file",
        "size_bytes": path.stat().st_size,
        "note": "metadata_only",
    }


def _profile_data_file(path: Path, root: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _profile_csv_like(path, root, delimiter=",")
    if suffix == ".tsv":
        return _profile_csv_like(path, root, delimiter="\t")
    if suffix == ".jsonl":
        return _profile_jsonl(path, root)
    if suffix == ".json":
        return _profile_json(path, root)
    return _profile_generic(path, root)


def _entrypoint_script(entrypoint: str) -> str:
    try:
        tokens = shlex.split(entrypoint or "")
    except Exception:
        tokens = (entrypoint or "").split()
    for tok in tokens:
        if tok.endswith(".py"):
            return tok
    return "runfile.py"


def collect_task_preparation_context(task: TaskSpec, repo_path: str | Path, max_data_files: int = 12) -> dict[str, Any]:
    root = Path(repo_path)
    files = _iter_repo_files(root)
    rels = [p.relative_to(root).as_posix() for p in files]

    py_files = [r for r in rels if r.endswith(".py")]
    data_candidates = [
        (p, _score_data_candidate(p.relative_to(root).as_posix(), p.suffix.lower()))
        for p in files
    ]
    data_candidates = [x for x in data_candidates if x[1] > 0]
    data_candidates.sort(
        key=lambda item: (
            -item[1],
            item[0].suffix.lower() not in DATA_FILE_EXTS,
            item[0].stat().st_size,
        )
    )
    selected_data = [p for p, _ in data_candidates[:max_data_files]]

    split_signals = {
        "train": [],
        "valid": [],
        "test": [],
    }
    for rel in rels:
        lower = rel.lower()
        if "train" in lower:
            split_signals["train"].append(rel)
        if "valid" in lower or "/val" in lower or "_val" in lower:
            split_signals["valid"].append(rel)
        if "test" in lower:
            split_signals["test"].append(rel)
    for key in split_signals:
        split_signals[key] = split_signals[key][:20]

    script_rel = _entrypoint_script(task.entrypoint)
    script_path = root / script_rel
    entry_preview = ""
    if script_path.exists():
        try:
            entry_preview = "\n".join(
                script_path.read_text(encoding="utf-8", errors="replace").splitlines()[:120]
            )[:7000]
        except Exception:
            entry_preview = ""

    ctx: dict[str, Any] = {
        "task": {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "metric_name": task.metric_name,
            "higher_is_better": task.higher_is_better,
            "wallclock_sec": task.wallclock_sec,
            "per_run_timeout_sec": task.per_run_timeout_sec,
            "entrypoint": task.entrypoint,
            "final_metric_regex": task.final_metric_regex,
        },
        "repo": {
            "path": str(root),
            "file_count": len(rels),
            "python_file_count": len(py_files),
            "top_python_files": py_files[:60],
        },
        "data_profiles": [_profile_data_file(p, root) for p in selected_data],
        "split_signals": split_signals,
        "entrypoint": {
            "script_relpath": script_rel,
            "exists": script_path.exists(),
            "preview": entry_preview,
        },
    }
    return ctx


def task_prep_context_text(ctx: dict[str, Any], max_chars: int = 12000) -> str:
    task = ctx.get("task", {}) or {}
    repo = ctx.get("repo", {}) or {}
    profiles = ctx.get("data_profiles", []) or []
    split = ctx.get("split_signals", {}) or {}
    entry = ctx.get("entrypoint", {}) or {}

    lines: list[str] = []
    lines.append("== Task Signals ==")
    lines.append(f"task_id: {task.get('task_id', '')}")
    lines.append(f"metric_name: {task.get('metric_name', '')}")
    lines.append(f"higher_is_better: {task.get('higher_is_better', '')}")
    lines.append(f"entrypoint: {task.get('entrypoint', '')}")
    lines.append(f"final_metric_regex: {task.get('final_metric_regex', '')}")
    lines.append(f"wallclock_sec: {task.get('wallclock_sec', '')}")
    lines.append(f"per_run_timeout_sec: {task.get('per_run_timeout_sec', '')}")

    lines.append("")
    lines.append("== Repo Metadata ==")
    lines.append(f"file_count: {repo.get('file_count', 0)}")
    lines.append(f"python_file_count: {repo.get('python_file_count', 0)}")
    top_py = repo.get("top_python_files", []) or []
    if top_py:
        lines.append("top_python_files:")
        lines.extend(f"- {x}" for x in top_py[:20])

    lines.append("")
    lines.append("== Data Profiles ==")
    if not profiles:
        lines.append("(none)")
    else:
        for p in profiles:
            lines.append(f"- path: {p.get('path')} type: {p.get('type')} size_bytes: {p.get('size_bytes')}")
            if p.get("columns"):
                lines.append(f"  columns: {', '.join(str(c) for c in (p.get('columns') or [])[:20])}")
            if p.get("keys"):
                lines.append(f"  keys: {', '.join(str(k) for k in (p.get('keys') or [])[:20])}")
            if p.get("line_count") is not None:
                lines.append(f"  line_count: {p.get('line_count')}")
            if p.get("note"):
                lines.append(f"  note: {p.get('note')}")

    lines.append("")
    lines.append("== Split Signals ==")
    lines.append(f"train_matches: {len(split.get('train', []) or [])}")
    lines.append(f"valid_matches: {len(split.get('valid', []) or [])}")
    lines.append(f"test_matches: {len(split.get('test', []) or [])}")

    lines.append("")
    lines.append("== Entrypoint Preview ==")
    lines.append(f"script_relpath: {entry.get('script_relpath')}")
    lines.append(f"exists: {entry.get('exists')}")
    preview = (entry.get("preview") or "").strip()
    if preview:
        lines.append(preview[:4000])

    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        return out[:max_chars]
    return out
