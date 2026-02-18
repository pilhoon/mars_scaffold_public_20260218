from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict


def ensure_run_dirs(run_dir: str | Path) -> None:
    p = Path(run_dir)
    (p / "state").mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir(parents=True, exist_ok=True)
    (p / "artifacts").mkdir(parents=True, exist_ok=True)
    (p / "repos").mkdir(parents=True, exist_ok=True)
    (p / "tree").mkdir(parents=True, exist_ok=True)


def write_text(run_dir: str | Path, relpath: str, text: str) -> str:
    p = Path(run_dir) / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)


def read_text(run_dir: str | Path, relpath: str) -> str:
    p = Path(run_dir) / relpath
    return p.read_text(encoding="utf-8")


def write_json(run_dir: str | Path, relpath: str, obj: Dict[str, Any]) -> str:
    p = Path(run_dir) / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def read_json(run_dir: str | Path, relpath: str) -> Dict[str, Any]:
    p = Path(run_dir) / relpath
    return json.loads(p.read_text(encoding="utf-8"))
