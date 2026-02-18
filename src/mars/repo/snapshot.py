from __future__ import annotations
import shutil
from pathlib import Path


def materialize_repo(run_dir: str | Path, node_id: int, template_path: str | Path) -> str:
    """Create a node repo directory by copying a source repo snapshot.
    This keeps the agent repository-level, like the paper's Eq (2).
    """
    run_dir = Path(run_dir)
    dst = run_dir / "repos" / f"node_{node_id:06d}"
    if dst.exists():
        # If resumed from an existing run, avoid reusing the same writable snapshot path.
        suffix = 1
        while True:
            candidate = run_dir / "repos" / f"node_{node_id:06d}_{suffix:02d}"
            if not candidate.exists():
                dst = candidate
                break
            suffix += 1
    src = Path(template_path)
    # Preserve symlinks so large immutable assets (e.g., parquet datasets) can stay linked.
    shutil.copytree(src, dst, dirs_exist_ok=False, symlinks=True)
    return str(dst)


def write_file(repo_path: str | Path, relpath: str, content: str) -> None:
    p = Path(repo_path) / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
