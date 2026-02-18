from __future__ import annotations
import re
import subprocess
from pathlib import Path


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _looks_like_unified_diff(text: str) -> bool:
    lines = text.splitlines()
    has_diff_header = any(line.startswith("diff --git ") for line in lines)
    has_file_headers = any(line.startswith("--- ") for line in lines) and any(line.startswith("+++ ") for line in lines)
    has_hunk = any(line.startswith("@@ ") or line.startswith("@@") for line in lines)
    return (has_diff_header or has_file_headers) and has_hunk


def _strip_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[^\n]*\n", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n```$", "", s)
    return s.strip()


def extract_unified_diff(llm_text: str) -> str:
    """Extract the first git-apply-compatible unified diff from LLM text.

    The model may wrap output in markdown fences; this helper tolerates that.
    Returns an empty string when no plausible unified diff is found.
    """
    if not llm_text:
        return ""

    fence_blocks = re.findall(r"```(?:diff|patch)?\s*\n(.*?)```", llm_text, flags=re.IGNORECASE | re.DOTALL)
    candidates = fence_blocks + [llm_text]
    for candidate in candidates:
        diff_text = _strip_fence(candidate)
        if _looks_like_unified_diff(diff_text):
            if not diff_text.endswith("\n"):
                return diff_text + "\n"
            return diff_text
    return ""


def _normalize_hunk_headers(diff_text: str) -> str:
    """Recompute unified-diff hunk counts from body lines.

    Some model-generated patches contain incorrect hunk counts (for example
    `@@ -0,0 +1,127 @@` with only 126 added lines). Git rejects those patches
    as malformed; this normalizer fixes only header counts, preserving content.
    """
    lines = diff_text.splitlines()
    if not lines:
        return diff_text

    normalized: list[str] = []
    i = 0
    changed = False
    while i < len(lines):
        header = lines[i]
        m = _HUNK_HEADER_RE.match(header)
        if m is None:
            normalized.append(header)
            i += 1
            continue

        old_start = m.group(1)
        new_start = m.group(3)
        tail = m.group(5)

        old_count = 0
        new_count = 0
        j = i + 1
        while j < len(lines):
            body_line = lines[j]
            if _HUNK_HEADER_RE.match(body_line) or body_line.startswith("diff --git "):
                break
            if (
                body_line.startswith("--- ")
                and j + 1 < len(lines)
                and lines[j + 1].startswith("+++ ")
            ):
                break
            if body_line.startswith("\\ No newline at end of file"):
                j += 1
                continue
            if body_line.startswith("+"):
                new_count += 1
                j += 1
                continue
            if body_line.startswith("-"):
                old_count += 1
                j += 1
                continue
            if body_line.startswith(" "):
                old_count += 1
                new_count += 1
                j += 1
                continue
            break

        normalized_header = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{tail}"
        if normalized_header != header:
            changed = True
        normalized.append(normalized_header)
        normalized.extend(lines[i + 1 : j])
        i = j

    out = "\n".join(normalized)
    if diff_text.endswith("\n"):
        out = out + "\n"
    if changed:
        return out
    return diff_text


def apply_unified_diff(repo_path: str | Path, diff_text: str) -> None:
    """Apply a unified diff atomically.
    Prefer `git apply`, and fall back to `patch -p1` when git is unavailable.
    """
    repo_path = str(repo_path)
    diff_to_apply = _normalize_hunk_headers(diff_text)
    git_missing = False
    try:
        p = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input=diff_to_apply,
            text=True,
            cwd=repo_path,
            capture_output=True,
        )
        if p.returncode == 0:
            return
        git_err = f"git apply failed\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    except FileNotFoundError:
        git_err = "git apply failed: git executable not found"
        git_missing = True

    # Do not fall back when git exists but rejects the diff.
    # Running `patch` after a failed `git apply` can leave partially applied changes.
    if not git_missing:
        raise RuntimeError(git_err)

    try:
        dry = subprocess.run(
            ["patch", "-p1", "--forward", "--silent", "--dry-run"],
            input=diff_to_apply,
            text=True,
            cwd=repo_path,
            capture_output=True,
        )
        if dry.returncode != 0:
            dry_err = f"patch dry-run failed\nSTDOUT:\n{dry.stdout}\nSTDERR:\n{dry.stderr}"
            raise RuntimeError(f"{git_err}\n\n{dry_err}")

        p2 = subprocess.run(
            ["patch", "-p1", "--forward", "--silent"],
            input=diff_to_apply,
            text=True,
            cwd=repo_path,
            capture_output=True,
        )
        if p2.returncode == 0:
            return
        patch_err = f"patch failed\nSTDOUT:\n{p2.stdout}\nSTDERR:\n{p2.stderr}"
    except FileNotFoundError:
        patch_err = "patch failed: patch executable not found"

    raise RuntimeError(f"{git_err}\n\n{patch_err}")
