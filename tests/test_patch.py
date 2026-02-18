from __future__ import annotations

import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mars.repo.patch import _normalize_hunk_headers, apply_unified_diff


class PatchApplyTests(unittest.TestCase):
    def test_normalize_hunk_header_counts(self) -> None:
        diff = (
            "diff --git a/a.txt b/a.txt\n"
            "new file mode 100644\n"
            "index 0000000..1111111\n"
            "--- /dev/null\n"
            "+++ b/a.txt\n"
            "@@ -0,0 +1,3 @@\n"
            "+a\n"
            "+b\n"
        )
        normalized = _normalize_hunk_headers(diff)
        self.assertIn("@@ -0,0 +1,2 @@", normalized)

    def test_apply_normalizes_hunk_before_git_apply(self) -> None:
        seen_inputs: list[str] = []

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if kwargs.get("input"):
                seen_inputs.append(str(kwargs["input"]))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        malformed = (
            "diff --git a/a.txt b/a.txt\n"
            "new file mode 100644\n"
            "index 0000000..1111111\n"
            "--- /dev/null\n"
            "+++ b/a.txt\n"
            "@@ -0,0 +1,3 @@\n"
            "+a\n"
            "+b\n"
        )

        with tempfile.TemporaryDirectory() as td:
            with patch("mars.repo.patch.subprocess.run", side_effect=fake_run):
                apply_unified_diff(td, malformed)

        self.assertEqual(len(seen_inputs), 1)
        self.assertIn("@@ -0,0 +1,2 @@", seen_inputs[0])

    def test_no_patch_fallback_when_git_apply_rejects_diff(self) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            if cmd and cmd[0] == "git":
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="invalid patch")
            raise AssertionError("patch fallback should not run when git exists")

        with tempfile.TemporaryDirectory() as td:
            with patch("mars.repo.patch.subprocess.run", side_effect=fake_run):
                with self.assertRaises(RuntimeError) as cm:
                    apply_unified_diff(td, "diff --git a/a.txt b/a.txt\n@@ -1 +1 @@\n-a\n+b\n")

        self.assertEqual(len(calls), 1)
        self.assertIn("git apply failed", str(cm.exception))

    def test_patch_fallback_runs_only_when_git_missing(self) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            if cmd and cmd[0] == "git":
                raise FileNotFoundError("git not found")
            if "--dry-run" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as td:
            with patch("mars.repo.patch.subprocess.run", side_effect=fake_run):
                apply_unified_diff(td, "diff --git a/a.txt b/a.txt\n@@ -1 +1 @@\n-a\n+b\n")

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0], "git")
        self.assertEqual(calls[1][0], "patch")
        self.assertIn("--dry-run", calls[1])
        self.assertEqual(calls[2][0], "patch")
        self.assertNotIn("--dry-run", calls[2])

    def test_patch_apply_stops_when_dry_run_fails(self) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            if cmd and cmd[0] == "git":
                raise FileNotFoundError("git not found")
            if "--dry-run" in cmd:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="dry-run fail")
            raise AssertionError("actual patch apply must not run after dry-run failure")

        with tempfile.TemporaryDirectory() as td:
            with patch("mars.repo.patch.subprocess.run", side_effect=fake_run):
                with self.assertRaises(RuntimeError) as cm:
                    apply_unified_diff(td, "diff --git a/a.txt b/a.txt\n@@ -1 +1 @@\n-a\n+b\n")

        self.assertEqual(len(calls), 2)
        self.assertIn("patch dry-run failed", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
