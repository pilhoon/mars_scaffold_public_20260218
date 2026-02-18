from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from mars.repo.diff import diff_summary


class DiffSummaryTests(unittest.TestCase):
    def test_diff_summary_handles_non_utf8_stdout_bytes(self) -> None:
        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            self.assertTrue(bool(kwargs.get("capture_output")))
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout=b"diff --git a/a b/a\n+ok\xfc\n",
                stderr=b"",
            )

        with patch("mars.repo.diff.subprocess.run", side_effect=fake_run):
            summary = diff_summary("/tmp/a", "/tmp/b")

        self.assertIn("diff --git a/a b/a", summary)
        self.assertIn("ok", summary)

    def test_diff_summary_truncates_to_max_lines(self) -> None:
        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout=b"line1\nline2\nline3\n",
                stderr=b"",
            )

        with patch("mars.repo.diff.subprocess.run", side_effect=fake_run):
            summary = diff_summary("/tmp/a", "/tmp/b", max_lines=2)

        self.assertEqual(summary, "line1\nline2\n... (truncated) ...")


if __name__ == "__main__":
    unittest.main()
