from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mars.config import LLMConfig
from mars.llm.client import _build_env, call_llm


class LLMClientEnvTests(unittest.TestCase):
    def test_codex_respects_existing_codex_home(self) -> None:
        cfg = LLMConfig(cli="codex", args=[], mode="stdin", timeout_sec=10)
        with tempfile.TemporaryDirectory() as td:
            custom_home = str(Path(td) / "custom_codex_home")
            with patch.dict(os.environ, {"CODEX_HOME": custom_home}, clear=True):
                env = _build_env(cfg)
        self.assertEqual(env.get("CODEX_HOME"), custom_home)

    def test_codex_defaults_to_user_home_codex_dir(self) -> None:
        cfg = LLMConfig(cli="codex", args=[], mode="stdin", timeout_sec=10)
        with tempfile.TemporaryDirectory() as td:
            fake_home = Path(td) / "fake_home"
            with patch.dict(os.environ, {}, clear=True), patch("mars.llm.client.Path.home", return_value=fake_home):
                env = _build_env(cfg)
            expected = fake_home / ".codex"
            self.assertEqual(env.get("CODEX_HOME"), str(expected))
            self.assertTrue(expected.exists())

    def test_non_codex_does_not_inject_codex_home(self) -> None:
        cfg = LLMConfig(cli="mock-llm", args=[], mode="stdin", timeout_sec=10)
        with patch.dict(os.environ, {}, clear=True):
            env = _build_env(cfg)
        self.assertNotIn("CODEX_HOME", env)

    def test_call_llm_passes_cwd_to_subprocess(self) -> None:
        cfg = LLMConfig(cli="mock-llm", args=["--flag"], mode="stdin", timeout_sec=10)
        seen_cwd: list[str | None] = []

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            seen_cwd.append(kwargs.get("cwd") if isinstance(kwargs.get("cwd"), str) else None)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with patch("mars.llm.client.subprocess.run", side_effect=fake_run):
            resp = call_llm("hello", cfg, cwd="/tmp")

        self.assertTrue(resp.ok)
        self.assertEqual(resp.text, "ok")
        self.assertEqual(seen_cwd, ["/tmp"])

    def test_call_llm_stdin_mode_runs_as_gemini_model_dash_p_dash(self) -> None:
        cfg = LLMConfig(
            cli="gemini",
            args=["-m", "gemini-3-pro-preview", "-p", "-"],
            mode="stdin",
            timeout_sec=10,
        )
        seen_cmd: list[list[str]] = []
        seen_has_input: list[bool] = []

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            seen_cmd.append(list(cmd))
            seen_has_input.append("input" in kwargs and kwargs.get("input") is not None)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with patch("mars.llm.client.subprocess.run", side_effect=fake_run):
            resp = call_llm("hello gemini", cfg, cwd="/tmp")

        self.assertTrue(resp.ok)
        self.assertEqual(resp.text, "ok")
        self.assertEqual(seen_cmd, [["gemini", "-m", "gemini-3-pro-preview", "-p", "-"]])
        self.assertEqual(seen_has_input, [True])


if __name__ == "__main__":
    unittest.main()
