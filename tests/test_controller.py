from __future__ import annotations

from datetime import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mars.config import GlobalConfig, LLMConfig, MCTSConfig, MemoryConfig, ExecConfig
from mars.artifacts import ensure_run_dirs
from mars.controller import (
    ControllerContext,
    _ensure_llm_profile_state_keys,
    _ensure_task_preparation,
    _extract_module_specs,
    _is_usage_limit_failure,
    _llm_error_summary,
    _normalize_llm_profiles,
    _request_llm,
    _usage_limit_wait_sec_from_response,
    _update_identical_metric_guard,
    run_controller,
)
from mars.fsqueue import ensure_queue_dirs
from mars.store import connect, create_node, get_kv, init_schema, set_kv
from mars.types import ActionType, NodeStatus, TaskSpec


def _write_task_yaml(path: Path, *, wallclock_sec: int) -> None:
    task = {
        "task_id": "demo_task",
        "instruction": "Toy task",
        "workdir": ".",
        "budget": {
            "wallclock_sec": wallclock_sec,
            "per_run_timeout_sec": 30,
        },
        "objective": {
            "metric_name": "demo_metric",
            "higher_is_better": True,
        },
        "execution": {
            "entrypoint": "python runfile.py",
            "final_metric_regex": r"^Final Validation Metric:\s*([0-9.eE+-]+)\s*$",
        },
        "repo_template": {
            "path": "./templates/demo_repo_template",
        },
    }
    path.write_text(json.dumps(task), encoding="utf-8")


class ControllerTests(unittest.TestCase):
    def test_detect_usage_limit_error(self) -> None:
        res = {
            "ok": False,
            "stderr": "ERROR: You've hit your usage limit. Please try again later.",
            "error": "",
            "text": "",
        }
        self.assertTrue(_is_usage_limit_failure(res))

        res2 = {"ok": False, "stderr": "random failure", "error": "", "text": ""}
        self.assertFalse(_is_usage_limit_failure(res2))

        res3 = {
            "ok": False,
            "stderr": "TerminalQuotaError: You have exhausted your capacity on this model. Your quota will reset after 6h22m36s.",
            "error": "",
            "text": "",
        }
        self.assertTrue(_is_usage_limit_failure(res3))

    def test_usage_limit_wait_uses_retry_at_hint(self) -> None:
        now_ts = datetime(2026, 2, 15, 14, 0, 0).timestamp()
        res = {
            "ok": False,
            "stderr": "ERROR: You've hit your usage limit for codex_bengalfox. Try again at 2:33 PM.",
            "error": "",
            "text": "",
        }
        wait_sec, source, hint = _usage_limit_wait_sec_from_response(res, default_wait_sec=1800.0, now_ts=now_ts)
        self.assertAlmostEqual(wait_sec, 33 * 60, delta=1.0)
        self.assertEqual(source, "parsed_retry_at")
        self.assertEqual(hint, "2:33 PM")

    def test_usage_limit_wait_rolls_to_next_day_for_past_hint(self) -> None:
        now_ts = datetime(2026, 2, 15, 15, 0, 0).timestamp()
        res = {
            "ok": False,
            "stderr": "ERROR: You've hit your usage limit for codex_bengalfox. Try again at 2:33 PM.",
            "error": "",
            "text": "",
        }
        wait_sec, source, hint = _usage_limit_wait_sec_from_response(res, default_wait_sec=1800.0, now_ts=now_ts)
        self.assertAlmostEqual(wait_sec, (23 * 3600) + (33 * 60), delta=1.0)
        self.assertEqual(source, "parsed_retry_at")
        self.assertEqual(hint, "2:33 PM")

    def test_usage_limit_wait_falls_back_to_default_without_hint(self) -> None:
        res = {
            "ok": False,
            "stderr": "ERROR: You've hit your usage limit for codex_bengalfox.",
            "error": "",
            "text": "",
        }
        wait_sec, source, hint = _usage_limit_wait_sec_from_response(res, default_wait_sec=1800.0)
        self.assertEqual(wait_sec, 1800.0)
        self.assertEqual(source, "default_config")
        self.assertIsNone(hint)

    def test_llm_error_summary_picks_meaningful_error_line(self) -> None:
        res = {
            "ok": False,
            "error": "",
            "stderr": "Reading prompt from stdin...\nOSError: [Errno 7] Argument list too long\nmore details",
            "text": "",
        }
        summary = _llm_error_summary(res)
        self.assertIn("Argument list too long", summary)

    def test_request_llm_switches_profile_on_usage_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            ensure_run_dirs(run_dir)
            conn = connect(run_dir)
            init_schema(conn)
            cfg = GlobalConfig(
                llm=LLMConfig(
                    cli="codex",
                    args=["-m", "primary", "exec"],
                    profiles=[
                        {"name": "p1", "cli": "codex", "args": ["-m", "primary", "exec"], "mode": "stdin"},
                        {"name": "p2", "cli": "codex", "args": ["-m", "backup", "exec"], "mode": "stdin"},
                    ],
                    mode="stdin",
                    timeout_sec=5,
                    usage_limit_wait_sec=1800.0,
                ),
                mcts=MCTSConfig(),
                memory=MemoryConfig(),
                exec=ExecConfig(),
            )

            jobs: list[dict[str, object]] = []
            wait_results = [
                {
                    "ok": False,
                    "returncode": 1,
                    "stderr": "ERROR: You've hit your usage limit for p1. Try again at 2:33 PM.",
                    "text": "",
                },
                {
                    "ok": True,
                    "returncode": 0,
                    "stderr": "",
                    "text": "{\"ok\": true}",
                },
            ]

            def fake_enqueue(_q, job):
                jobs.append(job)
                return f"job-{len(jobs)}"

            def fake_wait(_q, _job_id, timeout_sec):
                self.assertGreater(timeout_sec, 0)
                return wait_results.pop(0)

            with patch("mars.controller.enqueue", side_effect=fake_enqueue), patch(
                "mars.controller.wait_for_result", side_effect=fake_wait
            ), patch("mars.controller.time.sleep") as sleep_mock, patch(
                "mars.controller.send_telegram"
            ) as telegram_mock:
                job_id, res = _request_llm(
                    llm_q=object(),
                    run_dir=run_dir,
                    cfg=cfg,
                    job_type="TASK_PREPARE",
                    payload={"instruction": "x"},
                    conn=conn,
                    iter_index=1,
                    stage="task_prepare",
                    auth_retry_state={"consecutive": 0, "total": 0},
                )

            self.assertEqual(job_id, "job-2")
            self.assertTrue(bool(res.get("ok", False)))
            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0]["payload"]["llm_profile_name"], "p1")
            self.assertEqual(jobs[1]["payload"]["llm_profile_name"], "p2")
            self.assertEqual(jobs[0]["payload"]["llm_args"], ["-m", "primary", "exec"])
            self.assertEqual(jobs[1]["payload"]["llm_args"], ["-m", "backup", "exec"])
            sleep_mock.assert_not_called()
            sent_texts = [str(c.args[0]) for c in telegram_mock.call_args_list if c.args]
            self.assertTrue(any("usage limit" in s.lower() for s in sent_texts))

    def test_request_llm_switches_profile_on_generic_failure_then_proceeds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            ensure_run_dirs(run_dir)
            conn = connect(run_dir)
            init_schema(conn)
            cfg = GlobalConfig(
                llm=LLMConfig(
                    cli="codex",
                    args=["-m", "primary", "exec"],
                    profiles=[
                        {"name": "p1", "cli": "codex", "args": ["-m", "primary", "exec"], "mode": "stdin"},
                        {"name": "p2", "cli": "codex", "args": ["-m", "backup", "exec"], "mode": "stdin"},
                    ],
                    mode="stdin",
                    timeout_sec=5,
                    usage_limit_wait_sec=1800.0,
                    auth_retry_backoff_sec=2.0,
                ),
                mcts=MCTSConfig(),
                memory=MemoryConfig(),
                exec=ExecConfig(),
            )

            jobs: list[dict[str, object]] = []
            wait_results = [
                {"ok": False, "returncode": 1, "stderr": "temporary network failure", "text": ""},
                {"ok": True, "returncode": 0, "stderr": "", "text": "{\"ok\": true}"},
            ]

            def fake_enqueue(_q, job):
                jobs.append(job)
                return f"job-{len(jobs)}"

            def fake_wait(_q, _job_id, timeout_sec):
                self.assertGreater(timeout_sec, 0)
                return wait_results.pop(0)

            with patch("mars.controller.enqueue", side_effect=fake_enqueue), patch(
                "mars.controller.wait_for_result", side_effect=fake_wait
            ), patch(
                "mars.controller.time.time", return_value=1000.0
            ), patch("mars.controller.send_telegram") as telegram_mock:
                job_id, res = _request_llm(
                    llm_q=object(),
                    run_dir=run_dir,
                    cfg=cfg,
                    job_type="TASK_PREPARE",
                    payload={"instruction": "x"},
                    conn=conn,
                    iter_index=1,
                    stage="task_prepare",
                    auth_retry_state={"consecutive": 0, "total": 0},
                )

            self.assertEqual(job_id, "job-2")
            self.assertTrue(bool(res.get("ok", False)))
            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0]["payload"]["llm_profile_name"], "p1")
            self.assertEqual(jobs[1]["payload"]["llm_profile_name"], "p2")
            self.assertEqual(telegram_mock.call_count, 0)
            events_path = run_dir / "artifacts" / "llm_usage_limit_events.jsonl"
            self.assertTrue(events_path.exists())
            events_text = events_path.read_text(encoding="utf-8")
            self.assertIn('"event": "llm_failure_retry"', events_text)
            self.assertIn('"sleep_source": "default_config"', events_text)
            self.assertIn('"sleep_sec": 1800.0', events_text)
            self.assertEqual(get_kv(conn, "llm_profile.0.next_ready_unix"), "2800.000000")

    def test_request_llm_keeps_using_same_profile_until_limited(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            ensure_run_dirs(run_dir)
            conn = connect(run_dir)
            init_schema(conn)
            cfg = GlobalConfig(
                llm=LLMConfig(
                    cli="codex",
                    args=["-m", "primary", "exec"],
                    profiles=[
                        {"name": "p1", "cli": "codex", "args": ["-m", "primary", "exec"], "mode": "stdin"},
                        {"name": "p2", "cli": "codex", "args": ["-m", "backup", "exec"], "mode": "stdin"},
                    ],
                    mode="stdin",
                    timeout_sec=5,
                    usage_limit_wait_sec=1800.0,
                ),
                mcts=MCTSConfig(),
                memory=MemoryConfig(),
                exec=ExecConfig(),
            )

            jobs: list[dict[str, object]] = []
            wait_results = [
                {"ok": True, "returncode": 0, "stderr": "", "text": "{\"ok\": true}"},
                {"ok": True, "returncode": 0, "stderr": "", "text": "{\"ok\": true}"},
            ]

            def fake_enqueue(_q, job):
                jobs.append(job)
                return f"job-{len(jobs)}"

            def fake_wait(_q, _job_id, timeout_sec):
                self.assertGreater(timeout_sec, 0)
                return wait_results.pop(0)

            with patch("mars.controller.enqueue", side_effect=fake_enqueue), patch(
                "mars.controller.wait_for_result", side_effect=fake_wait
            ):
                _request_llm(
                    llm_q=object(),
                    run_dir=run_dir,
                    cfg=cfg,
                    job_type="TASK_PREPARE",
                    payload={"instruction": "x"},
                    conn=conn,
                    iter_index=1,
                    stage="task_prepare",
                    auth_retry_state={"consecutive": 0, "total": 0},
                )
                _request_llm(
                    llm_q=object(),
                    run_dir=run_dir,
                    cfg=cfg,
                    job_type="TASK_PREPARE",
                    payload={"instruction": "x"},
                    conn=conn,
                    iter_index=2,
                    stage="task_prepare",
                    auth_retry_state={"consecutive": 0, "total": 0},
                )

            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0]["payload"]["llm_profile_name"], "p1")
            self.assertEqual(jobs[1]["payload"]["llm_profile_name"], "p1")

    def test_request_llm_round_robin_rotates_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            ensure_run_dirs(run_dir)
            conn = connect(run_dir)
            init_schema(conn)
            cfg = GlobalConfig(
                llm=LLMConfig(
                    cli="codex",
                    args=["-m", "primary", "exec"],
                    profiles=[
                        {"name": "p1", "cli": "codex", "args": ["-m", "primary", "exec"], "mode": "stdin"},
                        {"name": "p2", "cli": "codex", "args": ["-m", "backup", "exec"], "mode": "stdin"},
                    ],
                    profile_selection_mode="round_robin",
                    mode="stdin",
                    timeout_sec=5,
                    usage_limit_wait_sec=1800.0,
                ),
                mcts=MCTSConfig(),
                memory=MemoryConfig(),
                exec=ExecConfig(),
            )

            jobs: list[dict[str, object]] = []
            wait_results = [
                {"ok": True, "returncode": 0, "stderr": "", "text": "{\"ok\": true}"},
                {"ok": True, "returncode": 0, "stderr": "", "text": "{\"ok\": true}"},
                {"ok": True, "returncode": 0, "stderr": "", "text": "{\"ok\": true}"},
            ]

            def fake_enqueue(_q, job):
                jobs.append(job)
                return f"job-{len(jobs)}"

            def fake_wait(_q, _job_id, timeout_sec):
                self.assertGreater(timeout_sec, 0)
                return wait_results.pop(0)

            with patch("mars.controller.enqueue", side_effect=fake_enqueue), patch(
                "mars.controller.wait_for_result", side_effect=fake_wait
            ):
                _request_llm(
                    llm_q=object(),
                    run_dir=run_dir,
                    cfg=cfg,
                    job_type="TASK_PREPARE",
                    payload={"instruction": "x"},
                    conn=conn,
                    iter_index=1,
                    stage="task_prepare",
                    auth_retry_state={"consecutive": 0, "total": 0},
                )
                _request_llm(
                    llm_q=object(),
                    run_dir=run_dir,
                    cfg=cfg,
                    job_type="TASK_PREPARE",
                    payload={"instruction": "x"},
                    conn=conn,
                    iter_index=2,
                    stage="task_prepare",
                    auth_retry_state={"consecutive": 0, "total": 0},
                )
                _request_llm(
                    llm_q=object(),
                    run_dir=run_dir,
                    cfg=cfg,
                    job_type="TASK_PREPARE",
                    payload={"instruction": "x"},
                    conn=conn,
                    iter_index=3,
                    stage="task_prepare",
                    auth_retry_state={"consecutive": 0, "total": 0},
                )

            self.assertEqual(len(jobs), 3)
            self.assertEqual(jobs[0]["payload"]["llm_profile_name"], "p1")
            self.assertEqual(jobs[1]["payload"]["llm_profile_name"], "p2")
            self.assertEqual(jobs[2]["payload"]["llm_profile_name"], "p1")

    def test_request_llm_selection_mode_can_be_overridden_from_state_kv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            ensure_run_dirs(run_dir)
            conn = connect(run_dir)
            init_schema(conn)
            cfg = GlobalConfig(
                llm=LLMConfig(
                    cli="codex",
                    args=["-m", "primary", "exec"],
                    profiles=[
                        {"name": "p1", "cli": "codex", "args": ["-m", "primary", "exec"], "mode": "stdin"},
                        {"name": "p2", "cli": "codex", "args": ["-m", "backup", "exec"], "mode": "stdin"},
                    ],
                    profile_selection_mode="sticky",
                    mode="stdin",
                    timeout_sec=5,
                    usage_limit_wait_sec=1800.0,
                ),
                mcts=MCTSConfig(),
                memory=MemoryConfig(),
                exec=ExecConfig(),
            )
            set_kv(conn, "llm_profile.selection_mode", "round_robin")

            jobs: list[dict[str, object]] = []
            wait_results = [
                {"ok": True, "returncode": 0, "stderr": "", "text": "{\"ok\": true}"},
                {"ok": True, "returncode": 0, "stderr": "", "text": "{\"ok\": true}"},
                {"ok": True, "returncode": 0, "stderr": "", "text": "{\"ok\": true}"},
            ]

            def fake_enqueue(_q, job):
                jobs.append(job)
                return f"job-{len(jobs)}"

            def fake_wait(_q, _job_id, timeout_sec):
                self.assertGreater(timeout_sec, 0)
                return wait_results.pop(0)

            with patch("mars.controller.enqueue", side_effect=fake_enqueue), patch(
                "mars.controller.wait_for_result", side_effect=fake_wait
            ):
                _request_llm(
                    llm_q=object(),
                    run_dir=run_dir,
                    cfg=cfg,
                    job_type="TASK_PREPARE",
                    payload={"instruction": "x"},
                    conn=conn,
                    iter_index=1,
                    stage="task_prepare",
                    auth_retry_state={"consecutive": 0, "total": 0},
                )
                _request_llm(
                    llm_q=object(),
                    run_dir=run_dir,
                    cfg=cfg,
                    job_type="TASK_PREPARE",
                    payload={"instruction": "x"},
                    conn=conn,
                    iter_index=2,
                    stage="task_prepare",
                    auth_retry_state={"consecutive": 0, "total": 0},
                )
                _request_llm(
                    llm_q=object(),
                    run_dir=run_dir,
                    cfg=cfg,
                    job_type="TASK_PREPARE",
                    payload={"instruction": "x"},
                    conn=conn,
                    iter_index=3,
                    stage="task_prepare",
                    auth_retry_state={"consecutive": 0, "total": 0},
                )

            self.assertEqual(len(jobs), 3)
            self.assertEqual(jobs[0]["payload"]["llm_profile_name"], "p1")
            self.assertEqual(jobs[1]["payload"]["llm_profile_name"], "p2")
            self.assertEqual(jobs[2]["payload"]["llm_profile_name"], "p1")

    def test_request_llm_waits_for_earliest_profile_when_all_limited(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            ensure_run_dirs(run_dir)
            conn = connect(run_dir)
            init_schema(conn)
            cfg = GlobalConfig(
                llm=LLMConfig(
                    cli="codex",
                    args=["-m", "primary", "exec"],
                    profiles=[
                        {"name": "p1", "cli": "codex", "args": ["-m", "primary", "exec"], "mode": "stdin"},
                        {"name": "p2", "cli": "codex", "args": ["-m", "backup", "exec"], "mode": "stdin"},
                    ],
                    mode="stdin",
                    timeout_sec=5,
                    usage_limit_wait_sec=1800.0,
                ),
                mcts=MCTSConfig(),
                memory=MemoryConfig(),
                exec=ExecConfig(),
            )

            jobs: list[dict[str, object]] = []
            wait_results = [
                {
                    "ok": False,
                    "returncode": 1,
                    "stderr": "ERROR: You've hit your usage limit for p1. Try again later.",
                    "text": "",
                },
                {
                    "ok": False,
                    "returncode": 1,
                    "stderr": "ERROR: You've hit your usage limit for p2. Try again later.",
                    "text": "",
                },
            ]

            def fake_enqueue(_q, job):
                jobs.append(job)
                return f"job-{len(jobs)}"

            def fake_wait(_q, _job_id, timeout_sec):
                self.assertGreater(timeout_sec, 0)
                if not wait_results:
                    raise AssertionError("unexpected extra llm call")
                return wait_results.pop(0)

            def fake_sleep(sec: float) -> None:
                raise RuntimeError(f"slept:{sec}")

            with patch("mars.controller.enqueue", side_effect=fake_enqueue), patch(
                "mars.controller.wait_for_result", side_effect=fake_wait
            ), patch(
                "mars.controller._usage_limit_wait_sec_from_response",
                side_effect=[(300.0, "default_config", None), (120.0, "default_config", None)],
            ), patch("mars.controller.time.time", return_value=1000.0), patch(
                "mars.controller.time.sleep",
                side_effect=fake_sleep,
            ), patch(
                "mars.controller.send_telegram"
            ) as telegram_mock:
                telegram_mock.return_value = True
                with self.assertRaisesRegex(RuntimeError, r"^slept:120.0$"):
                    _request_llm(
                        llm_q=object(),
                        run_dir=run_dir,
                        cfg=cfg,
                        job_type="TASK_PREPARE",
                        payload={"instruction": "x"},
                        conn=conn,
                        iter_index=1,
                        stage="task_prepare",
                        auth_retry_state={"consecutive": 0, "total": 0},
                    )
                sent_texts = [str(c.args[0]) for c in telegram_mock.call_args_list if c.args]
                self.assertTrue(any("all llm profiles are usage-limited" in s.lower() for s in sent_texts))

            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0]["payload"]["llm_profile_name"], "p1")
            self.assertEqual(jobs[1]["payload"]["llm_profile_name"], "p2")

    def test_normalize_llm_profiles_gemini_defaults_to_stdin_and_model_dash_p_dash_args(self) -> None:
        cfg = GlobalConfig(
            llm=LLMConfig(
                cli="codex",
                args=["-m", "gpt-5.3-codex-spark", "exec"],
                mode="stdin",
                profiles=[{"name": "gemini_backup", "cli": "gemini"}],
            ),
            mcts=MCTSConfig(),
            memory=MemoryConfig(),
            exec=ExecConfig(),
        )
        profiles = _normalize_llm_profiles(cfg)
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].name, "gemini_backup")
        self.assertEqual(profiles[0].cli, "gemini")
        self.assertEqual(profiles[0].args, ["-m", "gemini-3-pro-preview", "-p", "-"])
        self.assertEqual(profiles[0].mode, "stdin")

    def test_profile_signature_change_resets_active_index_and_cooldowns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            ensure_run_dirs(run_dir)
            conn = connect(run_dir)
            init_schema(conn)

            set_kv(conn, "llm_profile.signature", "[{\"name\":\"old\"}]")
            set_kv(conn, "llm_profile.active_index", "3")
            set_kv(conn, "llm_profile.0.next_ready_unix", "9999999999")
            set_kv(conn, "llm_profile.1.next_ready_unix", "9999999999")

            cfg = GlobalConfig(
                llm=LLMConfig(
                    cli="codex",
                    args=["-m", "gpt-5.3-codex-spark", "exec"],
                    mode="stdin",
                    profiles=[
                        {"name": "gemini_backup", "cli": "gemini", "args": ["-m", "gemini-3-pro-preview", "-p", "-"], "mode": "stdin"},
                        {"name": "gemini_flash_backup", "cli": "gemini", "args": ["-m", "gemini-3-flash-preview", "-p", "-"], "mode": "stdin"},
                    ],
                ),
                mcts=MCTSConfig(),
                memory=MemoryConfig(),
                exec=ExecConfig(),
            )

            _ensure_llm_profile_state_keys(conn, cfg)

            self.assertEqual(get_kv(conn, "llm_profile.count"), "2")
            self.assertEqual(get_kv(conn, "llm_profile.active_index"), "0")
            self.assertEqual(get_kv(conn, "llm_profile.0.next_ready_unix"), "0")
            self.assertEqual(get_kv(conn, "llm_profile.1.next_ready_unix"), "0")

    def test_identical_metric_guard_triggers_after_three(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            ensure_run_dirs(run_dir)
            conn = connect(run_dir)
            init_schema(conn)

            streak, triggered, reason = _update_identical_metric_guard(
                conn=conn,
                run_dir=run_dir,
                iter_index=1,
                metric_value=0.5,
                threshold=3,
            )
            self.assertEqual(streak, 1)
            self.assertFalse(triggered)
            self.assertEqual(reason, "")

            streak, triggered, reason = _update_identical_metric_guard(
                conn=conn,
                run_dir=run_dir,
                iter_index=2,
                metric_value=0.5,
                threshold=3,
            )
            self.assertEqual(streak, 2)
            self.assertFalse(triggered)

            streak, triggered, reason = _update_identical_metric_guard(
                conn=conn,
                run_dir=run_dir,
                iter_index=3,
                metric_value=0.5,
                threshold=3,
            )
            self.assertEqual(streak, 3)
            self.assertTrue(triggered)
            self.assertIn("forcing DEBUG", reason)
            self.assertEqual(get_kv(conn, "force_debug_next_iter"), "1")
            self.assertTrue((run_dir / "artifacts" / "guards" / "identical_metric_guard.jsonl").exists())

    def test_extract_module_specs_from_json(self) -> None:
        plan = """
        {
          "modules": [
            {"path": "src/features.py", "purpose": "feature engineering", "interfaces": ["build_features"]},
            {"path": "src/model.py", "purpose": "modeling", "interfaces": ["train_model", "predict"]},
            {"path": "runfile.py", "purpose": "entrypoint", "interfaces": ["main"]}
          ],
          "orchestration_notes": ["wire in runfile.py"]
        }
        """
        specs = _extract_module_specs(plan)
        self.assertEqual(len(specs), 2)
        self.assertTrue(any("src/features.py" in s for s in specs))
        self.assertTrue(any("src/model.py" in s for s in specs))
        self.assertFalse(any("runfile.py" in s.lower() for s in specs))

    def test_extract_module_specs_from_text_fallback(self) -> None:
        plan = """
        - src/features.py: feature engineering helpers
        - src/model.py: training and inference
        - runfile.py: orchestration only
        """
        specs = _extract_module_specs(plan)
        self.assertEqual(specs, ["src/features.py", "src/model.py"])

    def test_task_prep_falls_back_to_local_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            ensure_run_dirs(run_dir)
            repo = root / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            (repo / "runfile.py").write_text("print('Final Validation Metric: 0.1')\n", encoding="utf-8")

            conn = connect(run_dir)
            init_schema(conn)
            create_node(conn, parent_id=None, action=ActionType.DRAFT, status=NodeStatus.DONE, repo_path=str(repo))
            q = ensure_queue_dirs(run_dir, "llm")
            ctx = ControllerContext(
                run_dir=run_dir,
                worker_id="controller-1",
                task_yaml=Path("/tmp/unused_task.yaml"),
                default_conf_yaml=Path("/tmp/unused_conf.yaml"),
            )
            task = TaskSpec(
                task_id="t1",
                instruction="Toy",
                workdir=".",
                wallclock_sec=60,
                per_run_timeout_sec=30,
                metric_name="demo_metric",
                higher_is_better=True,
                entrypoint="python runfile.py",
                final_metric_regex=r"^Final Validation Metric:\s*([0-9.eE+-]+)\s*$",
                repo_template_path=str(repo),
            )
            cfg = GlobalConfig(
                llm=LLMConfig(cli="codex", args=[], mode="stdin", timeout_sec=5),
                mcts=MCTSConfig(),
                memory=MemoryConfig(),
                exec=ExecConfig(),
            )

            with patch("mars.controller._request_llm", return_value=("job-1", {"ok": False, "text": ""})):
                job_id, summary = _ensure_task_preparation(
                    conn=conn,
                    llm_q=q,
                    ctx=ctx,
                    cfg=cfg,
                    task=task,
                    root_repo=str(repo),
                )

            self.assertEqual(job_id, "job-1")
            self.assertIn("Task Signals", summary)
            self.assertIsNotNone(get_kv(conn, "task_prep_summary"))
            self.assertTrue((run_dir / "artifacts" / "task_prep" / "local_context.json").exists())

    def test_run_controller_stops_on_wallclock_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "runs" / "demo_task"
            run_dir.mkdir(parents=True, exist_ok=True)
            task_yaml = Path(td) / "task.yaml"
            _write_task_yaml(task_yaml, wallclock_sec=0)
            conf_yaml = Path(td) / "default.yaml"
            conf_yaml.write_text(
                """
mcts:
  c_uct: 1.4
  w_time_penalty: -0.07
  max_root_drafts: 1
  root_reactivate_after_valid_no_improve: 5
  max_debug_attempts: 2
  max_improve_children: 1
memory:
  keep_recent_k: 5
llm:
  cli: "codex"
  args: []
  mode: "stdin"
  timeout_sec: 10
exec:
  per_run_timeout_sec: 30
""".strip(),
                encoding="utf-8",
            )
            ctx = ControllerContext(
                run_dir=run_dir,
                worker_id="controller-1",
                task_yaml=task_yaml,
                default_conf_yaml=conf_yaml,
            )

            with patch("mars.controller._ensure_task_preparation", return_value=(None, "prep")), patch(
                "mars.controller._request_llm", side_effect=AssertionError("LLM should not be called when budget is 0")
            ), patch(
                "mars.controller._run_exec_once", side_effect=AssertionError("Exec should not be called when budget is 0")
            ):
                run_controller(ctx, max_iters=3)

            conn = connect(run_dir)
            self.assertIsNotNone(get_kv(conn, "root_node_id"))
            self.assertIsNotNone(get_kv(conn, "run_started_at"))
            # No iter JSON should be written because loop stops before iteration body.
            iter_files = list((run_dir / "tree").glob("iter_*.json"))
            self.assertEqual(iter_files, [])

    def test_no_patch_generated_skips_exec_and_marks_buggy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "runs" / "demo_task"
            run_dir.mkdir(parents=True, exist_ok=True)
            task_yaml = Path(td) / "task.yaml"
            _write_task_yaml(task_yaml, wallclock_sec=60)
            conf_yaml = Path(td) / "default.yaml"
            conf_yaml.write_text(
                """
mcts:
  c_uct: 1.4
  w_time_penalty: -0.07
  max_root_drafts: 1
  root_reactivate_after_valid_no_improve: 5
  max_debug_attempts: 1
  max_improve_children: 1
  identical_metric_streak_to_force_debug: 3
memory:
  keep_recent_k: 5
llm:
  cli: "codex"
  args: []
  mode: "stdin"
  timeout_sec: 10
  auth_retry_backoff_sec: 1
exec:
  per_run_timeout_sec: 30
""".strip(),
                encoding="utf-8",
            )
            ctx = ControllerContext(
                run_dir=run_dir,
                worker_id="controller-1",
                task_yaml=task_yaml,
                default_conf_yaml=conf_yaml,
            )

            seq = {"n": 0}
            captured_payloads: list[dict[str, object]] = []

            def fake_request_llm(*args, **kwargs):
                job_type = kwargs.get("job_type")
                if job_type is None and len(args) >= 4:
                    job_type = args[3]
                payload = kwargs.get("payload")
                if payload is None and len(args) >= 5:
                    payload = args[4]
                if isinstance(payload, dict):
                    captured_payloads.append(payload)
                seq["n"] += 1
                job_id = f"job-{seq['n']}"
                if job_type == "IDEA_PROPOSE":
                    return job_id, {"ok": False, "text": ""}
                if job_type == "GENERATE_PATCH":
                    return job_id, {"ok": False, "text": ""}
                if job_type == "EXEC_REVIEW":
                    return job_id, {"ok": False, "text": ""}
                return job_id, {"ok": False, "text": ""}

            with patch("mars.controller._ensure_task_preparation", return_value=(None, "prep")), patch(
                "mars.controller._request_llm", side_effect=fake_request_llm
            ), patch(
                "mars.controller._run_exec_once", side_effect=AssertionError("exec should be skipped when no patch is applied")
            ):
                run_controller(ctx, max_iters=1)

            iter_path = run_dir / "tree" / "iter_0001.json"
            self.assertTrue(iter_path.exists())
            obj = json.loads(iter_path.read_text(encoding="utf-8"))
            exec_obj = obj.get("exec", {})
            self.assertFalse(exec_obj.get("metric_found"))
            self.assertIn("No patch was generated/applied", str(exec_obj.get("error_summary", "")))
            self.assertGreaterEqual(len(captured_payloads), 2)
            for payload in captured_payloads:
                exp_ctx = str(payload.get("experiment_context", ""))
                self.assertIn("iter_index: 1", exp_ctx)
                self.assertIn("iteration_stats_total:", exp_ctx)


if __name__ == "__main__":
    unittest.main()
