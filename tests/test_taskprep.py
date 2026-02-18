from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mars.taskprep import collect_task_preparation_context, task_prep_context_text
from mars.types import TaskSpec


def _task(repo_path: str) -> TaskSpec:
    return TaskSpec(
        task_id="t1",
        instruction="Predict target from tabular data.",
        workdir=".",
        wallclock_sec=60,
        per_run_timeout_sec=30,
        metric_name="auc",
        higher_is_better=True,
        entrypoint="python runfile.py",
        final_metric_regex=r"^Final Validation Metric:\s*([0-9.eE+-]+)\s*$",
        repo_template_path=repo_path,
    )


class TaskPrepTests(unittest.TestCase):
    def test_collects_data_profiles_and_split_signals(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "runfile.py").write_text(
                "print('Final Validation Metric: 0.5')\n",
                encoding="utf-8",
            )
            (root / "data").mkdir(parents=True, exist_ok=True)
            (root / "data" / "train.csv").write_text(
                "x,y,label\n1,2,0\n2,3,1\n",
                encoding="utf-8",
            )
            (root / "data" / "valid.csv").write_text(
                "x,y,label\n1,1,0\n",
                encoding="utf-8",
            )

            ctx = collect_task_preparation_context(_task(str(root)), root)
            self.assertGreaterEqual(ctx["repo"]["file_count"], 3)
            self.assertTrue(ctx["entrypoint"]["exists"])
            self.assertIn("data/train.csv", ctx["split_signals"]["train"])
            self.assertIn("data/valid.csv", ctx["split_signals"]["valid"])

            profiles = {p["path"]: p for p in ctx["data_profiles"]}
            self.assertIn("data/train.csv", profiles)
            self.assertEqual(profiles["data/train.csv"]["columns"], ["x", "y", "label"])

            txt = task_prep_context_text(ctx)
            self.assertIn("Task Signals", txt)
            self.assertIn("data/train.csv", txt)


if __name__ == "__main__":
    unittest.main()
