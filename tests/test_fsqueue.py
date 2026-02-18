from __future__ import annotations

import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mars.fsqueue import claim, complete, ensure_queue_dirs, enqueue, read_job, read_result


class FSQueueTests(unittest.TestCase):
    def test_claim_falls_back_when_cross_dir_replace_is_not_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            q = ensure_queue_dirs(td, "llm")
            job_id = enqueue(q, {"kind": "llm", "type": "TASK_PREPARE", "payload": {"x": 1}})

            original_replace = os.replace

            def flaky_replace(src: str | bytes, dst: str | bytes) -> None:
                src_path = Path(src)
                dst_path = Path(dst)
                if src_path.parent != dst_path.parent:
                    raise OSError(errno.EXDEV, "Invalid cross-device link")
                original_replace(src, dst)

            with patch("mars.fsqueue.os.replace", side_effect=flaky_replace):
                running_job = claim(q, worker_id="w1")

            self.assertIsNotNone(running_job)
            assert running_job is not None
            self.assertEqual(running_job, q.running / f"{job_id}.json")
            self.assertTrue((q.running / f"{job_id}.json").exists())
            self.assertFalse((q.pending / f"{job_id}.json").exists())
            # Claim marker should not be left behind.
            leftovers = list(q.pending.glob("*.claim"))
            self.assertEqual(leftovers, [])

    def test_complete_moves_running_job_and_writes_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            q = ensure_queue_dirs(td, "llm")
            job_id = enqueue(q, {"kind": "llm", "type": "TASK_PREPARE", "payload": {"x": 1}})
            running_job = claim(q, worker_id="w1")
            assert running_job is not None

            result_path = complete(q, running_job, {"ok": True, "text": "done"}, ok=True)

            self.assertEqual(result_path, q.done / f"{job_id}.result.json")
            self.assertTrue(result_path.exists())
            self.assertEqual(read_result(result_path)["ok"], True)

            job_trace = q.done / f"{job_id}.job.json"
            self.assertTrue(job_trace.exists())
            self.assertEqual(read_job(job_trace)["job_id"], job_id)
            self.assertFalse((q.running / f"{job_id}.json").exists())


if __name__ == "__main__":
    unittest.main()
