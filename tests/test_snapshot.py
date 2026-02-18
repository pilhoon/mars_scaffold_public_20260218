from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from mars.repo.snapshot import materialize_repo


class SnapshotTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "symlink"), "symlink is not supported on this platform")
    def test_materialize_repo_preserves_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            template = root / "template"
            run_dir = root / "run"
            template.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            real_data = root / "train_counts.parquet"
            real_data.write_text("dummy parquet content", encoding="utf-8")

            linked_data = template / "train_counts.parquet"
            os.symlink(str(real_data), str(linked_data))

            copied_repo = Path(materialize_repo(run_dir, node_id=1, template_path=template))
            copied_link = copied_repo / "train_counts.parquet"

            self.assertTrue(copied_link.is_symlink())
            self.assertEqual(copied_link.resolve(), real_data.resolve())


if __name__ == "__main__":
    unittest.main()
