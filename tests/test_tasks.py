from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils.tasks import (
    build_task_context,
    default_local_dataset_path,
    load_dataset_records,
    read_task_ids,
    select_records_by_instance_id,
)


class TaskUtilsTests(unittest.TestCase):
    def test_read_task_ids_skips_comments_and_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.txt"
            path.write_text("# comment\n\nfoo\n  \nbar\n", encoding="utf-8")
            self.assertEqual(read_task_ids(path), ["foo", "bar"])

    def test_select_records_preserves_input_order(self) -> None:
        records = [{"instance_id": "b"}, {"instance_id": "a"}]
        selected = select_records_by_instance_id(records, ["a", "b"])
        self.assertEqual([record["instance_id"] for record in selected], ["a", "b"])

    def test_select_records_raises_for_missing_instance(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            select_records_by_instance_id([{"instance_id": "a"}], ["missing"])

    def test_build_task_context_supports_uppercase_test_fields(self) -> None:
        context = build_task_context(
            {
                "instance_id": "i1",
                "repo": "repo/name",
                "base_commit": "abc",
                "problem_statement": "fix it",
                "hints_text": "",
                "version": "1",
                "FAIL_TO_PASS": ["test_a"],
                "PASS_TO_PASS": ["test_b"],
            }
        )
        self.assertEqual(context.fail_to_pass, ["test_a"])
        self.assertEqual(context.pass_to_pass, ["test_b"])

    def test_default_local_dataset_path_converts_slash(self) -> None:
        self.assertEqual(
            default_local_dataset_path("princeton-nlp/SWE-bench_Lite"),
            Path("data/princeton-nlp__SWE-bench_Lite"),
        )

    def test_load_dataset_records_requires_local_path(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "download_dataset.py"):
            load_dataset_records("does-not-exist", "test")

    def test_load_dataset_records_uses_load_from_disk(self) -> None:
        fake_dataset = {"test": [{"instance_id": "i1", "repo": "r"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "dataset"
            local_path.mkdir()
            fake_datasets = SimpleNamespace(DatasetDict=dict, load_from_disk=lambda _: fake_dataset)
            with patch.dict("sys.modules", {"datasets": fake_datasets}):
                records = load_dataset_records(local_path, "test")
        self.assertEqual(records, [{"instance_id": "i1", "repo": "r"}])


if __name__ == "__main__":
    unittest.main()
