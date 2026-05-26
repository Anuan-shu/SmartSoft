from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

from evaluate import build_harness_command


class EvaluateTests(unittest.TestCase):
    def test_build_harness_command_uses_arm64_namespace_default(self) -> None:
        args = argparse.Namespace(
            predictions="predictions.jsonl",
            dataset="princeton-nlp/SWE-bench_Lite",
            split="test",
            run_id="demo",
            max_workers=1,
            timeout=None,
            namespace=None,
        )
        with patch("evaluate.platform.machine", return_value="arm64"):
            command = build_harness_command(args)
        self.assertEqual(command[-2:], ["--namespace", ""])

    def test_build_harness_command_respects_explicit_namespace(self) -> None:
        args = argparse.Namespace(
            predictions="predictions.jsonl",
            dataset="princeton-nlp/SWE-bench_Lite",
            split="test",
            run_id="demo",
            max_workers=1,
            timeout=300,
            namespace="custom",
        )
        command = build_harness_command(args)
        self.assertIn("--timeout", command)
        self.assertEqual(command[-2:], ["--namespace", "custom"])


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
