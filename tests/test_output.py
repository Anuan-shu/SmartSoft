from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.output import build_prediction, write_predictions


class OutputTests(unittest.TestCase):
    def test_build_prediction_has_required_fields(self) -> None:
        prediction = build_prediction("i1", "demo-model", "patch")
        self.assertEqual(sorted(prediction.keys()), ["instance_id", "model_name_or_path", "model_patch"])

    def test_write_predictions_outputs_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "predictions.jsonl"
            write_predictions(
                [
                    build_prediction("i1", "m", "p1"),
                    build_prediction("i2", "m", "p2"),
                ],
                output,
            )
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0])["instance_id"], "i1")
            self.assertEqual(json.loads(lines[1])["instance_id"], "i2")


if __name__ == "__main__":
    unittest.main()
