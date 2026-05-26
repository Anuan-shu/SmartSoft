from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils.models import ModelClient, load_model_config


class ModelTests(unittest.TestCase):
    def test_load_model_config_requires_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".env"
            path.write_text("apikey=x\nbase=y\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model"):
                load_model_config(path)

    def test_generate_updates_token_usage(self) -> None:
        mock_post = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "diff --git a/x b/x"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "apikey=test-key\nbase=https://example.com/v1\nmodel=test-model\n",
                encoding="utf-8",
            )
            client = ModelClient(env_path=env_path)
            fake_requests = SimpleNamespace(post=mock_post)
            with patch.dict("sys.modules", {"requests": fake_requests}):
                response = client.generate("system", "user")

        self.assertEqual(response.content, "diff --git a/x b/x")
        self.assertEqual(client.get_usage().total_tokens, 30)
        self.assertEqual(mock_post.call_args.kwargs["json"]["model"], "test-model")

    def test_save_usage_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "apikey=test-key\nbase=https://example.com/v1\nmodel=test-model\n",
                encoding="utf-8",
            )
            client = ModelClient(env_path=env_path)
            usage_path = Path(tmpdir) / "usage.json"
            client.save_usage(usage_path)
            payload = json.loads(usage_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"], "test-model")


if __name__ == "__main__":
    unittest.main()
