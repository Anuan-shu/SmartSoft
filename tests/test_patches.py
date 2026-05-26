from __future__ import annotations

import unittest

from utils.patches import extract_patch


class PatchTests(unittest.TestCase):
    def test_extract_patch_removes_markdown_fence(self) -> None:
        text = "```diff\ndiff --git a/a.py b/a.py\n```"
        self.assertEqual(extract_patch(text), "diff --git a/a.py b/a.py")

    def test_extract_patch_finds_diff_inside_text(self) -> None:
        text = "Here is the fix\n\ndiff --git a/a.py b/a.py\n+line"
        self.assertEqual(extract_patch(text), "diff --git a/a.py b/a.py\n+line")

    def test_extract_patch_returns_empty_for_blank_output(self) -> None:
        self.assertEqual(extract_patch("   "), "")


if __name__ == "__main__":
    unittest.main()
