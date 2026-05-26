from __future__ import annotations

import re


CODE_BLOCK_RE = re.compile(r"```(?:diff|patch)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
PATCH_STARTERS = ("diff --git", "--- ", "Index: ", "*** Begin Patch")


def strip_markdown_fences(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_patch(text: str) -> str:
    stripped = strip_markdown_fences(text)
    if not stripped:
        return ""

    for starter in PATCH_STARTERS:
        index = stripped.find(starter)
        if index >= 0:
            return stripped[index:].strip()
    return stripped
