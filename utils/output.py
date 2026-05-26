from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_prediction(instance_id: str, model_name_or_path: str, model_patch: str) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name_or_path,
        "model_patch": model_patch,
    }


def write_predictions(predictions: list[dict[str, Any]], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(prediction, ensure_ascii=False) for prediction in predictions]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
