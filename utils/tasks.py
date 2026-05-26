from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TaskContext:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    version: str
    fail_to_pass: Any
    pass_to_pass: Any


def read_task_ids(path: str | Path) -> list[str]:
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")

    task_ids: list[str] = []
    for raw_line in task_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        task_ids.append(line)
    if not task_ids:
        raise ValueError("No instance IDs found in swebench_tasks file.")
    return task_ids


def default_local_dataset_path(dataset_name: str) -> Path:
    safe_name = dataset_name.replace("/", "__")
    return Path("data") / safe_name


def load_dataset_records(dataset_path: str | Path, split: str) -> list[dict[str, Any]]:
    local_path = Path(dataset_path)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local dataset path not found: {local_path}. "
            "Please run scripts/download_dataset.py first."
        )

    from datasets import DatasetDict, load_from_disk

    dataset_obj = load_from_disk(str(local_path))
    if isinstance(dataset_obj, DatasetDict):
        if split not in dataset_obj:
            available_splits = ", ".join(dataset_obj.keys())
            raise ValueError(f"Split '{split}' not found in local dataset. Available splits: {available_splits}")
        dataset = dataset_obj[split]
    else:
        dataset = dataset_obj
    return [dict(record) for record in dataset]


def select_records_by_instance_id(records: list[dict[str, Any]], task_ids: list[str]) -> list[dict[str, Any]]:
    by_id = {record["instance_id"]: record for record in records}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"Instance IDs not found in dataset: {', '.join(missing)}")
    return [by_id[task_id] for task_id in task_ids]


def build_task_context(record: dict[str, Any]) -> TaskContext:
    return TaskContext(
        instance_id=record["instance_id"],
        repo=record["repo"],
        base_commit=record["base_commit"],
        problem_statement=record["problem_statement"],
        hints_text=record.get("hints_text") or "",
        version=record.get("version") or "",
        fail_to_pass=record.get("FAIL_TO_PASS") or record.get("fail_to_pass") or [],
        pass_to_pass=record.get("PASS_TO_PASS") or record.get("pass_to_pass") or [],
    )
