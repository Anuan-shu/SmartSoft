from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from utils.tasks import default_local_dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation via local SWE-bench harness without local image builds.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite", help="Hugging Face dataset name")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Local dataset directory created by scripts/download_dataset.py (defaults to data/<dataset_with__>)",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--run-id", required=True, help="Evaluation run ID")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel evaluation workers")
    parser.add_argument("--timeout", type=int, default=None, help="Optional per-instance timeout in seconds")
    parser.add_argument(
        "--namespace",
        default="swebench",
        help="Docker image namespace for remote images; keep non-empty to avoid local builds",
    )
    return parser.parse_args()


def build_harness_command(args: argparse.Namespace, dataset_path: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(dataset_path),
        "--split",
        args.split,
        "--predictions_path",
        args.predictions,
        "--max_workers",
        str(args.max_workers),
        "--run_id",
        args.run_id,
    ]
    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])

    if args.namespace:
        command.extend(["--namespace", args.namespace])
    return command


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset_path) if args.dataset_path else default_local_dataset_path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Local dataset path not found: {dataset_path}")

    swebench_repo = Path("SWE-bench")
    if not (swebench_repo / "swebench").exists():
        raise FileNotFoundError(f"Local SWE-bench repo not found at: {swebench_repo}")

    command = build_harness_command(args, dataset_path)
    run_env = os.environ.copy()
    existing_pythonpath = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = (
        f"{swebench_repo.resolve()}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(swebench_repo.resolve())
    )

    print("Running:", " ".join(repr(part) if part == "" else part for part in command))
    completed = subprocess.run(command, check=False, env=run_env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
