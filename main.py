from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import traceback

from src.agent import solve_task
from utils.docker_env import DockerEnv
from utils.models import ModelClient
from utils.output import build_prediction, write_predictions, write_text
from utils.tasks import (
    build_task_context,
    default_local_dataset_path,
    load_dataset_records,
    read_task_ids,
    select_records_by_instance_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SWE-bench predictions.jsonl with a minimal agent.")
    parser.add_argument("--tasks", required=True, help="Path to swebench_tasks.txt")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite", help="Canonical SWE-bench dataset name")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Local dataset directory created by scripts/download_dataset.py",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--output", default="predictions.jsonl", help="Output prediction jsonl path")
    parser.add_argument("--run-id", default=None, help="Run identifier for logs")
    parser.add_argument("--namespace", default=None, help="Optional SWE-bench image namespace")
    parser.add_argument("--max-workers", type=int, default=1, help="Max image build workers")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild Docker images")
    parser.add_argument("--instance-image-tag", default="latest", help="SWE-bench instance image tag")
    parser.add_argument("--env-image-tag", default="latest", help="SWE-bench environment image tag")
    return parser.parse_args()


def default_run_id() -> str:
    return datetime.now().strftime("run-%Y%m%d-%H%M%S")


def main() -> int:
    args = parse_args()
    run_id = args.run_id or default_run_id()
    run_dir = Path("runs") / run_id
    instance_root = run_dir / "instances"
    instance_root.mkdir(parents=True, exist_ok=True)

    task_ids = read_task_ids(args.tasks)
    dataset_path = Path(args.dataset_path) if args.dataset_path else default_local_dataset_path(args.dataset)
    records = load_dataset_records(dataset_path, args.split)
    selected_records = select_records_by_instance_id(records, task_ids)
    contexts = [build_task_context(record) for record in selected_records]
    model = ModelClient()

    predictions: list[dict[str, str]] = []
    for record, context in zip(selected_records, contexts, strict=True):
        instance_dir = instance_root / context.instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        try:
            with DockerEnv.create(
                record,
                run_id=run_id,
                log_dir=instance_dir,
                force_rebuild=args.force_rebuild,
                max_workers=args.max_workers,
                namespace=args.namespace,
                instance_image_tag=args.instance_image_tag,
                env_image_tag=args.env_image_tag,
            ) as env:
                patch = solve_task(context, env, model)
        except Exception as exc:
            patch = ""
            error_message = "".join(traceback.format_exception(exc))
            write_text(instance_dir / "error.txt", error_message)
            print(f"[WARN] {context.instance_id} failed: {exc}")

        predictions.append(build_prediction(context.instance_id, model.model_name, patch))

    write_predictions(predictions, args.output)
    model.save_usage(run_dir / "usage.json")
    usage = model.get_usage()
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    print(f"Loaded dataset from local path: {dataset_path}")
    print(
        "Token usage: "
        f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
