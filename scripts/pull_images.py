from __future__ import annotations

import argparse
import importlib
from typing import Any

from utils.tasks import read_task_ids

OFFICIAL_EVAL_IMAGE_PREFIX = "swebench/sweb.eval.x86_64."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull SWE-bench official eval images from task IDs (skip if already local)."
    )
    parser.add_argument("--tasks", default="swebench_tasks.txt", help="Path to swebench_tasks.txt")
    return parser.parse_args()


def instance_id_to_image(instance_id: str) -> str:
    return f"{OFFICIAL_EVAL_IMAGE_PREFIX}{instance_id.replace('__', '_1776_')}"


def image_exists(client: Any, docker_errors: Any, image_name: str) -> bool:
    try:
        client.images.get(image_name)
        return True
    except docker_errors.ImageNotFound:
        return False


def main() -> int:
    args = parse_args()
    task_ids = read_task_ids(args.tasks)
    image_names = sorted({instance_id_to_image(instance_id) for instance_id in task_ids})

    docker_module = importlib.import_module("docker")
    client = docker_module.from_env()
    docker_errors = docker_module.errors

    pulled_count = 0
    skipped_count = 0

    print(f"Resolved {len(image_names)} unique images from {len(task_ids)} tasks.")
    for image_name in image_names:
        if image_exists(client, docker_errors, image_name):
            skipped_count += 1
            print(f"[SKIP] {image_name}")
            continue
        client.images.pull(image_name)
        pulled_count += 1
        print(f"[PULL] {image_name}")

    print(f"Done. pulled={pulled_count}, skipped={skipped_count}, total={len(image_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
