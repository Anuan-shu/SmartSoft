from __future__ import annotations

import argparse
import json
from pathlib import Path


def default_output_dir(dataset_name: str) -> Path:
    return Path("data") / dataset_name.replace("/", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a SWE-bench dataset from Hugging Face and save it locally.")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite", help="Hugging Face dataset name")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local directory to save the dataset. Defaults to data/<dataset_name_with__>",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional list of splits to save. If omitted, saves all available splits.",
    )
    return parser.parse_args()


def save_manifest(output_dir: Path, dataset_name: str, splits: list[str]) -> None:
    manifest = {
        "dataset_name": dataset_name,
        "splits": splits,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    from datasets import DatasetDict, load_dataset

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.dataset)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.splits:
        datasets_by_split = {split: load_dataset(args.dataset, split=split) for split in args.splits}
        dataset_obj = DatasetDict(datasets_by_split)
        split_names = list(datasets_by_split.keys())
    else:
        dataset_obj = load_dataset(args.dataset)
        split_names = list(dataset_obj.keys()) if isinstance(dataset_obj, DatasetDict) else ["default"]

    dataset_obj.save_to_disk(str(output_dir))
    save_manifest(output_dir, args.dataset, split_names)
    print(f"Saved dataset '{args.dataset}' to {output_dir}")
    print(f"Available splits: {', '.join(split_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
