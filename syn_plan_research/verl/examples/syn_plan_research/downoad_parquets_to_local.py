"""
Download SynPlanResearch datasets from HuggingFace to local parquet files.

Datasets:
  - hzeng/syn-plan-research-data-sft   -> data/sft/
  - hzeng/syn-plan-research-data-eval  -> data/eval/
  - hzeng/syn-plan-research-data-rl    -> data/rl/

Usage:
  python downoad_parquets_to_local.py [--output_dir /path/to/output]
"""

import argparse
import os
from datasets import load_dataset


DATASETS = {
    "hzeng/syn-plan-research-data-sft": "sft",
    "hzeng/syn-plan-research-data-eval": "eval",
    "hzeng/syn-plan-research-data-rl": "rl",
}

DEFAULT_OUTPUT_DIR = "/gypsum/work1/zamani/hzeng/syn-plan-research/data"


def download_dataset(dataset_name: str, subset_dir: str, output_dir: str):
    """Download a HuggingFace dataset and save all splits as parquet files."""
    save_dir = os.path.join(output_dir, subset_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📥 Downloading: {dataset_name}")
    print(f"📁 Saving to:   {save_dir}")
    print(f"{'='*60}")

    # Load all available splits
    dataset_dict = load_dataset(dataset_name)

    for split_name, dataset in dataset_dict.items():
        output_path = os.path.join(save_dir, f"{split_name}.parquet")
        dataset.to_parquet(output_path)
        print(f"  ✅ Split '{split_name}': {len(dataset)} examples -> {output_path}")

    print(f"✅ Done: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Download SynPlanResearch datasets from HuggingFace")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
        help="Which datasets to download (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which datasets to download
    if "all" in args.datasets:
        to_download = DATASETS
    else:
        to_download = {k: v for k, v in DATASETS.items() if k in args.datasets}

    print(f"🚀 Downloading {len(to_download)} dataset(s) to: {args.output_dir}")

    for dataset_name, subset_dir in to_download.items():
        download_dataset(dataset_name, subset_dir, args.output_dir)

    print(f"\n{'='*60}")
    print(f"🎉 All downloads complete!")
    print(f"📁 Data saved to: {args.output_dir}")
    print(f"{'='*60}")

    # Print summary
    print("\nDirectory structure:")
    for subset_dir in to_download.values():
        full_path = os.path.join(args.output_dir, subset_dir)
        if os.path.exists(full_path):
            files = os.listdir(full_path)
            for f in sorted(files):
                fpath = os.path.join(full_path, f)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"  {full_path}/{f}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
