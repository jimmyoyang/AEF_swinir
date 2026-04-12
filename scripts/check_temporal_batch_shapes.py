#!/usr/bin/env python3
"""Quick sanity check for SwinIR temporal batching.

This script loads one split from a YAML config, fetches a small DataLoader batch,
and prints tensor shapes so you can confirm that variable-length temporal
samples are padded/truncated before collation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datapipe.datasets import create_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a SwinIR YAML config")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-num", type=int, default=8, help="Optional dataset sample limit for a faster sanity check")
    parser.add_argument("--pre-filter-sample-num", type=int, default=0, help="Randomly sample this many tiles before expensive filtering")
    parser.add_argument("--pre-filter-seed", type=int, default=0, help="Seed used for pre-filter random sampling")
    parser.add_argument("--keep-zero-filter", action="store_true", help="Keep the expensive HR zero-ratio filtering enabled")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        configs = yaml.safe_load(handle)

    dataset_config = configs["data"][args.split]
    dataset_config = dict(dataset_config)
    dataset_params = dict(dataset_config.get("params", {}))
    if args.sample_num > 0:
        dataset_params["sample_num"] = args.sample_num
    if args.pre_filter_sample_num > 0:
        dataset_params["pre_filter_sample_num"] = args.pre_filter_sample_num
        dataset_params["pre_filter_seed"] = args.pre_filter_seed
    if not args.keep_zero_filter:
        dataset_params["enable_hr_zero_filter"] = False
    dataset_config["params"] = dataset_params

    print(f"loading dataset from {config_path} split={args.split} sample_num={args.sample_num}")
    print(f"zero_filter={'on' if args.keep_zero_filter else 'off'}")
    if args.pre_filter_sample_num > 0:
        print(f"pre_filter_sample_num={args.pre_filter_sample_num} seed={args.pre_filter_seed}")
    dataset = create_dataset(dataset_config, parent_configs=configs)
    print(f"dataset length={len(dataset)}")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("building first batch...")
    batch = next(iter(loader))
    print(f"split={args.split}")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: {tuple(value.shape)}")
        else:
            print(f"{key}: {type(value).__name__}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
