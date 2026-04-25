"""Merge per-GPU IgFold shards into a single sidecar .pt file.

Each shard from predict_structures_igfold_launch.sh is a list of length
(end_idx - start_idx) covering a contiguous range of the input JSONL.
This script concatenates the shards in order and writes a single
List[dict | None] aligned with the JSONL, drop-in compatible with the
existing oas_vh_500k_coords.pt path.

Usage:
    python scripts/merge_igfold_shards.py \\
        --shard_prefix data/structures/oas_vh_500k_igfold \\
        --num_shards 8 \\
        --output data/structures/oas_vh_500k_igfold.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_prefix", type=str, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    merged: list[dict | None] = []
    total_succeeded = 0
    for g in range(args.num_shards):
        shard_path = Path(f"{args.shard_prefix}_shard{g}.pt")
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard: {shard_path}")

        shard = torch.load(shard_path, weights_only=False)
        merged.extend(shard)
        succeeded = sum(1 for e in shard if e is not None)
        total_succeeded += succeeded
        logger.info(
            "Shard %d: %d entries, %d succeeded (%.1f%%)",
            g, len(shard), succeeded, succeeded / max(len(shard), 1) * 100,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(output_path))

    logger.info(
        "Wrote %d entries to %s (overall coverage %d/%d = %.1f%%)",
        len(merged), output_path, total_succeeded, len(merged),
        total_succeeded / max(len(merged), 1) * 100,
    )


if __name__ == "__main__":
    main()
