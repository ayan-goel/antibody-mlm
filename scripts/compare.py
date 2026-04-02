"""CLI: compare multiple masking strategy experiments side by side.

Auto-discovers experiments from checkpoint and evaluation output directories,
prints a comparison table, and generates training curve and embedding plots.

Usage:
    python scripts/compare.py
    python scripts/compare.py --checkpoints-dir models/checkpoints --eval-dir evaluation_outputs
    python scripts/compare.py --output-dir comparison_outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.compare import (
    build_comparison_table,
    discover_experiments,
    format_table,
    plot_embedding_comparison,
    plot_training_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare masking strategy experiments")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="models/checkpoints",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="evaluation_outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_outputs",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding comparison plot (faster, avoids UMAP computation)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments(args.checkpoints_dir, args.eval_dir)
    if not experiments:
        logger.error("No experiments found. Check --checkpoints-dir and --eval-dir paths.")
        return

    table = build_comparison_table(experiments)

    print("\n" + format_table(table) + "\n")

    json_path = output_dir / "comparison_table.json"
    with json_path.open("w") as f:
        json.dump(table, f, indent=2)
    logger.info("JSON table saved to %s", json_path)

    csv_path = output_dir / "comparison_table.csv"
    if table:
        all_keys = list(dict.fromkeys(k for row in table for k in row))
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(table)
        logger.info("CSV table saved to %s", csv_path)

    has_training = any(e.training_summary for e in experiments.values())
    if has_training:
        plot_training_curves(experiments, output_dir / "training_curves.png")

    if not args.skip_embeddings:
        has_embeddings = any(
            e.eval_dir and (e.eval_dir / "embeddings.npy").exists()
            for e in experiments.values()
        )
        if has_embeddings:
            plot_embedding_comparison(experiments, output_dir / "embedding_comparison.png")
        else:
            logger.info("No embeddings found, skipping embedding comparison plot")

    logger.info("Comparison complete. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
