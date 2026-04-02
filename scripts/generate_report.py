"""Generate comparison reports from evaluation results.

Discovers all experiment results and produces terminal tables, CSV,
LaTeX tables, Markdown summaries, and comparison plots.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output-dir comparison_outputs
    python scripts/generate_report.py --checkpoints-dir models/checkpoints --eval-dir evaluation_outputs
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
    plot_training_curves,
)
from evaluation.report import (
    generate_latex_table,
    generate_markdown_summary,
    plot_metric_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison reports from evaluation results",
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="models/checkpoints",
    )
    parser.add_argument(
        "--eval-dir", type=str, default="evaluation_outputs",
    )
    parser.add_argument(
        "--downstream-dir", type=str, default="downstream_outputs",
    )
    parser.add_argument(
        "--output-dir", type=str, default="comparison_outputs",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--skip-latex", action="store_true",
        help="Skip LaTeX table generation",
    )
    parser.add_argument(
        "--skip-markdown", action="store_true",
        help="Skip Markdown summary generation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Discovering experiments...")
    experiments = discover_experiments(
        args.checkpoints_dir, args.eval_dir, args.downstream_dir,
    )

    if not experiments:
        logger.error(
            "No experiments found. Check --checkpoints-dir, --eval-dir, and --downstream-dir.",
        )
        return

    logger.info("Found %d experiments: %s", len(experiments), list(experiments.keys()))

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

    if not args.skip_latex:
        generate_latex_table(experiments, output_dir / "comparison_table.tex")

    if not args.skip_markdown:
        generate_markdown_summary(experiments, output_dir / "summary.md")

    if not args.skip_plots:
        has_training = any(e.training_summary for e in experiments.values())
        if has_training:
            plot_training_curves(experiments, output_dir / "training_curves.png")

        plot_metric_comparison(experiments, output_dir / "metric_comparison.png")

    logger.info("Report generation complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
