"""Generate comparison reports from evaluation results.

Discovers experiment results and produces terminal tables, CSV, LaTeX
tables, Markdown summaries, and comparison plots.

By default the report covers only the 7 single-chain masking strategies
(uniform / cdr / span / structure / interface / germline /
hybrid_curriculum). The paired-chain experiments (multispecific,
hybrid_paired) were trained on a different data regime and are not
directly comparable to single-chain models on the VH-only benchmarks;
they are excluded unless --include-paired is passed.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output-dir comparison_outputs
    python scripts/generate_report.py --include-paired   # include multispecific / hybrid_paired
    python scripts/generate_report.py --experiments uniform_medium cdr_medium  # custom subset
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


# Default set of experiments in the report: the 7 single-chain masking
# strategies that share a training and evaluation regime and are directly
# comparable on the VH-only benchmark suite.
DEFAULT_SINGLE_CHAIN_EXPERIMENTS = [
    "uniform_medium",
    "cdr_medium",
    "span_medium",
    "structure_medium",
    "interface_medium",
    "germline_medium",
    "hybrid_curriculum_medium",
]

# Paired-chain experiments that are excluded by default. Trained on
# different data (OAS paired VH+VL) with a different tokenizer / vocab,
# so VH-only benchmark numbers are not apples-to-apples.
PAIRED_EXPERIMENTS = [
    "multispecific_medium",
    "hybrid_paired_medium",
]


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
        "--experiments", type=str, nargs="*", default=None,
        help=(
            "Explicit list of experiment names to include. Defaults to the "
            "7 single-chain models. Paired-chain models are excluded by "
            "default because they were trained on a different data regime "
            "and the VH-only benchmarks are not directly comparable."
        ),
    )
    parser.add_argument(
        "--include-paired", action="store_true",
        help=(
            "Include paired-chain experiments (multispecific_medium, "
            "hybrid_paired_medium) alongside the single-chain defaults."
        ),
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

    # Resolve which experiments to include in the report
    if args.experiments is not None:
        allowed = set(args.experiments)
    elif args.include_paired:
        allowed = set(DEFAULT_SINGLE_CHAIN_EXPERIMENTS) | set(PAIRED_EXPERIMENTS)
    else:
        allowed = set(DEFAULT_SINGLE_CHAIN_EXPERIMENTS)

    discovered = set(experiments.keys())
    missing = sorted(allowed - discovered)
    if missing:
        logger.warning(
            "Requested experiments not found in eval outputs: %s (they will be skipped)",
            missing,
        )
    excluded = sorted(discovered - allowed)
    if excluded:
        logger.info("Excluding experiments from report: %s", excluded)

    experiments = {
        name: exp for name, exp in experiments.items() if name in allowed
    }
    if not experiments:
        logger.error(
            "No experiments remain after filtering. Check --experiments / --include-paired.",
        )
        return

    logger.info("Reporting on %d experiments: %s", len(experiments), sorted(experiments.keys()))

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
