"""Aggregate seed replicates from comparison_table.csv into mean +/- sd.

``evaluation/compare.py`` emits one row per experiment directory, so seed
replicates land as separate rows (``interface_medium``, ``interface_medium_s1``,
``interface_medium_s2``). This groups them by strategy and produces the two
things the reviewers asked for:

1. **Marginals** — mean, sd and n per (strategy, metric). Answers "what is the
   variance across pretraining seeds?" (#2, #5.1, #6).

2. **Paired contrasts** — for each seed, ``metric[strategy_s] - metric[ref_s]``,
   then the mean and sd of those per-seed differences. All replicates share an
   identical held-out split (``data/splits.py``) and a deterministic evaluation
   sample, so the paired difference is much tighter than comparing marginals.
   This is the number to quote for "interface beats random by X".

With n=3 per arm, no parametric test is reported — the paired sd and the
effect size in sd units are the honest summary.

Usage:
    python scripts/aggregate_seeds.py
    python scripts/aggregate_seeds.py --paper-metrics-only
    python scripts/aggregate_seeds.py --ref cdr_medium
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import statistics
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

#: Trailing ``_s<digits>`` marks a replicate; the bare name is seed 42.
_SEED_SUFFIX = re.compile(r"_s(\d+)$")

#: The seven metrics in main-text Tables 1 and 2, in column order.
#: Verified against the paper's `random` row (= uniform_medium).
PAPER_METRICS: list[tuple[str, str]] = [
    ("CDR3",     "infill_cdr3_exact_match"),
    ("P. AUPRC", "ds_paratope_auprc_mean"),
    ("P. MCC",   "ds_paratope_mcc_mean"),
    ("C. AUROC", "ds_contact_map_auroc_mean"),
    ("C. P@L",   "ds_contact_map_long_range_precision_at_L_mean"),
    ("Str. rho", "ds_structure_probe_spearman_distance_mean"),
    ("Dev. rho", "ds_developability_spearman_macro_mean"),
]

_NON_METRIC = {"experiment", "strategy", "model_size", "dataset"}


def split_experiment_name(name: str) -> tuple[str, int]:
    """``interface_medium_s1`` -> ``("interface_medium", 1)``; bare name -> seed 42."""
    m = _SEED_SUFFIX.search(name)
    return (name[: m.start()], int(m.group(1))) if m else (name, 42)


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(table: Path) -> dict[str, dict[int, dict[str, float]]]:
    """Return ``{strategy: {seed: {metric: value}}}``."""
    grouped: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    with table.open() as f:
        for row in csv.DictReader(f):
            strategy, seed = split_experiment_name(row["experiment"])
            metrics = {
                k: v for k, v in ((k, _as_float(v)) for k, v in row.items()
                                  if k not in _NON_METRIC)
                if v is not None
            }
            grouped[strategy][seed] = metrics
    return grouped


def summarize(values: list[float]) -> tuple[float, float | None, int]:
    """Mean, sample sd (None when n<2), n."""
    n = len(values)
    return (
        statistics.fmean(values),
        statistics.stdev(values) if n > 1 else None,
        n,
    )


def write_marginals(
    grouped: dict[str, dict[int, dict[str, float]]], out: Path, metrics: list[str],
) -> None:
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "metric", "mean", "sd", "n", "seeds"])
        for strategy in sorted(grouped):
            per_seed = grouped[strategy]
            seeds = sorted(per_seed)
            for metric in metrics:
                vals = [per_seed[s][metric] for s in seeds if metric in per_seed[s]]
                if not vals:
                    continue
                mean, sd, n = summarize(vals)
                w.writerow([
                    strategy, metric, f"{mean:.6f}",
                    "" if sd is None else f"{sd:.6f}", n,
                    ";".join(str(s) for s in seeds if metric in per_seed[s]),
                ])


def write_contrasts(
    grouped: dict[str, dict[int, dict[str, float]]],
    ref: str,
    out: Path,
    metrics: list[str],
) -> None:
    """Per-seed paired differences against ``ref``."""
    if ref not in grouped:
        raise KeyError(f"reference strategy {ref!r} not in table")
    ref_seeds = grouped[ref]

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "strategy", "metric", "mean_delta", "sd_delta", "n_paired",
            "effect_size_sd_units", "per_seed_deltas",
        ])
        for strategy in sorted(grouped):
            if strategy == ref:
                continue
            for metric in metrics:
                deltas = [
                    grouped[strategy][s][metric] - ref_seeds[s][metric]
                    for s in sorted(grouped[strategy])
                    if s in ref_seeds
                    and metric in grouped[strategy][s]
                    and metric in ref_seeds[s]
                ]
                if not deltas:
                    continue
                mean, sd, n = summarize(deltas)
                effect = f"{mean / sd:.2f}" if sd not in (None, 0.0) else ""
                w.writerow([
                    strategy, metric, f"{mean:+.6f}",
                    "" if sd is None else f"{sd:.6f}", n, effect,
                    ";".join(f"{d:+.4f}" for d in deltas),
                ])


def render_markdown(
    grouped: dict[str, dict[int, dict[str, float]]], ref: str, out: Path,
) -> None:
    """Paper-shaped table: mean +/- sd per strategy for the seven metrics."""
    lines = [
        "# Seed-replicate summary",
        "",
        f"Reference strategy for contrasts: `{ref}`. "
        "Cells are mean ± sd across pretraining seeds; "
        "`n=1` means the strategy has not been replicated yet.",
        "",
        "| Strategy | n | " + " | ".join(label for label, _ in PAPER_METRICS) + " |",
        "|---|---|" + "---|" * len(PAPER_METRICS),
    ]
    for strategy in sorted(grouped):
        per_seed = grouped[strategy]
        cells = []
        n_seen = 0
        for _, col in PAPER_METRICS:
            vals = [per_seed[s][col] for s in sorted(per_seed) if col in per_seed[s]]
            if not vals:
                cells.append("—")
                continue
            mean, sd, n = summarize(vals)
            n_seen = max(n_seen, n)
            cells.append(f"{mean:.3f}" if sd is None else f"{mean:.3f} ± {sd:.3f}")
        lines.append(f"| `{strategy}` | {n_seen} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Quote **paired** contrasts (`seed_contrasts.csv`), not differences of the "
        "means above: replicates share an identical held-out split and evaluation "
        "sample, so the per-seed difference is the tighter and more honest estimate.",
        "",
    ]
    out.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table", type=str, default="comparison_outputs/comparison_table.csv",
    )
    parser.add_argument("--output-dir", type=str, default="comparison_outputs")
    parser.add_argument(
        "--ref", type=str, default="uniform_medium",
        help="Baseline strategy for paired contrasts (paper's `random`).",
    )
    parser.add_argument(
        "--paper-metrics-only", action="store_true",
        help="Restrict to the seven main-text metrics instead of all columns.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    repo = Path(__file__).resolve().parent.parent
    table = repo / args.table
    if not table.exists():
        raise FileNotFoundError(f"comparison table not found: {table}")

    grouped = load_rows(table)

    if args.paper_metrics_only:
        metrics = [col for _, col in PAPER_METRICS]
    else:
        metrics = sorted({m for seeds in grouped.values()
                          for row in seeds.values() for m in row})

    out_dir = repo / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_marginals(grouped, out_dir / "seed_aggregate.csv", metrics)
    write_contrasts(grouped, args.ref, out_dir / "seed_contrasts.csv", metrics)
    render_markdown(grouped, args.ref, out_dir / "seed_summary.md")

    replicated = {s: len(v) for s, v in grouped.items() if len(v) > 1}
    logger.info("strategies found:      %d", len(grouped))
    logger.info("with >1 seed:          %d %s", len(replicated), replicated or "")
    logger.info("metrics aggregated:    %d", len(metrics))
    for f in ("seed_aggregate.csv", "seed_contrasts.csv", "seed_summary.md"):
        logger.info("wrote %s", (out_dir / f).relative_to(repo))

    if not replicated:
        logger.warning(
            "\nNo strategy has more than one seed yet — sd columns will be empty. "
            "Re-run after the replicates finish.",
        )


if __name__ == "__main__":
    main()
