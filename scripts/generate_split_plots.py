"""Generate split-by-strategy-family plots from the comparison table.

The default `generate_report.py` puts all experiments on a single bar
chart, which becomes unreadable past about 8 strategies. This script
produces three plots per metric, partitioned by family:

  1. individual_<metric>.png — the random-init control plus the 8
     single-pass masking strategies (untrained, uniform, cdr, span,
     structure_longrange, interface, germline, intersection, plus the
     legacy structure_medium for ablation).
  2. hybrid_<metric>.png — the 8 curriculum / continued-pretraining
     hybrid variants.
  3. all_sorted_<metric>.png — every strategy on a single horizontal
     bar chart, sorted from best to worst, color-coded by family.

Usage:
    python scripts/generate_split_plots.py
    python scripts/generate_split_plots.py --output-dir comparison_outputs
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Strategy taxonomy
INDIVIDUAL = [
    "untrained_medium",            # random-init control (no MLM pretraining)
    "uniform_medium",
    "cdr_medium",
    "span_medium",
    "structure_medium",            # legacy ESM-2 prior (kept for ablation)
    "structure_longrange_medium",  # IgFold + |i-j|>=4 filter
    "interface_medium",
    "germline_medium",
    "intersection_medium",
]
HYBRID = [
    "hybrid_curriculum_medium",
    "hybrid_stretched_medium",
    "hybrid_reverse_medium",
    "hybrid_warmstart_medium",
    "hybrid_intersection_medium",
    "hybrid_perbatch_medium",
    "hybrid_weighted_medium",
    "hybrid_adaptive_medium",
]
ALL = INDIVIDUAL + HYBRID

DISPLAY = {
    "untrained_medium": "untrained",
    "uniform_medium": "uniform",
    "cdr_medium": "cdr",
    "span_medium": "span",
    "structure_medium": "structure (ESM-2)",
    "structure_longrange_medium": "structure (long-range)",
    "interface_medium": "interface",
    "germline_medium": "germline",
    "intersection_medium": "intersection",
    "hybrid_curriculum_medium": "hybrid (original)",
    "hybrid_stretched_medium": "hybrid (stretched)",
    "hybrid_reverse_medium": "hybrid (reverse)",
    "hybrid_warmstart_medium": "hybrid (warm-start)",
    "hybrid_intersection_medium": "hybrid (+intersection)",
    "hybrid_perbatch_medium": "hybrid (per-batch)",
    "hybrid_weighted_medium": "hybrid (weighted)",
    "hybrid_adaptive_medium": "hybrid (adaptive)",
}

# Family colors — consistent across all plots
FAMILY_COLOR = {
    "untrained_medium": "#CCCCCC", # random-init control, light grey
    "uniform_medium": "#777777",   # baseline grey
    "cdr_medium": "#F58518",
    "span_medium": "#54A24B",
    "structure_medium": "#B279A2",
    "structure_longrange_medium": "#7B3F99",
    "interface_medium": "#4C78A8",
    "germline_medium": "#E45756",
    "intersection_medium": "#F2CF5B",
    "hybrid_curriculum_medium": "#9D755D",
    "hybrid_stretched_medium": "#B07AA1",
    "hybrid_reverse_medium": "#76B7B2",
    "hybrid_warmstart_medium": "#FF9DA7",
    "hybrid_intersection_medium": "#BAB0AC",
    "hybrid_perbatch_medium": "#59A14F",
    "hybrid_weighted_medium": "#EDC948",
    "hybrid_adaptive_medium": "#AF7AA1",
}

# Metrics to plot. Each entry: (csv_column, friendly_label, lower_is_better, group_tag)
METRICS = [
    # Pretraining / zero-shot
    ("mlm_accuracy",                       "MLM accuracy",                   False, "mlm_accuracy"),
    ("mlm_accuracy_cdr",                   "MLM accuracy (CDR)",             False, "mlm_accuracy_cdr"),
    ("mlm_accuracy_cdr3",                  "MLM accuracy (CDR3)",            False, "mlm_accuracy_cdr3"),
    ("perplexity_overall",                 "Perplexity (overall)",           True,  "perplexity"),
    ("perplexity_cdr",                     "Perplexity (CDR)",               True,  "perplexity_cdr"),
    ("perplexity_cdr3",                    "Perplexity (CDR3)",              True,  "perplexity_cdr3"),
    ("infill_cdr1_exact_match",            "CDR1 infill exact-match",        False, "cdr1_infill"),
    ("infill_cdr2_exact_match",            "CDR2 infill exact-match",        False, "cdr2_infill"),
    ("infill_cdr3_exact_match",            "CDR3 infill exact-match",        False, "cdr3_infill"),
    ("nterm_exact_match",                  "N-terminus exact-match",         False, "nterm"),
    ("cdr3_jsd",                           "CDR3 AA-frequency JSD",          True,  "cdr3_jsd"),
    # Mutation benchmark
    ("mut_mean_per_complex_spearman_rho",  "AB-Bind mean Spearman ρ",        False, "mutation_spearman"),
    ("mut_mean_per_complex_auroc",         "AB-Bind mean AUROC",             False, "mutation_auroc"),
    # Downstream linear probes
    ("ds_paratope_auroc_mean",             "Paratope AUROC",                 False, "paratope_auroc"),
    ("ds_paratope_auprc_mean",             "Paratope AUPRC",                 False, "paratope_auprc"),
    ("ds_paratope_f1_mean",                "Paratope F1",                    False, "paratope_f1"),
    ("ds_paratope_mcc_mean",               "Paratope MCC",                   False, "paratope_mcc"),
    ("ds_contact_map_auroc_mean",          "Contact map AUROC",              False, "contact_auroc"),
    ("ds_contact_map_long_range_auroc_mean", "Contact map long-range AUROC", False, "contact_long_auroc"),
    ("ds_contact_map_long_range_precision_at_L_mean", "Contact long-range P@L", False, "contact_long_pl"),
    ("ds_developability_spearman_macro_mean", "Developability macro Spearman", False, "developability"),
    ("ds_structure_probe_spearman_distance_mean",     "Structure probe Spearman ρ", False, "structure_rho"),
    ("ds_structure_probe_contact_precision_at_L_mean", "Structure probe P@L",       False, "structure_pl"),
]


def _read_table(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r["experiment"]
            row = {}
            for k, v in r.items():
                if k == "experiment" or v == "":
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
            rows[name] = row
    return rows


def _plot_horizontal_bars(
    title: str,
    models: list[str],
    values: list[float | None],
    lower_is_better: bool,
    out_path: Path,
    figsize: tuple[float, float] = (8, 4.5),
    sort_by_value: bool = True,
) -> None:
    """Horizontal bars sorted best-to-worst, family-colored, value-labeled."""
    pairs = [(m, v) for m, v in zip(models, values) if v is not None]
    if not pairs:
        return
    if sort_by_value:
        pairs.sort(key=lambda p: p[1], reverse=not lower_is_better)
    pairs = pairs[::-1]  # matplotlib draws bottom-up; reverse for top-down visual
    labels = [DISPLAY[m] for m, _ in pairs]
    vals = [v for _, v in pairs]
    colors = [FAMILY_COLOR.get(m, "#999") for m, _ in pairs]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    arrow = "↓ better" if lower_is_better else "↑ better"
    ax.set_xlabel(arrow, fontsize=8)

    # Pad x-range for value labels
    span = (max(vals) - min(vals)) if len(vals) > 1 else max(abs(vals[0]), 1e-3)
    pad = 0.10 * (abs(span) if span else 1.0)
    ax.set_xlim(min(0, min(vals)) - pad * 0.2, max(vals) + pad)

    for bar, v in zip(bars, vals):
        ax.text(
            v + pad * 0.05, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str,
        default="comparison_outputs/comparison_table.csv",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="comparison_outputs/split_plots",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    csv_path = Path(args.csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_table(csv_path)
    n_written = 0

    for col, label, lower, tag in METRICS:
        present_individual = [m for m in INDIVIDUAL if col in rows.get(m, {})]
        present_hybrid = [m for m in HYBRID if col in rows.get(m, {})]
        present_all = [m for m in ALL if col in rows.get(m, {})]

        # 1. Individual specialists
        if present_individual:
            _plot_horizontal_bars(
                f"{label} — individual masking strategies",
                present_individual,
                [rows[m][col] for m in present_individual],
                lower,
                out_dir / f"individual_{tag}.png",
            )
            n_written += 1

        # 2. Hybrid variants
        if present_hybrid:
            _plot_horizontal_bars(
                f"{label} — hybrid curriculum variants",
                present_hybrid,
                [rows[m][col] for m in present_hybrid],
                lower,
                out_dir / f"hybrid_{tag}.png",
            )
            n_written += 1

        # 3. All 16 sorted
        if len(present_all) >= 8:
            _plot_horizontal_bars(
                f"{label} — all strategies",
                present_all,
                [rows[m][col] for m in present_all],
                lower,
                out_dir / f"all_{tag}.png",
                figsize=(9, 7),
            )
            n_written += 1

    logger.info("Wrote %d plots to %s", n_written, out_dir)


if __name__ == "__main__":
    main()
