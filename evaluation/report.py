"""Report generation: LaTeX tables, Markdown summaries, and comparison plots.

Works with the ExperimentResult data from evaluation.compare to produce
publication-ready outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.compare import ExperimentResult

logger = logging.getLogger(__name__)

_PRETRAINING_METRICS = [
    ("mlm_accuracy", "MLM Acc"),
    ("mlm_accuracy_cdr", "MLM Acc (CDR)"),
    ("mlm_accuracy_cdr3", "MLM Acc (CDR3)"),
    ("mlm_accuracy_framework", "MLM Acc (FW)"),
]

_ZEROSHOT_METRICS = [
    ("perplexity_overall", "PPL"),
    ("perplexity_cdr", "PPL (CDR)"),
    ("perplexity_cdr3", "PPL (CDR3)"),
    ("infill_cdr1_accuracy", "Infill CDR1"),
    ("infill_cdr2_accuracy", "Infill CDR2"),
    ("infill_cdr3_accuracy", "Infill CDR3"),
    ("pll_normalized_mean", "PLL (norm)"),
]

_MUTATION_METRICS = [
    ("overall_spearman_rho", "Mut. Spearman"),
    ("overall_pearson_r", "Mut. Pearson"),
    ("binary_auroc", "Mut. AUROC"),
]

_DOWNSTREAM_METRICS = [
    ("paratope", "auroc_mean", "Para. AUROC"),
    ("paratope", "auprc_mean", "Para. AUPRC"),
    ("binding", "auroc_mean", "Bind. AUROC"),
    ("binding", "f1_mean", "Bind. F1"),
    ("developability", "spearman_macro_mean", "Dev. Spearman"),
]


def _get_metric(exp: ExperimentResult, source: str, key: str) -> float | None:
    """Retrieve a metric value from an ExperimentResult."""
    if source == "eval" and exp.eval_metrics:
        return exp.eval_metrics.get(key)
    if source == "zeroshot" and exp.zeroshot_metrics:
        return exp.zeroshot_metrics.get(key)
    if source == "mutation" and exp.mutation_metrics:
        return exp.mutation_metrics.get(key)
    if source.startswith("ds_"):
        task = source[3:]
        if task in exp.downstream_metrics:
            return exp.downstream_metrics[task].get(key)
    return None


def generate_latex_table(
    experiments: dict[str, ExperimentResult],
    output_path: str | Path,
) -> str:
    """Generate a booktabs-style LaTeX table comparing all experiments.

    Returns the LaTeX string and writes it to output_path.
    """
    names = sorted(experiments.keys())
    if not names:
        return "% No experiments found"

    metric_specs: list[tuple[str, str, str]] = []
    for key, label in _PRETRAINING_METRICS:
        metric_specs.append(("eval", key, label))
    for key, label in _ZEROSHOT_METRICS:
        metric_specs.append(("zeroshot", key, label))
    for key, label in _MUTATION_METRICS:
        metric_specs.append(("mutation", key, label))
    for task, key, label in _DOWNSTREAM_METRICS:
        metric_specs.append((f"ds_{task}", key, label))

    has_data: list[tuple[str, str, str]] = []
    for source, key, label in metric_specs:
        if any(_get_metric(experiments[n], source, key) is not None for n in names):
            has_data.append((source, key, label))

    if not has_data:
        return "% No metric data found"

    n_cols = 1 + len(names)
    col_spec = "l" + "r" * len(names)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of masking strategies across evaluation metrics.}",
        r"\label{tab:comparison}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Metric & " + " & ".join(n.replace("_", r"\_") for n in names) + r" \\",
        r"\midrule",
    ]

    current_section = ""
    for source, key, label in has_data:
        section = source.split("_")[0] if "_" in source else source
        if section != current_section:
            if current_section:
                lines.append(r"\midrule")
            current_section = section

        values = []
        float_vals: list[float] = []
        for n in names:
            v = _get_metric(experiments[n], source, key)
            if v is not None:
                float_vals.append(v)
                values.append(v)
            else:
                values.append(None)

        is_lower_better = "perplexity" in key or "loss" in key
        if float_vals:
            if is_lower_better:
                best_val = min(float_vals)
            else:
                best_val = max(float_vals)
        else:
            best_val = None

        formatted = []
        for v in values:
            if v is None:
                formatted.append("--")
            elif v == best_val and len(float_vals) > 1:
                formatted.append(rf"\textbf{{{v:.4f}}}")
            else:
                formatted.append(f"{v:.4f}")

        lines.append(label + " & " + " & ".join(formatted) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_str = "\n".join(lines)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str)
    logger.info("LaTeX table saved to %s", output_path)
    return latex_str


def generate_markdown_summary(
    experiments: dict[str, ExperimentResult],
    output_path: str | Path,
) -> str:
    """Generate a Markdown summary of all experiments."""
    names = sorted(experiments.keys())
    if not names:
        return "No experiments found."

    lines = [
        "# Evaluation Summary",
        "",
        f"**Experiments:** {', '.join(names)}",
        "",
    ]

    lines.append("## Pretraining Metrics")
    lines.append("")
    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|--------|" + "|".join("-------:" for _ in names) + "|"
    lines.extend([header, sep])
    for key, label in _PRETRAINING_METRICS:
        vals = []
        for n in names:
            v = _get_metric(experiments[n], "eval", key)
            vals.append(f"{v:.4f}" if v is not None else "--")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("## Zero-Shot Metrics")
    lines.append("")
    header = "| Metric | " + " | ".join(names) + " |"
    lines.extend([header, sep])
    for key, label in _ZEROSHOT_METRICS:
        vals = []
        for n in names:
            v = _get_metric(experiments[n], "zeroshot", key)
            vals.append(f"{v:.4f}" if v is not None else "--")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")

    has_mutation = any(exp.mutation_metrics for exp in experiments.values())
    if has_mutation:
        lines.append("## Mutation Benchmark")
        lines.append("")
        header = "| Metric | " + " | ".join(names) + " |"
        lines.extend([header, sep])
        for key, label in _MUTATION_METRICS:
            vals = []
            for n in names:
                v = _get_metric(experiments[n], "mutation", key)
                vals.append(f"{v:.4f}" if v is not None else "--")
            lines.append(f"| {label} | " + " | ".join(vals) + " |")
        lines.append("")

    has_downstream = any(exp.downstream_metrics for exp in experiments.values())
    if has_downstream:
        lines.append("## Downstream Tasks")
        lines.append("")
        header = "| Metric | " + " | ".join(names) + " |"
        lines.extend([header, sep])
        for task, key, label in _DOWNSTREAM_METRICS:
            vals = []
            for n in names:
                v = _get_metric(experiments[n], f"ds_{task}", key)
                vals.append(f"{v:.4f}" if v is not None else "--")
            lines.append(f"| {label} | " + " | ".join(vals) + " |")
        lines.append("")

    md_str = "\n".join(lines)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_str)
    logger.info("Markdown summary saved to %s", output_path)
    return md_str


def plot_metric_comparison(
    experiments: dict[str, ExperimentResult],
    output_path: str | Path,
) -> None:
    """Generate grouped bar charts comparing key metrics across experiments."""
    names = sorted(experiments.keys())
    if not names:
        return

    metric_groups: dict[str, list[tuple[str, str, str]]] = {
        "MLM Accuracy": [("eval", k, l) for k, l in _PRETRAINING_METRICS],
        "Perplexity": [
            ("zeroshot", k, l) for k, l in _ZEROSHOT_METRICS if "perplexity" in k
        ],
        "CDR Infilling": [
            ("zeroshot", k, l) for k, l in _ZEROSHOT_METRICS if "infill" in k
        ],
        "Downstream": [(f"ds_{t}", k, l) for t, k, l in _DOWNSTREAM_METRICS],
    }

    groups_with_data: list[tuple[str, list[tuple[str, str, str]]]] = []
    for group_name, specs in metric_groups.items():
        has_any = False
        for source, key, _ in specs:
            if any(_get_metric(experiments[n], source, key) is not None for n in names):
                has_any = True
                break
        if has_any:
            groups_with_data.append((group_name, specs))

    if not groups_with_data:
        logger.warning("No metric data for comparison plots")
        return

    n_groups = len(groups_with_data)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5), squeeze=False)

    x = np.arange(len(names))
    bar_width = 0.8 / max(1, len(names))

    for ax_idx, (group_name, specs) in enumerate(groups_with_data):
        ax = axes[0, ax_idx]
        labels = []
        data_matrix: list[list[float | None]] = []

        for source, key, label in specs:
            vals = [_get_metric(experiments[n], source, key) for n in names]
            if any(v is not None for v in vals):
                labels.append(label)
                data_matrix.append(vals)

        if not labels:
            continue

        metric_x = np.arange(len(labels))
        for exp_idx, name in enumerate(names):
            offset = (exp_idx - len(names) / 2 + 0.5) * bar_width
            heights = [
                row[exp_idx] if row[exp_idx] is not None else 0
                for row in data_matrix
            ]
            ax.bar(metric_x + offset, heights, bar_width, label=name, alpha=0.85)

        ax.set_title(group_name)
        ax.set_xticks(metric_x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Metric comparison plot saved to %s", output_path)
