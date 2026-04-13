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
    ("mlm_top5_accuracy", "MLM Top-5"),
    ("mlm_accuracy_cdr", "MLM Acc (CDR)"),
    ("mlm_accuracy_cdr3", "MLM Acc (CDR3)"),
    ("mlm_accuracy_framework", "MLM Acc (FW)"),
]

_ZEROSHOT_METRICS = [
    ("perplexity_overall", "PPL"),
    ("perplexity_cdr", "PPL (CDR)"),
    ("perplexity_cdr3", "PPL (CDR3)"),
    ("perplexity_framework", "PPL (FW)"),
    ("pll_normalized_mean", "PLL (norm)"),
    ("infill_cdr1_accuracy", "Infill CDR1"),
    ("infill_cdr2_accuracy", "Infill CDR2"),
    ("infill_cdr3_accuracy", "Infill CDR3"),
    ("infill_cdr1_exact_match", "Infill CDR1 Exact"),
    ("infill_cdr2_exact_match", "Infill CDR2 Exact"),
    ("infill_cdr3_exact_match", "Infill CDR3 Exact"),
    ("infill_cdr3_short_accuracy", "Infill CDR3 (short)"),
    ("infill_cdr3_medium_accuracy", "Infill CDR3 (med)"),
    ("infill_cdr3_long_accuracy", "Infill CDR3 (long)"),
    ("nterm_accuracy", "N-term Acc"),
    ("nterm_exact_match", "N-term Exact"),
    ("scattered_accuracy_k1", "Scattered k=1"),
    ("scattered_accuracy_k5", "Scattered k=5"),
    ("scattered_accuracy_k10", "Scattered k=10"),
    ("cdr1_jsd", "CDR1 JSD"),
    ("cdr2_jsd", "CDR2 JSD"),
    ("cdr3_jsd", "CDR3 JSD"),
]

_MUTATION_METRICS = [
    # Per-complex aggregations are the only honest measure of within-complex
    # ranking ability; pooled metrics are dominated by between-complex
    # variance (Simpson's paradox) and overstate model performance, so we
    # don't surface them in the table.
    ("mean_per_complex_spearman_rho", "Mut. Spearman (mean per-cplx)"),
    ("median_per_complex_spearman_rho", "Mut. Spearman (median per-cplx)"),
    ("mean_per_complex_auroc", "Mut. AUROC (mean per-cplx)"),
    ("median_per_complex_auroc", "Mut. AUROC (median per-cplx)"),
    ("n_complexes", "Mut. # complexes"),
    ("n_mutants_total", "Mut. # mutants"),
]

_DOWNSTREAM_METRICS = [
    # Paratope (sequence-level binary, per-token classification)
    ("paratope", "auroc_mean", "Para. AUROC"),
    ("paratope", "auprc_mean", "Para. AUPRC"),
    ("paratope", "f1_mean", "Para. F1"),
    ("paratope", "mcc_mean", "Para. MCC"),
    # Developability (5-target regression, macro Spearman + per-target)
    ("developability", "spearman_macro_mean", "Dev. Spearman (macro)"),
    ("developability", "spearman_CDR_Length_mean", "Dev. CDR Length"),
    ("developability", "spearman_PSH_mean", "Dev. PSH"),
    ("developability", "spearman_PPC_mean", "Dev. PPC"),
    ("developability", "spearman_PNC_mean", "Dev. PNC"),
    ("developability", "spearman_SFvCSP_mean", "Dev. SFvCSP"),
    ("developability", "mse_original_scale_mean", "Dev. MSE (orig scale)"),
    # Contact map (kNN-from-ESM2 contacts; long-range is the meaningful one)
    ("contact_map", "auroc_mean", "Contact AUROC"),
    ("contact_map", "long_range_auroc_mean", "Contact AUROC (long)"),
    ("contact_map", "long_range_precision_at_L_mean", "Contact P@L (long)"),
    ("contact_map", "long_range_precision_at_L5_mean", "Contact P@L/5 (long)"),
    ("contact_map", "medium_long_auroc_mean", "Contact AUROC (med+long)"),
    ("contact_map", "precision_at_L_mean", "Contact P@L (all)"),
    # Structure probe (Hewitt-Manning linear probe on Calpha distances)
    ("structure_probe", "spearman_distance_mean", "StructProbe Spearman"),
    ("structure_probe", "contact_precision_at_L_mean", "StructProbe P@L"),
    ("structure_probe", "rmse_distance_angstrom_mean", "StructProbe RMSE (Å)"),
]

_ATTENTION_METRICS = [
    ("attn_entropy_mean", "Attn entropy (mean)"),
    ("attn_entropy_layer0", "Attn entropy L0"),
    ("attn_entropy_layer5", "Attn entropy L5"),
    ("attn_entropy_layer11", "Attn entropy L11"),
]


def _get_metric(exp: ExperimentResult, source: str, key: str) -> float | None:
    """Retrieve a metric value from an ExperimentResult."""
    if source == "eval" and exp.eval_metrics:
        return exp.eval_metrics.get(key)
    if source == "zeroshot" and exp.zeroshot_metrics:
        return exp.zeroshot_metrics.get(key)
    if source == "mutation" and exp.mutation_metrics:
        return exp.mutation_metrics.get(key)
    if source == "attention":
        attn = getattr(exp, "attention_metrics", None)
        if attn:
            return attn.get(key)
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
    for key, label in _ATTENTION_METRICS:
        metric_specs.append(("attention", key, label))

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
    """Generate a Markdown summary of all experiments.

    Sections, in order:
      Pretraining → Zero-Shot → Mutation Benchmark → Downstream
      (per task: Paratope, Developability, Contact Map, Structure Probe)
      → Attention Analysis.

    Empty sections (no experiment has data) are silently dropped so the
    report doesn't show rows full of dashes.
    """
    names = sorted(experiments.keys())
    if not names:
        return "No experiments found."

    sep = "|--------|" + "|".join("-------:" for _ in names) + "|"
    header = "| Metric | " + " | ".join(names) + " |"

    lines: list[str] = [
        "# Evaluation Summary",
        "",
        f"**Experiments:** {', '.join(names)}",
        "",
    ]

    def _emit_section(
        title: str,
        rows: list[tuple[str, str, str]],
        getter,
    ) -> None:
        """Append a section if any of `rows` has data for any experiment."""
        if not any(getter(experiments[n], r[0], r[1]) is not None
                   for r in rows for n in names):
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.extend([header, sep])
        for source, key, label in rows:
            vals: list[str] = []
            for n in names:
                v = getter(experiments[n], source, key)
                if v is None:
                    vals.append("--")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append(f"| {label} | " + " | ".join(vals) + " |")
        lines.append("")

    _emit_section(
        "Pretraining Metrics",
        [("eval", k, l) for k, l in _PRETRAINING_METRICS],
        _get_metric,
    )
    _emit_section(
        "Zero-Shot Metrics",
        [("zeroshot", k, l) for k, l in _ZEROSHOT_METRICS],
        _get_metric,
    )
    _emit_section(
        "Mutation Benchmark",
        [("mutation", k, l) for k, l in _MUTATION_METRICS],
        _get_metric,
    )

    # Downstream: split by task so the table doesn't become unreadably wide
    # when we add the per-target developability columns and contact_map
    # long-range metrics.
    by_task: dict[str, list[tuple[str, str, str]]] = {}
    for task, key, label in _DOWNSTREAM_METRICS:
        by_task.setdefault(task, []).append((f"ds_{task}", key, label))

    task_titles = {
        "paratope": "Paratope (TDC SAbDab_Liberis)",
        "developability": "Developability (TDC TAP)",
        "contact_map": "Contact Map (kNN-from-ESM2)",
        "structure_probe": "Structure Probe (Hewitt-Manning, AB-Bind H/L)",
        "binding": "Binding (CoV-AbDab)",
    }
    for task, rows in by_task.items():
        _emit_section(
            f"Downstream — {task_titles.get(task, task)}",
            rows,
            _get_metric,
        )

    _emit_section(
        "Attention Analysis",
        [("attention", k, l) for k, l in _ATTENTION_METRICS],
        _get_metric,
    )

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
    """Generate grouped bar charts comparing key metrics across experiments.

    One PNG per metric group. ``output_path`` is treated as a stem: passing
    ``comparison_outputs/metric_comparison.png`` writes
    ``metric_comparison_mlm_accuracy.png``, ``metric_comparison_perplexity.png``,
    ``metric_comparison_cdr_infilling.png``, and ``metric_comparison_downstream.png``
    in the same directory.
    """
    names = sorted(experiments.keys())
    if not names:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem
    suffix = output_path.suffix or ".png"
    parent = output_path.parent

    # (filename slug, display title, [(source, key, label), ...])
    metric_groups: list[tuple[str, str, list[tuple[str, str, str]]]] = [
        (
            "mlm_accuracy",
            "MLM Accuracy",
            [("eval", k, l) for k, l in _PRETRAINING_METRICS],
        ),
        (
            "perplexity",
            "Perplexity",
            [("zeroshot", k, l) for k, l in _ZEROSHOT_METRICS if "perplexity" in k],
        ),
        (
            "cdr_infilling",
            "CDR Infilling",
            [("zeroshot", k, l) for k, l in _ZEROSHOT_METRICS if "infill" in k],
        ),
        (
            "infilling_quality",
            "Infilling AA Distribution (JSD)",
            [("zeroshot", k, l) for k, l in _ZEROSHOT_METRICS if k.endswith("_jsd")],
        ),
        (
            "mutation",
            "Mutation Effects (per-complex)",
            [("mutation", k, l) for k, l in _MUTATION_METRICS
             if "spearman" in k or "auroc" in k],
        ),
        (
            "paratope",
            "Paratope Prediction",
            [(f"ds_{t}", k, l) for t, k, l in _DOWNSTREAM_METRICS if t == "paratope"],
        ),
        (
            "developability",
            "Developability",
            [(f"ds_{t}", k, l) for t, k, l in _DOWNSTREAM_METRICS
             if t == "developability" and k != "mse_original_scale_mean"],
        ),
        (
            "contact_map",
            "Contact Map",
            [(f"ds_{t}", k, l) for t, k, l in _DOWNSTREAM_METRICS if t == "contact_map"],
        ),
        (
            "structure_probe",
            "Structure Probe",
            [(f"ds_{t}", k, l) for t, k, l in _DOWNSTREAM_METRICS
             if t == "structure_probe" and "rmse" not in k],
        ),
        (
            "attention",
            "Attention Entropy",
            [("attention", k, l) for k, l in _ATTENTION_METRICS],
        ),
    ]

    bar_width = 0.8 / max(1, len(names))
    any_plot_written = False

    for slug, title, specs in metric_groups:
        labels: list[str] = []
        data_matrix: list[list[float | None]] = []
        for source, key, label in specs:
            vals = [_get_metric(experiments[n], source, key) for n in names]
            if any(v is not None for v in vals):
                labels.append(label)
                data_matrix.append(vals)

        if not labels:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        metric_x = np.arange(len(labels))
        for exp_idx, name in enumerate(names):
            offset = (exp_idx - len(names) / 2 + 0.5) * bar_width
            heights = [
                row[exp_idx] if row[exp_idx] is not None else 0
                for row in data_matrix
            ]
            ax.bar(metric_x + offset, heights, bar_width, label=name, alpha=0.85)

        ax.set_title(title)
        ax.set_xticks(metric_x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out_file = parent / f"{stem}_{slug}{suffix}"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Metric comparison plot saved to %s", out_file)
        any_plot_written = True

    if not any_plot_written:
        logger.warning("No metric data for comparison plots")
