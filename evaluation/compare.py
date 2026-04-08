"""Model comparison utilities.

Discovers experiments from checkpoint and evaluation output directories,
builds side-by-side comparison tables, and generates overlay plots for
training curves and embedding visualizations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Aggregated results from a single experiment."""

    name: str
    training_summary: dict[str, Any] | None = None
    eval_metrics: dict[str, Any] | None = None
    zeroshot_metrics: dict[str, Any] | None = None
    mutation_metrics: dict[str, Any] | None = None
    downstream_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    all_metrics: dict[str, Any] | None = None
    checkpoint_dir: Path | None = None
    eval_dir: Path | None = None


def discover_experiments(
    checkpoints_dir: str | Path = "models/checkpoints",
    eval_dir: str | Path = "evaluation_outputs",
    downstream_dir: str | Path = "downstream_outputs",
) -> dict[str, ExperimentResult]:
    """Scan standard directories for completed experiments.

    Looks for training_summary.json in checkpoint dirs and metrics.json
    in evaluation output dirs. Experiments are matched by directory name
    (e.g. "uniform_medium"). Also discovers mutation benchmarks,
    downstream task results, and all_metrics.json from the batch runner.
    """
    checkpoints_dir = Path(checkpoints_dir)
    eval_dir = Path(eval_dir)
    downstream_dir = Path(downstream_dir)
    experiments: dict[str, ExperimentResult] = {}

    def _ensure(name: str) -> ExperimentResult:
        if name not in experiments:
            experiments[name] = ExperimentResult(name=name)
        return experiments[name]

    if checkpoints_dir.exists():
        for summary_path in sorted(checkpoints_dir.glob("*/training_summary.json")):
            name = summary_path.parent.name
            with summary_path.open() as f:
                training_summary = json.load(f)
            exp = _ensure(name)
            exp.training_summary = training_summary
            exp.checkpoint_dir = summary_path.parent

    if eval_dir.exists():
        for metrics_path in sorted(eval_dir.glob("*/metrics.json")):
            name = metrics_path.parent.name
            with metrics_path.open() as f:
                eval_metrics = json.load(f)
            exp = _ensure(name)
            exp.eval_metrics = eval_metrics
            exp.eval_dir = metrics_path.parent

        for zs_path in sorted(eval_dir.glob("*/metrics_zeroshot.json")):
            name = zs_path.parent.name
            with zs_path.open() as f:
                zeroshot_metrics = json.load(f)
            exp = _ensure(name)
            exp.zeroshot_metrics = zeroshot_metrics
            if exp.eval_dir is None:
                exp.eval_dir = zs_path.parent

        for mut_path in sorted(eval_dir.glob("*/mutation_benchmark.json")):
            name = mut_path.parent.name
            with mut_path.open() as f:
                mutation_metrics = json.load(f)
            exp = _ensure(name)
            exp.mutation_metrics = mutation_metrics

        for all_path in sorted(eval_dir.glob("*/all_metrics.json")):
            name = all_path.parent.name
            with all_path.open() as f:
                all_metrics = json.load(f)
            exp = _ensure(name)
            exp.all_metrics = all_metrics
            _unpack_all_metrics(exp, all_metrics)

    _discover_downstream(downstream_dir, experiments)

    logger.info("Discovered %d experiments: %s", len(experiments), list(experiments.keys()))
    return experiments


def _unpack_all_metrics(
    exp: ExperimentResult, all_metrics: dict[str, Any],
) -> None:
    """Unpack the sections of an ``all_metrics.json`` into ExperimentResult fields.

    ``run_all_evaluations.py`` writes every evaluation section (mlm,
    infilling, pll, perplexity, infilling_quality, mutation_benchmark,
    downstream) into a single ``all_metrics.json``. The report-generation
    code, however, reads from separate legacy fields (eval_metrics,
    zeroshot_metrics, mutation_metrics, downstream_metrics). This helper
    populates those legacy fields from the unified file so that
    ``build_comparison_table`` and ``_get_metric`` find the data.
    """
    mlm = all_metrics.get("mlm")
    if isinstance(mlm, dict) and "error" not in mlm:
        if exp.eval_metrics is None:
            exp.eval_metrics = {}
        exp.eval_metrics.update(mlm)

    # Zero-shot metrics: merge infilling, pll, perplexity, infilling_quality,
    # and also include MLM's perplexity_* keys (since report.py's
    # _ZEROSHOT_METRICS includes perplexity_overall/cdr/cdr3).
    zs: dict[str, Any] = {}
    for section_name in ("mlm", "infilling", "pll", "perplexity", "infilling_quality"):
        section = all_metrics.get(section_name)
        if isinstance(section, dict) and "error" not in section:
            zs.update(section)
    if zs:
        if exp.zeroshot_metrics is None:
            exp.zeroshot_metrics = {}
        exp.zeroshot_metrics.update(zs)

    mut = all_metrics.get("mutation_benchmark")
    if isinstance(mut, dict) and "error" not in mut:
        if exp.mutation_metrics is None:
            exp.mutation_metrics = {}
        exp.mutation_metrics.update(mut)

    downstream = all_metrics.get("downstream")
    if isinstance(downstream, dict) and "error" not in downstream:
        for task_name, task_results in downstream.items():
            if isinstance(task_results, dict) and not task_results.get("error"):
                exp.downstream_metrics[task_name] = task_results


def _discover_downstream(
    downstream_dir: Path,
    experiments: dict[str, ExperimentResult],
) -> None:
    """Scan downstream_outputs for task results and match to experiments by checkpoint path.

    Looks for results at ``downstream_outputs/<experiment>/<task>_<mode>/results.json``
    (the layout written by :meth:`BaseDownstreamTask.run`). Only fills in
    downstream metrics for experiments whose ``downstream_metrics`` dict
    hasn't already been populated from ``all_metrics.json`` — the batch
    runner's unified file takes priority.
    """
    if not downstream_dir.exists():
        return

    checkpoint_to_name: dict[str, str] = {}
    for name, exp in experiments.items():
        if exp.checkpoint_dir is not None:
            checkpoint_to_name[str(exp.checkpoint_dir)] = name

    for results_path in sorted(downstream_dir.glob("*/*/results.json")):
        task_dir_name = results_path.parent.name
        experiment_dir_name = results_path.parent.parent.name
        with results_path.open() as f:
            results = json.load(f)

        checkpoint = results.get("checkpoint", "")
        task_name = results.get("task", task_dir_name)

        matched_name: str | None = None
        # 1. Match by experiment directory name (most reliable).
        if experiment_dir_name in experiments:
            matched_name = experiment_dir_name
        # 2. Match by checkpoint path.
        if matched_name is None:
            for ckpt_path, exp_name in checkpoint_to_name.items():
                if checkpoint and (checkpoint in ckpt_path or ckpt_path in checkpoint):
                    matched_name = exp_name
                    break
        # 3. Last-resort: substring match of experiment names in checkpoint.
        if matched_name is None:
            for exp_name in experiments:
                if exp_name in checkpoint:
                    matched_name = exp_name
                    break

        if matched_name is not None:
            # all_metrics.json wins if it already populated this task.
            if task_name not in experiments[matched_name].downstream_metrics:
                experiments[matched_name].downstream_metrics[task_name] = results


_EVAL_KEYS = [
    "mlm_accuracy", "mlm_top5_accuracy",
    "mlm_accuracy_cdr", "mlm_top5_accuracy_cdr",
    "mlm_accuracy_cdr3", "mlm_top5_accuracy_cdr3",
    "mlm_accuracy_framework", "mlm_top5_accuracy_framework",
]

_ZEROSHOT_KEYS = [
    "pll_mean", "pll_normalized_mean",
    "perplexity_overall", "perplexity_cdr", "perplexity_cdr3",
    "perplexity_framework",
    "infill_cdr1_accuracy", "infill_cdr2_accuracy",
    "infill_cdr3_accuracy", "infill_cdr3_exact_match",
    "nterm_accuracy",
]

_MUTATION_KEYS = [
    "overall_spearman_rho", "overall_pearson_r", "binary_auroc",
]

_DOWNSTREAM_METRIC_KEYS = {
    "paratope": ["auroc_mean", "auprc_mean", "f1_mean", "mcc_mean"],
    "binding": ["auroc_mean", "auprc_mean", "f1_mean", "mcc_mean"],
    "developability": ["spearman_macro_mean"],
}


def build_comparison_table(
    experiments: dict[str, ExperimentResult],
) -> list[dict[str, Any]]:
    """Build a flat list of dicts suitable for tabular display or CSV export."""
    rows = []
    for name, exp in sorted(experiments.items()):
        row: dict[str, Any] = {"experiment": name}

        if exp.training_summary:
            info = exp.training_summary.get("experiment", {})
            row["strategy"] = info.get("masking_strategy", "")
            row["model_size"] = info.get("model_size", "")
            row["total_params"] = info.get("total_params", "")
            row["dataset"] = info.get("dataset", "")

            final = exp.training_summary.get("final_metrics", {})
            row["train_eval_loss"] = final.get("eval_loss")
            row["train_mlm_accuracy"] = final.get("eval_mlm_accuracy")

            history = exp.training_summary.get("eval_history", [])
            if history:
                row["total_train_steps"] = history[-1].get("step", 0)
                best = max(history, key=lambda e: e.get("eval_mlm_accuracy", 0))
                row["best_step"] = best.get("step", 0)
                row["best_train_mlm_accuracy"] = best.get("eval_mlm_accuracy")

        if exp.eval_metrics:
            for key in _EVAL_KEYS:
                if key in exp.eval_metrics:
                    row[key] = exp.eval_metrics[key]

        if exp.zeroshot_metrics:
            for key in _ZEROSHOT_KEYS:
                if key in exp.zeroshot_metrics:
                    row[key] = exp.zeroshot_metrics[key]

        if exp.mutation_metrics:
            for key in _MUTATION_KEYS:
                if key in exp.mutation_metrics:
                    row[f"mut_{key}"] = exp.mutation_metrics[key]

        for task_name, task_results in sorted(exp.downstream_metrics.items()):
            metric_keys = _DOWNSTREAM_METRIC_KEYS.get(
                task_name,
                [k for k in task_results if k.endswith("_mean") and isinstance(task_results[k], (int, float))],
            )
            for key in metric_keys:
                if key in task_results:
                    row[f"ds_{task_name}_{key}"] = task_results[key]

        rows.append(row)
    return rows


def format_table(rows: list[dict[str, Any]], float_fmt: str = ".4f") -> str:
    """Render a list of dicts as an aligned text table."""
    if not rows:
        return "(no experiments found)"

    all_keys = list(dict.fromkeys(k for row in rows for k in row))
    col_widths: dict[str, int] = {}
    for key in all_keys:
        values = [key] + [
            f"{row.get(key, ''):{float_fmt}}" if isinstance(row.get(key), float)
            else str(row.get(key, ""))
            for row in rows
        ]
        col_widths[key] = max(len(v) for v in values)

    def _fmt(val: Any) -> str:
        if isinstance(val, float):
            return f"{val:{float_fmt}}"
        return str(val) if val is not None else ""

    header = "  ".join(k.ljust(col_widths[k]) for k in all_keys)
    sep = "  ".join("-" * col_widths[k] for k in all_keys)
    lines = [header, sep]
    for row in rows:
        line = "  ".join(_fmt(row.get(k, "")).ljust(col_widths[k]) for k in all_keys)
        lines.append(line)
    return "\n".join(lines)


def plot_training_curves(
    experiments: dict[str, ExperimentResult],
    output_path: str | Path,
) -> None:
    """Overlay training curves (eval_loss and mlm_accuracy) for all experiments."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(16, 6))

    for name, exp in sorted(experiments.items()):
        if not exp.training_summary:
            continue
        history = exp.training_summary.get("eval_history", [])
        if not history:
            continue

        steps = [e["step"] for e in history]
        losses = [e.get("eval_loss") for e in history]
        accs = [e.get("eval_mlm_accuracy") for e in history]

        ax_loss.plot(steps, losses, label=name, linewidth=1.5)
        ax_acc.plot(steps, accs, label=name, linewidth=1.5)

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Eval Loss")
    ax_loss.set_title("Eval Loss vs Training Step")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("MLM Accuracy")
    ax_acc.set_title("MLM Accuracy vs Training Step")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Training curves saved to %s", output_path)


def plot_embedding_comparison(
    experiments: dict[str, ExperimentResult],
    output_path: str | Path,
) -> None:
    """Side-by-side UMAP plots for each experiment's embeddings."""
    import umap

    embed_data: list[tuple[str, np.ndarray]] = []
    for name, exp in sorted(experiments.items()):
        if exp.eval_dir is None:
            continue
        embed_path = exp.eval_dir / "embeddings.npy"
        if embed_path.exists():
            embed_data.append((name, np.load(embed_path)))

    if not embed_data:
        logger.warning("No embeddings found for comparison plot")
        return

    n = len(embed_data)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)

    for idx, (name, embeddings) in enumerate(embed_data):
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(embeddings)
        ax = axes[0, idx]
        ax.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.4)
        ax.set_title(name)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    fig.suptitle("Embedding Comparison (UMAP)", fontsize=14)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Embedding comparison saved to %s", output_path)
