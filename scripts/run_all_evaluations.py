"""Batch evaluation runner: run all evaluations for all experiments.

Loads each checkpoint once and runs MLM accuracy, zero-shot metrics,
mutation benchmarking, and downstream tasks. Results are merged into
a single all_metrics.json per experiment.

Usage:
    python scripts/run_all_evaluations.py
    python scripts/run_all_evaluations.py --experiments-yaml configs/experiments.yaml
    python scripts/run_all_evaluations.py --experiments uniform_medium --skip-downstream
    python scripts/run_all_evaluations.py --device cuda --skip-pll --skip-mutations
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from transformers import RoFormerForMaskedLM

from data.dataset import AntibodyDataset
from data.dataset_paired import PairedAntibodyDataset
from evaluation.downstream import DownstreamConfig, get_task, load_downstream_config
from evaluation.infilling import InfillingEvaluator
from evaluation.infilling_quality import InfillingQualityAnalyzer
from evaluation.mlm_accuracy import MLMAccuracyEvaluator
from evaluation.pseudo_loglikelihood import compute_pll, compute_pll_batch
from masking import get_strategy
from training.config import load_config
from utils.seed import set_seed
from utils.tokenizer import load_tokenizer, load_tokenizer_multispecific

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_experiments_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _run_mlm_eval(
    model: torch.nn.Module,
    tokenizer,
    config,
    eval_dataset,
    device: str,
    batch_size: int,
) -> dict:
    """Run MLM accuracy and region-stratified evaluation."""
    logger.info("  [MLM] Running MLM accuracy evaluation...")
    strategy = get_strategy(
        config.masking.strategy,
        tokenizer=tokenizer,
        mask_prob=config.masking.mask_prob,
        mask_token_ratio=config.masking.mask_token_ratio,
        random_token_ratio=config.masking.random_token_ratio,
        **config.masking.params,
    )
    evaluator = MLMAccuracyEvaluator(
        model=model, tokenizer=tokenizer, strategy=strategy, device=device,
    )
    return evaluator.evaluate(dataset=eval_dataset, batch_size=batch_size)


def _run_infilling(
    model: torch.nn.Module, tokenizer, eval_dataset, device: str, max_samples: int,
) -> dict:
    """Run CDR infilling evaluation."""
    logger.info("  [Infilling] Running infilling evaluation...")
    evaluator = InfillingEvaluator(model=model, tokenizer=tokenizer, device=device)
    return evaluator.evaluate(dataset=eval_dataset, max_samples=max_samples)


def _run_pll(
    model: torch.nn.Module, tokenizer, eval_dataset, device: str,
    max_sequences: int, batch_size: int,
) -> dict:
    """Run PLL scoring."""
    logger.info("  [PLL] Running pseudo-log-likelihood scoring...")
    n_pll = min(max_sequences, len(eval_dataset))
    sequences = []
    special = set(tokenizer.all_special_tokens)
    for i in range(n_pll):
        sample = eval_dataset[i]
        tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
        sequences.append("".join(t for t in tokens if t not in special))

    pll_results = compute_pll_batch(
        model=model, tokenizer=tokenizer, sequences=sequences,
        device=device, batch_size=batch_size,
    )
    pll_values = [r["pll"] for r in pll_results]
    pll_norm = [r["pll_normalized"] for r in pll_results]
    return {
        "pll_mean": sum(pll_values) / len(pll_values),
        "pll_normalized_mean": sum(pll_norm) / len(pll_norm),
        "pll_num_sequences": n_pll,
    }


def _run_perplexity(
    model: torch.nn.Module, tokenizer, config, eval_dataset, device: str, batch_size: int,
) -> dict:
    """Run region-stratified perplexity."""
    logger.info("  [Perplexity] Running region-stratified perplexity...")
    strategy = get_strategy(
        config.masking.strategy,
        tokenizer=tokenizer,
        mask_prob=config.masking.mask_prob,
        mask_token_ratio=config.masking.mask_token_ratio,
        random_token_ratio=config.masking.random_token_ratio,
        **config.masking.params,
    )
    evaluator = MLMAccuracyEvaluator(
        model=model, tokenizer=tokenizer, strategy=strategy, device=device,
    )
    metrics = evaluator.evaluate(dataset=eval_dataset, batch_size=batch_size)
    return {k: v for k, v in metrics.items() if k.startswith("perplexity")}


def _run_infilling_quality(
    model: torch.nn.Module, tokenizer, eval_dataset, device: str, max_samples: int,
) -> dict:
    """Run infilling quality analysis."""
    logger.info("  [InfillingQuality] Running AA frequency analysis...")
    analyzer = InfillingQualityAnalyzer(model=model, tokenizer=tokenizer, device=device)
    return analyzer.analyze(dataset=eval_dataset, max_samples=max_samples)


def _run_mutations(
    model: torch.nn.Module, tokenizer, device: str, data_dir: str, batch_size: int,
) -> dict:
    """Run zero-shot mutation benchmarking with AB-Bind."""
    logger.info("  [Mutations] Running AB-Bind mutation benchmark...")
    from data.benchmarks.ab_bind import download_ab_bind, load_ab_bind
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score

    download_ab_bind(data_dir)
    records = load_ab_bind(data_dir)
    if not records:
        logger.warning("  No mutation records loaded, skipping")
        return {}

    max_pos = getattr(model.config, "max_position_embeddings", 256)
    max_aa = max_pos - 2
    skipped_long = 0
    for rec in records:
        if len(rec["wildtype_seq"]) > max_aa or len(rec["mutant_seq"]) > max_aa:
            skipped_long += 1
    if skipped_long:
        logger.info(
            "  %d/%d records have sequences > %d AA; they will be truncated by compute_pll",
            skipped_long, len(records), max_aa,
        )

    complex_results: dict[str, list[dict]] = defaultdict(list)
    wt_pll_cache: dict[str, dict[str, float]] = {}
    n_errors = 0

    for i, rec in enumerate(records):
        try:
            cache_key = f"{rec['pdb_id']}_{rec['chain_id']}"
            if cache_key not in wt_pll_cache:
                wt_pll_cache[cache_key] = compute_pll(
                    model, tokenizer, rec["wildtype_seq"], device, batch_size,
                )
            wt_pll = wt_pll_cache[cache_key]
            mut_pll = compute_pll(
                model, tokenizer, rec["mutant_seq"], device, batch_size,
            )
            delta_pll = mut_pll["pll"] - wt_pll["pll"]
            complex_results[rec["pdb_id"]].append({
                "ddg": rec["ddg"], "delta_pll": delta_pll,
            })
        except RuntimeError as e:
            n_errors += 1
            if "CUDA" in str(e) or "device-side" in str(e):
                logger.error(
                    "  Fatal CUDA error at mutation %d (%s). "
                    "Aborting mutation benchmark to preserve GPU state.",
                    i, rec.get("mutation_str", "?"),
                )
                break
            logger.warning(
                "  Skipping mutation %d (%s): %s", i, rec.get("mutation_str", "?"), e,
            )
        except Exception as e:
            n_errors += 1
            logger.warning(
                "  Skipping mutation %d (%s): %s", i, rec.get("mutation_str", "?"), e,
            )

    if n_errors:
        logger.info("  Mutation benchmark: %d errors encountered", n_errors)

    all_ddg, all_delta = [], []
    per_complex: dict[str, dict] = {}
    for pdb_id, results in complex_results.items():
        ddgs = [r["ddg"] for r in results]
        deltas = [r["delta_pll"] for r in results]
        all_ddg.extend(ddgs)
        all_delta.extend(deltas)
        entry: dict = {"n_mutants": len(results)}
        if len(results) >= 3 and len(set(ddgs)) > 1:
            rho, _ = spearmanr(ddgs, deltas)
            entry["spearman_rho"] = float(rho)
        per_complex[pdb_id] = entry

    summary: dict = {"n_complexes": len(complex_results), "n_mutants_total": len(all_ddg)}
    if len(all_ddg) >= 3:
        rho, _ = spearmanr(all_ddg, all_delta)
        r, _ = pearsonr(all_ddg, all_delta)
        summary["overall_spearman_rho"] = float(rho)
        summary["overall_pearson_r"] = float(r)
    binary_labels = [1 if d > 0 else 0 for d in all_ddg]
    if len(set(binary_labels)) == 2:
        summary["binary_auroc"] = float(roc_auc_score(binary_labels, [-d for d in all_delta]))
    summary["per_complex"] = per_complex
    summary["n_errors"] = n_errors
    return summary


def _run_downstream(
    experiment_name: str,
    checkpoint: str,
    downstream_configs: list[str],
    device: str,
) -> dict[str, dict]:
    """Run all downstream tasks for a checkpoint."""
    results: dict[str, dict] = {}
    for cfg_path in downstream_configs:
        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            logger.warning("  Downstream config not found: %s", cfg_path)
            continue

        base_config = load_downstream_config(cfg_path)
        per_experiment_output = str(Path(base_config.output_dir) / experiment_name)
        config = DownstreamConfig(
            task=base_config.task,
            checkpoint=checkpoint,
            model_name=base_config.model_name,
            mode=base_config.mode,
            learning_rate=base_config.learning_rate,
            encoder_learning_rate=base_config.encoder_learning_rate,
            epochs=base_config.epochs,
            batch_size=base_config.batch_size,
            early_stopping_patience=base_config.early_stopping_patience,
            weight_decay=base_config.weight_decay,
            max_grad_norm=base_config.max_grad_norm,
            warmup_fraction=base_config.warmup_fraction,
            num_seeds=base_config.num_seeds,
            base_seed=base_config.base_seed,
            output_dir=per_experiment_output,
            device=device,
            num_workers=base_config.num_workers,
        )
        logger.info("  [Downstream] Running task=%s mode=%s", config.task, config.mode)
        try:
            task = get_task(config.task, config)
            task_results = task.run()
            results[config.task] = task_results
        except Exception:
            logger.exception("  Failed to run downstream task %s", config.task)
            results[config.task] = {"error": True, "task": config.task, "mode": config.mode}
    return results


def run_experiment(
    name: str,
    config_path: str,
    checkpoint_path: str,
    downstream_configs: list[str],
    args: argparse.Namespace,
) -> dict:
    """Run all evaluations for a single experiment."""
    logger.info("=" * 60)
    logger.info("Experiment: %s", name)
    logger.info("  Config: %s, Checkpoint: %s", config_path, checkpoint_path)
    logger.info("=" * 60)

    config = load_config(config_path)
    set_seed(config.seed)

    if config.data.paired:
        tokenizer = load_tokenizer_multispecific(config.model.model_name)
    else:
        tokenizer = load_tokenizer(config.model.model_name)

    model = RoFormerForMaskedLM.from_pretrained(checkpoint_path)
    model.to(args.device)
    model.eval()

    if config.data.paired:
        full_dataset = PairedAntibodyDataset(
            data_path=config.data.processed_path,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            paratope_path=config.data.paratope_path or None,
            interface_path=config.data.interface_path or None,
            germline_path=config.data.germline_path or None,
            bispecific=config.data.bispecific,
        )
    else:
        full_dataset = AntibodyDataset(
            data_path=config.data.processed_path,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            coords_path=config.data.coords_path or None,
            paratope_path=config.data.paratope_path or None,
            germline_path=config.data.germline_path or None,
        )
    eval_size = int(len(full_dataset) * (1 - config.data.train_split))
    _, eval_dataset = torch.utils.data.random_split(
        full_dataset,
        [len(full_dataset) - eval_size, eval_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    all_metrics: dict = {
        "experiment": name,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "num_eval_samples": len(eval_dataset),
    }

    eval_sections: list[tuple[str, bool, callable]] = [
        ("mlm", not args.skip_mlm, lambda: _run_mlm_eval(
            model, tokenizer, config, eval_dataset, args.device, args.batch_size,
        )),
        ("infilling", not args.skip_infilling, lambda: _run_infilling(
            model, tokenizer, eval_dataset, args.device, args.max_infilling_samples,
        )),
        ("pll", not args.skip_pll, lambda: _run_pll(
            model, tokenizer, eval_dataset, args.device,
            args.max_pll_sequences, args.pll_batch_size,
        )),
        ("perplexity", not args.skip_perplexity, lambda: _run_perplexity(
            model, tokenizer, config, eval_dataset, args.device, args.batch_size,
        )),
        ("infilling_quality", args.infilling_quality, lambda: _run_infilling_quality(
            model, tokenizer, eval_dataset, args.device, args.max_infilling_quality_samples,
        )),
        ("mutation_benchmark", not args.skip_mutations, lambda: _run_mutations(
            model, tokenizer, args.device, args.ab_bind_dir, args.pll_batch_size,
        )),
    ]

    for section_name, should_run, run_fn in eval_sections:
        if not should_run:
            continue
        try:
            all_metrics[section_name] = run_fn()
        except Exception:
            logger.exception("  Section '%s' failed -- continuing with remaining sections", section_name)
            all_metrics[section_name] = {"error": "section failed, see logs"}

    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        logger.warning("  torch.cuda.empty_cache() failed (CUDA context may be corrupted)")

    if not args.skip_downstream:
        try:
            downstream_results = _run_downstream(
                name, checkpoint_path, downstream_configs, args.device,
            )
            all_metrics["downstream"] = downstream_results
        except Exception:
            logger.exception("  Downstream tasks failed")
            all_metrics["downstream"] = {"error": "downstream failed, see logs"}

    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "all_metrics.json"

    if out_path.exists():
        with out_path.open() as f:
            existing = json.load(f)
        for key, value in all_metrics.items():
            if key in ("experiment", "config", "checkpoint", "num_eval_samples"):
                existing[key] = value
            elif value not in (None, {"error": "section failed, see logs"}):
                existing[key] = value
        all_metrics = existing

    with out_path.open("w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info("All metrics saved to %s", out_path)

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all evaluations for all experiments",
    )
    parser.add_argument(
        "--experiments-yaml", type=str, default="configs/experiments.yaml",
        help="Path to experiments registry YAML",
    )
    parser.add_argument(
        "--experiments", type=str, nargs="*", default=None,
        help="Specific experiment names to run (default: all)",
    )
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-pll-sequences", type=int, default=500)
    parser.add_argument("--max-infilling-samples", type=int, default=1000)
    parser.add_argument("--max-infilling-quality-samples", type=int, default=500)
    parser.add_argument("--pll-batch-size", type=int, default=8)
    parser.add_argument("--ab-bind-dir", type=str, default="data/ab_bind")

    parser.add_argument("--skip-mlm", action="store_true")
    parser.add_argument("--skip-infilling", action="store_true")
    parser.add_argument("--skip-pll", action="store_true")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--skip-mutations", action="store_true")
    parser.add_argument("--skip-downstream", action="store_true")
    parser.add_argument("--infilling-quality", action="store_true")

    args = parser.parse_args()

    registry = _load_experiments_yaml(args.experiments_yaml)
    experiment_defs = registry.get("experiments", {})
    downstream_configs = registry.get("downstream_tasks", [])

    if args.experiments:
        experiment_defs = {
            k: v for k, v in experiment_defs.items() if k in args.experiments
        }

    if not experiment_defs:
        logger.error("No experiments found in %s", args.experiments_yaml)
        return

    logger.info(
        "Running evaluations for %d experiments: %s",
        len(experiment_defs), list(experiment_defs.keys()),
    )

    all_results = {}
    for name, exp_def in experiment_defs.items():
        config_path = exp_def["config"]
        checkpoint_path = exp_def["checkpoint"]

        if not Path(checkpoint_path).exists():
            logger.warning("Checkpoint not found: %s -- skipping %s", checkpoint_path, name)
            continue

        try:
            results = run_experiment(
                name, config_path, checkpoint_path, downstream_configs, args,
            )
            all_results[name] = results
        except Exception:
            logger.exception("Experiment '%s' failed entirely -- moving to next", name)

    logger.info("=" * 60)
    logger.info("All experiments complete. %d/%d succeeded.", len(all_results), len(experiment_defs))
    logger.info("Results in: %s/<experiment>/all_metrics.json", args.output_dir)


if __name__ == "__main__":
    main()
