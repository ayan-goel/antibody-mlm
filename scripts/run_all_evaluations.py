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
from evaluation.attention_analysis import AttentionAnalyzer
from evaluation.pseudo_loglikelihood import compute_pll
from masking import get_strategy
from training.config import load_config
from utils.seed import set_seed
from utils.tokenizer import (
    is_paired_checkpoint,
    load_tokenizer,
    load_tokenizer_multispecific,
    tokenize_single_chain,
)

# Held-out split seed: every model uses the SAME generator seed for the
# train/eval random_split, regardless of its training-time `config.seed`.
# This guarantees the eval set is identical across models so cross-model
# metric comparisons are well-defined. Don't change without re-running
# every experiment.
EVAL_SPLIT_SEED = 42

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
    eval_dataset,
    device: str,
    batch_size: int,
) -> dict:
    """Run MLM accuracy and region-stratified evaluation.

    Always uses uniform masking as the *reference* strategy regardless of
    which strategy this model was trained with. Comparing each model's
    eval accuracy under its own training-time masking is unfair: span /
    cdr / germline / interface all skew toward harder positions, making
    those models look worse than uniform-trained models even when their
    representations are equally good.
    """
    logger.info("  [MLM] Running MLM accuracy evaluation (uniform reference masking)...")
    set_seed(42)  # ensure mask draws are deterministic across reruns
    strategy = get_strategy(
        "uniform",
        tokenizer=tokenizer,
        mask_prob=0.15,
        mask_token_ratio=0.8,
        random_token_ratio=0.1,
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
    set_seed(42)
    evaluator = InfillingEvaluator(model=model, tokenizer=tokenizer, device=device)
    return evaluator.evaluate(dataset=eval_dataset, max_samples=max_samples)


def _run_pll(
    model: torch.nn.Module, tokenizer, eval_dataset, device: str,
    max_sequences: int, batch_size: int,
) -> dict:
    """Run PLL scoring on the held-out set.

    Always uses the pre-tokenized path so that single-chain and paired
    models are scored on the exact same tokens the dataset produced.
    The previous single-chain path round-tripped through a string and
    called ``sanitize_sequence``, which silently dropped non-standard
    amino acids — making single-chain and paired PLL means incomparable.
    """
    logger.info("  [PLL] Running pseudo-log-likelihood scoring...")
    set_seed(42)
    n_pll = min(max_sequences, len(eval_dataset))

    pll_results = []
    for i in range(n_pll):
        sample = eval_dataset[i]
        ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        mask = torch.tensor(sample["attention_mask"], dtype=torch.long)
        result = compute_pll(
            model, tokenizer, device=device, batch_size=batch_size,
            pre_tokenized_ids=ids, pre_tokenized_mask=mask,
        )
        pll_results.append(result)

    pll_values = [r["pll"] for r in pll_results]
    pll_norm = [r["pll_normalized"] for r in pll_results]
    return {
        "pll_mean": sum(pll_values) / len(pll_values),
        "pll_normalized_mean": sum(pll_norm) / len(pll_norm),
        "pll_num_sequences": n_pll,
    }


def _run_perplexity(
    model: torch.nn.Module, tokenizer, eval_dataset, device: str, batch_size: int,
) -> dict:
    """Run region-stratified perplexity using a uniform reference masking.

    See `_run_mlm_eval` for the rationale: cross-model perplexity is only
    meaningful when every model is masked under the same distribution.
    """
    logger.info("  [Perplexity] Running region-stratified perplexity (uniform reference masking)...")
    set_seed(42)
    strategy = get_strategy(
        "uniform",
        tokenizer=tokenizer,
        mask_prob=0.15,
        mask_token_ratio=0.8,
        random_token_ratio=0.1,
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
    set_seed(42)
    analyzer = InfillingQualityAnalyzer(model=model, tokenizer=tokenizer, device=device)
    return analyzer.analyze(dataset=eval_dataset, max_samples=max_samples)


def _run_attention_analysis(
    model: torch.nn.Module, tokenizer, eval_dataset, device: str,
    max_samples: int, coords_data: list | None = None,
) -> dict:
    """Run zero-shot attention entropy, head importance, and contact correlation."""
    logger.info("  [Attention] Running attention perturbation analysis...")
    set_seed(42)
    analyzer = AttentionAnalyzer(
        model=model, tokenizer=tokenizer, device=device,
        coords_data=coords_data, max_samples=max_samples,
    )
    return analyzer.evaluate(dataset=eval_dataset)


def _wildtype_marginal_score(
    model: torch.nn.Module,
    tokenizer,
    wt_seq: str,
    mut_seq: str,
    device: str,
    max_pos: int,
) -> float | None:
    """Score a mutation via the ESM wildtype-marginal recipe.

    Mask the mutation positions in the WILDTYPE sequence in a single
    forward pass and compute the model's "destabilization score":

      Σ_i [log p(wt_aa[i] | wt_ctx_masked) − log p(mut_aa[i] | wt_ctx_masked)]

    Sign convention: HIGHER score = model thinks the mutation is more
    destabilizing (i.e. wildtype is more likely than the mutant). With
    AB-Bind's convention (ΔΔG > 0 = destabilizing) we therefore expect
    a POSITIVE spearman correlation between score and ΔΔG when the
    model is well calibrated.

    This is the standard mutation-effect metric for masked LMs (Meier et al.,
    NeurIPS 2021) and is much less noisy than the full-PLL difference for
    multi-point mutations because it isolates the per-position log-likelihood
    ratio at the actual mutation sites.

    Works for both standard and paired (multispecific) tokenizers: the
    AA → token position mapping is derived from ``special_tokens_mask``
    rather than assuming a fixed ``[CLS]``-only prefix, so paired models
    see the ``[CLS][MOD1][H] VH [SEP]`` framing they were trained on.

    ``max_pos`` is the model's ``max_position_embeddings`` (the function
    derives the framing-aware AA cap internally).

    Returns ``None`` if every mutation position lies past the truncation
    boundary (mutation effectively erased by length truncation).
    """
    if len(wt_seq) != len(mut_seq):
        return None  # indels not supported

    positions = [i for i in range(len(wt_seq)) if wt_seq[i] != mut_seq[i]]
    if not positions:
        return None

    # Reserve space for special tokens that the tokenizer prepends.
    # Standard: [CLS] ... [SEP]                  → 2 specials
    # Paired:   [CLS][MOD1][H] ... [SEP]         → 4 specials (heavy-only mode)
    additional = tokenizer.additional_special_tokens or []
    num_special = 4 if "[MOD1]" in additional else 2
    max_aa = max_pos - num_special

    # Reject the WHOLE record if ANY mutation site falls past the truncation
    # window — partially-scored multi-point mutations would silently inject
    # biased scores into the per-complex rank (a 3-mutation record reduced
    # to 2 sites is on a different scale than a fully-scored 3-mutation one).
    if any(p >= max_aa for p in positions):
        return None

    wt_seq_t = wt_seq[:max_aa]

    # Use tokenize_single_chain so paired models get [MOD1][H] framing.
    encoding = tokenize_single_chain(tokenizer, wt_seq_t, max_length=max_pos)
    input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)
    special_tokens_mask = encoding["special_tokens_mask"]

    # Build AA position → token position map by walking the special-tokens
    # mask. This is robust to either single-chain ([CLS] aa... [SEP]) or
    # paired ([CLS][MOD1][H] aa... [SEP]) framing, and any future variant.
    aa_to_token: list[int] = [
        tok_pos for tok_pos, is_special in enumerate(special_tokens_mask) if not is_special
    ]
    if len(aa_to_token) < len(wt_seq_t):
        # Tokenizer dropped some AAs (e.g. unexpected truncation); bail.
        return None

    masked_ids = input_ids.clone()
    for p in positions:
        masked_ids[aa_to_token[p]] = tokenizer.mask_token_id

    masked_ids = masked_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_ids=masked_ids, attention_mask=attention_mask).logits
    log_probs = torch.log_softmax(logits[0], dim=-1)

    score = 0.0
    for p in positions:
        wt_id = tokenizer.convert_tokens_to_ids(wt_seq_t[p])
        mut_id = tokenizer.convert_tokens_to_ids(mut_seq[p])
        if wt_id == tokenizer.unk_token_id or mut_id == tokenizer.unk_token_id:
            continue
        tok_pos = aa_to_token[p]
        # log p(wt) - log p(mut): high when model strongly prefers wildtype.
        score += float(log_probs[tok_pos, wt_id].item() - log_probs[tok_pos, mut_id].item())
    return score


def _run_mutations(
    model: torch.nn.Module, tokenizer, device: str, data_dir: str, batch_size: int,
) -> dict:
    """Run zero-shot mutation benchmarking with AB-Bind.

    Uses wildtype-marginal scoring (one forward pass per record, scores
    only the mutation positions). This is the standard ESM-style recipe
    and is dramatically faster + lower-noise than the previous full-PLL
    difference approach, especially for multi-point mutations where
    full-PLL accumulates O(K²) noise from non-mutated positions.
    """
    logger.info("  [Mutations] Running AB-Bind mutation benchmark (wildtype-marginal)...")
    from data.benchmarks.ab_bind import download_ab_bind, load_ab_bind
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score

    download_ab_bind(data_dir)
    records = load_ab_bind(data_dir)
    if not records:
        logger.warning("  No mutation records loaded, skipping")
        return {}

    max_pos = getattr(model.config, "max_position_embeddings", 256)
    # Reserve special-token slots: standard tokenizers add [CLS]+[SEP]
    # (2 tokens), multispecific tokenizers in heavy-only mode add
    # [CLS][MOD1][H]...[SEP] (4 tokens). Used here only for the
    # `skipped_long` warning; the per-record truncation check lives
    # inside _wildtype_marginal_score.
    additional = tokenizer.additional_special_tokens or []
    num_special = 4 if "[MOD1]" in additional else 2
    max_aa = max_pos - num_special
    skipped_long = 0
    for rec in records:
        if len(rec["wildtype_seq"]) > max_aa or len(rec["mutant_seq"]) > max_aa:
            skipped_long += 1
    if skipped_long:
        logger.info(
            "  %d/%d records have sequences > %d AA; mutations past pos %d will be skipped",
            skipped_long, len(records), max_aa, max_aa,
        )

    model.eval()
    complex_results: dict[str, list[dict]] = defaultdict(list)
    n_errors = 0
    n_skipped_truncation = 0

    for i, rec in enumerate(records):
        try:
            score = _wildtype_marginal_score(
                model, tokenizer,
                rec["wildtype_seq"], rec["mutant_seq"],
                device, max_pos,
            )
            if score is None:
                n_skipped_truncation += 1
                continue
            complex_results[rec["pdb_id"]].append({
                "ddg": rec["ddg"], "delta_pll": score,
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

    if n_skipped_truncation:
        logger.info(
            "  Skipped %d records whose mutation positions all fell past the truncation boundary",
            n_skipped_truncation,
        )

    if n_errors:
        logger.info("  Mutation benchmark: %d errors encountered", n_errors)

    import math
    from statistics import mean, median

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
            if not math.isnan(rho):
                entry["spearman_rho"] = float(rho)
        # Per-complex AUROC: requires both labels present in this complex.
        # The score (deltas) is wildtype-marginal log-likelihood ratio with
        # the convention HIGHER = model predicts destabilizing, which lines
        # up directly with binary_label=1 (destabilizing). No negation.
        binary = [1 if d > 0 else 0 for d in ddgs]
        if len(set(binary)) == 2:
            try:
                entry["binary_auroc"] = float(
                    roc_auc_score(binary, deltas)
                )
            except ValueError:
                pass
        per_complex[pdb_id] = entry

    summary: dict = {"n_complexes": len(complex_results), "n_mutants_total": len(all_ddg)}

    # PRIMARY metrics: per-complex aggregations.
    # The pooled metrics below are misleading because they're dominated by
    # between-complex variance (Simpson's paradox) and don't reflect the
    # model's actual within-complex predictive power.
    per_c_spearmans = [
        e["spearman_rho"] for e in per_complex.values() if "spearman_rho" in e
    ]
    per_c_aurocs = [
        e["binary_auroc"] for e in per_complex.values() if "binary_auroc" in e
    ]
    if per_c_spearmans:
        summary["mean_per_complex_spearman_rho"] = float(mean(per_c_spearmans))
        summary["median_per_complex_spearman_rho"] = float(median(per_c_spearmans))
        summary["n_complexes_with_spearman"] = len(per_c_spearmans)
    if per_c_aurocs:
        summary["mean_per_complex_auroc"] = float(mean(per_c_aurocs))
        summary["median_per_complex_auroc"] = float(median(per_c_aurocs))
        summary["n_complexes_with_auroc"] = len(per_c_aurocs)

    # SECONDARY metrics: pooled across all mutations regardless of complex.
    # Kept for backward compatibility, but treat with caution -- inflated by
    # between-complex baseline differences.
    if len(all_ddg) >= 3:
        rho, _ = spearmanr(all_ddg, all_delta)
        r, _ = pearsonr(all_ddg, all_delta)
        summary["pooled_spearman_rho"] = float(rho)
        summary["pooled_pearson_r"] = float(r)
        # Backward-compat aliases (pre-fix names)
        summary["overall_spearman_rho"] = float(rho)
        summary["overall_pearson_r"] = float(r)
    binary_labels = [1 if d > 0 else 0 for d in all_ddg]
    if len(set(binary_labels)) == 2:
        # Score is "destabilization prediction"; higher → destabilizing.
        pooled_auroc = float(roc_auc_score(binary_labels, all_delta))
        summary["pooled_binary_auroc"] = pooled_auroc
        summary["binary_auroc"] = pooled_auroc  # backward-compat alias

    summary["per_complex"] = per_complex
    summary["n_errors"] = n_errors
    return summary


def _run_downstream(
    experiment_name: str,
    checkpoint: str,
    downstream_configs: list[str],
    device: str,
    existing_results: dict[str, dict] | None = None,
    on_task_complete: callable | None = None,
) -> dict[str, dict]:
    """Run all downstream tasks for a checkpoint.

    ``existing_results`` lets a resumed run skip tasks that already
    completed (non-empty dicts without an error marker).
    ``on_task_complete`` is invoked after each task finishes so the caller
    can persist incremental progress.
    """
    results: dict[str, dict] = dict(existing_results or {})
    for cfg_path in downstream_configs:
        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            logger.warning("  Downstream config not found: %s", cfg_path)
            continue

        base_config = load_downstream_config(cfg_path)
        task_name = base_config.task
        existing = results.get(task_name)
        if (isinstance(existing, dict) and not existing.get("error") and existing):
            logger.info("  [Downstream] Task '%s' already complete, skipping", task_name)
            continue

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
        if on_task_complete is not None:
            on_task_complete(results)
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

    # Guard against the experiments registry wiring a paired checkpoint to a
    # single-chain config (or vice versa). The two paths use different
    # tokenizers and embedding tables; mismatching them silently produces
    # nonsense metrics rather than crashing, so detect it loudly here.
    ckpt_is_paired = is_paired_checkpoint(checkpoint_path)
    if ckpt_is_paired != config.data.paired:
        raise ValueError(
            f"Mismatch between checkpoint and config for experiment '{name}': "
            f"is_paired_checkpoint('{checkpoint_path}') = {ckpt_is_paired} "
            f"but config.data.paired = {config.data.paired}. "
            f"Check the experiments registry."
        )

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
        generator=torch.Generator().manual_seed(EVAL_SPLIT_SEED),
    )

    # Set up the output path and load any existing results so crashed/killed
    # runs can resume from where they stopped. The merge logic at the bottom
    # of this function preserves valid sections and overwrites only the ones
    # we've re-run.
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "all_metrics.json"

    if out_path.exists():
        with out_path.open() as f:
            all_metrics: dict = json.load(f)
    else:
        all_metrics = {}
    all_metrics.update({
        "experiment": name,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "num_eval_samples": len(eval_dataset),
    })

    def _save_progress() -> None:
        """Write the current all_metrics state to disk for incremental progress."""
        with out_path.open("w") as f:
            json.dump(all_metrics, f, indent=2, default=str)

    _save_progress()  # create the file early so progress is visible

    # Extract coords data for attention–contact correlation analysis.
    # Only available when the dataset has structure annotations (e.g.
    # structure_medium, hybrid_curriculum_medium). For Subset wrappers
    # produced by random_split, access the underlying dataset's coords.
    _base_dataset = getattr(eval_dataset, "dataset", eval_dataset)
    coords_data = getattr(_base_dataset, "coords", None)

    eval_sections: list[tuple[str, bool, callable]] = [
        ("mlm", not args.skip_mlm, lambda: _run_mlm_eval(
            model, tokenizer, eval_dataset, args.device, args.batch_size,
        )),
        ("infilling", not args.skip_infilling, lambda: _run_infilling(
            model, tokenizer, eval_dataset, args.device, args.max_infilling_samples,
        )),
        ("pll", not args.skip_pll, lambda: _run_pll(
            model, tokenizer, eval_dataset, args.device,
            args.max_pll_sequences, args.pll_batch_size,
        )),
        ("perplexity", not args.skip_perplexity, lambda: _run_perplexity(
            model, tokenizer, eval_dataset, args.device, args.batch_size,
        )),
        ("infilling_quality", not args.skip_infilling_quality, lambda: _run_infilling_quality(
            model, tokenizer, eval_dataset, args.device, args.max_infilling_quality_samples,
        )),
        ("attention_analysis", not args.skip_attention_analysis, lambda: _run_attention_analysis(
            model, tokenizer, eval_dataset, args.device,
            args.max_attention_samples, coords_data=coords_data,
        )),
        ("mutation_benchmark", not args.skip_mutations and not config.data.paired,
         lambda: _run_mutations(
            model, tokenizer, args.device, args.ab_bind_dir, args.pll_batch_size,
        )),
    ]

    if config.data.paired and not args.skip_mutations:
        logger.info(
            "  [Mutations] Skipping AB-Bind mutation benchmark for paired model "
            "(benchmark uses single-chain sequences only)"
        )

    for section_name, should_run, run_fn in eval_sections:
        if not should_run:
            continue
        # Skip sections that already succeeded in a previous run, identified
        # by the absence of an error marker. This lets you resume after a
        # crash without redoing expensive sections like mutations.
        existing_section = all_metrics.get(section_name)
        if (isinstance(existing_section, dict)
                and "error" not in existing_section
                and existing_section):
            logger.info("  [%s] Already present in all_metrics.json, skipping", section_name)
            continue
        try:
            all_metrics[section_name] = run_fn()
        except Exception:
            logger.exception("  Section '%s' failed -- continuing with remaining sections", section_name)
            all_metrics[section_name] = {"error": "section failed, see logs"}
        _save_progress()

    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        logger.warning("  torch.cuda.empty_cache() failed (CUDA context may be corrupted)")

    if not args.skip_downstream:
        # Resume: pass any previously-completed per-task downstream results
        # into _run_downstream so it can skip them. Each task-complete
        # callback updates all_metrics and flushes to disk.
        existing_downstream = all_metrics.get("downstream")
        if not isinstance(existing_downstream, dict) or "error" in existing_downstream:
            existing_downstream = {}

        def _on_downstream_task_complete(partial: dict) -> None:
            all_metrics["downstream"] = dict(partial)
            _save_progress()

        try:
            downstream_results = _run_downstream(
                name, checkpoint_path, downstream_configs, args.device,
                existing_results=existing_downstream,
                on_task_complete=_on_downstream_task_complete,
            )
            all_metrics["downstream"] = downstream_results
        except Exception:
            logger.exception("  Downstream tasks failed")
            all_metrics["downstream"] = {"error": "downstream failed, see logs"}
        _save_progress()

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
    parser.add_argument("--skip-infilling-quality", action="store_true")
    parser.add_argument("--skip-attention-analysis", action="store_true")
    parser.add_argument("--max-attention-samples", type=int, default=200)

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
