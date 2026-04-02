"""CLI: run zero-shot evaluations (PLL, infilling, perplexity) on a checkpoint.

Usage:
    python scripts/evaluate_zeroshot.py \
        --config configs/medium.yaml \
        --checkpoint models/checkpoints/uniform_medium \
        --output-dir evaluation_outputs/uniform_medium \
        --device cuda \
        --max-pll-sequences 500 \
        --max-infilling-samples 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import AntibodyDataset
from evaluation.infilling import InfillingEvaluator
from evaluation.infilling_quality import InfillingQualityAnalyzer
from evaluation.mlm_accuracy import MLMAccuracyEvaluator
from evaluation.pseudo_loglikelihood import compute_pll_batch
from masking import get_strategy
from training.config import load_config
from transformers import RoFormerForMaskedLM
from utils.seed import set_seed
from utils.tokenizer import load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation suite")
    parser.add_argument("--config", type=str, default="configs/medium.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-pll-sequences", type=int, default=500,
        help="Number of sequences for PLL scoring (each needs L forward passes)",
    )
    parser.add_argument(
        "--max-infilling-samples", type=int, default=1000,
        help="Number of samples for infilling evaluation",
    )
    parser.add_argument(
        "--pll-batch-size", type=int, default=64,
        help="Batch size for PLL masked copies within a single sequence",
    )
    parser.add_argument(
        "--skip-pll", action="store_true",
        help="Skip PLL scoring (slowest component)",
    )
    parser.add_argument(
        "--skip-infilling", action="store_true",
        help="Skip infilling evaluation",
    )
    parser.add_argument(
        "--skip-perplexity", action="store_true",
        help="Skip perplexity evaluation",
    )
    parser.add_argument(
        "--infilling-quality", action="store_true",
        help="Run infilling quality analysis (AA frequency comparison, JSD)",
    )
    parser.add_argument(
        "--max-infilling-quality-samples", type=int, default=500,
        help="Number of samples for infilling quality analysis",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(config.model.model_name)

    logger.info("Loading model from %s", args.checkpoint)
    model = RoFormerForMaskedLM.from_pretrained(args.checkpoint)

    logger.info("Loading eval dataset...")
    full_dataset = AntibodyDataset(
        data_path=config.data.processed_path,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
    )
    eval_size = int(len(full_dataset) * (1 - config.data.train_split))
    _, eval_dataset = torch.utils.data.random_split(
        full_dataset,
        [len(full_dataset) - eval_size, eval_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    all_metrics: dict = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "num_eval_samples": len(eval_dataset),
    }

    if not args.skip_infilling:
        logger.info("=== Infilling Evaluation ===")
        infilling_evaluator = InfillingEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
        )
        infilling_metrics = infilling_evaluator.evaluate(
            dataset=eval_dataset,
            max_samples=args.max_infilling_samples,
        )
        all_metrics.update(infilling_metrics)

    if not args.skip_pll:
        logger.info("=== Pseudo-Log-Likelihood Scoring ===")
        n_pll = min(args.max_pll_sequences, len(eval_dataset))
        sequences = []
        for i in range(n_pll):
            sample = eval_dataset[i]
            token_ids = sample["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            special = set(tokenizer.all_special_tokens)
            aa_seq = "".join(t for t in tokens if t not in special)
            sequences.append(aa_seq)

        pll_results = compute_pll_batch(
            model=model,
            tokenizer=tokenizer,
            sequences=sequences,
            device=args.device,
            batch_size=args.pll_batch_size,
        )
        pll_values = [r["pll"] for r in pll_results]
        pll_norm_values = [r["pll_normalized"] for r in pll_results]
        all_metrics["pll_mean"] = sum(pll_values) / len(pll_values)
        all_metrics["pll_normalized_mean"] = sum(pll_norm_values) / len(pll_norm_values)
        all_metrics["pll_num_sequences"] = n_pll

    if not args.skip_perplexity:
        logger.info("=== Region-Stratified Perplexity ===")
        strategy = get_strategy(
            config.masking.strategy,
            tokenizer=tokenizer,
            mask_prob=config.masking.mask_prob,
            mask_token_ratio=config.masking.mask_token_ratio,
            random_token_ratio=config.masking.random_token_ratio,
            **config.masking.params,
        )
        mlm_evaluator = MLMAccuracyEvaluator(
            model=model,
            tokenizer=tokenizer,
            strategy=strategy,
            device=args.device,
        )
        perplexity_metrics = mlm_evaluator.evaluate(
            dataset=eval_dataset,
            batch_size=args.batch_size,
        )
        for key, value in perplexity_metrics.items():
            if key.startswith("perplexity"):
                all_metrics[key] = value

    if args.infilling_quality:
        logger.info("=== Infilling Quality Analysis ===")
        quality_analyzer = InfillingQualityAnalyzer(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
        )
        quality_results = quality_analyzer.analyze(
            dataset=eval_dataset,
            max_samples=args.max_infilling_quality_samples,
        )
        all_metrics["infilling_quality"] = quality_results

    metrics_path = output_dir / "metrics_zeroshot.json"
    with metrics_path.open("w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Zero-shot metrics saved to %s", metrics_path)

    logger.info("Zero-shot evaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
