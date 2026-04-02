"""CLI entry point: load a trained model and run the evaluation suite."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import AntibodyDataset
from evaluation.embeddings import PoolingStrategy, extract_embeddings, save_embeddings
from evaluation.mlm_accuracy import MLMAccuracyEvaluator
from evaluation.visualize import plot_pca, plot_umap
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
    parser = argparse.ArgumentParser(description="Evaluate antibody MLM model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_outputs",
        help="Directory to save evaluation results",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=5000,
        help="Max samples for embedding visualization (subsample for speed)",
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

    logger.info("=== MLM Accuracy ===")
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
    mlm_metrics = mlm_evaluator.evaluate(
        dataset=eval_dataset,
        batch_size=args.batch_size,
    )
    logger.info("MLM metrics: %s", mlm_metrics)

    logger.info("=== Embedding Extraction ===")
    n_embed = min(args.max_eval_samples, len(eval_dataset))
    embed_subset = torch.utils.data.Subset(eval_dataset, list(range(n_embed)))

    embeddings = extract_embeddings(
        model=model,
        dataset=embed_subset,
        tokenizer=tokenizer,
        pooling=PoolingStrategy.MEAN,
        batch_size=args.batch_size,
        device=args.device,
    )
    save_embeddings(embeddings, output_dir / "embeddings.npy")

    logger.info("=== Visualization ===")
    plot_umap(embeddings, output_dir / "umap.png", title="UMAP - Antibody Embeddings")
    plot_pca(embeddings, output_dir / "pca.png", title="PCA - Antibody Embeddings")

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "config": args.config,
                "num_eval_samples": len(eval_dataset),
                "num_embed_samples": n_embed,
                **mlm_metrics,
            },
            f,
            indent=2,
        )
    logger.info("Metrics saved to %s", metrics_path)

    logger.info("Evaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
