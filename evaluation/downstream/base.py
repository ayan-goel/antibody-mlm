"""Abstract base class for downstream evaluation tasks.

Subclasses implement task-specific data loading, head construction,
loss, and metrics. The base class orchestrates multi-seed training
and aggregation for both probe and fine-tune modes.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from evaluation.downstream.config import DownstreamConfig
from evaluation.downstream.embedding_cache import (
    CachedEmbeddingDataset, cache_is_valid, extract_and_cache,
)
from evaluation.downstream.encoder import EncoderWrapper
from evaluation.downstream.trainer import DownstreamTrainer
from utils.seed import set_seed
from utils.tokenizer import is_paired_checkpoint, load_tokenizer_for_checkpoint

logger = logging.getLogger(__name__)


class BaseDownstreamTask(ABC):
    """Abstract base for all downstream evaluation tasks."""

    def __init__(self, config: DownstreamConfig) -> None:
        self.config = config

    @abstractmethod
    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        """Return (train_dataset, val_dataset, test_dataset).

        Each dataset item must be a dict with at least 'labels'.
        For probe mode: items should also work with AntibodyDataset format
        (input_ids, attention_mask, etc.) for embedding extraction.
        For finetune mode: items must include tokenized fields.
        """
        ...

    @abstractmethod
    def build_head(self, hidden_size: int) -> nn.Module:
        """Build a fresh (randomly initialized) task-specific head."""
        ...

    @abstractmethod
    def compute_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        """Compute task-specific metrics from raw predictions and labels."""
        ...

    @property
    @abstractmethod
    def loss_fn(self) -> nn.Module:
        """Return the loss function for this task."""
        ...

    @property
    def monitor_metric(self) -> str:
        """Metric name to monitor for early stopping (override per task)."""
        return "val_loss"

    @property
    def higher_is_better(self) -> bool:
        """Whether higher monitor_metric is better (override per task)."""
        return False

    def extract_labels(self, dataset: Dataset) -> list[Any]:
        """Extract labels from a dataset for cached-embedding mode.

        Default: reads 'labels' key from each item. Override if the
        dataset stores labels differently.
        """
        return [dataset[i]["labels"] for i in range(len(dataset))]

    def run(self) -> dict[str, Any]:
        """End-to-end: load data, train across seeds, evaluate, aggregate."""
        logger.info("=== Running task: %s (mode=%s) ===", self.config.task, self.config.mode)

        train_data, val_data, test_data = self.load_data()
        logger.info(
            "Data loaded: train=%d, val=%d, test=%d",
            len(train_data), len(val_data), len(test_data),
        )

        tokenizer = load_tokenizer_for_checkpoint(
            self.config.checkpoint, self.config.model_name,
        )
        if is_paired_checkpoint(self.config.checkpoint):
            logger.info(
                "Paired checkpoint detected: downstream inputs will be tokenized "
                "with the multispecific tokenizer ([CLS][MOD1][H] VH [SEP] format) "
                "so the model sees its trained framing tokens."
            )

        trainer = DownstreamTrainer(self.config)

        output_dir = Path(self.config.output_dir) / f"{self.config.task}_{self.config.mode}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.mode == "probe":
            cache_dir = output_dir / "embedding_cache"
            train_labels = self.extract_labels(train_data)
            val_labels = self.extract_labels(val_data)
            test_labels = self.extract_labels(test_data)

            train_cache = cache_dir / "train.pt"
            val_cache = cache_dir / "val.pt"
            test_cache = cache_dir / "test.pt"

            ckpt = self.config.checkpoint
            tok_type = "paired" if is_paired_checkpoint(ckpt) else "standard"
            needs_caching = not (
                cache_is_valid(train_cache, ckpt, tok_type)
                and cache_is_valid(val_cache, ckpt, tok_type)
                and cache_is_valid(test_cache, ckpt, tok_type)
            )
            if needs_caching:
                encoder = EncoderWrapper.from_checkpoint(
                    self.config.checkpoint, device=self.config.device,
                )
                if not cache_is_valid(train_cache, ckpt, tok_type):
                    logger.info("Caching train embeddings...")
                    extract_and_cache(encoder, train_data, tokenizer, train_cache,
                                      batch_size=self.config.batch_size, device=self.config.device,
                                      checkpoint_path=ckpt, tokenizer_type=tok_type)
                if not cache_is_valid(val_cache, ckpt, tok_type):
                    logger.info("Caching val embeddings...")
                    extract_and_cache(encoder, val_data, tokenizer, val_cache,
                                      batch_size=self.config.batch_size, device=self.config.device,
                                      checkpoint_path=ckpt, tokenizer_type=tok_type)
                if not cache_is_valid(test_cache, ckpt, tok_type):
                    logger.info("Caching test embeddings...")
                    extract_and_cache(encoder, test_data, tokenizer, test_cache,
                                      batch_size=self.config.batch_size, device=self.config.device,
                                      checkpoint_path=ckpt, tokenizer_type=tok_type)
                del encoder
                torch.cuda.empty_cache()
                logger.info("Encoder freed after caching embeddings")

            cached_train = CachedEmbeddingDataset(
                train_cache, train_labels, device=self.config.device,
            )
            cached_val = CachedEmbeddingDataset(
                val_cache, val_labels, device=self.config.device,
            )
            cached_test = CachedEmbeddingDataset(
                test_cache, test_labels, device=self.config.device,
            )
            hidden_size = cached_train.hidden_states.size(-1)
        else:
            encoder = EncoderWrapper.from_checkpoint(
                self.config.checkpoint, device=self.config.device,
            )
            hidden_size = encoder.hidden_size

        all_seed_metrics: list[dict[str, float]] = []

        for seed_idx in range(self.config.num_seeds):
            seed = self.config.base_seed + seed_idx
            set_seed(seed)
            logger.info("--- Seed %d/%d (seed=%d) ---", seed_idx + 1, self.config.num_seeds, seed)

            head = self.build_head(hidden_size)

            if self.config.mode == "probe":
                train_result = trainer.train_probe(
                    head=head,
                    train_data=cached_train,
                    val_data=cached_val,
                    loss_fn=self.loss_fn,
                    compute_metrics=self.compute_metrics,
                    monitor_metric=self.monitor_metric,
                    higher_is_better=self.higher_is_better,
                )
                test_metrics = self._evaluate_probe(head, cached_test)
            else:
                encoder_copy = EncoderWrapper.from_checkpoint(
                    self.config.checkpoint, device=self.config.device,
                )
                train_result = trainer.train_finetune(
                    encoder=encoder_copy,
                    head=head,
                    train_data=train_data,
                    val_data=val_data,
                    tokenizer=tokenizer,
                    loss_fn=self.loss_fn,
                    compute_metrics=self.compute_metrics,
                    monitor_metric=self.monitor_metric,
                    higher_is_better=self.higher_is_better,
                )
                test_metrics = self._evaluate_finetune(encoder_copy, head, test_data, tokenizer)

            test_metrics["best_epoch"] = train_result["best_epoch"]
            all_seed_metrics.append(test_metrics)
            logger.info("Seed %d test metrics: %s", seed, test_metrics)

        aggregated = self._aggregate_seeds(all_seed_metrics)
        aggregated["task"] = self.config.task
        aggregated["mode"] = self.config.mode
        aggregated["checkpoint"] = self.config.checkpoint
        aggregated["num_seeds"] = self.config.num_seeds

        results_path = output_dir / "results.json"
        with results_path.open("w") as f:
            json.dump(aggregated, f, indent=2)
        logger.info("Results saved to %s", results_path)
        logger.info("Aggregated metrics: %s", {
            k: v for k, v in aggregated.items()
            if isinstance(v, (int, float)) and not k.endswith("_std")
        })

        return aggregated

    def _evaluate_probe(
        self, head: nn.Module, test_data: CachedEmbeddingDataset
    ) -> dict[str, float]:
        """Evaluate a trained head on cached test embeddings."""
        head.eval()
        all_preds, all_labels = [], []
        n = len(test_data)
        batch_size = self.config.batch_size
        with torch.no_grad():
            for i in range(0, n, batch_size):
                hidden = test_data.hidden_states[i : i + batch_size]
                mask = test_data.attention_mask[i : i + batch_size]
                special_mask = test_data.special_tokens_mask[i : i + batch_size]
                labels = test_data.labels_tensor[i : i + batch_size]
                logits = DownstreamTrainer._forward_head(head, hidden, mask, special_mask)
                all_preds.append(logits.cpu())
                all_labels.append(labels.cpu())
        return self.compute_metrics(torch.cat(all_preds), torch.cat(all_labels))

    def _evaluate_finetune(
        self,
        encoder: nn.Module,
        head: nn.Module,
        test_data: Dataset,
        tokenizer: Any,
    ) -> dict[str, float]:
        """Evaluate encoder+head on raw test data."""
        from torch.utils.data import DataLoader
        from evaluation.downstream.collator import DownstreamCollator

        collator = DownstreamCollator(tokenizer=tokenizer)
        loader = DataLoader(
            test_data, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, collate_fn=collator,
        )
        encoder.eval()
        head.eval()
        all_preds, all_labels = [], []
        special_ids = torch.tensor(
            list(tokenizer.all_special_ids), device=self.config.device,
        )
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                special_tokens_mask = torch.isin(input_ids, special_ids).long()
                hidden = encoder(input_ids, attention_mask)
                logits = DownstreamTrainer._forward_head(head, hidden, attention_mask, special_tokens_mask)
                all_preds.append(logits.cpu())
                all_labels.append(labels.cpu())
        return self.compute_metrics(torch.cat(all_preds), torch.cat(all_labels))

    @staticmethod
    def _aggregate_seeds(
        seed_metrics: list[dict[str, float]],
    ) -> dict[str, Any]:
        """Compute mean and sample std across seeds for each metric.

        Uses ``ddof=1`` (sample std) which is the unbiased estimator and
        what most papers report. With N=3 seeds the default ``ddof=0``
        underreports the std by ~22%.
        """
        all_keys = {k for m in seed_metrics for k in m if isinstance(m[k], (int, float))}
        result: dict[str, Any] = {"per_seed": seed_metrics}
        for key in sorted(all_keys):
            values = [m[key] for m in seed_metrics if key in m]
            if values:
                result[f"{key}_mean"] = float(np.mean(values))
                result[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return result
