"""Lightweight downstream training harness.

Supports two modes:
  - probe:    Train a small head on cached (frozen-encoder) embeddings.
  - finetune: Train encoder + head end-to-end with separate LRs.

Uses a simple PyTorch loop with cosine LR schedule, early stopping,
and best-model checkpointing.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from evaluation.downstream.config import DownstreamConfig

logger = logging.getLogger(__name__)


def _cosine_with_warmup(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """LR multiplier: linear warmup then cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


class DownstreamTrainer:
    """Training harness for downstream probe and fine-tune tasks."""

    def __init__(self, config: DownstreamConfig) -> None:
        self.config = config
        self.device = config.device

    def train_probe(
        self,
        head: nn.Module,
        train_data: Dataset,
        val_data: Dataset,
        loss_fn: nn.Module,
        compute_metrics: Callable,
        monitor_metric: str = "loss",
        higher_is_better: bool = False,
    ) -> dict[str, Any]:
        """Train a head on cached embeddings (frozen encoder).

        Args:
            head: Prediction head module.
            train_data: CachedEmbeddingDataset for training.
            val_data: CachedEmbeddingDataset for validation.
            loss_fn: Loss function.
            compute_metrics: fn(predictions, labels) -> dict[str, float].
            monitor_metric: Which metric to monitor for early stopping.
            higher_is_better: Whether higher monitor_metric is better.

        Returns:
            Dict with best_val_metrics, best_epoch, and training_history.
        """
        head.to(self.device)
        if isinstance(loss_fn, nn.Module):
            loss_fn = loss_fn.to(self.device)

        # Probe data is GPU-resident: iterate via direct indexing instead of
        # a DataLoader to avoid per-batch CPU→GPU transfer overhead.
        n_train = len(train_data)
        batch_size = self.config.batch_size
        n_batches_per_epoch = (n_train + batch_size - 1) // batch_size

        optimizer = AdamW(
            head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps = n_batches_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_fraction)
        scheduler = LambdaLR(optimizer, _cosine_with_warmup(warmup_steps, total_steps))

        best_val_score = float("-inf") if higher_is_better else float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.config.epochs + 1):
            head.train()
            epoch_loss = 0.0
            n_batches = 0
            perm = torch.randperm(n_train, device=self.device)
            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                hidden = train_data.hidden_states[idx]
                mask = train_data.attention_mask[idx]
                labels = train_data.labels_tensor[idx]

                logits = self._forward_head(head, hidden, mask)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1

            val_metrics = self._evaluate(head, val_data, loss_fn, compute_metrics)
            val_metrics["train_loss"] = epoch_loss / max(n_batches, 1)
            history.append({"epoch": epoch, **val_metrics})

            score = val_metrics.get(monitor_metric, val_metrics["val_loss"])
            improved = (score > best_val_score) if higher_is_better else (score < best_val_score)
            if improved:
                best_val_score = score
                best_state = copy.deepcopy(head.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "  Epoch %d/%d  train_loss=%.4f  %s=%.4f  patience=%d/%d",
                    epoch, self.config.epochs,
                    val_metrics["train_loss"], monitor_metric, score,
                    patience_counter, self.config.early_stopping_patience,
                )

            if patience_counter >= self.config.early_stopping_patience > 0:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            head.load_state_dict(best_state)

        return {
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
            "training_history": history,
        }

    def train_finetune(
        self,
        encoder: nn.Module,
        head: nn.Module,
        train_data: Dataset,
        val_data: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        loss_fn: nn.Module,
        compute_metrics: Callable,
        monitor_metric: str = "loss",
        higher_is_better: bool = False,
    ) -> dict[str, Any]:
        """Train encoder + head end-to-end.

        Uses two parameter groups with separate learning rates.
        """
        from evaluation.downstream.collator import DownstreamCollator

        collator = DownstreamCollator(tokenizer=tokenizer)

        encoder.to(self.device)
        head.to(self.device)
        encoder.train()

        train_loader = DataLoader(
            train_data, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, collate_fn=collator,
        )
        val_loader = DataLoader(
            val_data, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, collate_fn=collator,
        )

        optimizer = AdamW([
            {"params": encoder.parameters(), "lr": self.config.encoder_learning_rate},
            {"params": head.parameters(), "lr": self.config.learning_rate},
        ], weight_decay=self.config.weight_decay)

        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_fraction)
        scheduler = LambdaLR(optimizer, _cosine_with_warmup(warmup_steps, total_steps))

        best_val_score = float("-inf") if higher_is_better else float("inf")
        best_encoder_state = None
        best_head_state = None
        best_epoch = 0
        patience_counter = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.config.epochs + 1):
            encoder.train()
            head.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                hidden = encoder(input_ids, attention_mask)
                logits = self._forward_head(head, hidden, attention_mask)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()),
                    self.config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1

            val_metrics = self._evaluate_finetune(
                encoder, head, val_loader, loss_fn, compute_metrics,
            )
            val_metrics["train_loss"] = epoch_loss / max(n_batches, 1)
            history.append({"epoch": epoch, **val_metrics})

            score = val_metrics.get(monitor_metric, val_metrics["val_loss"])
            improved = (score > best_val_score) if higher_is_better else (score < best_val_score)
            if improved:
                best_val_score = score
                best_encoder_state = copy.deepcopy(encoder.state_dict())
                best_head_state = copy.deepcopy(head.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "  Epoch %d/%d  train_loss=%.4f  %s=%.4f  patience=%d/%d",
                    epoch, self.config.epochs,
                    val_metrics["train_loss"], monitor_metric, score,
                    patience_counter, self.config.early_stopping_patience,
                )

            if patience_counter >= self.config.early_stopping_patience > 0:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if best_encoder_state is not None:
            encoder.load_state_dict(best_encoder_state)
        if best_head_state is not None:
            head.load_state_dict(best_head_state)

        return {
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
            "training_history": history,
        }

    _head_needs_mask_cache: dict[type, bool] = {}

    @staticmethod
    def _forward_head(
        head: nn.Module, hidden: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward through head, passing attention_mask if the head accepts it."""
        head_type = type(head)
        if head_type not in DownstreamTrainer._head_needs_mask_cache:
            import inspect
            sig = inspect.signature(head.forward)
            DownstreamTrainer._head_needs_mask_cache[head_type] = "attention_mask" in sig.parameters
        if DownstreamTrainer._head_needs_mask_cache[head_type]:
            return head(hidden, mask)
        return head(hidden)

    def _evaluate(
        self,
        head: nn.Module,
        data: Dataset,
        loss_fn: nn.Module,
        compute_metrics: Callable,
    ) -> dict[str, Any]:
        """Evaluate head on cached-embedding data via direct GPU indexing."""
        head.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        n_batches = 0
        n = len(data)
        batch_size = self.config.batch_size

        with torch.no_grad():
            for i in range(0, n, batch_size):
                hidden = data.hidden_states[i : i + batch_size]
                mask = data.attention_mask[i : i + batch_size]
                labels = data.labels_tensor[i : i + batch_size]

                logits = self._forward_head(head, hidden, mask)
                total_loss += loss_fn(logits, labels).item()
                n_batches += 1
                all_preds.append(logits.cpu())
                all_labels.append(labels.cpu())

        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_labels))
        metrics["val_loss"] = total_loss / max(n_batches, 1)
        return metrics

    def _evaluate_finetune(
        self,
        encoder: nn.Module,
        head: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        compute_metrics: Callable,
    ) -> dict[str, Any]:
        """Evaluate encoder+head on raw data."""
        encoder.eval()
        head.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                hidden = encoder(input_ids, attention_mask)
                logits = self._forward_head(head, hidden, attention_mask)
                total_loss += loss_fn(logits, labels).item()
                n_batches += 1
                all_preds.append(logits.cpu())
                all_labels.append(labels.cpu())

        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_labels))
        metrics["val_loss"] = total_loss / max(n_batches, 1)
        return metrics
