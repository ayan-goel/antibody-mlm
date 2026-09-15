"""Binding-specificity downstream task (CoV-AbDab).

Sequence-level binary classification: does this VH neutralize SARS-CoV-2?

This is the heavy-chain analogue of the specificity evaluation in Ng &
Briney (Patterns 2025), which reviewer #5.3 cites as the kind of functional
evidence missing from the paper. Their version classifies ~25k *paired*
VH/VL sequences as CoV-specific or not, with a single feedforward head on
frozen embeddings under 5-fold CV. Ours is VH-only because every model here
is heavy-chain-only, so the numbers are not directly comparable to theirs —
but the task is the same question, and it is the closest thing to a
functional binding readout the current models can support.

Splits are grouped by approximate clonotype (see
``data/benchmarks/binding.py``), so no clonal family straddles train and
test. Metrics mirror the reference: accuracy, AUROC, AUPRC, F1, MCC.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import Dataset

from data.benchmarks.binding import compute_class_weights, load_binding_splits
from evaluation.downstream import register_task
from evaluation.downstream._metric_utils import find_youdens_threshold
from evaluation.downstream.base import BaseDownstreamTask
from evaluation.downstream.heads import SequenceClassificationHead
from utils.tokenizer import load_tokenizer_for_checkpoint

logger = logging.getLogger(__name__)


@register_task("binding")
class BindingSpecificityTask(BaseDownstreamTask):
    """Binary SARS-CoV-2 neutralization prediction from VH sequence."""

    _class_weights: list[float] | None = None

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        tokenizer = load_tokenizer_for_checkpoint(
            self.config.checkpoint, self.config.model_name,
        )
        train, val, test = load_binding_splits(tokenizer, max_length=160)
        self._class_weights = compute_class_weights(train)
        return train, val, test

    def build_head(self, hidden_size: int) -> nn.Module:
        return SequenceClassificationHead(hidden_size, num_labels=2, dropout=0.1)

    @property
    def loss_fn(self) -> nn.Module:
        weights = self._class_weights or [1.0, 1.0]
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float))

    @property
    def monitor_metric(self) -> str:
        return "auprc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def fit_threshold(
        self, predictions: torch.Tensor, labels: torch.Tensor,
    ) -> None:
        """Fit the decision threshold on validation via Youden's J.

        Accuracy, F1 and MCC are threshold-sensitive and the classes are
        imbalanced (~70% positive), so the cut is tuned on held-out
        validation rather than left at 0.5 or read off the test set.
        """
        probs = self._positive_probs(predictions)
        y = labels.detach().cpu().numpy().astype(int)
        if len(np.unique(y)) == 2:
            self._fitted_threshold = float(find_youdens_threshold(y, probs))

    @staticmethod
    def _positive_probs(predictions: torch.Tensor) -> np.ndarray:
        logits = predictions.detach().cpu()
        if logits.dim() == 1:  # already a single score per example
            return torch.sigmoid(logits).numpy()
        return torch.softmax(logits.float(), dim=-1)[:, 1].numpy()

    def compute_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor,
    ) -> dict[str, Any]:
        probs = self._positive_probs(predictions)
        y = labels.detach().cpu().numpy().astype(int)
        preds = (probs >= self._fitted_threshold).astype(int)

        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y, preds)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, preds)),
            "threshold": float(self._fitted_threshold),
            "n": float(len(y)),
            "positive_rate": float(y.mean()) if len(y) else 0.0,
        }
        # Threshold-free metrics need both classes present in the split.
        if len(np.unique(y)) == 2:
            metrics["auroc"] = float(roc_auc_score(y, probs))
            metrics["auprc"] = float(average_precision_score(y, probs))
        else:
            metrics["auroc"] = 0.0
            metrics["auprc"] = 0.0
        return metrics
