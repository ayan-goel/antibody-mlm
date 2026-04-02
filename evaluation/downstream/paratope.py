"""Paratope prediction downstream task.

Per-token binary classification: which residues contact antigen?
Uses TDC SAbDab_Liberis dataset and evaluates with AUPRC, AUROC,
F1, and MCC.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import Dataset

from data.benchmarks.paratope import compute_class_weight, load_paratope_splits
from evaluation.downstream import register_task
from evaluation.downstream.base import BaseDownstreamTask
from evaluation.downstream.heads import TokenClassificationHead
from utils.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE loss that ignores positions where labels == -100."""

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self._pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            logits = logits.squeeze(-1)

        valid = labels != -100
        if not valid.any():
            return logits.sum() * 0.0

        logits_v = logits[valid]
        labels_v = labels[valid].float()

        pw = torch.tensor(self._pos_weight, device=logits.device, dtype=logits.dtype)
        return nn.functional.binary_cross_entropy_with_logits(
            logits_v, labels_v, pos_weight=pw,
        )


def _find_youdens_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Find optimal threshold via Youden's J statistic on ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


@register_task("paratope")
class ParatopePredictionTask(BaseDownstreamTask):
    """Paratope prediction: per-residue binary classification."""

    _train_data: Dataset | None = None
    _cached_pos_weight: float | None = None

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        tokenizer = load_tokenizer(self.config.model_name)
        train, val, test = load_paratope_splits(tokenizer, max_length=160)
        self._train_data = train
        self._cached_pos_weight = compute_class_weight(train)
        return train, val, test

    def build_head(self, hidden_size: int) -> nn.Module:
        return TokenClassificationHead(hidden_size, num_labels=1, dropout=0.1)

    def compute_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor,
    ) -> dict[str, float]:
        if predictions.dim() == 3:
            predictions = predictions.squeeze(-1)

        preds_flat = predictions.view(-1)
        labels_flat = labels.view(-1)

        valid = labels_flat != -100
        preds_v = torch.sigmoid(preds_flat[valid]).numpy()
        labels_v = labels_flat[valid].numpy()

        if len(np.unique(labels_v)) < 2:
            return {"auroc": 0.0, "auprc": 0.0, "f1": 0.0, "mcc": 0.0}

        auroc = float(roc_auc_score(labels_v, preds_v))
        auprc = float(average_precision_score(labels_v, preds_v))

        threshold = _find_youdens_threshold(labels_v, preds_v)
        binary_preds = (preds_v >= threshold).astype(int)
        f1 = float(f1_score(labels_v, binary_preds, zero_division=0))
        mcc = float(matthews_corrcoef(labels_v, binary_preds))

        return {"auroc": auroc, "auprc": auprc, "f1": f1, "mcc": mcc}

    @property
    def loss_fn(self) -> nn.Module:
        pw = self._cached_pos_weight if self._cached_pos_weight is not None else 1.0
        return MaskedBCEWithLogitsLoss(pos_weight=pw)

    @property
    def monitor_metric(self) -> str:
        return "auprc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def extract_labels(self, dataset: Dataset) -> list[Any]:
        return [torch.tensor(dataset[i]["labels"]) for i in range(len(dataset))]
