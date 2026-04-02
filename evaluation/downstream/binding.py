"""Binding specificity downstream task.

Sequence-level binary classification: does this antibody neutralize SARS-CoV-2?
Uses CoV-AbDab dataset and evaluates with AUROC, AUPRC, macro-F1, and MCC.
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
)
from torch.utils.data import Dataset

from data.benchmarks.binding import compute_class_weights, load_binding_splits
from evaluation.downstream import register_task
from evaluation.downstream.base import BaseDownstreamTask
from evaluation.downstream.heads import SequenceClassificationHead
from utils.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


@register_task("binding")
class BindingSpecificityTask(BaseDownstreamTask):
    """SARS-CoV-2 neutralization prediction: sequence-level binary classification."""

    _class_weights: list[float] | None = None

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        tokenizer = load_tokenizer(self.config.model_name)
        train, val, test = load_binding_splits(tokenizer, max_length=160)
        self._class_weights = compute_class_weights(train)
        return train, val, test

    def build_head(self, hidden_size: int) -> nn.Module:
        return SequenceClassificationHead(hidden_size, num_labels=2, dropout=0.1)

    def compute_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor,
    ) -> dict[str, float]:
        probs = torch.softmax(predictions, dim=-1)[:, 1].numpy()
        labels_np = labels.numpy()

        if len(np.unique(labels_np)) < 2:
            return {"auroc": 0.0, "auprc": 0.0, "f1": 0.0, "mcc": 0.0}

        auroc = float(roc_auc_score(labels_np, probs))
        auprc = float(average_precision_score(labels_np, probs))

        binary_preds = (probs >= 0.5).astype(int)
        f1 = float(f1_score(labels_np, binary_preds, average="macro", zero_division=0))
        mcc = float(matthews_corrcoef(labels_np, binary_preds))

        return {"auroc": auroc, "auprc": auprc, "f1": f1, "mcc": mcc}

    @property
    def loss_fn(self) -> nn.Module:
        if self._class_weights is not None:
            weight = torch.tensor(self._class_weights, dtype=torch.float)
        else:
            weight = None
        return nn.CrossEntropyLoss(weight=weight)

    @property
    def monitor_metric(self) -> str:
        return "auroc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def extract_labels(self, dataset: Dataset) -> list[Any]:
        return [dataset[i]["labels"] for i in range(len(dataset))]
