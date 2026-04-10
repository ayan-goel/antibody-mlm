"""Contact map prediction downstream task.

Pairwise binary classification: which residue pairs are in spatial contact?
Uses kNN annotations from predicted Calpha coordinates and evaluates with
AUROC and precision@L metrics (standard protein contact prediction metrics).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

from data.benchmarks.contact_map import load_contact_map_splits
from evaluation.downstream import register_task
from evaluation.downstream.base import BaseDownstreamTask
from evaluation.downstream.heads import ContactMapHead, MaskedBCEWithLogitsLoss
from utils.tokenizer import load_tokenizer_for_checkpoint

logger = logging.getLogger(__name__)


@register_task("contact_map")
class ContactMapTask(BaseDownstreamTask):
    """Pairwise contact prediction from frozen or fine-tuned embeddings."""

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        tokenizer = load_tokenizer_for_checkpoint(
            self.config.checkpoint, self.config.model_name,
        )
        return load_contact_map_splits(tokenizer)

    def build_head(self, hidden_size: int) -> nn.Module:
        return ContactMapHead(hidden_size, dropout=0.1)

    def compute_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor,
    ) -> dict[str, float]:
        preds_flat = predictions.view(-1).numpy()
        labels_flat = labels.view(-1).numpy()

        valid = labels_flat != -100
        if valid.sum() < 10:
            return {"auroc": 0.0, "precision_at_L": 0.0, "precision_at_L2": 0.0, "precision_at_L5": 0.0}

        preds_v = preds_flat[valid]
        labels_v = labels_flat[valid]

        # AUROC
        if len(np.unique(labels_v)) < 2:
            auroc = 0.0
        else:
            auroc = float(roc_auc_score(labels_v, preds_v))

        # Precision@L/k: sort by predicted score descending, take top N,
        # compute fraction that are true contacts.
        # L ≈ sqrt(2 * n_valid_pairs) (approximate sequence length)
        n_pairs = int(valid.sum())
        L = int(np.ceil((-1 + np.sqrt(1 + 8 * n_pairs)) / 2))

        sorted_idx = np.argsort(-preds_v)
        prec_L = _precision_at_k(labels_v, sorted_idx, L)
        prec_L2 = _precision_at_k(labels_v, sorted_idx, max(1, L // 2))
        prec_L5 = _precision_at_k(labels_v, sorted_idx, max(1, L // 5))

        return {
            "auroc": auroc,
            "precision_at_L": prec_L,
            "precision_at_L2": prec_L2,
            "precision_at_L5": prec_L5,
        }

    @property
    def loss_fn(self) -> nn.Module:
        return MaskedBCEWithLogitsLoss()

    @property
    def monitor_metric(self) -> str:
        return "precision_at_L"

    @property
    def higher_is_better(self) -> bool:
        return True

    def extract_labels(self, dataset: Dataset) -> list[Any]:
        return [dataset[i]["labels"] for i in range(len(dataset))]


def _precision_at_k(
    labels: np.ndarray, sorted_idx: np.ndarray, k: int,
) -> float:
    """Compute precision among the top-k predictions."""
    if k <= 0 or len(sorted_idx) == 0:
        return 0.0
    top_k = sorted_idx[:k]
    return float(labels[top_k].mean())
