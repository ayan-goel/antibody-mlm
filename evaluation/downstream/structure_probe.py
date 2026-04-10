"""Structure probe downstream task.

Hewitt & Manning (2019) structural probe: learn a linear transformation B
such that ||Bh_i - Bh_j||^2 ≈ d(i,j)^2 (squared Euclidean distance between
Calpha atoms).  Evaluates with Spearman correlation, contact precision,
and MSE.  Uses real PDB structures from the AB-Bind collection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import Dataset

from data.benchmarks.structure_probe import load_structure_probe_splits
from evaluation.downstream import register_task
from evaluation.downstream.base import BaseDownstreamTask
from evaluation.downstream.heads import MaskedMSELoss, StructureProbeHead
from utils.tokenizer import load_tokenizer_for_checkpoint

logger = logging.getLogger(__name__)

CONTACT_THRESHOLD_ANGSTROM = 8.0


@register_task("structure_probe")
class StructureProbeTask(BaseDownstreamTask):
    """Pairwise distance prediction via structural probe."""

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        tokenizer = load_tokenizer_for_checkpoint(
            self.config.checkpoint, self.config.model_name,
        )
        return load_structure_probe_splits(tokenizer)

    def build_head(self, hidden_size: int) -> nn.Module:
        return StructureProbeHead(hidden_size, probe_rank=64, dropout=0.1)

    def compute_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor,
    ) -> dict[str, float]:
        preds_flat = predictions.view(-1).numpy()
        labels_flat = labels.view(-1).numpy()

        valid = labels_flat != -100
        if valid.sum() < 10:
            return {"spearman_distance": 0.0, "contact_precision_at_L": 0.0, "mse_distance": 0.0}

        preds_v = preds_flat[valid]
        labels_v = labels_flat[valid]

        # Spearman correlation between predicted and true squared distances
        rho, _ = spearmanr(preds_v, labels_v)
        if np.isnan(rho):
            rho = 0.0

        # MSE on squared distances
        mse = float(np.mean((preds_v - labels_v) ** 2))

        # Contact precision@L: threshold true distances at 8A (squared = 64)
        # to get binary contacts. Rank predicted distances ascending
        # (low distance = predicted contact).
        threshold_sq = CONTACT_THRESHOLD_ANGSTROM ** 2
        true_contacts = (labels_v < threshold_sq).astype(float)
        n_pairs = int(valid.sum())
        L = int(np.ceil((-1 + np.sqrt(1 + 8 * n_pairs)) / 2))

        sorted_idx = np.argsort(preds_v)  # ascending — smallest predicted distance first
        top_L = sorted_idx[:L]
        contact_prec = float(true_contacts[top_L].mean()) if L > 0 else 0.0

        return {
            "spearman_distance": float(rho),
            "contact_precision_at_L": contact_prec,
            "mse_distance": mse,
        }

    @property
    def loss_fn(self) -> nn.Module:
        return MaskedMSELoss()

    @property
    def monitor_metric(self) -> str:
        return "spearman_distance"

    @property
    def higher_is_better(self) -> bool:
        return True

    def extract_labels(self, dataset: Dataset) -> list[Any]:
        return [dataset[i]["labels"] for i in range(len(dataset))]
