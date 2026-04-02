"""Developability downstream task.

Multi-target regression: predict 5 computed developability metrics
(CDR_Length, PSH, PPC, PNC, SFvCSP) from VH sequence.  Uses TDC TAP
dataset with z-score standardized labels.  Evaluates per-target and
macro-average Spearman correlation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import Dataset

from data.benchmarks.developability import LabelScaler, load_developability_splits
from evaluation.downstream import register_task
from evaluation.downstream.base import BaseDownstreamTask
from evaluation.downstream.heads import RegressionHead
from utils.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


@register_task("developability")
class DevelopabilityTask(BaseDownstreamTask):
    """Multi-target regression for antibody developability metrics."""

    _scaler: LabelScaler | None = None
    _label_names: list[str] | None = None

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        tokenizer = load_tokenizer(self.config.model_name)
        train, val, test, label_names, scaler = load_developability_splits(
            tokenizer, max_length=160,
        )
        self._scaler = scaler
        self._label_names = label_names
        return train, val, test

    def build_head(self, hidden_size: int) -> nn.Module:
        return RegressionHead(hidden_size, num_targets=5, dropout=0.1)

    def compute_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor,
    ) -> dict[str, float]:
        preds_np = predictions.detach().numpy()
        labels_np = labels.detach().numpy()

        if self._scaler is not None:
            preds_np = self._scaler.inverse_transform(preds_np)
            labels_np = self._scaler.inverse_transform(labels_np)

        label_names = self._label_names or [f"target_{i}" for i in range(preds_np.shape[1])]
        spearman_values: list[float] = []

        metrics: dict[str, float] = {}
        for i, name in enumerate(label_names):
            if labels_np.shape[0] < 3:
                rho = 0.0
            else:
                rho, _ = spearmanr(labels_np[:, i], preds_np[:, i])
                if np.isnan(rho):
                    rho = 0.0
            metrics[f"spearman_{name}"] = float(rho)
            spearman_values.append(float(rho))

        metrics["spearman_macro"] = float(np.mean(spearman_values))

        mse = float(np.mean((preds_np - labels_np) ** 2))
        metrics["mse_original_scale"] = mse

        return metrics

    @property
    def loss_fn(self) -> nn.Module:
        return nn.MSELoss()

    @property
    def monitor_metric(self) -> str:
        return "spearman_macro"

    @property
    def higher_is_better(self) -> bool:
        return True

    def extract_labels(self, dataset: Dataset) -> list[Any]:
        return [dataset[i]["labels"] for i in range(len(dataset))]
