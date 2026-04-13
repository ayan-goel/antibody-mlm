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

from data.benchmarks.contact_map import load_sabdab_contact_splits
from evaluation.downstream import register_task
from evaluation.downstream.base import BaseDownstreamTask
from evaluation.downstream.heads import ContactMapHead, MaskedBCEWithLogitsLoss
from utils.tokenizer import load_tokenizer_for_checkpoint

logger = logging.getLogger(__name__)


@register_task("contact_map")
class ContactMapTask(BaseDownstreamTask):
    """Pairwise contact prediction from frozen or fine-tuned embeddings.

    Labels are real Calpha-Calpha contacts at the 8 Å threshold,
    derived from X-ray crystal structures in SAbDab Liberis (built by
    ``scripts/build_sabdab_real_coords.py``).
    """

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        tokenizer = load_tokenizer_for_checkpoint(
            self.config.checkpoint, self.config.model_name,
        )
        train, val, test = load_sabdab_contact_splits(tokenizer)
        self._max_pairs = train.max_pairs
        return train, val, test

    def build_head(self, hidden_size: int) -> nn.Module:
        return ContactMapHead(hidden_size, dropout=0.1, max_pairs=self._max_pairs)

    def compute_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor,
    ) -> dict[str, float]:
        # Predictions and labels arrive as (n_samples, max_pairs) from the
        # trainer (before any flattening). We keep that shape so we can
        # reconstruct each sample's (i, j) index pairs and compute
        # separation-stratified metrics — short-range contacts dominate
        # the global metric and inflate precision; long-range contacts
        # are the meaningful structural signal.
        preds_2d = predictions.detach().cpu().numpy()
        labels_2d = labels.detach().cpu().numpy()
        if preds_2d.ndim == 1:
            preds_2d = preds_2d.reshape(1, -1)
            labels_2d = labels_2d.reshape(1, -1)

        # Global (all separations) metrics — kept for backwards
        # compatibility, but precision_at_L on all pairs is dominated by
        # the trivial structurally-close ESM2 contact predictions and
        # saturates near 1.0 across models.
        flat_valid_mask = labels_2d != -100
        preds_v = preds_2d[flat_valid_mask]
        labels_v = labels_2d[flat_valid_mask]

        if labels_v.size < 10 or len(np.unique(labels_v)) < 2:
            return {
                "auroc": 0.0, "precision_at_L": 0.0,
                "precision_at_L2": 0.0, "precision_at_L5": 0.0,
                "long_range_auroc": 0.0,
                "long_range_precision_at_L": 0.0,
                "long_range_precision_at_L5": 0.0,
                "medium_long_precision_at_L": 0.0,
            }

        auroc = float(roc_auc_score(labels_v, preds_v))

        n_pairs = int(labels_v.size)
        L = int(np.ceil((-1 + np.sqrt(1 + 8 * n_pairs)) / 2))
        sorted_idx = np.argsort(-preds_v)
        prec_L = _precision_at_k(labels_v, sorted_idx, L)
        prec_L2 = _precision_at_k(labels_v, sorted_idx, max(1, L // 2))
        prec_L5 = _precision_at_k(labels_v, sorted_idx, max(1, L // 5))

        # Per-sample reconstruction of (i, j) for separation filtering.
        # Each sample's labels are upper-triangle pairs of an n_s × n_s
        # matrix where n_s = solve(K = n*(n-1)/2) and K = #valid labels
        # in that row. The first K entries of row[s] map to (i, j) given
        # by triu_indices(n_s, n_s, 1).
        all_preds_lr: list[np.ndarray] = []
        all_labels_lr: list[np.ndarray] = []
        all_preds_ml: list[np.ndarray] = []  # |i-j| >= 12 (medium+long)
        all_labels_ml: list[np.ndarray] = []
        per_sample_l_long: list[float] = []
        per_sample_l5_long: list[float] = []

        for s in range(preds_2d.shape[0]):
            row_labels = labels_2d[s]
            row_preds = preds_2d[s]
            row_valid = row_labels != -100
            K = int(row_valid.sum())
            if K < 3:
                continue
            # K = n*(n-1)/2  →  n = (1 + sqrt(1 + 8K)) / 2.
            # Round to nearest integer to absorb any floating-point dust.
            n_s = int(round((1.0 + np.sqrt(1.0 + 8.0 * K)) / 2.0))
            if n_s * (n_s - 1) // 2 != K:
                # Defensive: K wasn't a triangular number for any n.
                # Skip rather than misalign indices.
                continue
            tri_i, tri_j = np.triu_indices(n_s, k=1)
            sep = tri_j - tri_i

            # Pull only the valid prefix (first K entries) — that's where
            # this sample's pairs live before -100 padding.
            sample_preds = row_preds[:K]
            sample_labels = row_labels[:K]

            long_mask = sep >= 24
            ml_mask = sep >= 12

            if long_mask.sum() >= 1:
                all_preds_lr.append(sample_preds[long_mask])
                all_labels_lr.append(sample_labels[long_mask])
                # Per-sample top-L_long precision (L = sample sequence length).
                long_preds = sample_preds[long_mask]
                long_labels = sample_labels[long_mask]
                lp_sorted = np.argsort(-long_preds)
                if long_labels.size > 0:
                    per_sample_l_long.append(
                        _precision_at_k(long_labels, lp_sorted, n_s)
                    )
                    per_sample_l5_long.append(
                        _precision_at_k(long_labels, lp_sorted, max(1, n_s // 5))
                    )

            if ml_mask.sum() >= 1:
                all_preds_ml.append(sample_preds[ml_mask])
                all_labels_ml.append(sample_labels[ml_mask])

        def _safe_auroc(preds_list, labels_list) -> float:
            if not preds_list:
                return 0.0
            p = np.concatenate(preds_list)
            l = np.concatenate(labels_list)
            if len(np.unique(l)) < 2:
                return 0.0
            return float(roc_auc_score(l, p))

        long_auroc = _safe_auroc(all_preds_lr, all_labels_lr)
        long_prec_L = float(np.mean(per_sample_l_long)) if per_sample_l_long else 0.0
        long_prec_L5 = float(np.mean(per_sample_l5_long)) if per_sample_l5_long else 0.0
        ml_auroc = _safe_auroc(all_preds_ml, all_labels_ml)

        return {
            "auroc": auroc,
            "precision_at_L": prec_L,
            "precision_at_L2": prec_L2,
            "precision_at_L5": prec_L5,
            "long_range_auroc": long_auroc,
            "long_range_precision_at_L": long_prec_L,
            "long_range_precision_at_L5": long_prec_L5,
            "medium_long_auroc": ml_auroc,
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
