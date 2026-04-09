"""Shared metric helpers used by multiple downstream tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def find_youdens_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Find the optimal binary-classification threshold via Youden's J statistic.

    Picks the threshold that maximizes ``tpr - fpr`` on the ROC curve.
    Used by tasks where the loss is class-weighted (so model logits are
    asymmetrically pushed and a fixed 0.5 threshold under-estimates F1/MCC).
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])
