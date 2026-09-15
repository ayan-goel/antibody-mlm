"""AB-Bind binding-affinity (ddG) downstream task.

Supervised regression on frozen embeddings: predict the experimental change
in binding free energy for an antibody point/multi-point mutant.

Why this exists: the repo already scores AB-Bind zero-shot via delta-PLL
(``evaluation/mutation_scoring.py`` -> the ``mut_*`` columns), but every
trained model lands at rho ~ 0 with below-chance AUROC there, while the
random-init control scores best. That is a property of the *protocol*, not
of the models — delta-PLL ranks mutants by likelihood under an OAS repertoire
prior, and affinity-improving mutations are typically germline-divergent and
therefore unlikely. Both works the paper compares against (Ng & Briney;
Talaei et al.) instead use supervised probes on frozen embeddings. This task
matches that protocol.

Splits are grouped by complex (``data/benchmarks/ab_bind.py``), so a model
cannot memorise a per-complex ddG offset. The headline metric is the mean
per-complex Spearman rho, chosen to be directly comparable to the existing
zero-shot ``mut_mean_per_complex_spearman_rho``.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import Dataset

from data.benchmarks.ab_bind import load_ab_bind_splits
from evaluation.downstream import register_task
from evaluation.downstream.base import BaseDownstreamTask
from evaluation.downstream.heads import RegressionHead
from utils.tokenizer import load_tokenizer_for_checkpoint

logger = logging.getLogger(__name__)

#: Per-complex Spearman needs at least this many mutants with label variance.
_MIN_MUTANTS = 3


class DdgMSELoss(nn.Module):
    """MSE on the ddG column only.

    ``labels`` is ``[ddg_z, group_index]``; column 1 is metadata that rides
    along so ``compute_metrics`` can group by complex, and must never enter
    the loss.
    """

    def forward(
        self, predictions: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        return nn.functional.mse_loss(predictions, labels[:, 0].float())


@register_task("ab_bind")
class ABBindTask(BaseDownstreamTask):
    """ddG regression on frozen embeddings, scored per complex."""

    _ddg_std: float = 1.0

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        tokenizer = load_tokenizer_for_checkpoint(
            self.config.checkpoint, self.config.model_name,
        )
        train, val, test = load_ab_bind_splits(tokenizer, max_length=160)
        # Kept so RMSE can be reported in kcal/mol rather than z-units.
        self._ddg_std = train.std
        return train, val, test

    def build_head(self, hidden_size: int) -> nn.Module:
        return RegressionHead(hidden_size, num_targets=1, dropout=0.1)

    @property
    def loss_fn(self) -> nn.Module:
        return DdgMSELoss()

    @property
    def monitor_metric(self) -> str:
        return "mean_per_complex_spearman"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor,
    ) -> dict[str, float]:
        preds = predictions.detach().cpu().numpy().reshape(-1)
        labels_np = labels.detach().cpu().numpy()
        ddg = labels_np[:, 0].astype(float)
        groups = labels_np[:, 1].astype(int)

        by_complex: dict[int, list[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            by_complex[g].append(i)

        rhos: list[float] = []
        for idx in by_complex.values():
            if len(idx) < _MIN_MUTANTS:
                continue
            y = ddg[idx]
            if len(np.unique(y)) < 2:
                continue
            rho, _ = spearmanr(y, preds[idx])
            if not np.isnan(rho):
                rhos.append(float(rho))

        metrics: dict[str, float] = {
            "n_complexes_scored": float(len(rhos)),
            "n_mutants": float(len(ddg)),
        }
        # Empty when a split has no complex with enough labelled variance;
        # returning 0.0 keeps early stopping well-defined rather than NaN.
        metrics["mean_per_complex_spearman"] = float(np.mean(rhos)) if rhos else 0.0
        metrics["median_per_complex_spearman"] = float(np.median(rhos)) if rhos else 0.0

        if len(np.unique(ddg)) > 1:
            overall, _ = spearmanr(ddg, preds)
            metrics["overall_spearman"] = float(overall) if not np.isnan(overall) else 0.0
        else:
            metrics["overall_spearman"] = 0.0

        rmse_z = float(np.sqrt(np.mean((preds - ddg) ** 2)))
        metrics["rmse_z"] = rmse_z
        metrics["rmse_kcal_per_mol"] = rmse_z * self._ddg_std
        return metrics
