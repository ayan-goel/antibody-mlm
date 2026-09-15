"""Tests for the AB-Bind ddG probe task.

The properties that matter: complexes never straddle splits (otherwise the
probe memorises a per-complex ddG offset), the group index rides along in the
label tensor without entering the loss, and per-complex Spearman is computed
over the right groupings.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.benchmarks.ab_bind import _assign_complexes_to_splits
from evaluation.downstream.ab_bind import ABBindTask, DdgMSELoss
from evaluation.downstream.config import DownstreamConfig


# ---------------------------------------------------------------------------
# Grouped splitting
# ---------------------------------------------------------------------------

def test_complexes_never_straddle_splits() -> None:
    sizes = {f"pdb{i}": (i * 7) % 23 + 3 for i in range(31)}
    assignment = _assign_complexes_to_splits(sizes, (0.6, 0.2, 0.2), seed=42)

    assert set(assignment) == set(sizes)
    # Each complex lands in exactly one split by construction of the dict.
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_skewed_data_still_balances_by_record_count() -> None:
    """One AB-Bind complex holds ~35% of all mutants.

    Splitting by complex *count* would put wildly different record counts in
    each split; the bin-packer targets record share instead.
    """
    sizes = {"giant": 492, **{f"p{i}": 20 for i in range(30)}}
    assignment = _assign_complexes_to_splits(sizes, (0.6, 0.2, 0.2), seed=0)

    totals = {"train": 0, "val": 0, "test": 0}
    for pdb, split in assignment.items():
        totals[split] += sizes[pdb]
    total = sum(sizes.values())

    # Every split gets a non-trivial share despite the 492-record outlier.
    for split, target in [("train", 0.6), ("val", 0.2), ("test", 0.2)]:
        assert totals[split] / total == pytest.approx(target, abs=0.20)
        assert totals[split] > 0


def test_assignment_is_deterministic_for_a_seed() -> None:
    sizes = {f"p{i}": i + 3 for i in range(20)}
    a = _assign_complexes_to_splits(sizes, (0.6, 0.2, 0.2), seed=7)
    b = _assign_complexes_to_splits(sizes, (0.6, 0.2, 0.2), seed=7)
    assert a == b


# ---------------------------------------------------------------------------
# Loss must ignore the group column
# ---------------------------------------------------------------------------

def test_loss_ignores_group_column() -> None:
    preds = torch.tensor([0.5, -1.0, 2.0])
    # Same ddG values, wildly different group indices.
    a = torch.tensor([[0.5, 0.0], [-1.0, 1.0], [2.0, 2.0]])
    b = torch.tensor([[0.5, 99.0], [-1.0, 50.0], [2.0, 7.0]])

    loss = DdgMSELoss()
    assert loss(preds, a).item() == pytest.approx(0.0)
    assert loss(preds, a).item() == pytest.approx(loss(preds, b).item())


def test_loss_handles_trailing_singleton_dim() -> None:
    """RegressionHead emits (batch, 1); the loss must squeeze it."""
    preds = torch.tensor([[0.5], [-1.0]])
    labels = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    assert DdgMSELoss()(preds, labels).item() == pytest.approx(0.625)


# ---------------------------------------------------------------------------
# Per-complex metrics
# ---------------------------------------------------------------------------

@pytest.fixture()
def task() -> ABBindTask:
    t = ABBindTask(DownstreamConfig(task="ab_bind"))
    t._ddg_std = 2.0
    return t


def test_perfect_ranking_within_each_complex(task: ABBindTask) -> None:
    ddg = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    groups = [0, 0, 0, 1, 1, 1]
    preds = torch.tensor(ddg)
    labels = torch.tensor([[d, g] for d, g in zip(ddg, groups)])

    m = task.compute_metrics(preds, labels)
    assert m["mean_per_complex_spearman"] == pytest.approx(1.0)
    assert m["n_complexes_scored"] == 2.0
    assert m["n_mutants"] == 6.0


def test_per_complex_differs_from_overall(task: ABBindTask) -> None:
    """Ranking inverted inside each complex but preserved globally.

    Catches an implementation that silently pools all complexes together.
    """
    ddg = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    groups = [0, 0, 0, 1, 1, 1]
    preds = torch.tensor([3.0, 2.0, 1.0, 30.0, 20.0, 10.0])
    labels = torch.tensor([[d, g] for d, g in zip(ddg, groups)])

    m = task.compute_metrics(preds, labels)
    assert m["mean_per_complex_spearman"] == pytest.approx(-1.0)
    assert m["overall_spearman"] > 0.0


def test_small_and_constant_complexes_are_skipped(task: ABBindTask) -> None:
    ddg = [1.0, 2.0, 3.0,   5.0, 5.0, 5.0,   9.0, 9.9]
    groups = [0, 0, 0,      1, 1, 1,         2, 2]
    preds = torch.tensor(ddg)
    labels = torch.tensor([[d, g] for d, g in zip(ddg, groups)])

    m = task.compute_metrics(preds, labels)
    # Complex 1 has no label variance; complex 2 has only 2 mutants.
    assert m["n_complexes_scored"] == 1.0
    assert m["mean_per_complex_spearman"] == pytest.approx(1.0)


def test_no_scorable_complex_returns_zero_not_nan(task: ABBindTask) -> None:
    """Early stopping monitors this metric — NaN would poison it."""
    preds = torch.tensor([1.0, 2.0])
    labels = torch.tensor([[5.0, 0.0], [5.0, 0.0]])

    m = task.compute_metrics(preds, labels)
    assert m["mean_per_complex_spearman"] == 0.0
    assert not np.isnan(m["mean_per_complex_spearman"])


def test_rmse_reported_in_kcal_per_mol(task: ABBindTask) -> None:
    preds = torch.tensor([1.0, 1.0, 1.0])
    labels = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    m = task.compute_metrics(preds, labels)
    assert m["rmse_z"] == pytest.approx(1.0)
    assert m["rmse_kcal_per_mol"] == pytest.approx(2.0)  # _ddg_std = 2.0


def test_monitor_metric_is_maximised(task: ABBindTask) -> None:
    assert task.monitor_metric == "mean_per_complex_spearman"
    assert task.higher_is_better is True
