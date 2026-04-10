"""Tests for structure probe downstream task."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from evaluation.downstream.heads import (
    MaskedMSELoss,
    StructureProbeHead,
    get_aa_pair_indices,
)
from data.benchmarks.structure_probe import compute_distance_matrix


class TestStructureProbeHead:
    def test_output_shape(self):
        B, L, H = 2, 20, 32
        head = StructureProbeHead(hidden_size=H, probe_rank=16, dropout=0.0)
        hidden = torch.randn(B, L, H)
        attn_mask = torch.ones(B, L, dtype=torch.long)
        special_mask = torch.zeros(B, L, dtype=torch.long)
        special_mask[:, 0] = 1
        special_mask[:, -1] = 1

        output = head(hidden, attn_mask, special_mask)
        # 18 AA positions → 18*17/2 = 153 pairs
        assert output.shape == (B, 153)

    def test_output_nonnegative(self):
        """Squared distances should always be non-negative."""
        B, L, H = 3, 10, 16
        head = StructureProbeHead(hidden_size=H, probe_rank=8, dropout=0.0)
        hidden = torch.randn(B, L, H)
        attn_mask = torch.ones(B, L, dtype=torch.long)
        output = head(hidden, attn_mask)
        assert (output >= 0).all()

    def test_empty_with_all_special(self):
        B, L, H = 1, 5, 16
        head = StructureProbeHead(hidden_size=H, probe_rank=8, dropout=0.0)
        hidden = torch.randn(B, L, H)
        attn_mask = torch.ones(B, L, dtype=torch.long)
        special_mask = torch.ones(B, L, dtype=torch.long)
        output = head(hidden, attn_mask, special_mask)
        assert output.shape == (B, 0)


class TestMaskedMSELoss:
    def test_ignores_padding(self):
        loss_fn = MaskedMSELoss()
        preds = torch.tensor([1.0, 2.0, 3.0])
        labels = torch.tensor([1.0, 2.0, -100.0])
        loss = loss_fn(preds, labels)
        # Only first two contribute: MSE = 0.0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_all_padding_zero_loss(self):
        loss_fn = MaskedMSELoss()
        preds = torch.tensor([1.0, 2.0])
        labels = torch.tensor([-100.0, -100.0])
        assert loss_fn(preds, labels).item() == 0.0

    def test_known_mse(self):
        loss_fn = MaskedMSELoss()
        preds = torch.tensor([1.0, 3.0])
        labels = torch.tensor([2.0, 4.0])
        # MSE = ((1-2)^2 + (3-4)^2) / 2 = 1.0
        assert loss_fn(preds, labels).item() == pytest.approx(1.0, abs=1e-6)


class TestComputeDistanceMatrix:
    def test_symmetric(self):
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        dist = compute_distance_matrix(coords)
        assert dist.shape == (3, 3)
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_diagonal_zero(self):
        coords = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)]
        dist = compute_distance_matrix(coords)
        assert dist[0, 0] == pytest.approx(0.0)
        assert dist[1, 1] == pytest.approx(0.0)

    def test_known_distance(self):
        coords = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)]
        dist = compute_distance_matrix(coords)
        assert dist[0, 1] == pytest.approx(5.0)
        assert dist[1, 0] == pytest.approx(5.0)


class TestTaskRegistered:
    def test_structure_probe_in_registry(self):
        from evaluation.downstream import TASK_REGISTRY
        assert "structure_probe" in TASK_REGISTRY
