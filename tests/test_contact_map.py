"""Tests for contact map prediction task."""

from __future__ import annotations

import torch
import pytest

from evaluation.downstream.heads import (
    ContactMapHead,
    MaskedBCEWithLogitsLoss,
    get_aa_pair_indices,
)
from data.benchmarks.contact_map import knn_to_contact_matrix


class TestKnnToContactMatrix:
    def test_symmetric(self):
        knn = torch.tensor([[1, 2], [0, 2], [0, 1]])
        contact = knn_to_contact_matrix(knn, seq_len=3)
        assert contact.shape == (3, 3)
        assert (contact == contact.T).all(), "Contact matrix should be symmetric"

    def test_diagonal_zero(self):
        knn = torch.tensor([[0, 1], [0, 1]])
        contact = knn_to_contact_matrix(knn, seq_len=2)
        assert contact[0, 0] == 0.0
        assert contact[1, 1] == 0.0

    def test_known_contacts(self):
        # Residue 0 has neighbor 2, residue 1 has neighbor 0
        knn = torch.tensor([[2], [0], [1]])
        contact = knn_to_contact_matrix(knn, seq_len=3)
        assert contact[0, 2] == 1.0
        assert contact[2, 0] == 1.0  # symmetric
        assert contact[1, 0] == 1.0
        assert contact[0, 1] == 1.0  # symmetric

    def test_out_of_range_ignored(self):
        knn = torch.tensor([[1, 99], [0, -1]])
        contact = knn_to_contact_matrix(knn, seq_len=2)
        assert contact[0, 1] == 1.0
        assert contact.sum() == 2.0  # only (0,1) and (1,0)


class TestContactMapHead:
    def test_output_shape(self):
        B, L, H = 2, 20, 32
        # 18 AA positions → 18*17/2 = 153 pairs; pad to 200
        head = ContactMapHead(hidden_size=H, dropout=0.0, max_pairs=200)
        hidden = torch.randn(B, L, H)
        attn_mask = torch.ones(B, L, dtype=torch.long)
        special_mask = torch.zeros(B, L, dtype=torch.long)
        special_mask[:, 0] = 1  # [CLS]
        special_mask[:, -1] = 1  # [SEP]

        output = head(hidden, attn_mask, special_mask)
        assert output.shape == (B, 200)
        # Positions beyond 153 real pairs should be zero
        assert output[:, 153:].abs().sum() == 0

    def test_empty_with_all_special(self):
        B, L, H = 1, 5, 16
        head = ContactMapHead(hidden_size=H, dropout=0.0, max_pairs=10)
        hidden = torch.randn(B, L, H)
        attn_mask = torch.ones(B, L, dtype=torch.long)
        special_mask = torch.ones(B, L, dtype=torch.long)  # all special
        output = head(hidden, attn_mask, special_mask)
        assert output.shape == (B, 10)
        assert output.abs().sum() == 0  # all zeros


class TestMaskedBCEWithLogitsLoss:
    def test_ignores_padding(self):
        loss_fn = MaskedBCEWithLogitsLoss()
        logits = torch.tensor([1.0, -1.0, 0.5])
        labels = torch.tensor([1.0, 0.0, -100.0])
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

        # All padding → zero loss
        labels_all_pad = torch.tensor([-100.0, -100.0, -100.0])
        loss_zero = loss_fn(logits, labels_all_pad)
        assert loss_zero.item() == 0.0

    def test_perfect_prediction_low_loss(self):
        loss_fn = MaskedBCEWithLogitsLoss()
        logits = torch.tensor([10.0, -10.0])
        labels = torch.tensor([1.0, 0.0])
        loss = loss_fn(logits, labels)
        assert loss.item() < 0.01


class TestTaskRegistered:
    def test_contact_map_in_registry(self):
        from evaluation.downstream import TASK_REGISTRY
        assert "contact_map" in TASK_REGISTRY

    def test_binding_not_in_registry(self):
        from evaluation.downstream import TASK_REGISTRY
        assert "binding" not in TASK_REGISTRY
