"""Tests for attention perturbation analysis."""

from __future__ import annotations

import math

import numpy as np
import torch
import pytest
from unittest.mock import MagicMock

from evaluation.attention_analysis import AttentionAnalyzer, _register_zero_head_hook


class TestEntropyKnownDistribution:
    def test_uniform_attention_entropy(self):
        """Uniform attention over L positions should have entropy = log(L)."""
        L = 16
        # Create a mock attention output: uniform distribution
        uniform_attn = torch.ones(L, L) / L
        attn_clamped = uniform_attn.clamp(min=1e-12)
        entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)
        expected = math.log(L)
        for val in entropy:
            assert val.item() == pytest.approx(expected, rel=1e-4)

    def test_peaked_attention_low_entropy(self):
        """Attention concentrated on one position should have near-zero entropy."""
        L = 16
        peaked = torch.zeros(L, L)
        peaked[:, 0] = 1.0  # all queries attend to position 0
        attn_clamped = peaked.clamp(min=1e-12)
        entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)
        for val in entropy:
            assert val.item() < 0.01


class TestAblationHook:
    def test_hook_zeros_head_slice(self):
        """The zeroing hook should zero out exactly the target head's slice."""
        hidden_size = 32
        n_heads = 4
        head_dim = hidden_size // n_heads
        target_head = 1

        # Simulate an attention module
        module = MagicMock()
        handle = _register_zero_head_hook(module, target_head, head_dim)

        # Simulate the output: (B, L, hidden_size)
        B, L = 2, 10
        output = torch.ones(B, L, hidden_size)
        # The hook receives (module, input, output)
        result = module.register_forward_hook.call_args[0][0](
            module, None, (output, torch.ones(B, n_heads, L, L))
        )
        modified = result[0]

        # Target head slice should be zeroed
        start = target_head * head_dim
        end = start + head_dim
        assert (modified[:, :, start:end] == 0).all()
        # Other heads should be untouched
        assert (modified[:, :, :start] == 1).all()
        assert (modified[:, :, end:] == 1).all()

        handle.remove()


class TestGetAaPairIndices:
    def test_basic_pair_count(self):
        from evaluation.downstream.heads import get_aa_pair_indices

        B, L = 1, 10
        attn_mask = torch.ones(B, L, dtype=torch.long)
        special_mask = torch.zeros(B, L, dtype=torch.long)
        special_mask[0, 0] = 1  # [CLS]
        special_mask[0, -1] = 1  # [SEP]

        idx_i, idx_j, aa_mask = get_aa_pair_indices(attn_mask, special_mask)
        # 8 AA positions → 8*7/2 = 28 pairs
        assert idx_i.shape[0] == 28
        assert idx_j.shape[0] == 28
        assert aa_mask.sum().item() == 8

    def test_all_special_no_pairs(self):
        from evaluation.downstream.heads import get_aa_pair_indices

        B, L = 1, 5
        attn_mask = torch.ones(B, L, dtype=torch.long)
        special_mask = torch.ones(B, L, dtype=torch.long)
        idx_i, idx_j, _ = get_aa_pair_indices(attn_mask, special_mask)
        assert idx_i.numel() == 0
