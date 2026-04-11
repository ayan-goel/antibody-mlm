"""Reusable prediction heads for downstream tasks.

Three head types covering token classification (paratope), sequence
classification (binding specificity), and regression (developability,
mutation effect).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenClassificationHead(nn.Module):
    """Per-token classification head (e.g. paratope prediction).

    Input:  (batch, seq_len, hidden_size)
    Output: (batch, seq_len, num_labels)
    """

    def __init__(
        self, hidden_size: int, num_labels: int = 1, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(hidden_states))


def _mean_pool_excluding_specials(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-pool hidden states over real (non-special, non-pad) positions.

    For paired models, the input contains framing tokens [MOD1]/[H]/[L]
    that are real attended-to positions but carry no amino-acid identity.
    Pooling over them biases the pooled vector and makes paired/single-chain
    comparisons unfair. When ``special_tokens_mask`` is provided we
    exclude any position where it equals 1 (special / pad).
    """
    pool_mask = attention_mask
    if special_tokens_mask is not None:
        pool_mask = attention_mask * (1 - special_tokens_mask)
    mask = pool_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class SequenceClassificationHead(nn.Module):
    """Sequence-level classification head with mean pooling.

    Input:  hidden_states (batch, seq_len, hidden_size),
            attention_mask (batch, seq_len),
            special_tokens_mask (batch, seq_len) — optional, excluded from pool
    Output: (batch, num_labels)
    """

    def __init__(
        self, hidden_size: int, num_labels: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = _mean_pool_excluding_specials(
            hidden_states, attention_mask, special_tokens_mask,
        )
        return self.classifier(self.dropout(pooled))


class RegressionHead(nn.Module):
    """Sequence-level regression head with mean pooling.

    Input:  hidden_states (batch, seq_len, hidden_size),
            attention_mask (batch, seq_len),
            special_tokens_mask (batch, seq_len) — optional, excluded from pool
    Output: (batch, num_targets)
    """

    def __init__(
        self, hidden_size: int, num_targets: int = 1, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, num_targets)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = _mean_pool_excluding_specials(
            hidden_states, attention_mask, special_tokens_mask,
        )
        return self.regressor(self.dropout(pooled))


# ---------------------------------------------------------------------------
# Pairwise utilities
# ---------------------------------------------------------------------------

def get_aa_pair_indices(
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return upper-triangle index pairs for AA-only positions.

    Args:
        attention_mask: (B, L) — 1 for real tokens, 0 for padding.
        special_tokens_mask: (B, L) — 1 for special tokens, 0 for AAs.

    Returns:
        idx_i, idx_j: (n_pairs,) index tensors for the upper triangle of
            the *full* L×L matrix, restricted to positions that are AA
            (non-special, non-pad) in the *first* sample of the batch.
            All samples in a batch share the same padded length so we
            compute indices once and mask invalid pairs via labels=-100.
        aa_mask: (L,) bool tensor — True for AA positions.
    """
    L = attention_mask.size(1)
    if special_tokens_mask is not None:
        aa_mask = (attention_mask[0] == 1) & (special_tokens_mask[0] == 0)
    else:
        aa_mask = attention_mask[0] == 1
    aa_positions = aa_mask.nonzero(as_tuple=True)[0]
    n_aa = aa_positions.size(0)
    if n_aa < 2:
        empty = torch.zeros(0, dtype=torch.long, device=attention_mask.device)
        return empty, empty, aa_mask.bool()
    tri_i, tri_j = torch.triu_indices(n_aa, n_aa, offset=1, device=attention_mask.device)
    idx_i = aa_positions[tri_i]
    idx_j = aa_positions[tri_j]
    return idx_i, idx_j, aa_mask.bool()


# ---------------------------------------------------------------------------
# Pairwise heads
# ---------------------------------------------------------------------------

class ContactMapHead(nn.Module):
    """Bilinear pairwise contact prediction head.

    Input:  hidden_states (B, L, H), attention_mask (B, L),
            special_tokens_mask (B, L)
    Output: (B, max_pairs) — logits for upper-triangle AA-only pairs,
            padded per-sample to max_pairs so that variable-length
            sequences in a batch all produce the same output size.
    """

    def __init__(
        self, hidden_size: int, dropout: float = 0.1, max_pairs: int = 0,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1, bias=True)
        self.max_pairs = max_pairs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.dropout(hidden_states)
        B, device = h.size(0), h.device
        results: list[torch.Tensor] = []
        for b in range(B):
            aa_mask = attention_mask[b] == 1
            if special_tokens_mask is not None:
                aa_mask = aa_mask & (special_tokens_mask[b] == 0)
            aa_pos = aa_mask.nonzero(as_tuple=True)[0]
            n_aa = aa_pos.size(0)
            if n_aa < 2:
                results.append(torch.zeros(self.max_pairs, device=device))
                continue
            tri_i, tri_j = torch.triu_indices(n_aa, n_aa, offset=1, device=device)
            h_i = h[b, aa_pos[tri_i]].unsqueeze(0)  # (1, n_pairs, H)
            h_j = h[b, aa_pos[tri_j]].unsqueeze(0)  # (1, n_pairs, H)
            preds = self.bilinear(h_i, h_j).squeeze(-1).squeeze(0)  # (n_pairs,)
            padded = torch.zeros(self.max_pairs, device=device)
            padded[: preds.size(0)] = preds
            results.append(padded)
        return torch.stack(results)  # (B, max_pairs)


class StructureProbeHead(nn.Module):
    """Hewitt & Manning (2019) linear structural probe.

    Learns B (probe_rank × H) such that ||Bh_i − Bh_j||² ≈ d(i,j)².

    Input:  hidden_states (B, L, H), attention_mask (B, L),
            special_tokens_mask (B, L)
    Output: (B, max_pairs) — predicted squared distances for AA-only
            pairs, padded per-sample to max_pairs so that
            variable-length sequences produce the same output size.
    """

    def __init__(
        self, hidden_size: int, probe_rank: int = 64, dropout: float = 0.1,
        max_pairs: int = 0,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.transform = nn.Linear(hidden_size, probe_rank, bias=False)
        self.max_pairs = max_pairs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.dropout(hidden_states)
        Bh = self.transform(h)  # (B, L, rank)
        B, device = h.size(0), h.device
        results: list[torch.Tensor] = []
        for b in range(B):
            aa_mask = attention_mask[b] == 1
            if special_tokens_mask is not None:
                aa_mask = aa_mask & (special_tokens_mask[b] == 0)
            aa_pos = aa_mask.nonzero(as_tuple=True)[0]
            n_aa = aa_pos.size(0)
            if n_aa < 2:
                results.append(torch.zeros(self.max_pairs, device=device))
                continue
            tri_i, tri_j = torch.triu_indices(n_aa, n_aa, offset=1, device=device)
            diff = Bh[b, aa_pos[tri_i]] - Bh[b, aa_pos[tri_j]]  # (n_pairs, rank)
            preds = (diff ** 2).sum(dim=-1)  # (n_pairs,)
            padded = torch.zeros(self.max_pairs, device=device)
            padded[: preds.size(0)] = preds
            results.append(padded)
        return torch.stack(results)  # (B, max_pairs)


# ---------------------------------------------------------------------------
# Pairwise losses
# ---------------------------------------------------------------------------

class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE with logits loss that ignores positions where labels == -100."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels != -100
        if not valid.any():
            return logits.sum() * 0.0
        return nn.functional.binary_cross_entropy_with_logits(
            logits[valid], labels[valid].float(),
        )


class MaskedMSELoss(nn.Module):
    """MSE loss that ignores positions where labels == -100."""

    def forward(
        self, predictions: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        valid = labels != -100
        if not valid.any():
            return predictions.sum() * 0.0
        return nn.functional.mse_loss(predictions[valid], labels[valid].float())
