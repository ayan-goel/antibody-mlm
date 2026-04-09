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
