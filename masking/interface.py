"""Interface and paratope masking using antibody–antigen complex data.

Biases the MLM mask budget toward paratope (antigen-contacting) residues
using per-residue labels in [0, 1]. Labels can be hard (0/1 from
structure-derived contacts) or soft (predicted paratope probabilities).
Falls back to uniform masking when paratope labels are unavailable.

Default weights (paratope=6, non-paratope=1) concentrate ~50-60% of
masked positions on paratope residues despite paratopes being only
~15-25% of the sequence, focusing learning on binding determinants.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from masking.base import BaseMaskingStrategy, register_strategy


@register_strategy("interface")
class InterfaceMasking(BaseMaskingStrategy):
    """Paratope-biased masking with weighted Bernoulli sampling.

    Allocates the mask budget (default 15%) disproportionately toward
    paratope positions using per-residue weights derived from paratope
    labels. Supports both hard labels (0/1) and soft probabilities via
    a single float field.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_prob: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        paratope_weight: float = 6.0,
        non_paratope_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            mask_prob=mask_prob,
            mask_token_ratio=mask_token_ratio,
            random_token_ratio=random_token_ratio,
            **kwargs,
        )
        self.paratope_weight = paratope_weight
        self.non_paratope_weight = non_paratope_weight

    def select_mask_positions(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        paratope_labels = metadata.get("paratope_labels") if metadata else None

        if paratope_labels is None:
            probability_matrix = torch.full(input_ids.shape, self.mask_prob)
            probability_matrix[special_tokens_mask.bool()] = 0.0
            return torch.bernoulli(probability_matrix).bool()

        weights = self.non_paratope_weight + (
            self.paratope_weight - self.non_paratope_weight
        ) * paratope_labels.float()

        non_special = ~special_tokens_mask.bool()
        weights[~non_special] = 0.0

        mean_weight = weights[non_special].mean()
        if mean_weight > 0:
            probability_matrix = (self.mask_prob * weights / mean_weight).clamp(max=1.0)
        else:
            probability_matrix = torch.full(input_ids.shape, self.mask_prob)

        probability_matrix[~non_special] = 0.0
        return torch.bernoulli(probability_matrix).bool()
