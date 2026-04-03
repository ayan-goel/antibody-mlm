"""Structure-aware masking using 3D spatial neighborhoods.

Uses predicted Calpha coordinates to define kNN neighborhoods in 3D
space, then masks spatially clustered residues via a seed-and-grow
algorithm. Falls back to uniform masking when coordinates are
unavailable for a sequence.

The seed-and-grow approach:
  1. Pick a random seed residue from maskable positions.
  2. Add the seed and its k nearest 3D neighbors to the mask set
     (closest first).
  3. If budget is not yet filled, pick another seed and repeat.
  4. Stop once the mask budget is reached.

Default k_neighbors=32 is supported by ProteinMPNN's finding that
structural context saturates around 32-48 nearest Calpha neighbors.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from masking.base import BaseMaskingStrategy, register_strategy


@register_strategy("structure")
class StructureMasking(BaseMaskingStrategy):
    """3D neighborhood masking with seed-and-grow kNN selection.

    Requires per-token Calpha coordinates passed as metadata["coords_ca"]
    (shape: seq_len x 3, dtype: float). Positions with zero coordinates
    (special tokens, padding) are excluded from masking.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_prob: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        k_neighbors: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            mask_prob=mask_prob,
            mask_token_ratio=mask_token_ratio,
            random_token_ratio=random_token_ratio,
            **kwargs,
        )
        self.k_neighbors = k_neighbors

    def _uniform_fallback(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        probability_matrix[special_tokens_mask.bool()] = 0.0
        return torch.bernoulli(probability_matrix).bool()

    def select_mask_positions(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        coords = metadata.get("coords_ca") if metadata else None

        if coords is None:
            return self._uniform_fallback(input_ids, special_tokens_mask)

        seq_len = input_ids.size(0)
        is_special = special_tokens_mask.bool()

        has_coords = (coords.abs().sum(dim=-1) > 1e-6) & (~is_special)
        maskable_idx = has_coords.nonzero(as_tuple=True)[0]
        num_maskable = maskable_idx.numel()

        if num_maskable == 0:
            return self._uniform_fallback(input_ids, special_tokens_mask)

        budget = max(1, round(self.mask_prob * num_maskable))

        maskable_coords = coords[maskable_idx]
        dists = torch.cdist(
            maskable_coords.unsqueeze(0),
            maskable_coords.unsqueeze(0),
        ).squeeze(0)

        k = min(self.k_neighbors, num_maskable - 1)
        if k == 0:
            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask[maskable_idx[0]] = True
            return mask

        _, knn_local = dists.topk(k + 1, dim=1, largest=False)
        knn_local = knn_local[:, 1:]

        mask_local = torch.zeros(num_maskable, dtype=torch.bool)
        masked_count = 0
        used_as_seed = torch.zeros(num_maskable, dtype=torch.bool)

        while masked_count < budget:
            available_seeds = (~used_as_seed & ~mask_local).nonzero(as_tuple=True)[0]
            if available_seeds.numel() == 0:
                available_seeds = (~mask_local).nonzero(as_tuple=True)[0]
                if available_seeds.numel() == 0:
                    break

            seed = available_seeds[torch.randint(available_seeds.numel(), (1,)).item()].item()
            used_as_seed[seed] = True

            if not mask_local[seed]:
                mask_local[seed] = True
                masked_count += 1

            for neighbor in knn_local[seed].tolist():
                if masked_count >= budget:
                    break
                if not mask_local[neighbor]:
                    mask_local[neighbor] = True
                    masked_count += 1

        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[maskable_idx[mask_local]] = True
        return mask
