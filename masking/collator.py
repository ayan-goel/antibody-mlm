"""MLM Data Collator that delegates masking to a pluggable strategy.

This collator handles padding/batching and calls the masking strategy's
apply() method on each sequence. It is strategy-agnostic — swap the
strategy and the collator works unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from masking.base import BaseMaskingStrategy


@dataclass
class MLMDataCollator:
    """Collator for masked language modeling with pluggable masking strategy.

    Pads sequences to equal length within a batch, then applies the
    masking strategy to produce (input_ids, attention_mask, labels).
    """

    tokenizer: PreTrainedTokenizerBase
    strategy: BaseMaskingStrategy
    pad_to_multiple_of: int | None = 8
    return_metadata: bool = False

    _CORE_KEYS = {"input_ids", "attention_mask", "special_tokens_mask"}

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids_list = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]
        attention_mask_list = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in examples]
        special_tokens_mask_list = [torch.tensor(ex["special_tokens_mask"], dtype=torch.long) for ex in examples]

        extra_keys = {k for ex in examples for k in ex if k not in self._CORE_KEYS}
        extra_lists: dict[str, list[torch.Tensor]] = {
            k: [torch.tensor(ex[k], dtype=torch.long) for ex in examples if k in ex]
            for k in extra_keys
        }

        max_len = max(ids.size(0) for ids in input_ids_list)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_metadata: dict[str, list[torch.Tensor]] = {
            k: [] for k in extra_keys
        }

        for i, (input_ids, attn_mask, special_mask) in enumerate(zip(
            input_ids_list, attention_mask_list, special_tokens_mask_list
        )):
            pad_len = max_len - input_ids.size(0)
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
                attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)])
                special_mask = torch.cat([special_mask, torch.ones(pad_len, dtype=torch.long)])

            metadata: dict[str, torch.Tensor] | None = None
            if extra_keys:
                metadata = {}
                for k, tensors in extra_lists.items():
                    if i < len(tensors):
                        t = tensors[i]
                        if pad_len > 0 and t.dim() >= 1:
                            t = torch.cat([t, torch.zeros(pad_len, dtype=t.dtype)])
                        metadata[k] = t
                        batch_metadata[k].append(t)

            masked_ids, labels = self.strategy.apply(
                input_ids, special_mask, metadata=metadata
            )

            batch_input_ids.append(masked_ids)
            batch_attention_mask.append(attn_mask)
            batch_labels.append(labels)

        result = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }

        if self.return_metadata:
            for k, tensors in batch_metadata.items():
                if len(tensors) == len(batch_input_ids):
                    result[k] = torch.stack(tensors)

        return result
