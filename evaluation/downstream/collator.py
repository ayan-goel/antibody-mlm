"""Collator for downstream tasks (no masking).

Pads input_ids, attention_mask, special_tokens_mask, and task labels
to uniform length within a batch. Handles both token-level labels
(variable-length tensors padded with -100) and scalar labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DownstreamCollator:
    """Pad tokenized inputs and task labels without any masking."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8
    label_pad_value: int = -100

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids_list = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]
        attention_mask_list = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in examples]
        labels_list = [
            ex["labels"] if isinstance(ex["labels"], torch.Tensor) else torch.tensor(ex["labels"])
            for ex in examples
        ]

        max_len = max(ids.size(0) for ids in input_ids_list)
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        token_level = labels_list[0].dim() >= 1 and labels_list[0].size(0) > max_len // 2

        batch_ids, batch_mask, batch_labels = [], [], []

        for input_ids, attn_mask, label in zip(
            input_ids_list, attention_mask_list, labels_list
        ):
            pad_len = max_len - input_ids.size(0)
            if pad_len > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long),
                ])
                attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)])
                if token_level:
                    label = torch.cat([
                        label,
                        torch.full((pad_len,), self.label_pad_value, dtype=label.dtype),
                    ])

            batch_ids.append(input_ids)
            batch_mask.append(attn_mask)
            batch_labels.append(label)

        return {
            "input_ids": torch.stack(batch_ids),
            "attention_mask": torch.stack(batch_mask),
            "labels": torch.stack(batch_labels),
        }
