"""Encoder wrapper for downstream tasks.

Extracts the base RoFormer encoder from a RoFormerForMaskedLM checkpoint
and exposes a clean forward() that returns last-layer hidden states.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import RoFormerForMaskedLM

logger = logging.getLogger(__name__)


class EncoderWrapper(nn.Module):
    """Thin wrapper around the RoFormer encoder for downstream use.

    Provides a uniform interface: forward(input_ids, attention_mask)
    returns hidden states of shape (batch, seq_len, hidden_size).
    """

    def __init__(self, model: RoFormerForMaskedLM) -> None:
        super().__init__()
        self.encoder = model.roformer
        self.hidden_size: int = model.config.hidden_size

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: str = "cuda"
    ) -> EncoderWrapper:
        """Load encoder from a RoFormerForMaskedLM checkpoint directory."""
        logger.info("Loading encoder from %s", checkpoint_path)
        model = RoFormerForMaskedLM.from_pretrained(checkpoint_path)
        wrapper = cls(model)
        wrapper.to(device)
        wrapper.eval()
        logger.info("Encoder loaded (hidden_size=%d)", wrapper.hidden_size)
        return wrapper
