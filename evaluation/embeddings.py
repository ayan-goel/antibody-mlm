"""Embedding extraction from trained antibody MLM models."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class PoolingStrategy(Enum):
    CLS = "cls"
    MEAN = "mean"


def extract_embeddings(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    pooling: PoolingStrategy = PoolingStrategy.MEAN,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
) -> np.ndarray:
    """Extract embeddings from all sequences in a dataset.

    Args:
        model: Trained RoFormer model.
        dataset: AntibodyDataset to extract embeddings from.
        tokenizer: Tokenizer for padding.
        pooling: Pooling strategy (CLS token or mean of non-special tokens).
        batch_size: Batch size for inference.
        num_workers: DataLoader workers.
        device: Device to run inference on.

    Returns:
        NumPy array of shape (num_sequences, hidden_size).
    """
    from masking.collator import MLMDataCollator
    from masking.base import BaseMaskingStrategy

    class _NoOpStrategy(BaseMaskingStrategy):
        """Dummy strategy that masks nothing — used for embedding extraction."""

        def select_mask_positions(self, input_ids, special_tokens_mask, metadata=None):
            return torch.zeros_like(input_ids, dtype=torch.bool)

    no_mask = _NoOpStrategy(tokenizer=tokenizer, mask_prob=0.0)
    collator = MLMDataCollator(tokenizer=tokenizer, strategy=no_mask)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    model.eval()
    model.to(device)
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]

            if pooling == PoolingStrategy.CLS:
                embeddings = hidden_states[:, 0, :]
            else:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1)
                embeddings = sum_hidden / count

            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def save_embeddings(embeddings: np.ndarray, path: str | Path) -> None:
    """Save embeddings to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    logger.info("Saved embeddings of shape %s to %s", embeddings.shape, path)


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load embeddings from a .npy file."""
    return np.load(path)
