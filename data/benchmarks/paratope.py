"""Paratope prediction dataset from TDC SAbDab_Liberis.

Each sample is an antibody chain with per-residue binary labels indicating
whether a residue is part of the paratope (contacts antigen within 4.5A).

TDC provides:
  X = amino acid sequence (str)
  Y = list of 0-indexed paratope residue positions
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ParatopeDataset(Dataset):
    """Token-level binary classification dataset for paratope prediction.

    Each item returns input_ids, attention_mask, special_tokens_mask,
    and labels (per-token: 1=paratope, 0=non-paratope, -100=ignore).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 160,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        seq_col = "Antibody" if "Antibody" in df.columns else "X"
        self.sequences: list[str] = df[seq_col].tolist()
        self.paratope_indices: list[list[int]] = [
            _parse_indices(y) for y in df["Y"]
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sequence = self.sequences[idx]
        paratope_set = set(self.paratope_indices[idx])

        spaced = " ".join(list(sequence))
        encoding = self.tokenizer(
            spaced,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_special_tokens_mask=True,
        )

        num_tokens = len(encoding["input_ids"])
        labels = [-100] * num_tokens

        for aa_idx in range(len(sequence)):
            token_pos = aa_idx + 1  # +1 for [CLS]
            if token_pos >= num_tokens - 1:
                break
            labels[token_pos] = 1 if aa_idx in paratope_set else 0

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
            "labels": labels,
        }


def _parse_indices(y: Any) -> list[int]:
    """Parse TDC's Y field into a list of integer indices."""
    if isinstance(y, list):
        return [int(i) for i in y]
    if isinstance(y, str):
        import ast
        return [int(i) for i in ast.literal_eval(y)]
    return []


def load_paratope_splits(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 160,
) -> tuple[ParatopeDataset, ParatopeDataset, ParatopeDataset]:
    """Download TDC SAbDab_Liberis and return (train, val, test) datasets."""
    from tdc.single_pred import Paratope

    logger.info("Loading TDC SAbDab_Liberis paratope dataset...")
    data = Paratope(name="SAbDab_Liberis")
    split = data.get_split()

    train_ds = ParatopeDataset(split["train"], tokenizer, max_length)
    val_ds = ParatopeDataset(split["valid"], tokenizer, max_length)
    test_ds = ParatopeDataset(split["test"], tokenizer, max_length)

    logger.info(
        "Paratope splits: train=%d, val=%d, test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_ds, val_ds, test_ds


def compute_class_weight(dataset: ParatopeDataset) -> float:
    """Compute pos_weight for BCE loss from training set label distribution."""
    n_pos, n_neg = 0, 0
    for idx in range(len(dataset)):
        item = dataset[idx]
        labels = torch.tensor(item["labels"])
        valid = labels >= 0
        n_pos += (labels[valid] == 1).sum().item()
        n_neg += (labels[valid] == 0).sum().item()
    weight = n_neg / max(n_pos, 1)
    logger.info("Paratope class balance: %d pos, %d neg -> pos_weight=%.2f", n_pos, n_neg, weight)
    return weight
