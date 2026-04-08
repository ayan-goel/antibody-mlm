"""Developability prediction dataset from TDC TAP.

Multi-target regression: predict 5 computed developability metrics
from VH sequence alone.  Labels are z-score standardized so MSE loss
treats all targets equally.

Source: TDC TAP (242 antibodies, paired VH/VL, 5 metrics).
We use VH only since our model is trained on heavy chains.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from utils.tokenizer import tokenize_single_chain

logger = logging.getLogger(__name__)

TAP_LABEL_NAMES = ["CDR_Length", "PSH", "PPC", "PNC", "SFvCSP"]


@dataclass
class LabelScaler:
    """Z-score scaler parameters for inverse transform."""

    mean: np.ndarray
    std: np.ndarray

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def _extract_vh(antibody_str: str) -> str | None:
    """Extract the VH sequence (first chain) from TDC's paired string format."""
    seqs = re.findall(r"'([A-Z]+)'", antibody_str)
    if not seqs:
        return None
    return seqs[0]


class DevelopabilityDataset(Dataset):
    """Sequence-level multi-target regression dataset for developability.

    Each item returns input_ids, attention_mask, special_tokens_mask,
    and labels (float tensor of shape (5,) -- standardized).
    """

    def __init__(
        self,
        sequences: list[str],
        labels: np.ndarray,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 160,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = sequences
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sequence = self.sequences[idx]
        encoding = tokenize_single_chain(self.tokenizer, sequence, self.max_length)
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
            "labels": self.labels[idx].tolist(),
        }


def load_developability_splits(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 160,
) -> tuple[DevelopabilityDataset, DevelopabilityDataset, DevelopabilityDataset, list[str], LabelScaler]:
    """Load TDC TAP, merge 5 labels, z-score standardize, return datasets + scaler.

    Returns:
        (train_ds, val_ds, test_ds, label_names, scaler)
    """
    from tdc.single_pred import Develop
    from tdc.utils import retrieve_label_name_list

    label_names = retrieve_label_name_list("TAP")
    logger.info("Loading TDC TAP with labels: %s", label_names)

    splits_per_label: dict[str, dict] = {}
    for label_name in label_names:
        data = Develop(name="TAP", label_name=label_name)
        splits_per_label[label_name] = data.get_split()

    def _build_split(split_name: str) -> tuple[list[str], np.ndarray]:
        first_label = label_names[0]
        df = splits_per_label[first_label][split_name]
        ids = df["Antibody_ID"].tolist()
        ab_strs = df["Antibody"].tolist()

        sequences: list[str] = []
        labels_rows: list[list[float]] = []

        id_to_labels: dict[str, dict[str, float]] = {}
        for label_name in label_names:
            split_df = splits_per_label[label_name][split_name]
            for _, row in split_df.iterrows():
                aid = row["Antibody_ID"]
                if aid not in id_to_labels:
                    id_to_labels[aid] = {}
                id_to_labels[aid][label_name] = float(row["Y"])

        for aid, ab_str in zip(ids, ab_strs):
            vh = _extract_vh(ab_str)
            if vh is None:
                continue
            label_vals = id_to_labels.get(aid, {})
            if len(label_vals) < len(label_names):
                continue
            sequences.append(vh)
            labels_rows.append([label_vals[ln] for ln in label_names])

        return sequences, np.array(labels_rows, dtype=np.float32)

    train_seqs, train_labels = _build_split("train")
    val_seqs, val_labels = _build_split("valid")
    test_seqs, test_labels = _build_split("test")

    train_mean = train_labels.mean(axis=0)
    train_std = train_labels.std(axis=0)
    train_std[train_std == 0] = 1.0
    scaler = LabelScaler(mean=train_mean, std=train_std)

    train_labels_z = (train_labels - train_mean) / train_std
    val_labels_z = (val_labels - train_mean) / train_std
    test_labels_z = (test_labels - train_mean) / train_std

    logger.info(
        "TAP splits: train=%d, val=%d, test=%d, targets=%d",
        len(train_seqs), len(val_seqs), len(test_seqs), len(label_names),
    )

    return (
        DevelopabilityDataset(train_seqs, train_labels_z, tokenizer, max_length),
        DevelopabilityDataset(val_seqs, val_labels_z, tokenizer, max_length),
        DevelopabilityDataset(test_seqs, test_labels_z, tokenizer, max_length),
        label_names,
        scaler,
    )
