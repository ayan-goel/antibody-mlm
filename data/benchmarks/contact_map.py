"""Contact map dataset from kNN structure annotations.

Converts kNN neighbor indices (derived from predicted Calpha coordinates)
into binary contact matrices.  Each sample provides flattened upper-triangle
contact labels aligned to tokenized antibody sequences.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from utils.io import load_jsonl
from utils.tokenizer import tokenize_single_chain

logger = logging.getLogger(__name__)


def knn_to_contact_matrix(knn_indices: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Convert kNN indices to a symmetric binary contact matrix.

    Args:
        knn_indices: (L, k) — per-residue neighbor indices.
        seq_len: number of residues (= L).

    Returns:
        (L, L) binary tensor — 1 where residues i and j are kNN neighbors.
    """
    contact = torch.zeros(seq_len, seq_len, dtype=torch.float)
    for i in range(seq_len):
        for j in knn_indices[i]:
            j = j.item()
            if 0 <= j < seq_len:
                contact[i, j] = 1.0
                contact[j, i] = 1.0
    contact.fill_diagonal_(0.0)
    return contact


def _upper_triangle_flat(matrix: torch.Tensor) -> torch.Tensor:
    """Extract strict upper triangle as a flat 1-D tensor."""
    n = matrix.size(0)
    idx_i, idx_j = torch.triu_indices(n, n, offset=1)
    return matrix[idx_i, idx_j]


def max_pairs_for_length(max_length: int) -> int:
    """Maximum number of upper-triangle pairs for a given max token length.

    Accounts for [CLS] and [SEP] tokens (2 fewer AA positions).
    """
    max_aa = max_length - 2
    return max_aa * (max_aa - 1) // 2


class ContactMapDataset(Dataset):
    """Pairwise binary contact prediction dataset.

    Each item returns input_ids, attention_mask, special_tokens_mask,
    and labels (flat upper-triangle binary contacts, padded to max_pairs).
    """

    def __init__(
        self,
        records: list[dict],
        knn_entries: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 160,
    ) -> None:
        self.records = records
        self.knn_entries = knn_entries
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_pairs = max_pairs_for_length(max_length)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        sequence = record["sequence"]

        encoding = tokenize_single_chain(self.tokenizer, sequence, self.max_length)

        knn = self.knn_entries[idx]["knn_indices"]
        seq_len = min(len(sequence), knn.size(0))
        knn = knn[:seq_len]
        contact = knn_to_contact_matrix(knn, seq_len)
        flat_labels = _upper_triangle_flat(contact)

        # Pad to max_pairs with -100 (ignore index)
        padded = torch.full((self.max_pairs,), -100.0)
        n = min(flat_labels.size(0), self.max_pairs)
        padded[:n] = flat_labels[:n]

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
            "labels": padded.tolist(),
        }


def load_contact_map_splits(
    tokenizer: PreTrainedTokenizerBase,
    data_path: str = "data/processed/oas_vh_500k.jsonl",
    coords_path: str = "data/structures/oas_vh_500k_coords.pt",
    max_length: int = 160,
    max_samples: int = 5000,
    seed: int = 42,
) -> tuple[ContactMapDataset, ContactMapDataset, ContactMapDataset]:
    """Load kNN data, filter valid entries, and split 60/20/20.

    Returns (train, val, test) ContactMapDatasets.
    """
    logger.info("Loading contact map data from %s + %s", data_path, coords_path)
    records = load_jsonl(data_path)
    coords = torch.load(coords_path, weights_only=False)

    # Filter to samples that have kNN data
    valid_indices = [
        i for i in range(min(len(records), len(coords)))
        if coords[i] is not None and "knn_indices" in coords[i]
    ]
    logger.info("Contact map: %d/%d samples have kNN data", len(valid_indices), len(records))

    # Subsample
    rng = random.Random(seed)
    if len(valid_indices) > max_samples:
        valid_indices = rng.sample(valid_indices, max_samples)
    else:
        rng.shuffle(valid_indices)

    # Split 60/20/20
    n = len(valid_indices)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train_idx = valid_indices[:n_train]
    val_idx = valid_indices[n_train:n_train + n_val]
    test_idx = valid_indices[n_train + n_val:]

    def _make_dataset(indices: list[int]) -> ContactMapDataset:
        recs = [records[i] for i in indices]
        knns = [coords[i] for i in indices]
        return ContactMapDataset(recs, knns, tokenizer, max_length)

    train_ds = _make_dataset(train_idx)
    val_ds = _make_dataset(val_idx)
    test_ds = _make_dataset(test_idx)

    logger.info(
        "Contact map splits: train=%d, val=%d, test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_ds, val_ds, test_ds
