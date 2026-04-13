"""Contact map dataset for downstream antibody structure evaluation.

Two label sources are supported:

1. **Real X-ray crystal coordinates** from SAbDab Liberis (default):
   ``load_sabdab_contact_splits`` reads
   ``data/structures/sabdab_liberis_coords.pt`` (built by
   ``scripts/build_sabdab_real_coords.py``) and turns each chain's real
   Calpha coordinates into a binary contact matrix at the standard
   8 Å threshold (CASP / CAMEO convention).

2. **ESM-2 predicted contact kNN** (legacy / deprecated):
   ``load_contact_map_splits`` reads
   ``data/structures/oas_vh_500k_coords.pt``, which contains the top-k
   ESM-2-predicted contacts per residue. This is a circular metric
   (we measure whether our antibody LM imitates a general protein LM)
   and is kept only for backwards compatibility — prefer the SAbDab
   loader for evaluation.
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


CONTACT_THRESHOLD_ANGSTROM = 8.0


def coords_to_contact_matrix(
    coords_ca: torch.Tensor,
    seq_len: int | None = None,
    threshold_a: float = CONTACT_THRESHOLD_ANGSTROM,
) -> torch.Tensor:
    """Build a binary Calpha contact matrix from real coordinates.

    Args:
        coords_ca: (L, 3) tensor of Calpha xyz positions in angstroms.
        seq_len: clip coords to the first ``seq_len`` residues. If
            None, uses the full coords.
        threshold_a: distance cutoff in angstroms. 8 Å is the standard
            protein contact prediction threshold (CASP, CAMEO).

    Returns:
        (n, n) float32 binary tensor where ``n = seq_len`` (or
        ``coords_ca.size(0)``). Symmetric, diagonal zeroed.
    """
    coords = coords_ca[: seq_len if seq_len else coords_ca.size(0)].float()
    n = coords.size(0)
    if n == 0:
        return torch.zeros(0, 0, dtype=torch.float)
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist_sq = (diff ** 2).sum(dim=-1)
    contact = (dist_sq < threshold_a * threshold_a).float()
    contact.fill_diagonal_(0.0)
    return contact


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

    Two construction modes:

    1. **Real-coordinates mode** (preferred): pass ``sequences`` and
       ``coords_list`` (Calpha xyz tensors). Contacts are computed at
       the 8 Å threshold.

    2. **Legacy ESM-2 kNN mode** (deprecated): pass ``records`` and
       ``knn_entries`` (each dict has a ``knn_indices`` tensor). Contacts
       are reconstructed via :func:`knn_to_contact_matrix`.
    """

    def __init__(
        self,
        records: list[dict] | None = None,
        knn_entries: list[dict] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_length: int = 160,
        *,
        sequences: list[str] | None = None,
        coords_list: list[torch.Tensor] | None = None,
    ) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_pairs = max_pairs_for_length(max_length)

        if sequences is not None and coords_list is not None:
            self._mode = "coords"
            if len(sequences) != len(coords_list):
                raise ValueError(
                    f"sequences ({len(sequences)}) and coords_list "
                    f"({len(coords_list)}) length mismatch"
                )
            self.sequences = sequences
            self.coords_list = coords_list
            self.records = None
            self.knn_entries = None
        elif records is not None and knn_entries is not None:
            self._mode = "knn"
            self.records = records
            self.knn_entries = knn_entries
            self.sequences = None
            self.coords_list = None
        else:
            raise ValueError(
                "Must provide either (sequences + coords_list) or (records + knn_entries)"
            )

    def __len__(self) -> int:
        if self._mode == "coords":
            return len(self.sequences)
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._mode == "coords":
            sequence = self.sequences[idx]
            coords = self.coords_list[idx]
        else:
            sequence = self.records[idx]["sequence"]
            coords = None

        encoding = tokenize_single_chain(self.tokenizer, sequence, self.max_length)

        # Count actual AA tokens that survived tokenization (the model
        # may add framing specials like [MOD1][H] for paired tokenizers).
        # Clip the contact matrix to that prefix so each label pair (i, j)
        # lines up with the same pair (i, j) the head will compute from
        # hidden states.
        n_aa = sum(1 for sm in encoding["special_tokens_mask"] if sm == 0)

        if self._mode == "coords":
            n_aa = min(n_aa, coords.size(0))
            contact = coords_to_contact_matrix(coords, seq_len=n_aa)
        else:
            knn = self.knn_entries[idx]["knn_indices"]
            n_aa = min(n_aa, knn.size(0))
            knn = knn[:n_aa]
            contact = knn_to_contact_matrix(knn, n_aa)

        flat_labels = _upper_triangle_flat(contact)

        padded = torch.full((self.max_pairs,), -100.0)
        n = min(flat_labels.size(0), self.max_pairs)
        padded[:n] = flat_labels[:n]

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
            "labels": padded.tolist(),
        }


def load_sabdab_contact_splits(
    tokenizer: PreTrainedTokenizerBase,
    coords_path: str = "data/structures/sabdab_liberis_coords.pt",
    max_length: int = 160,
    seed: int = 42,
) -> tuple[ContactMapDataset, ContactMapDataset, ContactMapDataset]:
    """Real-X-ray-coordinate contact map splits from SAbDab Liberis.

    Reads ``data/structures/sabdab_liberis_coords.pt`` (built by
    ``scripts/build_sabdab_real_coords.py``) and splits 60/20/20 by
    PDB so heavy and light chains from the same complex never cross
    splits.

    Each chain provides:
      - sequence: AA from the PDB ATOM records
      - coords_ca: real Calpha xyz from the crystal
    Contact labels are computed at the 8 Å Calpha-Calpha threshold.
    """
    coords_path = Path(coords_path)
    if not coords_path.exists():
        raise FileNotFoundError(
            f"SAbDab real-coords file not found: {coords_path}\n"
            "Run `python scripts/build_sabdab_real_coords.py` first."
        )
    logger.info("Loading SAbDab real-coordinate contact map data from %s", coords_path)
    entries = torch.load(coords_path, weights_only=False)
    if not entries:
        raise ValueError(f"No entries in {coords_path}")

    # Group chains by PDB so we can split by complex (not by chain).
    by_pdb: dict[str, list[dict]] = {}
    for e in entries:
        by_pdb.setdefault(e["pdb_id"], []).append(e)
    pdb_ids = sorted(by_pdb.keys())

    rng = random.Random(seed)
    rng.shuffle(pdb_ids)
    n = len(pdb_ids)
    n_train = max(1, int(0.6 * n))
    n_val = max(1, int(0.2 * n))
    train_pdbs = pdb_ids[:n_train]
    val_pdbs = pdb_ids[n_train : n_train + n_val]
    test_pdbs = pdb_ids[n_train + n_val :]
    if not test_pdbs:
        test_pdbs = [pdb_ids[-1]]

    def _collect(pdbs: list[str]) -> tuple[list[str], list[torch.Tensor]]:
        seqs: list[str] = []
        coords: list[torch.Tensor] = []
        for p in pdbs:
            for e in by_pdb[p]:
                seqs.append(e["sequence"])
                coords.append(e["coords_ca"])
        return seqs, coords

    train_seqs, train_coords = _collect(train_pdbs)
    val_seqs, val_coords = _collect(val_pdbs)
    test_seqs, test_coords = _collect(test_pdbs)

    train_ds = ContactMapDataset(
        sequences=train_seqs, coords_list=train_coords,
        tokenizer=tokenizer, max_length=max_length,
    )
    val_ds = ContactMapDataset(
        sequences=val_seqs, coords_list=val_coords,
        tokenizer=tokenizer, max_length=max_length,
    )
    test_ds = ContactMapDataset(
        sequences=test_seqs, coords_list=test_coords,
        tokenizer=tokenizer, max_length=max_length,
    )

    logger.info(
        "SAbDab contact map: %d antibody chains from %d PDBs (train=%d, val=%d, test=%d)",
        len(train_seqs) + len(val_seqs) + len(test_seqs),
        len(train_pdbs) + len(val_pdbs) + len(test_pdbs),
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_ds, val_ds, test_ds


def load_contact_map_splits(
    tokenizer: PreTrainedTokenizerBase,
    data_path: str = "data/processed/oas_vh_500k.jsonl",
    coords_path: str = "data/structures/oas_vh_500k_coords.pt",
    max_length: int = 160,
    max_samples: int = 5000,
    seed: int = 42,
) -> tuple[ContactMapDataset, ContactMapDataset, ContactMapDataset]:
    """[DEPRECATED] Legacy ESM-2 kNN contact map splits.

    This loader uses ESM-2-predicted contact maps as ground truth, which
    is a circular metric. Prefer :func:`load_sabdab_contact_splits` for
    real X-ray crystal coordinates.

    Returns (train, val, test) ContactMapDatasets.
    """
    logger.warning(
        "load_contact_map_splits is deprecated; use load_sabdab_contact_splits "
        "for real X-ray crystal coordinates from SAbDab."
    )
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
        return ContactMapDataset(records=recs, knn_entries=knns,
                                 tokenizer=tokenizer, max_length=max_length)

    train_ds = _make_dataset(train_idx)
    val_ds = _make_dataset(val_idx)
    test_ds = _make_dataset(test_idx)

    logger.info(
        "Contact map splits: train=%d, val=%d, test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_ds, val_ds, test_ds
