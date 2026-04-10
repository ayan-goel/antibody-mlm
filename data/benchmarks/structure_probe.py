"""Structure probe dataset from AB-Bind PDB structures.

Parses Calpha coordinates from PDB files and computes pairwise Euclidean
distance matrices.  Each sample provides flattened upper-triangle squared
distances aligned to tokenized antibody sequences.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from utils.tokenizer import tokenize_single_chain

logger = logging.getLogger(__name__)


def extract_ca_coords(
    pdb_path: Path,
) -> dict[str, list[tuple[float, float, float]]]:
    """Parse PDB file and return Calpha coordinates per chain.

    Returns:
        {chain_id: [(x, y, z), ...]} for standard amino acid residues.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    model = structure[0]

    result: dict[str, list[tuple[float, float, float]]] = {}
    for chain in model.get_chains():
        coords: list[tuple[float, float, float]] = []
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue
            if "CA" not in residue:
                continue
            atom = residue["CA"]
            x, y, z = atom.get_vector().get_array()
            coords.append((float(x), float(y), float(z)))
        if coords:
            result[chain.id] = coords
    return result


def extract_sequence_from_pdb(pdb_path: Path, chain_id: str) -> str:
    """Extract amino acid sequence for a specific chain from a PDB file."""
    from Bio.PDB import PDBParser

    _THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    model = structure[0]
    chain = model[chain_id]

    parts: list[str] = []
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        resname = residue.get_resname().strip()
        aa = _THREE_TO_ONE.get(resname)
        if aa and "CA" in residue:
            parts.append(aa)
    return "".join(parts)


def compute_distance_matrix(coords: list[tuple[float, float, float]]) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix from Calpha coordinates.

    Returns:
        (L, L) float64 array of distances in Angstroms.
    """
    arr = np.array(coords, dtype=np.float64)
    diff = arr[:, None, :] - arr[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def max_pairs_for_length(max_length: int) -> int:
    """Maximum number of upper-triangle pairs for a given max token length."""
    max_aa = max_length - 2
    return max_aa * (max_aa - 1) // 2


class StructureProbeDataset(Dataset):
    """Pairwise squared-distance regression dataset.

    Each item returns input_ids, attention_mask, special_tokens_mask,
    and labels (flat upper-triangle squared distances, padded to max_pairs).
    """

    def __init__(
        self,
        sequences: list[str],
        distance_matrices: list[np.ndarray],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> None:
        self.sequences = sequences
        self.distance_matrices = distance_matrices
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_pairs = max_pairs_for_length(max_length)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sequence = self.sequences[idx]
        dist_mat = self.distance_matrices[idx]

        encoding = tokenize_single_chain(self.tokenizer, sequence, self.max_length)

        # Squared distances (probe predicts ||Bh_i - Bh_j||^2)
        sq_dist = dist_mat ** 2
        n = sq_dist.shape[0]
        tri_i, tri_j = np.triu_indices(n, k=1)
        flat_labels = torch.tensor(sq_dist[tri_i, tri_j], dtype=torch.float)

        # Pad to max_pairs with -100
        padded = torch.full((self.max_pairs,), -100.0)
        n_pairs = min(flat_labels.size(0), self.max_pairs)
        padded[:n_pairs] = flat_labels[:n_pairs]

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
            "labels": padded.tolist(),
        }


def load_structure_probe_splits(
    tokenizer: PreTrainedTokenizerBase,
    pdb_dir: str = "data/ab_bind/pdbs",
    max_length: int = 256,
    seed: int = 42,
) -> tuple[StructureProbeDataset, StructureProbeDataset, StructureProbeDataset]:
    """Parse PDB files, extract heavy-chain Calpha distances, split for training.

    Uses the AB-Bind PDB collection (32 structures). For each PDB, extracts
    the longest chain's Calpha coordinates and sequence.

    Returns (train, val, test) StructureProbeDatasets.
    """
    pdb_dir = Path(pdb_dir)
    if not pdb_dir.exists():
        raise FileNotFoundError(
            f"PDB directory not found: {pdb_dir}. "
            "Run the AB-Bind download first."
        )

    sequences: list[str] = []
    dist_matrices: list[np.ndarray] = []

    for pdb_path in sorted(pdb_dir.glob("*.pdb")):
        try:
            chain_coords = extract_ca_coords(pdb_path)
            if not chain_coords:
                continue
            # Use the longest chain (most likely the antibody heavy chain)
            best_chain = max(chain_coords, key=lambda c: len(chain_coords[c]))
            coords = chain_coords[best_chain]
            if len(coords) < 10:
                continue
            seq = extract_sequence_from_pdb(pdb_path, best_chain)
            if len(seq) != len(coords):
                logger.warning(
                    "Sequence/coords length mismatch in %s chain %s: %d vs %d",
                    pdb_path.name, best_chain, len(seq), len(coords),
                )
                min_len = min(len(seq), len(coords))
                seq = seq[:min_len]
                coords = coords[:min_len]
            dist_mat = compute_distance_matrix(coords)
            sequences.append(seq)
            dist_matrices.append(dist_mat)
        except Exception as e:
            logger.warning("Failed to process %s: %s", pdb_path.name, e)

    logger.info("Structure probe: loaded %d structures from %s", len(sequences), pdb_dir)

    if len(sequences) < 3:
        raise ValueError(
            f"Too few valid structures ({len(sequences)}) for train/val/test split"
        )

    # Deterministic shuffle and split: 60/20/20
    indices = list(range(len(sequences)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n = len(indices)
    n_train = max(1, int(0.6 * n))
    n_val = max(1, int(0.2 * n))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    if not test_idx:
        test_idx = [indices[-1]]

    def _make_ds(idxs: list[int]) -> StructureProbeDataset:
        return StructureProbeDataset(
            [sequences[i] for i in idxs],
            [dist_matrices[i] for i in idxs],
            tokenizer, max_length,
        )

    train_ds = _make_ds(train_idx)
    val_ds = _make_ds(val_idx)
    test_ds = _make_ds(test_idx)

    logger.info(
        "Structure probe splits: train=%d, val=%d, test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_ds, val_ds, test_ds
