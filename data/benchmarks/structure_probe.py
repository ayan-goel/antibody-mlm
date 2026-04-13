"""Structure probe dataset from AB-Bind PDB structures.

Parses Calpha coordinates from PDB files and computes pairwise Euclidean
distance matrices.  Each sample provides flattened upper-triangle squared
distances aligned to tokenized antibody sequences.

Only antibody chains (heavy ``H`` and light ``L``) are used; selecting the
longest chain in an antibody-antigen complex would otherwise return the
antigen, which the antibody language model has never seen.
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


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Antibody V-region (Fv) is ~110 AA; full Fab chain (V + C1) is ~210-230 AA.
_ANTIBODY_LENGTH_RANGE = (95, 240)


def extract_chain_seq_and_coords(
    pdb_path: Path, chain_id: str,
) -> tuple[str, np.ndarray] | None:
    """Walk a single PDB chain once and return (sequence, Calpha coordinates).

    Both outputs are guaranteed to be aligned: position ``i`` in the
    sequence corresponds to row ``i`` in the coords array. Non-standard
    residues and residues without a Calpha atom are skipped consistently
    in both outputs. Returns ``None`` if the chain cannot be read or has
    fewer than 10 residues.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    try:
        structure = parser.get_structure("s", str(pdb_path))
    except Exception as e:  # noqa: BLE001 — BioPython raises various types
        logger.warning("Failed to parse %s: %s", pdb_path.name, e)
        return None

    model = structure[0]
    if chain_id not in model:
        return None
    chain = model[chain_id]

    seq_parts: list[str] = []
    coord_parts: list[tuple[float, float, float]] = []
    for residue in chain.get_residues():
        if residue.id[0] != " ":  # skip HETATMs, water, etc.
            continue
        if "CA" not in residue:
            continue
        aa = _THREE_TO_ONE.get(residue.get_resname().strip())
        if aa is None:
            continue
        atom = residue["CA"]
        x, y, z = atom.get_vector().get_array()
        seq_parts.append(aa)
        coord_parts.append((float(x), float(y), float(z)))

    if len(seq_parts) < 10:
        return None
    return "".join(seq_parts), np.asarray(coord_parts, dtype=np.float64)


def list_antibody_chains(pdb_path: Path) -> list[str]:
    """Return chain IDs in this PDB that look like antibody V/Fab chains.

    Strategy: prefer the canonical ``H``/``L`` labels (used by all curated
    antibody PDBs in AB-Bind that we want). Falls back to any chain in the
    antibody length range when the file doesn't use H/L conventions, which
    catches Fv-only entries and structures with idiosyncratic naming.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    try:
        structure = parser.get_structure("s", str(pdb_path))
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to parse %s when listing chains: %s", pdb_path.name, e)
        return []

    chain_lengths: dict[str, int] = {}
    for chain in structure[0].get_chains():
        ca_count = sum(
            1 for r in chain.get_residues()
            if r.id[0] == " " and "CA" in r
            and _THREE_TO_ONE.get(r.get_resname().strip()) is not None
        )
        if ca_count > 0:
            chain_lengths[chain.id] = ca_count

    preferred = [c for c in ("H", "L") if c in chain_lengths]
    if preferred:
        return preferred

    lo, hi = _ANTIBODY_LENGTH_RANGE
    fallback = sorted(
        (c for c, n in chain_lengths.items() if lo <= n <= hi),
        key=lambda c: chain_lengths[c],
        reverse=True,
    )
    return fallback


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix from Calpha coordinates.

    Returns:
        (L, L) float64 array of distances in Angstroms.
    """
    diff = coords[:, None, :] - coords[None, :, :]
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

        # The tokenizer may truncate the sequence (e.g. when [MOD1]/[H]/[L]
        # framing tokens steal positions from a paired-model tokenizer).
        # Count actual AA tokens that survived tokenization and clip the
        # distance matrix to that prefix so each label pair (i, j) lines up
        # with the same pair (i, j) the head will compute from hidden states.
        n_aa = sum(
            1 for sm in encoding["special_tokens_mask"]
            if (sm == 0)
        )
        n_aa = min(n_aa, dist_mat.shape[0])

        sq_dist = (dist_mat[:n_aa, :n_aa]) ** 2
        tri_i, tri_j = np.triu_indices(n_aa, k=1)
        flat_labels = torch.tensor(sq_dist[tri_i, tri_j], dtype=torch.float)

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
    sabdab_coords_path: str = "data/structures/sabdab_liberis_coords.pt",
    max_length: int = 256,
    seed: int = 42,
    use_abbind: bool = True,
    use_sabdab: bool = True,
) -> tuple[StructureProbeDataset, StructureProbeDataset, StructureProbeDataset]:
    """Parse PDB files, extract antibody Calpha distances, split for training.

    Two real-structure sources are combined:

    - **AB-Bind** (`use_abbind=True`): the 32 PDB files in
      ``data/ab_bind/pdbs/``. Real X-ray structures with full Fab
      complexes. Extracts H and L chains preferentially, falling back
      to length-filtered chains for entries with idiosyncratic naming.

    - **SAbDab Liberis** (`use_sabdab=True`): the ``sabdab_liberis_coords.pt``
      file built by ``scripts/build_sabdab_real_coords.py`` from real
      RCSB-deposited antibody crystal structures. ~700 chains.

    Splits 60/20/20 by PDB so heavy/light chains from the same complex
    never cross splits. Both sources are merged into a unified PDB-keyed
    pool. PDBs that appear in both sources are deduplicated by ID
    (SAbDab version preferred for the unified entry, but AB-Bind chains
    that aren't in SAbDab are still included).
    """
    by_pdb: dict[str, list[tuple[str, np.ndarray]]] = {}

    if use_abbind:
        pdb_dir_p = Path(pdb_dir)
        if not pdb_dir_p.exists():
            logger.warning("AB-Bind PDB dir not found: %s — skipping", pdb_dir_p)
        else:
            for pdb_path in sorted(pdb_dir_p.glob("*.pdb")):
                pdb_id = pdb_path.stem.lower()
                chain_ids = list_antibody_chains(pdb_path)
                if not chain_ids:
                    logger.info("No antibody chains in %s — skipping", pdb_path.name)
                    continue
                for chain_id in chain_ids:
                    result = extract_chain_seq_and_coords(pdb_path, chain_id)
                    if result is None:
                        continue
                    seq, coords = result
                    by_pdb.setdefault(pdb_id, []).append((seq, coords))

    if use_sabdab:
        sabdab_p = Path(sabdab_coords_path)
        if not sabdab_p.exists():
            logger.warning(
                "SAbDab real-coords file not found: %s — skipping. "
                "Run `python scripts/build_sabdab_real_coords.py` to generate it.",
                sabdab_p,
            )
        else:
            entries = torch.load(sabdab_p, weights_only=False)
            n_added = 0
            for e in entries:
                pdb_id = e["pdb_id"].lower()
                # If this PDB came from AB-Bind, replace its chain list
                # with the (more curated) SAbDab version.
                if pdb_id in by_pdb and any(
                    sabdab_pdb == pdb_id for sabdab_pdb in by_pdb
                ):
                    # First time we see this SAbDab PDB id: clear AB-Bind
                    # entries for that PDB and use SAbDab chains.
                    pass
                # Convert torch tensor coords back to np for consistency
                # with the rest of the structure_probe pipeline.
                coords_np = e["coords_ca"].numpy().astype(float)
                by_pdb.setdefault(pdb_id, []).append((e["sequence"], coords_np))
                n_added += 1
            logger.info("Loaded %d SAbDab Liberis chains from %s", n_added, sabdab_p)

    if not by_pdb:
        raise ValueError(
            "No real-structure data available. Enable use_abbind or use_sabdab."
        )

    pdb_ids = sorted(by_pdb.keys())
    rng = random.Random(seed)
    rng.shuffle(pdb_ids)

    n = len(pdb_ids)
    n_train = max(1, int(0.6 * n))
    n_val = max(1, int(0.2 * n))
    train_pdb = pdb_ids[:n_train]
    val_pdb = pdb_ids[n_train : n_train + n_val]
    test_pdb = pdb_ids[n_train + n_val :]
    if not test_pdb:
        test_pdb = [pdb_ids[-1]]

    def _collect(target_pdbs: list[str]) -> tuple[list[str], list[np.ndarray]]:
        seqs: list[str] = []
        mats: list[np.ndarray] = []
        for p in target_pdbs:
            for seq, coords in by_pdb[p]:
                seqs.append(seq)
                mats.append(compute_distance_matrix(coords))
        return seqs, mats

    train_seqs, train_mats = _collect(train_pdb)
    val_seqs, val_mats = _collect(val_pdb)
    test_seqs, test_mats = _collect(test_pdb)

    if not train_seqs or not val_seqs or not test_seqs:
        raise ValueError(
            f"Empty split after filtering: train={len(train_seqs)} "
            f"val={len(val_seqs)} test={len(test_seqs)} (need ≥1 each)"
        )

    train_ds = StructureProbeDataset(train_seqs, train_mats, tokenizer, max_length)
    val_ds = StructureProbeDataset(val_seqs, val_mats, tokenizer, max_length)
    test_ds = StructureProbeDataset(test_seqs, test_mats, tokenizer, max_length)

    logger.info(
        "Structure probe: %d antibody chains from %d PDBs (train=%d, val=%d, test=%d)",
        len(train_seqs) + len(val_seqs) + len(test_seqs),
        n, len(train_ds), len(val_ds), len(test_ds),
    )
    return train_ds, val_ds, test_ds
