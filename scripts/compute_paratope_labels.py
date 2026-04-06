"""Compute paratope labels from antibody–antigen complex structures.

Reads SAbDab/AACDB PDB complex files, identifies antibody residues
within a distance threshold of antigen atoms (paratope), and produces
a sidecar .pt file aligned by index with the training JSONL.

Output .pt format:
    List[dict | None] of length N (same as JSONL record count).
    Each entry is:
        {"paratope_labels": FloatTensor(L,)} — per-residue [0, 1] labels

Label generation uses minimum heavy-atom distance between antibody
and antigen residues. A residue is labeled paratope (1.0) if its
minimum distance to any antigen atom is <= delta (default 4.5 Å).

Usage:
    python scripts/compute_paratope_labels.py \
        --input data/processed/oas_vh_500k.jsonl \
        --output data/structures/oas_vh_500k_paratope.pt \
        --pdb_dir data/structures/sabdab_complexes \
        --mapping_file data/structures/sequence_to_pdb.csv \
        --distance_threshold 4.5
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.io import load_jsonl

logger = logging.getLogger(__name__)


def compute_paratope_residues(
    structure,
    ab_chain_ids: list[str],
    ag_chain_ids: list[str],
    delta: float = 4.5,
) -> set[int]:
    """Identify paratope residues from a PDB structure.

    For each antibody residue, computes the minimum heavy-atom distance
    to any antigen atom. Returns the set of antibody residue indices
    (0-based, within the antibody chain) where this distance is <= delta.

    Args:
        structure: BioPython Structure object.
        ab_chain_ids: Chain IDs for antibody (e.g., ["H"] or ["H", "L"]).
        ag_chain_ids: Chain IDs for antigen.
        delta: Distance threshold in angstroms.

    Returns:
        Set of 0-based residue indices in the antibody chain(s) that are
        within delta of antigen atoms.
    """
    from Bio.PDB import Selection

    model = structure[0]

    ag_atoms = []
    for chain_id in ag_chain_ids:
        if chain_id not in model:
            continue
        for residue in model[chain_id]:
            if residue.id[0] != " ":
                continue  # skip hetero residues
            for atom in residue:
                if atom.element == "H":
                    continue  # skip hydrogen
                ag_atoms.append(atom.get_vector().get_array())

    if not ag_atoms:
        return set()

    import numpy as np
    ag_coords = np.array(ag_atoms)

    paratope_indices: set[int] = set()
    residue_idx = 0

    for chain_id in ab_chain_ids:
        if chain_id not in model:
            continue
        for residue in model[chain_id]:
            if residue.id[0] != " ":
                continue  # skip hetero residues
            min_dist = float("inf")
            for atom in residue:
                if atom.element == "H":
                    continue
                ab_coord = atom.get_vector().get_array()
                dists = np.linalg.norm(ag_coords - ab_coord, axis=1)
                min_dist = min(min_dist, dists.min())
            if min_dist <= delta:
                paratope_indices.add(residue_idx)
            residue_idx += 1

    return paratope_indices


def extract_sequence_from_chain(structure, chain_ids: list[str]) -> str:
    """Extract amino acid sequence from PDB chain(s)."""
    from Bio.PDB.Polypeptide import protein_letters_3to1

    model = structure[0]
    sequence = []
    for chain_id in chain_ids:
        if chain_id not in model:
            continue
        for residue in model[chain_id]:
            if residue.id[0] != " ":
                continue
            resname = residue.get_resname().strip()
            aa = protein_letters_3to1.get(resname, "X")
            sequence.append(aa)
    return "".join(sequence)


def load_mapping(mapping_path: str) -> dict[str, dict]:
    """Load sequence-to-PDB mapping from CSV.

    Expected columns: sequence, pdb_file, ab_chains, ag_chains
    ab_chains and ag_chains are comma-separated chain IDs (e.g., "H,L" and "A").
    """
    mapping: dict[str, dict] = {}
    with open(mapping_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row["sequence"]
            mapping[seq] = {
                "pdb_file": row["pdb_file"],
                "ab_chains": row["ab_chains"].split(","),
                "ag_chains": row["ag_chains"].split(","),
            }
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paratope labels from antibody-antigen complex PDBs"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to processed JSONL file",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output .pt sidecar file",
    )
    parser.add_argument(
        "--pdb_dir", type=str, required=True,
        help="Directory containing antibody-antigen complex PDB files",
    )
    parser.add_argument(
        "--mapping_file", type=str, required=True,
        help="CSV mapping training sequences to PDB files and chain IDs",
    )
    parser.add_argument(
        "--distance_threshold", type=float, default=4.5,
        help="Distance threshold in angstroms for paratope definition (default: 4.5)",
    )
    parser.add_argument(
        "--max_sequences", type=int, default=0,
        help="Max sequences to process (0 = all)",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to existing .pt file to resume from",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Loading sequences from %s", args.input)
    records = load_jsonl(args.input)
    total = len(records)
    if args.max_sequences > 0:
        total = min(total, args.max_sequences)
    logger.info("Will process %d / %d sequences", total, len(records))

    logger.info("Loading sequence-to-PDB mapping from %s", args.mapping_file)
    mapping = load_mapping(args.mapping_file)
    logger.info("Mapping contains %d entries", len(mapping))

    results: list[dict[str, torch.Tensor] | None] = [None] * len(records)

    if args.resume and Path(args.resume).exists():
        logger.info("Resuming from %s", args.resume)
        existing = torch.load(args.resume, weights_only=False)
        for i, entry in enumerate(existing):
            if i < len(results):
                results[i] = entry
        done = sum(1 for e in existing[:total] if e is not None)
        logger.info("Loaded %d existing predictions", done)

    from Bio.PDB import PDBParser
    pdb_parser = PDBParser(QUIET=True)
    pdb_dir = Path(args.pdb_dir)

    processed = 0
    matched = 0

    for i in range(total):
        if results[i] is not None:
            processed += 1
            continue

        sequence = records[i]["sequence"]
        entry = mapping.get(sequence)

        if entry is None:
            processed += 1
            continue

        pdb_path = pdb_dir / entry["pdb_file"]
        if not pdb_path.exists():
            logger.warning("PDB file not found: %s", pdb_path)
            processed += 1
            continue

        try:
            structure = pdb_parser.get_structure("complex", str(pdb_path))
            paratope_set = compute_paratope_residues(
                structure,
                ab_chain_ids=entry["ab_chains"],
                ag_chain_ids=entry["ag_chains"],
                delta=args.distance_threshold,
            )

            seq_len = len(sequence)
            labels = torch.zeros(seq_len, dtype=torch.float)
            for idx in paratope_set:
                if idx < seq_len:
                    labels[idx] = 1.0

            results[i] = {"paratope_labels": labels}
            matched += 1

        except Exception as e:
            logger.warning("Failed for sequence %d: %s", i, e)

        processed += 1

        if processed % 10000 < 1:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(results, str(output_path))
            logger.info("Checkpoint saved to %s (%d/%d processed, %d matched)",
                        output_path, processed, total, matched)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(output_path))

    logger.info("Done. Saved %d entries to %s", len(results), output_path)
    logger.info("Coverage: %d/%d (%.1f%%)", matched, total,
                matched / total * 100 if total > 0 else 0)


if __name__ == "__main__":
    main()
