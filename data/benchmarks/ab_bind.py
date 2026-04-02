"""AB-Bind mutation-effect dataset loader.

Downloads AB-Bind experimental data and PDB structures from GitHub,
extracts chain sequences, and applies mutations to produce
wildtype/mutant sequence pairs with experimental ddG labels.

Source: https://github.com/sarahsirin/AB-Bind-Database
Reference: Sirin et al., Protein Science (2016)
"""

from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_GITHUB_RAW = "https://raw.githubusercontent.com/sarahsirin/AB-Bind-Database/master"
_CSV_FILENAME = "AB-Bind_experimental_data.csv"

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}

_MUT_PATTERN = re.compile(r"^([A-Za-z]):([A-Z])(-?\d+)([A-Z])$")


def download_ab_bind(data_dir: str | Path) -> Path:
    """Download AB-Bind CSV and PDB files from GitHub."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / _CSV_FILENAME
    if not csv_path.exists():
        url = f"{_GITHUB_RAW}/{_CSV_FILENAME}"
        logger.info("Downloading AB-Bind CSV from %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            csv_path.write_bytes(resp.read())

    df = pd.read_csv(csv_path, encoding="latin-1")
    pdb_ids = df["#PDB"].unique()

    pdb_dir = data_dir / "pdbs"
    pdb_dir.mkdir(exist_ok=True)
    for pdb_id in pdb_ids:
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        if pdb_path.exists():
            continue
        url = f"{_GITHUB_RAW}/{pdb_id}.pdb"
        logger.info("Downloading %s.pdb", pdb_id)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                pdb_path.write_bytes(resp.read())
        except Exception as e:
            logger.warning("Could not download %s: %s", pdb_id, e)

    return data_dir


def _extract_chain_sequence(
    pdb_path: Path, chain_id: str
) -> tuple[str, dict[int, int]]:
    """Extract amino acid sequence and residue number mapping from a PDB chain.

    Returns:
        sequence: one-letter AA string
        resnum_to_pos: mapping from PDB residue number -> 0-indexed position
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    model = structure[0]

    if chain_id not in [c.id for c in model.get_chains()]:
        raise KeyError(f"Chain {chain_id!r} not found in {pdb_path.name}")

    chain = model[chain_id]
    sequence_parts: list[str] = []
    resnum_to_pos: dict[int, int] = {}
    pos = 0

    for residue in chain.get_residues():
        het_flag = residue.id[0]
        if het_flag != " ":
            continue
        resname = residue.get_resname().strip()
        aa = _THREE_TO_ONE.get(resname, "X")
        if aa == "X":
            continue
        resnum = residue.id[1]
        resnum_to_pos[resnum] = pos
        sequence_parts.append(aa)
        pos += 1

    return "".join(sequence_parts), resnum_to_pos


def parse_mutations(
    mutation_str: str,
) -> list[tuple[str, str, int, str]]:
    """Parse AB-Bind mutation string into structured mutations.

    Input format: 'D:A488G' or 'D:A488G,D:V486P,...'
    Returns: list of (chain_id, wildtype_aa, resnum, mutant_aa)
    """
    mutations: list[tuple[str, str, int, str]] = []
    for part in mutation_str.split(","):
        part = part.strip()
        m = _MUT_PATTERN.match(part)
        if m is None:
            logger.debug("Could not parse mutation: %r", part)
            continue
        chain_id, wt_aa, resnum_str, mut_aa = m.groups()
        mutations.append((chain_id, wt_aa, int(resnum_str), mut_aa))
    return mutations


def apply_mutations(
    sequence: str,
    mutations: list[tuple[str, int, str]],
    resnum_to_pos: dict[int, int],
) -> str | None:
    """Apply mutations to a wildtype sequence.

    Args:
        sequence: wildtype amino acid string
        mutations: list of (wt_aa, resnum, mut_aa) for this chain
        resnum_to_pos: PDB residue number -> sequence position mapping

    Returns:
        mutant sequence, or None if any mutation can't be applied
    """
    seq_list = list(sequence)
    for wt_aa, resnum, mut_aa in mutations:
        pos = resnum_to_pos.get(resnum)
        if pos is None:
            logger.debug("Residue %d not found in chain", resnum)
            return None
        if seq_list[pos] != wt_aa:
            logger.debug(
                "Wildtype mismatch at pos %d: expected %s, got %s",
                resnum, wt_aa, seq_list[pos],
            )
            return None
        seq_list[pos] = mut_aa
    return "".join(seq_list)


def load_ab_bind(
    data_dir: str | Path = "data/ab_bind",
) -> list[dict[str, Any]]:
    """Load AB-Bind data, extract sequences, and return mutation records.

    Each record contains:
        pdb_id, chain_id, wildtype_seq, mutant_seq, ddg, mutation_str, n_mutations
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / _CSV_FILENAME
    if not csv_path.exists():
        data_dir = download_ab_bind(data_dir)
        csv_path = data_dir / _CSV_FILENAME

    df = pd.read_csv(csv_path, encoding="latin-1")
    pdb_dir = data_dir / "pdbs"

    chain_cache: dict[tuple[str, str], tuple[str, dict[int, int]]] = {}
    records: list[dict[str, Any]] = []
    skipped = 0

    for _, row in df.iterrows():
        pdb_id = row["#PDB"]
        mutation_str = row["Mutation"]
        ddg = row["ddG(kcal/mol)"]

        mutations = parse_mutations(mutation_str)
        if not mutations:
            skipped += 1
            continue

        chains_in_row = {m[0] for m in mutations}

        for chain_id in chains_in_row:
            cache_key = (pdb_id, chain_id)
            if cache_key not in chain_cache:
                pdb_path = pdb_dir / f"{pdb_id}.pdb"
                if not pdb_path.exists():
                    continue
                try:
                    seq, rmap = _extract_chain_sequence(pdb_path, chain_id)
                    chain_cache[cache_key] = (seq, rmap)
                except (KeyError, Exception) as e:
                    logger.debug("Cannot extract %s chain %s: %s", pdb_id, chain_id, e)
                    chain_cache[cache_key] = ("", {})

            wt_seq, resnum_to_pos = chain_cache[cache_key]
            if not wt_seq:
                continue

            chain_muts = [(wt, rn, mut) for ch, wt, rn, mut in mutations if ch == chain_id]
            if not chain_muts:
                continue

            mutant_seq = apply_mutations(wt_seq, chain_muts, resnum_to_pos)
            if mutant_seq is None:
                skipped += 1
                continue

            records.append({
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "wildtype_seq": wt_seq,
                "mutant_seq": mutant_seq,
                "ddg": ddg,
                "mutation_str": mutation_str,
                "n_mutations": len(chain_muts),
            })

    logger.info(
        "AB-Bind loaded: %d mutation records from %d complexes (%d skipped)",
        len(records),
        len({r["pdb_id"] for r in records}),
        skipped,
    )
    return records
