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
from typing import TYPE_CHECKING, Any

import pandas as pd
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

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


# ---------------------------------------------------------------------------
# Supervised probe splits (ddG regression on frozen embeddings)
# ---------------------------------------------------------------------------

class ABBindDataset(Dataset):
    """ddG regression dataset over AB-Bind mutants.

    Encodes the **mutant** sequence only. The wildtype is constant within a
    complex, so it contributes a constant offset that cannot affect the
    within-complex ranking that per-complex Spearman measures — which keeps
    this inside the standard single-sequence probe pipeline.

    ``labels`` is a 2-vector ``[ddg_z, group_index]``. The group index rides
    along in the label tensor because the base task's ``compute_metrics``
    receives only (predictions, labels), and per-complex metrics need to know
    which complex each row came from. The loss reads column 0 only.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: "PreTrainedTokenizerBase",
        group_to_index: dict[str, int],
        mean: float = 0.0,
        std: float = 1.0,
        max_length: int = 160,
    ) -> None:
        from utils.tokenizer import tokenize_single_chain

        self._tokenize = tokenize_single_chain
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = records
        self.group_to_index = group_to_index
        self.mean = mean
        self.std = std if std > 0 else 1.0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        encoding = self._tokenize(
            self.tokenizer, rec["mutant_seq"], self.max_length,
        )
        ddg_z = (float(rec["ddg"]) - self.mean) / self.std
        encoding["labels"] = [ddg_z, float(self.group_to_index[rec["pdb_id"]])]
        return encoding

    @property
    def groups(self) -> list[str]:
        """PDB id per record, in dataset order."""
        return [r["pdb_id"] for r in self.records]


def _assign_complexes_to_splits(
    sizes: dict[str, int], fracs: tuple[float, float, float], seed: int,
) -> dict[str, str]:
    """Greedily bin-pack complexes into train/val/test by record count.

    AB-Bind is severely skewed — one complex holds ~35% of all mutants — so a
    naive split *by complex count* produces wildly unbalanced *record* counts.
    Complexes are shuffled, then assigned largest-first to whichever split is
    furthest below its target share. Complexes never straddle splits, so the
    model cannot memorise a per-complex ddG offset and reuse it at test time.
    """
    import random

    total = sum(sizes.values())
    targets = {
        "train": fracs[0] * total, "val": fracs[1] * total, "test": fracs[2] * total,
    }
    current = {"train": 0, "val": 0, "test": 0}
    assignment: dict[str, str] = {}

    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    for pdb in sorted(order, key=lambda p: -sizes[p]):
        split = max(targets, key=lambda s: targets[s] - current[s])
        assignment[pdb] = split
        current[split] += sizes[pdb]
    return assignment


def load_ab_bind_splits(
    tokenizer: "PreTrainedTokenizerBase",
    data_dir: str | Path = "data/ab_bind",
    max_length: int = 160,
    fracs: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
    min_mutants_per_complex: int = 3,
) -> tuple[ABBindDataset, ABBindDataset, ABBindDataset]:
    """Return (train, val, test) ddG datasets split by complex.

    Complexes with fewer than ``min_mutants_per_complex`` records are dropped:
    per-complex Spearman is undefined for them, matching the zero-shot
    benchmark in ``scripts/benchmark_mutations.py``.

    Labels are z-scored using **training-split** statistics only.
    """
    from collections import Counter

    records = load_ab_bind(data_dir)

    counts = Counter(r["pdb_id"] for r in records)
    keep = {p for p, n in counts.items() if n >= min_mutants_per_complex}
    records = [r for r in records if r["pdb_id"] in keep]

    sizes = {p: counts[p] for p in keep}
    assignment = _assign_complexes_to_splits(sizes, fracs, seed)
    group_to_index = {p: i for i, p in enumerate(sorted(keep))}

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for rec in records:
        by_split[assignment[rec["pdb_id"]]].append(rec)

    train_ddg = [float(r["ddg"]) for r in by_split["train"]]
    mean = sum(train_ddg) / len(train_ddg)
    var = sum((d - mean) ** 2 for d in train_ddg) / max(len(train_ddg) - 1, 1)
    std = var ** 0.5

    logger.info(
        "AB-Bind ddG splits (by complex): train=%d/%dc val=%d/%dc test=%d/%dc "
        "| train ddG mean=%.2f sd=%.2f",
        len(by_split["train"]), sum(1 for v in assignment.values() if v == "train"),
        len(by_split["val"]), sum(1 for v in assignment.values() if v == "val"),
        len(by_split["test"]), sum(1 for v in assignment.values() if v == "test"),
        mean, std,
    )

    return tuple(  # type: ignore[return-value]
        ABBindDataset(
            by_split[s], tokenizer, group_to_index, mean, std, max_length,
        )
        for s in ("train", "val", "test")
    )
