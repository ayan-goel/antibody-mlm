"""Check the benchmark's paratope labels against contacts computed from crystals.

Motivated by reviewer #1.5, who asks what the secondary non-CDR peak in the
`interface` mask distribution is. Tracing it back: the paratope teacher assigns
probability ~1.0 to IMGT positions 92, 93 and 94 in 99.9% of corpus sequences,
because those three positions are labelled paratope in **100% of antibodies** in
TDC SAbDab_Liberis — more consistently than any real CDR position (the best,
IMGT 58, reaches 94.1%). The median antibody carries only 6 paratope labels, so
half of a typical label set is this fixed framework triple, and a rule predicting
exactly those three positions scores precision 1.000 / recall 0.405 on the test
split using no sequence information at all.

That pattern is not physically plausible — diverse antibodies binding diverse
antigens cannot all contact antigen at the same three FR3 positions — but the
repo has no independent paratope source to confirm it with:
``build_sabdab_real_coords.py:118`` re-anchors the same TDC indices rather than
deriving contacts, so it cannot serve as a check.

This computes ground truth directly from geometry. For each SAbDab complex it
identifies antibody chains by ANARCI numbering, treats every other chain as
antigen, and labels a heavy-chain residue as paratope when any of its atoms lies
within ``--cutoff`` A of any antigen atom — the standard definition, and the one
Parapred/Liberis states it uses. The resulting IMGT profile is compared against
the benchmark labels.

Usage:
    PATH=$CONDA_PREFIX/bin:$PATH python scripts/verify_paratope_labels.py --n 150
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

PDB_DIR = "data/sabdab/pdbs"
OUT_DIR = Path("comparison_outputs")
CDR = set(range(27, 39)) | set(range(56, 66)) | set(range(105, 118))
TRIPLE = (92, 93, 94)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def chain_residues(chain):
    """Standard amino-acid residues of a chain, in order."""
    out = []
    for res in chain:
        if res.id[0] != " ":
            continue
        name = res.get_resname().upper()
        if name in THREE_TO_ONE:
            out.append(res)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-dir", default=PDB_DIR)
    parser.add_argument("--n", type=int, default=150, help="complexes to sample")
    parser.add_argument("--cutoff", type=float, default=4.5, help="contact distance, A")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore")
    env_bin = str(Path(sys.executable).parent)
    if env_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")

    from Bio.PDB import PDBParser
    from anarci import run_anarci

    paths = sorted(Path(args.pdb_dir).glob("*.pdb"))
    rng = np.random.default_rng(args.seed)
    if len(paths) > args.n:
        paths = [paths[i] for i in sorted(rng.choice(len(paths), args.n, replace=False))]
    logger.info("parsing %d complexes", len(paths))

    parser_pdb = PDBParser(QUIET=True)
    hit = Counter()
    tot = Counter()
    n_complexes = 0
    n_heavy = 0

    for path in paths:
        try:
            model = next(iter(parser_pdb.get_structure(path.stem, str(path))))
        except Exception:
            continue

        chains = {}
        for ch in model:
            res = chain_residues(ch)
            if len(res) >= 30:
                chains[ch.id] = res
        if len(chains) < 2:
            continue

        seqs = {cid: "".join(THREE_TO_ONE[r.get_resname().upper()] for r in res)
                for cid, res in chains.items()}
        ids = list(seqs)
        numbered, details = run_anarci(
            [(cid, seqs[cid]) for cid in ids], scheme="imgt", ncpu=4
        )[1:3]

        heavy, antibody = [], set()
        for cid, num, det in zip(ids, numbered, details):
            if num and det:
                antibody.add(cid)
                if det[0].get("chain_type") == "H":
                    heavy.append((cid, num, det))
        antigen = [cid for cid in ids if cid not in antibody]
        if not heavy or not antigen:
            continue
        n_complexes += 1

        ag_atoms = np.array([a.coord for cid in antigen for r in chains[cid]
                             for a in r if a.element != "H"])
        if ag_atoms.size == 0:
            continue

        for cid, num, det in heavy:
            n_heavy += 1
            res_list = chains[cid]
            # min distance from each residue to any antigen atom
            contact = []
            for r in res_list:
                atoms = np.array([a.coord for a in r if a.element != "H"])
                d = np.sqrt(((atoms[:, None, :] - ag_atoms[None, :, :]) ** 2).sum(-1))
                contact.append(bool((d <= args.cutoff).any()))
            ptr = det[0]["query_start"]
            for (pos, _ins), aa in num[0][0]:
                if aa == "-":
                    continue
                if 1 <= pos <= 128 and ptr < len(contact):
                    tot[pos] += 1
                    hit[pos] += contact[ptr]
                ptr += 1

    logger.info("usable complexes %d, heavy chains %d", n_complexes, n_heavy)
    freq = {p: hit[p] / tot[p] for p in tot if tot[p] >= 20}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "paratope_label_verification.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["imgt_position", "region", "true_contact_freq", "n_chains"])
        for p in sorted(freq):
            w.writerow([p, "CDR" if p in CDR else "FRAMEWORK",
                        round(freq[p], 4), tot[p]])
    logger.info("wrote %s", path)

    print(f"\n=== TRUE antigen contact from crystal geometry "
          f"({args.cutoff} A), {n_heavy} heavy chains ===")
    print("top 10 positions by contact frequency:")
    for p in sorted(freq, key=lambda q: -freq[q])[:10]:
        print(f"   IMGT {p:>3}: {freq[p]:>6.1%}  {'CDR' if p in CDR else 'FRAMEWORK'}")
    print("\nat the benchmark's invariant triple:")
    for p in TRIPLE:
        if p in freq:
            print(f"   IMGT {p:>3}: {freq[p]:>6.1%}  (benchmark labels it in 100% of antibodies)")
    cdr_f = np.mean([freq[p] for p in freq if p in CDR])
    fr_f = np.mean([freq[p] for p in freq if p not in CDR])
    print(f"\nmean contact frequency — CDR {cdr_f:.1%} | framework {fr_f:.1%}")


if __name__ == "__main__":
    main()
