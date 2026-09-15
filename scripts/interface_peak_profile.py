"""Characterise the secondary non-CDR peak in the interface mask distribution.

Reviewer #1.5 asks what the secondary, non-CDR peak in the `interface` masking
distribution is and whether it is functionally meaningful. The paper's Figure 2
fingerprints are computed on a **single** VH chain, so that peak could equally be
a property of that one sequence. This settles it by building a canonical profile.

Method
------
Sequences vary in length, so a raw positional average is meaningless. Every
sampled sequence is numbered with ANARCI under the IMGT scheme, which assigns
each residue a position in a fixed antibody frame, and the paratope teacher's
per-residue probability is accumulated into that frame. The result is the mean
teacher output at each IMGT position over the corpus — i.e. exactly the shape of
the mask budget `masking/interface.py` allocates, since its Bernoulli matrix is
proportional to these weights.

IMGT region boundaries (Lefranc): FR1 1-26, CDR1 27-38, FR2 39-55, CDR2 56-65,
FR3 66-104, CDR3 105-117, FR4 118-128.

Reported per position: mean probability, the fraction of sequences exceeding a
threshold (consistency — a peak driven by a few outlier sequences is not a real
feature), and the ranked non-CDR peaks.

Usage:
    PATH=$CONDA_PREFIX/bin:$PATH python scripts/interface_peak_profile.py
    python scripts/interface_peak_profile.py --n 40000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

CORPUS = "data/processed/oas_vh_500k_annotated.jsonl"
PARATOPE = "data/structures/oas_vh_500k_paratope.pt"
OUT_DIR = Path("comparison_outputs")

IMGT_MAX = 128
REGIONS = [
    ("FR1", 1, 26), ("CDR1", 27, 38), ("FR2", 39, 55), ("CDR2", 56, 65),
    ("FR3", 66, 104), ("CDR3", 105, 117), ("FR4", 118, 128),
]
CDR_POSITIONS = set(range(27, 39)) | set(range(56, 66)) | set(range(105, 118))


def region_of(pos: int) -> str:
    for name, lo, hi in REGIONS:
        if lo <= pos <= hi:
            return name
    return "?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20000, help="sequences to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ncpu", type=int, default=16)
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="probability above which a residue counts as teacher-called paratope",
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env_bin = str(Path(sys.executable).parent)
    if env_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")

    import torch
    from anarci import run_anarci

    logger.info("loading corpus + paratope labels")
    seqs = []
    with open(CORPUS) as f:
        for line in f:
            seqs.append(json.loads(line)["sequence"])
    labels = torch.load(PARATOPE, weights_only=False)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(seqs), size=min(args.n, len(seqs)), replace=False)
    logger.info("numbering %d sequences with ANARCI (IMGT)", len(idx))

    numbered = run_anarci(
        [(f"s{i}", seqs[i]) for i in idx], scheme="imgt", ncpu=args.ncpu
    )[1]

    total = np.zeros(IMGT_MAX + 1)
    count = np.zeros(IMGT_MAX + 1)
    over = np.zeros(IMGT_MAX + 1)
    n_used = 0

    for i, num in zip(idx, numbered):
        if not num:
            continue
        prob = labels[int(i)]["paratope_labels"].numpy()
        # ANARCI returns [((pos, insertion), aa), ...] over the numbered domain
        # only, so walk it while tracking the offset into the raw sequence.
        aa_ptr = 0
        for (pos, _ins), aa in num[0][0]:
            if aa == "-":
                continue
            if aa_ptr >= len(prob):
                break
            if 1 <= pos <= IMGT_MAX:
                total[pos] += prob[aa_ptr]
                count[pos] += 1
                over[pos] += prob[aa_ptr] >= args.threshold
            aa_ptr += 1
        n_used += 1

    logger.info("used %d numbered sequences", n_used)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    frac = np.divide(over, count, out=np.zeros_like(over), where=count > 0)

    rows = [
        {
            "imgt_position": p,
            "region": region_of(p),
            "is_cdr": p in CDR_POSITIONS,
            "mean_paratope_prob": round(float(mean[p]), 5),
            "frac_above_threshold": round(float(frac[p]), 5),
            "n_sequences": int(count[p]),
        }
        for p in range(1, IMGT_MAX + 1) if count[p] > 0
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "interface_peak_profile.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s", path)

    # --- report -----------------------------------------------------------
    print(f"\n=== mean teacher paratope probability by IMGT position (n={n_used}) ===")
    print(f"{'region':<6}{'positions':>12}{'mean prob':>12}{'peak pos':>10}{'peak prob':>11}")
    for name, lo, hi in REGIONS:
        sel = [p for p in range(lo, hi + 1) if count[p] > 0]
        if not sel:
            continue
        pk = max(sel, key=lambda p: mean[p])
        print(f"{name:<6}{f'{lo}-{hi}':>12}{np.mean([mean[p] for p in sel]):>12.4f}"
              f"{pk:>10}{mean[pk]:>11.4f}")

    noncdr = [p for p in range(1, IMGT_MAX + 1) if count[p] > 0 and p not in CDR_POSITIONS]
    noncdr.sort(key=lambda p: -mean[p])
    print("\n=== top 12 NON-CDR positions ===")
    print(f"{'IMGT':>6}{'region':>8}{'mean prob':>12}{'frac>=thr':>11}{'n':>8}")
    for p in noncdr[:12]:
        print(f"{p:>6}{region_of(p):>8}{mean[p]:>12.4f}{frac[p]:>11.4f}{int(count[p]):>8}")

    cdr_mean = float(np.mean([mean[p] for p in CDR_POSITIONS if count[p] > 0]))
    fr_mean = float(np.mean([mean[p] for p in noncdr]))
    print(f"\nCDR mean {cdr_mean:.4f} | framework mean {fr_mean:.4f} | ratio {cdr_mean/max(fr_mean,1e-9):.2f}x")


if __name__ == "__main__":
    main()
