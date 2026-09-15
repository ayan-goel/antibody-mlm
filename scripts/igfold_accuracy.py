"""Measure IgFold's accuracy on the signal our masking actually consumes.

Reviewer #4.5.1 asks how good IgFold's predictions are on our corpus. We never
measured it, and we cannot measure it retrospectively: ``predict_structures_igfold.py``
keeps only ``{"knn_indices": Int16Tensor(L, 32)}`` and discards the coordinates
and IgFold's per-residue predicted RMSD, so no confidence estimate survives in
``oas_vh_500k_igfold.pt``.

Re-running is cheap, and there is a sharper target than raw confidence. The
structure strategies never see coordinates — ``masking/structure.py::_mask_from_knn``
reads ``knn_indices`` and keeps the first ``k_neighbors`` columns:

    structure      k_neighbors=5, min_seq_separation=0  -> the 5 nearest residues
    structure-LR   k_neighbors=5, min_seq_separation=4  -> nearest 5, with |i-j|<=4
                                                           nullified to self-references

So the question that matters is not "what is IgFold's global RMSD" but "does
IgFold put the same residues in each residue's neighbour list as the crystal
does". A model can be mediocre at global fold and still supply a perfectly good
mask prior, or vice versa. This scores the neighbour lists directly, and reports
RMSD alongside as the familiar summary.

Ground truth is ``data/structures/sabdab_liberis_coords.pt`` — 618 chains with
real RCSB-deposited Calpha coordinates, the same source behind the contact-map
and structure-probe benchmarks.

CHAINS ARE TRUNCATED TO THE VARIABLE DOMAIN, AND THAT IS NOT A CONVENIENCE
-------------------------------------------------------------------------
546 of those 618 chains are longer than 150 residues (median 217): they are full
Fab chains carrying a constant domain. IgFold predicts the **Fv only**. Scoring a
whole Fab against it measures a task nobody asked IgFold to do, and the numbers
are not subtly wrong but catastrophically so — the constant domain gets folded
back onto the variable one, so radius of gyration collapses from 20.6 to 13.2 A
and Calpha RMSD reads ~21 A. On the ANARCI-numbered Fv of the same chains, RMSD
is 0.6-2.0 A. Measured on three chains during development:

    2hh0H   full chain 18.53 A / knn5 0.275     Fv 1.03 A / knn5 0.929
    4ydlH   full chain 21.84 A / knn5 0.186     Fv 2.01 A / knn5 0.897
    4ydlL   full chain 21.15 A / knn5 0.216     Fv 0.60 A / knn5 0.926

The Fv span is also the right comparison on its own terms: the pretraining corpus
is isolated ~119-residue VH domains, so the Fv is exactly what IgFold was asked
to fold when the corpus labels were built. Backbone geometry corroborates the
truncation — consecutive Calpha-Calpha spacing is 3.82 A on the Fv (correct) and
drifts to 4.2-4.4 A when the constant domain is included.

PROTOCOL MATCHES PRODUCTION
--------------------------
Chains are folded in isolation (``sequences={"H": seq}``), ``num_models=1``,
``do_refine=False`` — identical to how the corpus labels were generated. Folding
a VH without its light-chain partner is a real source of error, and since that
is what we did to the corpus, it belongs inside the measured number rather than
outside it. Crystal coordinates come from the full complex, so this comparison
is honest about that gap rather than hiding it.

Heavy chains are the headline: pretraining is VH-only, so heavy-chain accuracy
is what bears on the paper. Light chains are folded as L and reported separately.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/igfold_accuracy.py
    CUDA_VISIBLE_DEVICES=3 python scripts/igfold_accuracy.py --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scripts.predict_structures_igfold import _patch_torch_load_for_igfold, coords_to_knn

logger = logging.getLogger(__name__)

K_STORED = 32          # what predict_structures_igfold.py writes
K_USED = 5             # masking/structure.py k_neighbors
MIN_SEP_LR = 4         # structure_longrange_medium.yaml min_seq_separation
CONTACT_ANGSTROM = 8.0
CONTACT_MIN_SEP = 5    # ignore trivial backbone-adjacent contacts

COORDS_PATH = "data/structures/sabdab_liberis_coords.pt"
OUT_DIR = Path("comparison_outputs")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def knn_agreement(pred: torch.Tensor, true: torch.Tensor, k: int) -> float:
    """Mean fraction of each residue's top-k neighbours that both agree on."""
    p, t = pred[:, :k].long(), true[:, :k].long()
    n = p.size(0)
    if n == 0:
        return float("nan")
    return float(
        np.mean([len(set(p[i].tolist()) & set(t[i].tolist())) / k for i in range(n)])
    )


def longrange_knn_agreement(
    pred_d: torch.Tensor, true_d: torch.Tensor, k: int, min_sep: int
) -> float:
    """Top-k agreement after excluding |i-j| <= min_sep, the structure-LR view.

    Computed from distance matrices rather than the stored 32-column lists so
    that excluding near-sequence neighbours pulls genuine long-range ones in,
    which is what min_seq_separation is for.
    """
    n = pred_d.size(0)
    if n <= min_sep + k + 1:
        return float("nan")
    idx = torch.arange(n)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    blocked = sep <= min_sep
    pd = pred_d.masked_fill(blocked, float("inf"))
    td = true_d.masked_fill(blocked, float("inf"))
    pk = pd.topk(k, dim=1, largest=False).indices
    tk = td.topk(k, dim=1, largest=False).indices
    return float(
        np.mean([len(set(pk[i].tolist()) & set(tk[i].tolist())) / k for i in range(n)])
    )


def contact_prf(pred_d: torch.Tensor, true_d: torch.Tensor) -> tuple[float, float, float]:
    """Precision/recall/F1 of predicted 8 A contacts at |i-j| >= 5."""
    n = pred_d.size(0)
    idx = torch.arange(n)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    keep = torch.triu(sep >= CONTACT_MIN_SEP)
    p = (pred_d < CONTACT_ANGSTROM) & keep
    t = (true_d < CONTACT_ANGSTROM) & keep
    tp = float((p & t).sum())
    prec = tp / max(1.0, float(p.sum()))
    rec = tp / max(1.0, float(t.sum()))
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def kabsch_rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    """Calpha RMSD after optimal superposition."""
    x = (a - a.mean(0)).double().numpy()
    y = (b - b.mean(0)).double().numpy()
    u, _, vt = np.linalg.svd(x.T @ y)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(((x @ rot.T - y) ** 2).sum(axis=1).mean()))


def random_knn_baseline(n: int, k: int, rng: np.random.Generator, trials: int = 20) -> float:
    """Expected top-k agreement if neighbours were picked at random.

    Without this the raw agreement number is uninterpretable: for a short chain
    even random neighbour lists overlap a fair amount.
    """
    if n <= k + 1:
        return float("nan")
    vals = []
    for _ in range(trials):
        a = np.array([rng.choice(n - 1, size=k, replace=False) for _ in range(n)])
        b = np.array([rng.choice(n - 1, size=k, replace=False) for _ in range(n)])
        vals.append(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(n)]))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------

def classify_chains(sequences: list[str]) -> list[tuple[str, int, int] | None]:
    """Per sequence: (chain_type, fv_start, fv_end), or None if unnumberable.

    ANARCI numbers only the variable domain, so its aligned span *is* the Fv
    boundary — which is what lets a full Fab chain be cut down to the part
    IgFold actually models.
    """
    env_bin = str(Path(sys.executable).parent)
    if env_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")
    from anarci import run_anarci

    details = run_anarci(
        [(f"s{i}", s) for i, s in enumerate(sequences)], scheme="imgt", ncpu=8
    )[2]
    out: list[tuple[str, int, int] | None] = []
    for d in details:
        if not d:
            out.append(None)
            continue
        out.append(
            (d[0].get("chain_type") or "?", int(d[0]["query_start"]), int(d[0]["query_end"]))
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coords", default=COORDS_PATH)
    parser.add_argument("--limit", type=int, default=0, help="Debug: first N chains only.")
    parser.add_argument("--num-models", type=int, default=1, help="Match production (1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    entries = torch.load(args.coords, weights_only=False)
    if args.limit:
        entries = entries[: args.limit]
    logger.info("loaded %d chains from %s", len(entries), args.coords)

    annotations = classify_chains([e["sequence"] for e in entries])
    kinds = [a[0] if a else "?" for a in annotations]
    logger.info("chain types: %s", {t: kinds.count(t) for t in set(kinds)})

    _patch_torch_load_for_igfold()
    if not torch.cuda.is_available():
        logger.error("CUDA not available; IgFold is unusably slow on CPU.")
        sys.exit(1)
    from igfold import IgFoldRunner

    igfold = IgFoldRunner(num_models=args.num_models)
    logger.info("IgFold loaded on cuda:%d", torch.cuda.current_device())

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    skipped = 0

    for i, (e, ann) in enumerate(zip(entries, annotations)):
        full_seq = e["sequence"]
        full_true = e["coords_ca"].float()
        if ann is None or full_true.size(0) != len(full_seq):
            skipped += 1
            continue
        # Cut to the variable domain. Without this the constant domain dominates
        # and the measurement is meaningless — see the module docstring.
        ctype, fv_start, fv_end = ann
        seq = full_seq[fv_start:fv_end]
        true_ca = full_true[fv_start:fv_end]
        if len(seq) < 80:
            skipped += 1
            continue
        # Fold as the chain actually is; folding a light chain as "H" would
        # measure a mistake we never made in production.
        key = "H" if ctype == "H" else "L"
        try:
            out = igfold.fold(
                "/dev/null", sequences={key: seq},
                do_refine=False, use_openmm=False, do_renum=False, skip_pdb=True,
            )
            pred_ca = out.coords[0, :, 1, :].detach().cpu().float()
        except Exception as ex:
            logger.warning("fold failed (%s len=%d): %s", e.get("pdb_id"), len(seq), ex)
            skipped += 1
            continue
        if pred_ca.size(0) != true_ca.size(0):
            skipped += 1
            continue

        pred_d = torch.cdist(pred_ca.unsqueeze(0), pred_ca.unsqueeze(0))[0]
        true_d = torch.cdist(true_ca.unsqueeze(0), true_ca.unsqueeze(0))[0]
        pred_knn = coords_to_knn(pred_ca, K_STORED)
        true_knn = coords_to_knn(true_ca, K_STORED)
        prec, rec, f1 = contact_prf(pred_d, true_d)

        rows.append({
            "pdb_id": e.get("pdb_id"),
            "chain": e.get("chain"),
            "chain_type": ctype,
            "fv_length": len(seq),
            "full_chain_length": len(full_seq),
            "knn5_agreement": round(knn_agreement(pred_knn, true_knn, K_USED), 4),
            "knn32_agreement": round(knn_agreement(pred_knn, true_knn, K_STORED), 4),
            "knn5_lr_agreement": round(
                longrange_knn_agreement(pred_d, true_d, K_USED, MIN_SEP_LR), 4
            ),
            "knn5_random_baseline": round(random_knn_baseline(len(seq), K_USED, rng), 4),
            "contact_precision": round(prec, 4),
            "contact_recall": round(rec, 4),
            "contact_f1": round(f1, 4),
            "ca_rmsd": round(kabsch_rmsd(pred_ca, true_ca), 3),
        })
        if (i + 1) % 50 == 0:
            logger.info("%d/%d folded (%d skipped)", i + 1, len(entries), skipped)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = out_dir / "igfold_accuracy_per_chain.csv"
    with detail.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d chains, %d skipped)", detail, len(rows), skipped)

    def summarise(subset: list[dict[str, Any]], label: str) -> dict[str, Any]:
        def med(key: str) -> float | None:
            vals = [r[key] for r in subset if r[key] == r[key]]  # drop NaN
            return round(float(np.median(vals)), 4) if vals else None
        return {
            "subset": label,
            "n_chains": len(subset),
            "median_knn5_agreement": med("knn5_agreement"),
            "median_knn5_lr_agreement": med("knn5_lr_agreement"),
            "median_knn32_agreement": med("knn32_agreement"),
            "median_knn5_random_baseline": med("knn5_random_baseline"),
            "median_contact_precision": med("contact_precision"),
            "median_contact_recall": med("contact_recall"),
            "median_contact_f1": med("contact_f1"),
            "median_ca_rmsd": med("ca_rmsd"),
        }

    summaries = [
        summarise([r for r in rows if r["chain_type"] == "H"], "heavy (VH — the paper's setting)"),
        summarise([r for r in rows if r["chain_type"] != "H"], "light"),
        summarise(rows, "all chains"),
    ]
    summary_path = out_dir / "igfold_accuracy.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    logger.info("wrote %s", summary_path)

    print("\n=== IgFold accuracy vs SAbDab crystal structures ===")
    for s in summaries:
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
