"""Site-resolved AB-Bind ddG probing over frozen encoders.

Why this is a standalone script rather than a registered downstream task:
the shared probe pipeline assumes one mean-pooled vector per sequence, and
``extract_and_cache`` re-derives ``special_tokens_mask`` from ``input_ids``,
so a dataset cannot supply a per-token mutation mask. Mean pooling is also
demonstrably the wrong representation here — measured on uniform_medium,
within-complex spread is 1.81 mean-pooled versus 11.71 at the mutation site,
against a between-complex spread of 3.01. A head trained across complexes on
pooled vectors therefore fits complex identity and goes flat within a
complex, which is exactly what the registered ``ab_bind`` task produces
(mean per-complex Spearman 0.008).

This script instead represents a mutant by its embeddings **at the mutated
positions**, relative to the wildtype:

    features = [ h_mut[sites] , h_mut[sites] - h_wt[sites] ]

Protocol notes:
  * Splits are grouped by complex, so no complex straddles train and test.
  * With only ~31 complexes a single split is very high variance, so the
    whole fit is repeated over N grouped splits and we report mean +/- sd.
  * Ridge regression (closed form) rather than SGD: n ~ 850 training rows,
    no early-stopping noise, alpha selected on the validation split.
  * Headline metric is mean per-complex Spearman rho, chosen to be directly
    comparable to the zero-shot ``mut_mean_per_complex_spearman_rho``.

Usage:
    python scripts/run_ab_bind_probe.py --checkpoints models/checkpoints/*/final
    python scripts/run_ab_bind_probe.py --n-splits 20 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import RoFormerForMaskedLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.benchmarks.ab_bind import _assign_complexes_to_splits, load_ab_bind
from utils.tokenizer import load_tokenizer_for_checkpoint, tokenize_single_chain

logger = logging.getLogger(__name__)

_MIN_MUTANTS = 3
_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def mutated_positions(wildtype: str, mutant: str) -> list[int]:
    """Sequence indices where the two differ.

    AB-Bind mutations are substitutions, so the sequences are the same
    length and differing indices are exactly the mutated sites.
    """
    return [i for i, (a, b) in enumerate(zip(wildtype, mutant)) if a != b]


@torch.no_grad()
def embed_sites(
    encoder, tokenizer, sequence: str, sites: list[int], device: str,
) -> np.ndarray:
    """Mean of the encoder's hidden states at ``sites`` (amino-acid indices)."""
    enc = tokenize_single_chain(tokenizer, sequence, 160)
    ids = torch.tensor([enc["input_ids"]], device=device)
    attn = torch.tensor([enc["attention_mask"]], device=device)
    hidden = encoder(input_ids=ids, attention_mask=attn).last_hidden_state[0]

    # Map amino-acid index -> token index via the special-token mask, so this
    # stays correct for paired tokenizations with extra framing tokens.
    special = np.asarray(enc["special_tokens_mask"])
    real = np.where(special == 0)[0]
    valid = [s for s in sites if s < len(real)]
    if not valid:
        return hidden[real].mean(0).cpu().numpy()
    return hidden[real[valid]].mean(0).cpu().numpy()


def build_features(
    checkpoint: str, records: list[dict], device: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (features, ddg, complex_ids) for one checkpoint."""
    tokenizer = load_tokenizer_for_checkpoint(checkpoint, "alchemab/antiberta2")
    encoder = RoFormerForMaskedLM.from_pretrained(checkpoint).roformer
    encoder.to(device).eval()

    wt_cache: dict[tuple[str, str, tuple[int, ...]], np.ndarray] = {}
    feats, ddgs, groups = [], [], []
    for rec in records:
        wt, mut = rec["wildtype_seq"], rec["mutant_seq"]
        sites = mutated_positions(wt, mut)
        if not sites:
            continue

        key = (rec["pdb_id"], rec["chain_id"], tuple(sites))
        if key not in wt_cache:
            wt_cache[key] = embed_sites(encoder, tokenizer, wt, sites, device)
        h_wt = wt_cache[key]
        h_mut = embed_sites(encoder, tokenizer, mut, sites, device)

        feats.append(np.concatenate([h_mut, h_mut - h_wt]))
        ddgs.append(float(rec["ddg"]))
        groups.append(rec["pdb_id"])

    del encoder
    torch.cuda.empty_cache()
    return np.asarray(feats, dtype=np.float64), np.asarray(ddgs), groups


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with an intercept (the bias is not penalised)."""
    Xc = np.hstack([X, np.ones((len(X), 1))])
    n_features = Xc.shape[1]
    penalty = alpha * np.eye(n_features)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(Xc.T @ Xc + penalty, Xc.T @ y)


def per_complex_spearman(
    y: np.ndarray, pred: np.ndarray, groups: list[str],
) -> float:
    """Mean Spearman rho within each complex that has enough label variance."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)

    rhos = []
    for idx in by.values():
        if len(idx) < _MIN_MUTANTS or len(np.unique(y[idx])) < 2:
            continue
        rho, _ = spearmanr(y[idx], pred[idx])
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def evaluate_checkpoint(
    X: np.ndarray, y: np.ndarray, groups: list[str], n_splits: int,
) -> dict[str, float]:
    """Repeat the grouped-split fit ``n_splits`` times and summarise."""
    sizes: dict[str, int] = defaultdict(int)
    for g in groups:
        sizes[g] += 1
    sizes = {g: n for g, n in sizes.items() if n >= _MIN_MUTANTS}

    scores = []
    for split_seed in range(n_splits):
        assign = _assign_complexes_to_splits(sizes, (0.6, 0.2, 0.2), seed=split_seed)
        idx = {s: [] for s in ("train", "val", "test")}
        for i, g in enumerate(groups):
            if g in assign:
                idx[assign[g]].append(i)
        if not idx["train"] or not idx["test"]:
            continue

        tr, va, te = (np.asarray(idx[s]) for s in ("train", "val", "test"))
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xva, Xte = ((X[s] - mu) / sd for s in (tr, va, te))
        ymu, ysd = y[tr].mean(), y[tr].std() or 1.0

        best_alpha, best_val = _ALPHAS[0], -np.inf
        for alpha in _ALPHAS:
            w = ridge_fit(Xtr, (y[tr] - ymu) / ysd, alpha)
            pred_va = np.hstack([Xva, np.ones((len(Xva), 1))]) @ w
            score = per_complex_spearman(y[va], pred_va, [groups[i] for i in va])
            if not np.isnan(score) and score > best_val:
                best_val, best_alpha = score, alpha

        w = ridge_fit(Xtr, (y[tr] - ymu) / ysd, best_alpha)
        pred_te = np.hstack([Xte, np.ones((len(Xte), 1))]) @ w
        score = per_complex_spearman(y[te], pred_te, [groups[i] for i in te])
        if not np.isnan(score):
            scores.append(score)

    if not scores:
        return {"mean_per_complex_spearman": float("nan"), "sd": float("nan"), "n": 0}
    return {
        "mean_per_complex_spearman": float(np.mean(scores)),
        "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "n": len(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--data-dir", default="data/ab_bind")
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="comparison_outputs/ab_bind_probe.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    repo = Path(__file__).resolve().parent.parent
    checkpoints = args.checkpoints or sorted(
        str(p) for p in (repo / "models/checkpoints").glob("*/final")
    )
    records = load_ab_bind(args.data_dir)
    logger.info("AB-Bind: %d records over %d checkpoints",
                len(records), len(checkpoints))

    rows = []
    for ckpt in checkpoints:
        name = Path(ckpt).parent.name
        logger.info("[%s] embedding...", name)
        try:
            X, y, groups = build_features(ckpt, records, args.device)
        except Exception:
            logger.exception("[%s] failed — skipping", name)
            continue
        res = evaluate_checkpoint(X, y, groups, args.n_splits)
        logger.info(
            "[%s] mean per-complex rho = %.4f +/- %.4f over %d splits",
            name, res["mean_per_complex_spearman"], res["sd"], res["n"],
        )
        rows.append({"experiment": name, **res})

    out = repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["experiment", "mean_per_complex_spearman", "sd", "n"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s (%d checkpoints)", out, len(rows))


if __name__ == "__main__":
    main()
