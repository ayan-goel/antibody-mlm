"""SARS-CoV-2 specificity classification on Ng & Briney's Dataset E.

Reviewer #5.3 asks for the binding evaluations used in the cited preferential-
masking work. This runs *their* benchmark rather than an approximation of it:
``E_hd-0_cov-1.csv`` from Zenodo record 14019655 (CC-BY-4.0), 24,969 antibodies
balanced 12,485 healthy-donor (label 0) vs 12,484 CoV-specific (label 1).

Matched to their protocol:
  * their dataset and their labels, unmodified;
  * stratified 5-fold cross-validation, as in the paper;
  * a single linear layer on frozen mean-pooled embeddings (logistic
    regression is exactly that, fit to convergence rather than by SGD);
  * their metric set: accuracy, AUROC, AUPRC, F1, MCC.

ONE UNAVOIDABLE DIFFERENCE: they concatenate paired VH+VL; every encoder in
this repo is heavy-chain-only, so only ``h_sequence`` is used. This is the
heavy-chain-only scope limitation reviewer #5.6 raises, and it means our
absolute numbers are not comparable to theirs -- only the comparison *between
our own masking strategies* is meaningful. Do not present these as matching
their reported values.

Two further caveats worth stating rather than hiding:
  * The classes come from different sources -- positives are named antibodies
    from published studies, negatives are 10x single-cell repertoire sequences
    from healthy donors. Part of what a classifier can learn is provenance,
    not specificity. That is a property of their benchmark, inherited by
    matching it.
  * Stratified CV is not clonotype-grouped, so clonal relatives may straddle
    folds. We follow their protocol; a grouped variant would score lower.

Usage:
    python scripts/run_ngbriney_cov.py --checkpoints models/checkpoints/*/final
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from transformers import RoFormerForMaskedLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.tokenizer import load_tokenizer_for_checkpoint, tokenize_single_chain

logger = logging.getLogger(__name__)

_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
_C_GRID = (0.01, 0.1, 1.0)


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    keep = df["h_sequence"].apply(lambda s: set(s) <= _STANDARD_AA)
    df = df[keep].reset_index(drop=True)
    if len(df) != before:
        logger.info("dropped %d row(s) with non-standard residues", before - len(df))
    logger.info(
        "Dataset E: %d antibodies (label 0=%d healthy-donor, 1=%d CoV)",
        len(df), int((df.label == 0).sum()), int((df.label == 1).sum()),
    )
    return df


@torch.no_grad()
def embed(encoder, tokenizer, sequences: list[str], device: str, batch: int = 64):
    """Mean-pooled last-layer embeddings, special/pad positions excluded."""
    out = []
    for start in range(0, len(sequences), batch):
        chunk = sequences[start : start + batch]
        encoded = [tokenize_single_chain(tokenizer, s, 160) for s in chunk]
        width = max(len(e["input_ids"]) for e in encoded)
        pad_id = tokenizer.pad_token_id or 0

        ids, attn, keep = [], [], []
        for e in encoded:
            n = width - len(e["input_ids"])
            ids.append(e["input_ids"] + [pad_id] * n)
            attn.append(e["attention_mask"] + [0] * n)
            # 1 where the position is a real amino acid.
            keep.append([1 - m for m in e["special_tokens_mask"]] + [0] * n)

        ids_t = torch.tensor(ids, device=device)
        attn_t = torch.tensor(attn, device=device)
        keep_t = torch.tensor(keep, device=device, dtype=torch.float).unsqueeze(-1)

        hidden = encoder(input_ids=ids_t, attention_mask=attn_t).last_hidden_state
        pooled = (hidden * keep_t).sum(1) / keep_t.sum(1).clamp(min=1)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0)


def cross_validate(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> dict:
    """Stratified k-fold, matching the reference protocol."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    per_fold: dict[str, list[float]] = {
        k: [] for k in ("accuracy", "auroc", "auprc", "f1", "mcc")
    }

    for train_idx, test_idx in skf.split(X, y):
        X_tr, y_tr, X_te, y_te = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd

        # Pick C on an inner split of the training folds only.
        cut = int(0.85 * len(X_tr))
        best_c, best_score = _C_GRID[0], -np.inf
        for c in _C_GRID:
            clf = LogisticRegression(C=c, max_iter=2000)
            clf.fit(X_tr[:cut], y_tr[:cut])
            score = roc_auc_score(y_tr[cut:], clf.predict_proba(X_tr[cut:])[:, 1])
            if score > best_score:
                best_score, best_c = score, c

        clf = LogisticRegression(C=best_c, max_iter=2000)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)

        per_fold["accuracy"].append(float(accuracy_score(y_te, pred)))
        per_fold["auroc"].append(float(roc_auc_score(y_te, prob)))
        per_fold["auprc"].append(float(average_precision_score(y_te, prob)))
        per_fold["f1"].append(float(f1_score(y_te, pred)))
        per_fold["mcc"].append(float(matthews_corrcoef(y_te, pred)))

    result = {}
    for k, v in per_fold.items():
        result[k] = float(np.mean(v))
        result[f"{k}_sd"] = float(np.std(v, ddof=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--data", default="data/ngbriney/E_hd-0_cov-1.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="comparison_outputs/ngbriney_cov.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    repo = Path(__file__).resolve().parent.parent

    df = load_dataset(args.data)
    sequences, y = df["h_sequence"].tolist(), df["label"].to_numpy()

    checkpoints = args.checkpoints or sorted(
        str(p) for p in (repo / "models/checkpoints").glob("*/final")
    )

    rows = []
    for ckpt in checkpoints:
        name = Path(ckpt).parent.name
        try:
            tokenizer = load_tokenizer_for_checkpoint(ckpt, "alchemab/antiberta2")
            encoder = RoFormerForMaskedLM.from_pretrained(ckpt).roformer
            encoder.to(args.device).eval()
            X = embed(encoder, tokenizer, sequences, args.device)
            del encoder
            torch.cuda.empty_cache()
        except Exception:
            logger.exception("[%s] failed — skipping", name)
            continue

        res = cross_validate(X, y, args.folds, args.seed)
        rows.append({"experiment": name, "n": len(y), **res})
        logger.info(
            "[%s] AUROC %.4f±%.4f  AUPRC %.4f  acc %.4f  F1 %.4f  MCC %.4f",
            name, res["auroc"], res["auroc_sd"], res["auprc"],
            res["accuracy"], res["f1"], res["mcc"],
        )

    out = repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s (%d checkpoints)", out, len(rows))


if __name__ == "__main__":
    main()
