"""Developability probing via ridge regression with nested k-fold CV.

The SGD linear probe in ``evaluation/downstream/developability.py`` is
unstable on this task, and the cause is data starvation rather than a bad
hyperparameter. TAP has 241 antibodies, split 169 / 24 / 48. The head is
hidden_size x 5 (2,560 parameters at hidden 512) fit on 169 examples and
model-selected on 24. Some head initialisations overfit by epoch 2 and
validation never recovers; others train normally. Measured across the run
matrix, 7 checkpoint/task pairs had at least one probe seed collapse, all of
them developability, including two values reported in the paper.

Neither a lower learning rate (3e-4) nor selecting on validation MSE instead
of macro-Spearman fixes it; ``patience: 0`` does correctly disable early
stopping, so the runs are not being truncated.

This replaces that estimator with one appropriate to the sample size:

  * closed-form ridge instead of SGD -- no initialisation, no epoch budget,
    no early-stopping decision, so the collapse mode cannot occur;
  * nested k-fold CV over all 241 antibodies instead of one 169/24/48 split
    -- every antibody contributes an out-of-fold prediction, so the score is
    computed on 241 points rather than 48, and alpha is chosen on inner folds
    so the outer fold stays untouched.

Spearman is rank-based, so the label z-scoring applied by the TAP loader does
not affect the reported metric.

Usage:
    python scripts/run_developability_ridge.py --checkpoints models/checkpoints/*/final
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import RoFormerForMaskedLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.benchmarks.developability import load_developability_splits
from utils.tokenizer import load_tokenizer_for_checkpoint

logger = logging.getLogger(__name__)

_ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)


@torch.no_grad()
def embed_all(encoder, datasets, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pooled embeddings (specials excluded) and labels for every antibody."""
    feats, labels = [], []
    for dataset in datasets:
        for i in range(len(dataset)):
            item = dataset[i]
            ids = torch.tensor([item["input_ids"]], device=device)
            attn = torch.tensor([item["attention_mask"]], device=device)
            special = np.asarray(item["special_tokens_mask"])

            hidden = encoder(input_ids=ids, attention_mask=attn).last_hidden_state[0]
            real = np.where(special == 0)[0]
            feats.append(hidden[real].mean(0).cpu().numpy())
            labels.append(item["labels"])
    return np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form multi-target ridge with an unpenalised intercept."""
    Xc = np.hstack([X, np.ones((len(X), 1))])
    penalty = alpha * np.eye(Xc.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(Xc.T @ Xc + penalty, Xc.T @ Y)


def ridge_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def _folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return [f for f in np.array_split(idx, k)]


def nested_cv_predictions(
    X: np.ndarray, Y: np.ndarray, k: int, seed: int,
) -> np.ndarray:
    """Out-of-fold predictions; alpha chosen on inner folds only."""
    oof = np.zeros_like(Y)
    for fold in _folds(len(X), k, seed):
        mask = np.ones(len(X), dtype=bool)
        mask[fold] = False
        X_tr, Y_tr, X_te = X[mask], Y[mask], X[fold]

        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr_s, X_te_s = (X_tr - mu) / sd, (X_te - mu) / sd

        # Inner CV over the training folds to pick alpha.
        best_alpha, best_score = _ALPHAS[0], -np.inf
        inner = _folds(len(X_tr_s), k, seed + 1)
        for alpha in _ALPHAS:
            scores = []
            for inner_fold in inner:
                inner_mask = np.ones(len(X_tr_s), dtype=bool)
                inner_mask[inner_fold] = False
                W = ridge_fit(X_tr_s[inner_mask], Y_tr[inner_mask], alpha)
                pred = ridge_predict(X_tr_s[inner_fold], W)
                rhos = [
                    spearmanr(Y_tr[inner_fold][:, t], pred[:, t]).statistic
                    for t in range(Y.shape[1])
                ]
                rhos = [r for r in rhos if not np.isnan(r)]
                if rhos:
                    scores.append(float(np.mean(rhos)))
            if scores and np.mean(scores) > best_score:
                best_score, best_alpha = float(np.mean(scores)), alpha

        W = ridge_fit(X_tr_s, Y_tr, best_alpha)
        oof[fold] = ridge_predict(X_te_s, W)
    return oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", default="comparison_outputs/developability_ridge.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    repo = Path(__file__).resolve().parent.parent

    checkpoints = args.checkpoints or sorted(
        str(p) for p in (repo / "models/checkpoints").glob("*/final")
    )

    rows = []
    for ckpt in checkpoints:
        name = Path(ckpt).parent.name
        try:
            tokenizer = load_tokenizer_for_checkpoint(ckpt, "alchemab/antiberta2")
            train, val, test, target_names, _ = load_developability_splits(
                tokenizer, max_length=160,
            )
            encoder = RoFormerForMaskedLM.from_pretrained(ckpt).roformer
            encoder.to(args.device).eval()
            X, Y = embed_all(encoder, (train, val, test), args.device)
            del encoder
            torch.cuda.empty_cache()
        except Exception:
            logger.exception("[%s] failed — skipping", name)
            continue

        # Repeat the whole CV with different fold assignments; the spread is
        # the honest uncertainty for a 241-sample task.
        macro, per_target = [], {t: [] for t in target_names}
        for repeat in range(args.repeats):
            oof = nested_cv_predictions(X, Y, args.folds, seed=repeat)
            rhos = []
            for t, tname in enumerate(target_names):
                rho = spearmanr(Y[:, t], oof[:, t]).statistic
                if not np.isnan(rho):
                    rhos.append(float(rho))
                    per_target[tname].append(float(rho))
            macro.append(float(np.mean(rhos)) if rhos else float("nan"))

        row = {
            "experiment": name,
            "spearman_macro": float(np.mean(macro)),
            "sd": float(np.std(macro, ddof=1)) if len(macro) > 1 else 0.0,
            "n_antibodies": len(X),
            "repeats": args.repeats,
        }
        row.update(
            {f"spearman_{t}": float(np.mean(v)) for t, v in per_target.items() if v}
        )
        rows.append(row)
        logger.info(
            "[%s] macro Spearman = %.4f +/- %.4f  (n=%d, %d-fold x %d)",
            name, row["spearman_macro"], row["sd"], len(X), args.folds, args.repeats,
        )

    out = repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "experiment", k))
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s (%d checkpoints)", out, len(rows))


if __name__ == "__main__":
    main()
