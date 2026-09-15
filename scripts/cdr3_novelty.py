"""Novelty and variety of infilled CDR3s (reviewer #1.4).

The reviewer asks about "the novelty and variety of the recovered CDR3s",
i.e. whether a high exact-match score conceals degenerate or repetitive
generation. ``scripts/inspect_cdr3_infilling.py`` already answers the crude
form of that (zero homopolymers, well-formed AR…/…DL motifs, realistic
lengths). This adds the two metrics that make the answer quantitative:

  nn_edit_to_train   mean Levenshtein distance from each generated CDR3 to
                     its NEAREST neighbour in the training split. Exact-match
                     novelty is too weak -- a generation one residue away
                     from a training CDR3 counts as "novel" under it.
  mean_pairwise_edit mean Levenshtein distance between generated CDR3s
                     (sampled pairs). Distinct-fraction is brittle for the
                     same reason: 1,000 near-identical strings differing by
                     one residue score as fully distinct.

Both are reported against two anchors so the numbers are interpretable:
  * the TRUE held-out CDR3s (ceiling -- real antibody diversity), and
  * a MODE baseline that always emits the corpus-modal CDR3 (floor).

Runs post-hoc on the dumps written by ``inspect_cdr3_infilling.py``; no GPU.

Usage:
    python scripts/cdr3_novelty.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def edit_distance(a: str, b: str, cap: int | None = None) -> int:
    """Levenshtein distance, with an optional early-exit bound.

    ``cap`` lets the nearest-neighbour search abandon candidates that cannot
    beat the best distance found so far, which is what makes an all-pairs
    search over tens of thousands of training CDR3s tractable in pure Python.
    """
    if a == b:
        return 0
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(v)
            best = min(best, v)
        if cap is not None and best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def nearest_neighbour_distance(query: str, corpus_by_len: dict[int, list[str]]) -> int:
    """Smallest edit distance from ``query`` to anything in the corpus.

    Searches lengths outward from |query|; once the best distance found is
    <= the length gap to the next band, no closer match can exist there.
    """
    best = len(query)
    for gap in range(0, best + 1):
        if gap > best:
            break
        for length in {len(query) - gap, len(query) + gap}:
            for cand in corpus_by_len.get(length, ()):
                d = edit_distance(query, cand, cap=best)
                if d < best:
                    best = d
                    if best == 0:
                        return 0
    return best


def mean_pairwise(seqs: list[str], n_pairs: int, rng: random.Random) -> float:
    if len(seqs) < 2:
        return 0.0
    total = 0
    for _ in range(n_pairs):
        a, b = rng.sample(seqs, 2)
        total += edit_distance(a, b)
    return total / n_pairs


def load_train_cdr3s(data_path: str, split_seed: int = 42) -> list[str]:
    """CDR3s from the TRAINING split only (never the eval prefix)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    from torch.utils.data import Dataset

    from data.splits import make_train_eval_split

    cdr3 = [json.loads(line).get("cdr3_aa") for line in open(data_path)]

    class _Len(Dataset):
        def __len__(self): return len(cdr3)
        def __getitem__(self, i): return i

    train, _ = make_train_eval_split(_Len(), 0.9, split_seed)
    return [cdr3[i] for i in train.indices if cdr3[i]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", default="comparison_outputs/cdr3_generations")
    parser.add_argument("--data", default="data/processed/oas_vh_500k.jsonl")
    parser.add_argument(
        "--corpus-sample", type=int, default=20000,
        help="Distinct training CDR3s to search for nearest neighbours.",
    )
    parser.add_argument("--n-pairs", type=int, default=4000)
    parser.add_argument("--output", default="comparison_outputs/cdr3_novelty.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo = Path(__file__).resolve().parent.parent
    rng = random.Random(0)

    train_cdr3 = load_train_cdr3s(args.data)
    counts = Counter(train_cdr3)
    # Most-frequent distinct CDR3s: the nearest neighbour of a generation is
    # overwhelmingly likely to sit in the dense part of the distribution.
    corpus = [c for c, _ in counts.most_common(args.corpus_sample)]
    by_len: dict[int, list[str]] = {}
    for c in corpus:
        by_len.setdefault(len(c), []).append(c)
    mode_cdr3 = counts.most_common(1)[0][0]
    logger.info(
        "training CDR3s: %d total, %d distinct; searching top %d. mode=%s (%d x)",
        len(train_cdr3), len(counts), len(corpus), mode_cdr3, counts[mode_cdr3],
    )

    rows = []
    dumps = sorted(Path(repo / args.dump_dir).glob("*.csv"))
    truths: list[str] | None = None

    for dump in dumps:
        recs = list(csv.DictReader(dump.open()))
        gen = [r["generated_cdr3"] for r in recs]
        if truths is None:
            truths = [r["true_cdr3"] for r in recs]
        rows.append(_score(dump.stem, gen, by_len, counts, rng, args.n_pairs))

    # Anchors: real antibodies (ceiling) and always-emit-the-mode (floor).
    if truths:
        rows.append(_score("[ANCHOR] true CDR3s", truths, by_len, counts, rng, args.n_pairs))
        rows.append(_score("[ANCHOR] always-mode", [mode_cdr3] * len(truths),
                           by_len, counts, rng, args.n_pairs))

    out = repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    logger.info(
        "\n%-24s %10s %10s %10s %10s", "strategy", "nn_edit", "pair_edit", "distinct", "in_train",
    )
    for r in rows:
        logger.info(
            "%-24s %10.2f %10.2f %10.3f %10.3f",
            r["strategy"], r["nn_edit_to_train"], r["mean_pairwise_edit"],
            r["distinct_frac"], r["frac_exact_in_train"],
        )
    logger.info("\nwrote %s", out)


def _score(name, seqs, by_len, counts, rng, n_pairs) -> dict:
    nn = [nearest_neighbour_distance(s, by_len) for s in seqs]
    return {
        "strategy": name,
        "n": len(seqs),
        "nn_edit_to_train": sum(nn) / len(nn),
        "frac_nn_zero": sum(d == 0 for d in nn) / len(nn),
        "mean_pairwise_edit": mean_pairwise(seqs, n_pairs, rng),
        "distinct_frac": len(set(seqs)) / len(seqs),
        "frac_exact_in_train": sum(s in counts for s in seqs) / len(seqs),
        "mean_len": sum(len(s) for s in seqs) / len(seqs),
    }


if __name__ == "__main__":
    main()
