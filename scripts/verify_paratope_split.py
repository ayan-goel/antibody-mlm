"""Verify the paratope teacher and probe cannot leak into each other.

Reviewers #6.3 and #2.3.2 raise the paratope teacher as a circularity risk: the
teacher is trained on TDC SAbDab_Liberis, and the paratope AUPRC/MCC we report
is scored on the same source. Our defence in the paper is that both call the
same split function, so the teacher never sees a test antibody. That is a claim
about code, and a reviewer is entitled to want it checked rather than asserted.

This checks it, and also checks the thing that argument does NOT cover.

  A. DETERMINISM        two independent calls return byte-identical partitions,
                        so "the same function" really does mean the same split.
  B. DISJOINTNESS       no sequence appears in more than one split.
  C. NO TEACHER LEAK    teacher-train ∩ probe-test = empty. This is the direct
                        statement of what #6.3 asks about.
  D. NEAR-DUPLICATES    highest sequence identity between each test antibody and
                        any training antibody. A shared split function proves
                        the teacher never saw a test *label*; it says nothing
                        about whether a near-identical antibody sits in train.
                        SAbDab_Liberis is redundant and TDC's default split is
                        random, so this is the real exposure and we should
                        report the number rather than let a reviewer find it.

C follows from A and B, but is asserted separately because it is the specific
claim the response makes.

Usage:
    python scripts/verify_paratope_split.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

OUT_DIR = Path("comparison_outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--identity-threshold", type=float, default=95.0,
        help="Identity at or above which a test/train pair is called a near-duplicate.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from utils.tokenizer import ANTIBERTA2_MODEL_NAME, load_tokenizer
    from data.benchmarks.paratope import load_paratope_splits
    from scripts.contamination_audit import make_aligner, percent_identity

    tokenizer = load_tokenizer(ANTIBERTA2_MODEL_NAME)

    # --- A. determinism -----------------------------------------------------
    # The teacher (scripts/train_paratope_teacher.py:159) and the probe
    # (evaluation/downstream/paratope.py:26) both call this. Calling it twice
    # here stands in for calling it from those two places.
    s1 = load_paratope_splits(tokenizer)
    s2 = load_paratope_splits(tokenizer)
    names = ("train", "valid", "test")
    deterministic = all(a.sequences == b.sequences for a, b in zip(s1, s2))

    train, valid, test = (list(d.sequences) for d in s1)
    sets = {"train": set(train), "valid": set(valid), "test": set(test)}
    logger.info("split sizes: train=%d valid=%d test=%d", len(train), len(valid), len(test))

    # --- B. disjointness ----------------------------------------------------
    overlaps = {
        f"{a}&{b}": len(sets[a] & sets[b])
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }

    # --- C. the direct claim ------------------------------------------------
    teacher_train_seen = sets["train"] | sets["valid"]  # teacher fits on train, early-stops on valid
    leaked = sets["test"] & teacher_train_seen

    # --- D. near-duplicates -------------------------------------------------
    aligner = make_aligner()
    logger.info("scoring %d test x %d train pairs for near-duplicates", len(test), len(train))
    best: list[float] = []
    for i, t in enumerate(test):
        best.append(max(percent_identity(aligner, t, tr) for tr in train))
        if (i + 1) % 50 == 0:
            logger.info("  %d/%d", i + 1, len(test))
    best_arr = np.array(best)
    thr = args.identity_threshold

    report = {
        "split_sizes": {n: len(s) for n, s in zip(names, (train, valid, test))},
        "A_deterministic_across_calls": deterministic,
        "B_pairwise_exact_overlaps": overlaps,
        "C_test_sequences_seen_by_teacher": len(leaked),
        "D_test_vs_train_identity": {
            "median": round(float(np.median(best_arr)), 2),
            "mean": round(float(best_arr.mean()), 2),
            "max": round(float(best_arr.max()), 2),
            f"n_at_or_above_{thr:g}pct": int((best_arr >= thr).sum()),
            f"pct_at_or_above_{thr:g}pct": round(100.0 * float((best_arr >= thr).mean()), 2),
            "n_identical_100pct": int((best_arr >= 99.99).sum()),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "paratope_split_verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("wrote %s", path)

    print("\n=== paratope split verification ===")
    print(json.dumps(report, indent=2))
    verdict_ok = deterministic and not any(overlaps.values()) and not leaked
    print(
        "\nA-C: "
        + ("PASS — teacher and probe share one deterministic, disjoint partition"
           if verdict_ok else "FAIL — investigate before citing the leakage claim")
    )
    print(
        f"D:   {report['D_test_vs_train_identity'][f'n_at_or_above_{thr:g}pct']} of {len(test)} "
        f"test antibodies have a >={thr:g}% identical antibody in train "
        f"(median nearest-train identity {report['D_test_vs_train_identity']['median']}%)"
    )


if __name__ == "__main__":
    main()
