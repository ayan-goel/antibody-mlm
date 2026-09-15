"""Build the label-permutation control for the interface masking strategy.

Reviewers #2.3 and #5.2 both ask whether function-aware masking works because
the masks land on *biologically meaningful* residues, or merely because they
change the *statistics* of corruption (rate, clustering, weight spread). The
hybrid arms already have an answer — the Dirichlet control in
``configs/hybrid_random_medium.yaml``. The specialist arms, which carry the
paper's main claim, have none. This builds it.

The control shuffles each sequence's per-residue paratope probabilities
**within that sequence**. Because a permutation is a bijection on the multiset
of values, every per-sequence statistic the masker can see is preserved exactly:

    - the marginal mask rate (``weights.mean()`` normalises the Bernoulli
      matrix in ``masking/interface.py:100-102``, and a permutation leaves the
      mean untouched),
    - the full distribution of per-residue weights, hence the variance and the
      shape of the mask-count distribution,
    - the sequence length and the special-token layout.

The only thing destroyed is *which* residue carries which probability. So a
permuted-label teacher carries exactly as much supervision as the real one, and
the contrast isolates paratope identity alone:

    interface > interface-permuted  ->  the biological prior does real work
    interface ~ interface-permuted  ->  the gain was mask statistics

Usage:
    python scripts/permute_paratope_labels.py \\
        --input data/structures/oas_vh_500k_paratope.pt \\
        --output data/structures/oas_vh_500k_paratope_permuted.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

#: Kept in step with ``data.max_length`` in configs/interface_medium.yaml.
#: ``data/dataset.py:155`` keeps only the first ``max_length - 2`` residues, so
#: permuting across that boundary would move probability mass into the
#: discarded tail and change the effective marginal. We permute within the
#: retained window and leave any truncated tail in place.
DEFAULT_MAX_LENGTH = 160


def permute_entry(
    labels: torch.Tensor, window: int, generator: torch.Generator
) -> torch.Tensor:
    """Return ``labels`` with its first ``window`` positions shuffled."""
    out = labels.clone()
    n = min(window, labels.numel())
    if n > 1:
        out[:n] = labels[torch.randperm(n, generator=generator)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/structures/oas_vh_500k_paratope.pt")
    parser.add_argument(
        "--output", default="data/structures/oas_vh_500k_paratope_permuted.pt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Permutation RNG seed. Fixed so the control is reproducible and "
        "identical across the pretraining seeds it is compared at.",
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    window = args.max_length - 2
    logger.info("loading %s", args.input)
    entries = torch.load(args.input, weights_only=False)
    logger.info("%d entries; permuting within the first %d residues", len(entries), window)

    generator = torch.Generator().manual_seed(args.seed)

    permuted: list[dict | None] = []
    n_truncated = 0
    sum_before = sum_after = 0.0
    n_res = 0

    for entry in entries:
        if entry is None:
            permuted.append(None)
            continue
        labels = entry["paratope_labels"]
        if labels.numel() > window:
            n_truncated += 1
        new = permute_entry(labels, window, generator)
        sum_before += float(labels[:window].sum())
        sum_after += float(new[:window].sum())
        n_res += min(window, labels.numel())
        permuted.append({"paratope_labels": new})

    # A permutation cannot change the sum, so any difference here is float32
    # accumulation order, not a real change. Check relatively — the absolute
    # drift grows with corpus size and is meaningless on its own. A genuine
    # mismatch means the control is not matched and the experiment is
    # uninterpretable, so fail loudly rather than burn 5 GPU-hours on it.
    rel_drift = abs(sum_before - sum_after) / max(abs(sum_before), 1e-12)
    if rel_drift > 1e-6:
        raise AssertionError(
            f"marginal not preserved: {sum_before:.6f} -> {sum_after:.6f} "
            f"(relative drift {rel_drift:.2e})"
        )

    logger.info(
        "marginal preserved: mean label %.6f over %d residues (%d sequences longer than the window)",
        sum_after / max(n_res, 1),
        n_res,
        n_truncated,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(permuted, args.output)
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
