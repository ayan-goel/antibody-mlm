"""Zero-shot mutation-effect scoring via pseudo-log-likelihood.

Scores a mutant relative to its wildtype: delta_PLL = PLL(mutant) - PLL(wt).
Positive delta means the model considers the mutant more likely than wildtype.

Note: compute_pll automatically truncates sequences that exceed the model's
max_position_embeddings, so long antigen chains won't cause CUDA errors.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from evaluation.pseudo_loglikelihood import compute_pll

logger = logging.getLogger(__name__)


def score_mutation(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    wildtype: str,
    mutant: str,
    device: str = "cuda",
    batch_size: int = 64,
    wt_pll: dict[str, float] | None = None,
) -> dict[str, float]:
    """Score a single mutant against a wildtype sequence.

    Args:
        model: Trained masked LM.
        tokenizer: Matching tokenizer.
        wildtype: Raw wildtype amino acid string.
        mutant: Raw mutant amino acid string.
        device: Torch device.
        batch_size: Batch size for PLL computation.
        wt_pll: Pre-computed wildtype PLL (pass to avoid recomputation
                 when scoring many mutants of the same wildtype).

    Returns:
        Dict with 'pll_wt', 'pll_mut', 'delta_pll', normalized variants,
        and 'truncated' flag.
    """
    if wt_pll is None:
        wt_pll = compute_pll(model, tokenizer, wildtype, device, batch_size)
    mut_pll = compute_pll(model, tokenizer, mutant, device, batch_size)

    return {
        "pll_wt": wt_pll["pll"],
        "pll_wt_normalized": wt_pll["pll_normalized"],
        "pll_mut": mut_pll["pll"],
        "pll_mut_normalized": mut_pll["pll_normalized"],
        "delta_pll": mut_pll["pll"] - wt_pll["pll"],
        "delta_pll_normalized": mut_pll["pll_normalized"] - wt_pll["pll_normalized"],
        "truncated": wt_pll.get("was_truncated", False) or mut_pll.get("was_truncated", False),
    }


def score_mutations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    wildtype: str,
    mutants: list[str],
    device: str = "cuda",
    batch_size: int = 64,
) -> list[dict[str, float]]:
    """Score multiple mutants against the same wildtype.

    Caches the wildtype PLL so it is only computed once.
    """
    wt_pll = compute_pll(model, tokenizer, wildtype, device, batch_size)
    logger.info("Wildtype PLL: %.4f (normalized: %.4f)", wt_pll["pll"], wt_pll["pll_normalized"])

    results = []
    for mutant in mutants:
        results.append(
            score_mutation(model, tokenizer, wildtype, mutant, device, batch_size, wt_pll=wt_pll)
        )
    return results
