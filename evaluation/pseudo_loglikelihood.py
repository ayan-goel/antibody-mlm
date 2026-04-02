"""Pseudo-log-likelihood (PLL) scoring for antibody sequences.

Exact PLL: for each non-special position i, mask it, run a forward pass,
and accumulate log p(true_token_i | context). This is the standard
zero-shot fitness metric used in ESM and protein LM benchmarks.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_sequence(sequence: str) -> str:
    """Remove non-standard amino acid characters from a sequence."""
    return "".join(c for c in sequence if c in STANDARD_AAS)


def _get_max_seq_length(model: torch.nn.Module) -> int | None:
    """Extract max_position_embeddings from a model's config, if available."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    return getattr(config, "max_position_embeddings", None)


def compute_pll(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sequence: str,
    device: str = "cuda",
    batch_size: int = 64,
) -> dict[str, float]:
    """Compute the exact pseudo-log-likelihood for a single sequence.

    Creates L copies of the tokenized sequence (one per non-special position),
    each with a different position masked, and batches the forward passes.

    Sequences longer than the model's max_position_embeddings are truncated
    to avoid out-of-bounds position embedding lookups.

    Args:
        model: Trained masked LM.
        tokenizer: Matching tokenizer.
        sequence: Raw amino acid string (e.g. "EVQLVES...").
        device: Torch device.
        batch_size: How many masked copies to process per forward pass.

    Returns:
        Dict with 'pll' (sum of log probs), 'pll_normalized' (pll / L),
        and 'was_truncated' (bool).
    """
    sequence = sanitize_sequence(sequence)
    if not sequence:
        return {"pll": 0.0, "pll_normalized": 0.0, "was_truncated": False}

    max_model_len = _get_max_seq_length(model)
    was_truncated = False
    if max_model_len is not None:
        max_aa = max_model_len - 2  # reserve for [CLS] and [SEP]
        if len(sequence) > max_aa:
            logger.debug(
                "Truncating sequence from %d to %d AA (model max_position_embeddings=%d)",
                len(sequence), max_aa, max_model_len,
            )
            sequence = sequence[:max_aa]
            was_truncated = True

    spaced = " ".join(list(sequence))
    encoding = tokenizer(
        spaced, return_tensors="pt", padding=False,
        truncation=True, max_length=max_model_len or 512,
    )
    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)

    special_tokens = set(tokenizer.all_special_ids)
    mask_token_id = tokenizer.mask_token_id
    vocab_size = model.config.vocab_size

    maskable = [
        i for i in range(len(input_ids))
        if input_ids[i].item() not in special_tokens
        and input_ids[i].item() < vocab_size
    ]

    if not maskable:
        return {"pll": 0.0, "pll_normalized": 0.0, "was_truncated": was_truncated}

    model.eval()
    total_log_prob = 0.0

    for start in range(0, len(maskable), batch_size):
        chunk = maskable[start : start + batch_size]
        batch_ids = input_ids.unsqueeze(0).expand(len(chunk), -1).clone()
        true_tokens = torch.tensor(
            [input_ids[pos].item() for pos in chunk], dtype=torch.long, device=device
        )

        for batch_idx, pos in enumerate(chunk):
            batch_ids[batch_idx, pos] = mask_token_id

        batch_ids = batch_ids.to(device)
        batch_attn = attention_mask.unsqueeze(0).expand(len(chunk), -1).to(device)

        with torch.no_grad():
            logits = model(input_ids=batch_ids, attention_mask=batch_attn).logits

        log_probs = F.log_softmax(logits, dim=-1)
        for batch_idx, pos in enumerate(chunk):
            total_log_prob += log_probs[batch_idx, pos, true_tokens[batch_idx]].item()

    seq_len = len(maskable)
    return {
        "pll": total_log_prob,
        "pll_normalized": total_log_prob / seq_len,
        "was_truncated": was_truncated,
    }


def compute_pll_batch(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sequences: list[str],
    device: str = "cuda",
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[dict[str, float]]:
    """Compute exact PLL for multiple sequences.

    Each sequence is processed independently (L forward passes per sequence,
    batched internally).

    Returns:
        List of dicts, one per sequence, each with 'pll' and 'pll_normalized'.
    """
    results = []
    iterator = tqdm(sequences, desc="Computing PLL") if show_progress else sequences
    for seq in iterator:
        results.append(compute_pll(model, tokenizer, seq, device, batch_size))
    return results
