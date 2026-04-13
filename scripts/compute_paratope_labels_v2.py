"""Compute per-sequence paratope labels for all OAS training sequences.

Uses the fine-tuned AntiBERTa2 paratope teacher (produced by
``scripts/train_paratope_teacher.py``) to predict per-residue paratope
probabilities for every sequence in the OAS training corpus.

Replaces the marginal ``P(paratope | amino_acid, cdr_region)`` lookup
used by the original ``compute_paratope_labels.py``, which had no
per-sequence signal and caused interface_medium to underperform on the
paratope downstream probe.

Output format (matches the original file for drop-in replacement):

    List[dict | None] of length N (same as JSONL record count).
    Each entry: ``{"paratope_labels": FloatTensor(L,)}`` in [0, 1]
    where L is the raw AA sequence length.

Usage:
    python scripts/compute_paratope_labels_v2.py \\
        --teacher models/paratope_teacher/final \\
        --input data/processed/oas_vh_500k.jsonl \\
        --output data/structures/oas_vh_500k_paratope_v2.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import RoFormerForMaskedLM

from scripts.train_paratope_teacher import ParatopeTeacher
from utils.io import load_jsonl
from utils.tokenizer import ANTIBERTA2_MODEL_NAME, load_tokenizer, tokenize_sequence

logger = logging.getLogger(__name__)


def _load_teacher(teacher_dir: Path, device: str) -> tuple[ParatopeTeacher, dict]:
    """Rebuild the teacher from the saved encoder + head state dicts.

    The encoder is re-initialized from alchemab/antiberta2 to get the
    correct architecture, then loaded with the fine-tuned weights.
    """
    metadata_path = teacher_dir / "teacher_metadata.json"
    with metadata_path.open() as f:
        metadata = json.load(f)

    logger.info("Loading base AntiBERTa2 architecture (%s)...", metadata["base_model"])
    base = RoFormerForMaskedLM.from_pretrained(metadata["base_model"])
    encoder = base.roformer
    hidden_size = int(metadata["hidden_size"])
    del base

    logger.info("Loading fine-tuned encoder weights from %s", teacher_dir / "encoder.pt")
    encoder_state = torch.load(teacher_dir / "encoder.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(encoder_state)

    model = ParatopeTeacher(encoder=encoder, hidden_size=hidden_size)
    logger.info("Loading fine-tuned head weights from %s", teacher_dir / "head.pt")
    head_state = torch.load(teacher_dir / "head.pt", map_location="cpu", weights_only=True)
    model.head.load_state_dict(head_state)

    model.to(device)
    model.eval()
    return model, metadata


def _predict_batch(
    model: ParatopeTeacher,
    sequences: list[str],
    tokenizer,
    device: str,
    max_length: int,
) -> list[torch.Tensor]:
    """Predict per-AA paratope probabilities for a batch of sequences.

    Returns one FloatTensor of shape (L_i,) per sequence, where L_i is
    the number of residues actually covered by the tokenized input
    (i.e. min(len(seq), max_length - 2)). Sequences longer than
    ``max_length - 2`` are truncated, matching how training data is
    tokenized.
    """
    spaced = [tokenize_sequence(seq, tokenizer) for seq in sequences]
    enc = tokenizer(
        spaced,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    special_mask = enc["special_tokens_mask"]  # keep on CPU

    with torch.amp.autocast(device_type="cuda", enabled=(device != "cpu")):
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
    probs = torch.sigmoid(logits.float()).cpu()

    out: list[torch.Tensor] = []
    for i, seq in enumerate(sequences):
        # Non-special positions only, in tokenization order = AA order.
        non_special = (special_mask[i] == 0)
        per_aa = probs[i][non_special]
        # Cap at raw sequence length in case of tokenizer edge cases.
        per_aa = per_aa[: len(seq)]
        out.append(per_aa.contiguous().clone())
    return out


def run(args: argparse.Namespace) -> None:
    device = args.device
    teacher_dir = Path(args.teacher)

    logger.info("Loading training sequences from %s", args.input)
    records = load_jsonl(args.input)
    n_records = len(records)
    logger.info("Loaded %d sequences", n_records)

    tokenizer = load_tokenizer(ANTIBERTA2_MODEL_NAME)
    model, metadata = _load_teacher(teacher_dir, device)

    max_length = args.max_length or int(metadata.get("max_length", 160))
    logger.info(
        "Teacher metadata: base=%s hidden_size=%d max_length=%d test_auprc=%.4f",
        metadata["base_model"], metadata["hidden_size"], max_length,
        metadata.get("test_metrics", {}).get("auprc", -1.0),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, torch.Tensor] | None] = [None] * n_records

    # Resume from existing file
    if args.resume and output_path.exists():
        logger.info("Resuming from %s", output_path)
        existing = torch.load(output_path, weights_only=False)
        for i, entry in enumerate(existing):
            if i < n_records:
                results[i] = entry
        done = sum(1 for r in results if r is not None)
        logger.info("Loaded %d existing entries", done)

    pending_idx = [i for i, r in enumerate(results) if r is None]
    logger.info("Predicting soft labels for %d sequences...", len(pending_idx))

    batch_size = args.batch_size
    checkpoint_every = args.checkpoint_every

    processed_since_checkpoint = 0
    pbar = tqdm(total=len(pending_idx), desc="Inference")
    for batch_start in range(0, len(pending_idx), batch_size):
        batch_idx = pending_idx[batch_start : batch_start + batch_size]
        batch_seqs = [records[i]["sequence"] for i in batch_idx]
        per_aa_list = _predict_batch(model, batch_seqs, tokenizer, device, max_length)
        for i, per_aa in zip(batch_idx, per_aa_list):
            results[i] = {"paratope_labels": per_aa.float()}
        pbar.update(len(batch_idx))
        processed_since_checkpoint += len(batch_idx)

        if processed_since_checkpoint >= checkpoint_every:
            torch.save(results, str(output_path))
            pbar.write(f"Checkpoint saved ({sum(1 for r in results if r is not None)}/{n_records})")
            processed_since_checkpoint = 0
    pbar.close()

    torch.save(results, str(output_path))

    done = sum(1 for r in results if r is not None)
    logger.info("Done. Saved %d/%d entries to %s", done, n_records, output_path)

    # Quick sanity check on the distribution — verify it's per-sequence
    # (unlike the old marginal-lookup file where every sequence had the
    # same set of unique values).
    if done >= 3:
        u0 = torch.unique(results[0]["paratope_labels"]).numel()
        u1 = torch.unique(results[1]["paratope_labels"]).numel()
        pos_rate = torch.cat([
            r["paratope_labels"] for r in results[:1000] if r is not None
        ]).mean().item()
        logger.info(
            "Sanity: sample 0 unique values=%d, sample 1 unique values=%d, "
            "mean prob over first 1000 seqs=%.4f",
            u0, u1, pos_rate,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher", type=str, default="models/paratope_teacher/final",
        help="Directory containing encoder.pt + head.pt + teacher_metadata.json",
    )
    parser.add_argument(
        "--input", type=str, default="data/processed/oas_vh_500k.jsonl",
    )
    parser.add_argument(
        "--output", type=str, default="data/structures/oas_vh_500k_paratope_v2.pt",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-length", type=int, default=0,
        help="Override max sequence length (default: from teacher metadata)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50000,
        help="Save partial results every N sequences (for crash recovery)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip sequences already present in the output file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
