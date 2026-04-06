"""Predict spatial neighborhoods via ESM2 contact maps.

Runs ESM2 on batches of sequences, extracts attention-based contact
probability maps, converts them to kNN neighbor lists, and saves a
sidecar .pt file aligned by index with the JSONL.

Output .pt format:
    List[dict | None] of length N (same as JSONL record count).
    Each entry is:
        {"knn_indices": LongTensor(L, k)} — per-residue k nearest neighbors

Usage:
    python scripts/predict_structures.py \
        --input data/processed/oas_vh_500k.jsonl \
        --output data/structures/oas_vh_500k_coords.pt \
        --batch_size 64 --k_neighbors 32

ESM2 is only imported here — training code never depends on it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.io import load_jsonl

logger = logging.getLogger(__name__)


def contacts_to_knn(contact_probs: torch.Tensor, k: int) -> torch.Tensor:
    """Convert a contact probability matrix to kNN indices.

    Args:
        contact_probs: [L, L] contact probabilities (higher = closer).
        k: number of neighbors per residue.

    Returns:
        [L, k] LongTensor of neighbor indices (excluding self).
    """
    seq_len = contact_probs.size(0)
    # Zero out the diagonal so self is not a neighbor
    contact_probs = contact_probs.clone()
    contact_probs.fill_diagonal_(0.0)
    actual_k = min(k, seq_len - 1)
    _, indices = contact_probs.topk(actual_k, dim=1, largest=True)
    indices = indices.to(torch.int16)
    # Pad if seq is shorter than k
    if actual_k < k:
        pad = torch.zeros(seq_len, k - actual_k, dtype=torch.int16)
        indices = torch.cat([indices, pad], dim=1)
    return indices


def predict_batch(
    sequences: list[str],
    model: torch.nn.Module,
    batch_converter: object,
    k: int,
    device: str,
) -> list[dict[str, torch.Tensor] | None]:
    """Run ESM2 contact prediction on a batch and return kNN lists."""
    data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[], return_contacts=True)
    contacts = out["contacts"]  # [B, L, L] — already cropped to seq length

    results = []
    for i, seq in enumerate(sequences):
        try:
            seq_len = len(seq)
            contact_map = contacts[i, :seq_len, :seq_len].cpu().float()
            knn = contacts_to_knn(contact_map, k)
            results.append({"knn_indices": knn})
        except Exception as e:
            logger.warning("Failed for seq %d (len=%d): %s", i, len(seq), e)
            results.append(None)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict spatial neighborhoods via ESM2 contact maps"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to processed JSONL file",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output .pt sidecar file",
    )
    parser.add_argument(
        "--k_neighbors", type=int, default=32,
        help="Number of nearest neighbors per residue (default: 32)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for ESM2 inference (default: 64)",
    )
    parser.add_argument(
        "--max_sequences", type=int, default=0,
        help="Max sequences to process (0 = all)",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to existing .pt file to resume from",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Loading sequences from %s", args.input)
    records = load_jsonl(args.input)
    total = len(records)
    if args.max_sequences > 0:
        total = min(total, args.max_sequences)
    logger.info("Will process %d / %d sequences", total, len(records))

    results: list[dict[str, torch.Tensor] | None] = [None] * len(records)

    if args.resume and Path(args.resume).exists():
        logger.info("Resuming from %s", args.resume)
        existing = torch.load(args.resume, weights_only=False)
        for i, entry in enumerate(existing):
            if i < len(results):
                results[i] = entry
        done = sum(1 for e in existing[:total] if e is not None)
        logger.info("Loaded %d existing predictions", done)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading ESM2 model...")
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device).half()
    logger.info("ESM2 loaded on %s", device)

    todo_indices = [i for i in range(total) if results[i] is None]
    logger.info("Sequences to predict: %d (skipping %d already done)",
                len(todo_indices), total - len(todo_indices))

    processed = 0
    for batch_start in range(0, len(todo_indices), args.batch_size):
        batch_indices = todo_indices[batch_start : batch_start + args.batch_size]
        batch_sequences = [records[i]["sequence"] for i in batch_indices]

        batch_results = predict_batch(
            batch_sequences, model, batch_converter,
            args.k_neighbors, device,
        )

        for j, idx in enumerate(batch_indices):
            results[idx] = batch_results[j]

        processed += len(batch_indices)
        if processed % (args.batch_size * 50) == 0 or processed >= len(todo_indices):
            logger.info("Progress: %d/%d", processed, len(todo_indices))

        if processed % 10000 < args.batch_size:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(results, str(output_path))
            logger.info("Checkpoint saved to %s", output_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(output_path))

    succeeded = sum(1 for e in results[:total] if e is not None)
    logger.info("Done. Saved %d entries to %s", len(results), output_path)
    logger.info("Coverage: %d/%d (%.1f%%)", succeeded, total, succeeded / total * 100)


if __name__ == "__main__":
    main()
