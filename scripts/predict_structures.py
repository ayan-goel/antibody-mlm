"""Predict antibody structures with IgFold and store Calpha coordinates.

Runs IgFold on each sequence in a processed JSONL, extracts Calpha
backbone coordinates and pLDDT confidence scores, applies a quality
filter (mean pLDDT >= threshold), and saves a sidecar .pt file aligned
by index with the JSONL.

Output .pt format:
    List[dict | None] of length N (same as JSONL record count).
    Each entry is None (prediction failed or filtered) or:
        {"coords_ca": FloatTensor(L, 3), "plddt": FloatTensor(L)}

Usage:
    python scripts/predict_structures.py \
        --input data/processed/oas_vh_tiny.jsonl \
        --output data/structures/oas_vh_tiny_coords.pt \
        --plddt_threshold 70

IgFold is only imported here — training code never depends on it.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.io import load_jsonl

logger = logging.getLogger(__name__)


def predict_single(
    sequence: str,
    igfold_runner: object,
    tmp_dir: Path,
) -> dict[str, torch.Tensor] | None:
    """Run IgFold on a single VH sequence and extract Calpha coords + pLDDT.

    Returns None if prediction fails.
    """
    try:
        pdb_path = tmp_dir / "pred.pdb"
        output = igfold_runner.fold(
            pdb_path,
            sequences={"H": sequence},
            do_refine=False,
            do_renum=False,
        )

        coords_ca = output.coords[:, 1, :]  # Calpha is atom index 1 in IgFold
        plddt = output.plddt[:, 1] if output.plddt.dim() == 2 else output.plddt

        return {
            "coords_ca": coords_ca.detach().cpu().float(),
            "plddt": plddt.detach().cpu().float(),
        }
    except Exception as e:
        logger.warning("Prediction failed for sequence (len=%d): %s", len(sequence), e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict antibody structures and store Calpha coordinates"
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
        "--plddt_threshold", type=float, default=70.0,
        help="Minimum mean pLDDT to keep a structure (default: 70)",
    )
    parser.add_argument(
        "--max_sequences", type=int, default=0,
        help="Max sequences to process (0 = all, useful for development)",
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
    start_idx = 0

    if args.resume and Path(args.resume).exists():
        logger.info("Resuming from %s", args.resume)
        existing = torch.load(args.resume, weights_only=False)
        for i, entry in enumerate(existing):
            if i < len(results):
                results[i] = entry
        start_idx = sum(1 for e in existing if e is not None)
        logger.info("Loaded %d existing predictions", start_idx)

    try:
        from igfold import IgFoldRunner
    except ImportError:
        logger.error(
            "IgFold is not installed. Install it with: pip install igfold\n"
            "See https://github.com/Graylab/IgFold for details."
        )
        sys.exit(1)

    logger.info("Initializing IgFold model...")
    igfold_runner = IgFoldRunner()

    passed = 0
    failed = 0
    filtered = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for i in range(total):
            if results[i] is not None:
                passed += 1
                continue

            sequence = records[i]["sequence"]
            result = predict_single(sequence, igfold_runner, tmp_path)

            if result is None:
                failed += 1
                results[i] = None
            elif result["plddt"].mean().item() < args.plddt_threshold:
                filtered += 1
                results[i] = None
                logger.debug(
                    "Filtered idx=%d: mean pLDDT=%.1f < %.1f",
                    i, result["plddt"].mean().item(), args.plddt_threshold,
                )
            else:
                passed += 1
                results[i] = result

            if (i + 1) % 100 == 0 or (i + 1) == total:
                logger.info(
                    "Progress: %d/%d (passed=%d, failed=%d, filtered=%d)",
                    i + 1, total, passed, failed, filtered,
                )

            if (i + 1) % 1000 == 0:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(results, str(output_path))
                logger.info("Checkpoint saved to %s", output_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(output_path))

    logger.info("Done. Saved %d entries to %s", len(results), output_path)
    logger.info("Statistics: passed=%d, failed=%d, filtered=%d", passed, failed, filtered)
    coverage = passed / total * 100 if total > 0 else 0
    logger.info("Coverage: %.1f%% of sequences have valid structures", coverage)


if __name__ == "__main__":
    main()
