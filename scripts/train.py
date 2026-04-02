"""CLI entry point: load config, build components, train."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import load_config
from training.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train antibody MLM model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_dir = train(config)
    print(f"\nTraining complete. Checkpoint saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
