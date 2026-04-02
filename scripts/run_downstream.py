"""CLI: run a downstream evaluation task for a pretrained checkpoint.

Usage:
    python scripts/run_downstream.py --config configs/downstream/paratope_probe.yaml

The config specifies the task name, checkpoint, mode (probe/finetune),
and all hyperparameters. The task class is looked up from the registry.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.downstream import get_task, load_downstream_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run downstream evaluation task")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to downstream task YAML config",
    )
    args = parser.parse_args()

    config = load_downstream_config(args.config)
    logger.info("Task: %s | Mode: %s | Checkpoint: %s", config.task, config.mode, config.checkpoint)
    logger.info("Seeds: %d | Epochs: %d | LR: %s", config.num_seeds, config.epochs, config.learning_rate)

    task = get_task(config.task, config)
    results = task.run()

    metric_summary = {
        k: f"{v:.4f}" for k, v in results.items()
        if isinstance(v, float) and k.endswith("_mean")
    }
    logger.info("=== Final Results ===")
    for k, v in sorted(metric_summary.items()):
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
