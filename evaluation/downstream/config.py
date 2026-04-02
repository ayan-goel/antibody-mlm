"""Downstream task configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DownstreamConfig:
    """Configuration for downstream fine-tuning / probing tasks."""

    task: str = "paratope"
    checkpoint: str = "models/checkpoints/uniform_medium"
    model_name: str = "alchemab/antiberta2"
    mode: str = "probe"

    learning_rate: float = 1e-3
    encoder_learning_rate: float = 1e-5
    epochs: int = 50
    batch_size: int = 32
    early_stopping_patience: int = 10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.1

    num_seeds: int = 3
    base_seed: int = 42

    output_dir: str = "downstream_outputs"

    device: str = "cuda"
    num_workers: int = 4


_FLOAT_FIELDS = {
    "learning_rate", "encoder_learning_rate", "weight_decay",
    "max_grad_norm", "warmup_fraction",
}
_INT_FIELDS = {
    "epochs", "batch_size", "early_stopping_patience",
    "num_seeds", "base_seed", "num_workers",
}


def load_downstream_config(path: str | Path) -> DownstreamConfig:
    """Load a DownstreamConfig from a YAML file.

    Unspecified fields use their dataclass defaults.
    Coerces numeric fields because PyYAML parses '1e-3' as a string.
    """
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    for k in _FLOAT_FIELDS:
        if k in raw and isinstance(raw[k], str):
            raw[k] = float(raw[k])
    for k in _INT_FIELDS:
        if k in raw and isinstance(raw[k], str):
            raw[k] = int(raw[k])
    return DownstreamConfig(**raw)
