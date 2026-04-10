"""Downstream fine-tuning and probing infrastructure.

Provides base classes, reusable heads, embedding caching, and a training
harness for all supervised downstream evaluation tasks. New tasks are
registered via the @register_task decorator and discovered by the CLI.
"""

from __future__ import annotations

from typing import Type

from evaluation.downstream.base import BaseDownstreamTask
from evaluation.downstream.config import DownstreamConfig, load_downstream_config
from evaluation.downstream.embedding_cache import CachedEmbeddingDataset, extract_and_cache
from evaluation.downstream.encoder import EncoderWrapper
from evaluation.downstream.heads import (
    RegressionHead,
    SequenceClassificationHead,
    TokenClassificationHead,
)
from evaluation.downstream.collator import DownstreamCollator
from evaluation.downstream.trainer import DownstreamTrainer

TASK_REGISTRY: dict[str, Type[BaseDownstreamTask]] = {}


def register_task(name: str):
    """Decorator to register a downstream task class by name."""
    def decorator(cls: Type[BaseDownstreamTask]) -> Type[BaseDownstreamTask]:
        TASK_REGISTRY[name] = cls
        return cls
    return decorator


def get_task(name: str, config: DownstreamConfig) -> BaseDownstreamTask:
    """Instantiate a registered downstream task by name."""
    if name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY)) or "(none registered)"
        raise ValueError(f"Unknown task {name!r}. Available: {available}")
    return TASK_REGISTRY[name](config)


import evaluation.downstream.contact_map as _contact_map_module  # noqa: F401 — registers task
import evaluation.downstream.developability as _developability_module  # noqa: F401 — registers task
import evaluation.downstream.paratope as _paratope_module  # noqa: F401 — registers task
import evaluation.downstream.structure_probe as _structure_probe_module  # noqa: F401 — registers task

__all__ = [
    "BaseDownstreamTask",
    "CachedEmbeddingDataset",
    "DownstreamCollator",
    "DownstreamConfig",
    "DownstreamTrainer",
    "EncoderWrapper",
    "RegressionHead",
    "SequenceClassificationHead",
    "TASK_REGISTRY",
    "TokenClassificationHead",
    "extract_and_cache",
    "get_task",
    "load_downstream_config",
    "register_task",
]
