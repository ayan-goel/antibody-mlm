from masking.base import BaseMaskingStrategy, get_strategy, register_strategy
from masking.collator import MLMDataCollator

import masking.cdr  # noqa: F401 — triggers registration
import masking.uniform  # noqa: F401 — triggers registration

__all__ = [
    "BaseMaskingStrategy",
    "MLMDataCollator",
    "get_strategy",
    "register_strategy",
]
