"""MLM token-level accuracy and perplexity evaluation with per-region breakdown.

When the dataset provides CDR region labels (cdr_mask), the evaluator
reports accuracy and perplexity sliced by region so every masking strategy
can be compared on CDR vs framework prediction quality.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from evaluation.base import BaseEvaluator
from masking.base import BaseMaskingStrategy
from masking.collator import MLMDataCollator

logger = logging.getLogger(__name__)

_REGION_FILTERS: dict[str, Any] = {
    "cdr": lambda cdr: cdr > 0,
    "cdr3": lambda cdr: cdr == 3,
    "framework": lambda cdr: cdr == 0,
}


class MLMAccuracyEvaluator(BaseEvaluator):
    """Evaluate masked language model token prediction accuracy and perplexity.

    Computes overall and (when cdr_mask is available) per-region accuracy,
    top-5 accuracy, and perplexity on a held-out dataset.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        strategy: BaseMaskingStrategy,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.device = device

    def evaluate(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute MLM accuracy and perplexity metrics on a dataset.

        Returns a dict with overall metrics and, when cdr_mask is present
        in the data, per-region (cdr, cdr3, framework) breakdowns.
        """
        collator = MLMDataCollator(
            tokenizer=self.tokenizer,
            strategy=self.strategy,
            return_metadata=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
        )

        self.model.eval()
        self.model.to(self.device)

        total_correct = 0
        total_top5_correct = 0
        total_masked = 0
        total_nll = 0.0

        region_correct: dict[str, int] = {r: 0 for r in _REGION_FILTERS}
        region_top5_correct: dict[str, int] = {r: 0 for r in _REGION_FILTERS}
        region_masked: dict[str, int] = {r: 0 for r in _REGION_FILTERS}
        region_nll: dict[str, float] = {r: 0.0 for r in _REGION_FILTERS}
        has_regions = False

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating MLM accuracy"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                mask = labels != -100
                predictions = logits.argmax(dim=-1)
                correct = predictions == labels

                total_correct += correct[mask].sum().item()
                total_masked += mask.sum().item()

                top5 = logits.topk(5, dim=-1).indices
                labels_expanded = labels.unsqueeze(-1).expand_as(top5)
                top5_hit = (top5 == labels_expanded).any(dim=-1)
                total_top5_correct += top5_hit[mask].sum().item()

                per_token_nll = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view(labels.shape)
                total_nll += per_token_nll.sum().item()

                if "cdr_mask" in batch:
                    has_regions = True
                    cdr_mask = batch["cdr_mask"].to(self.device)
                    for region, filter_fn in _REGION_FILTERS.items():
                        region_mask = mask & filter_fn(cdr_mask)
                        region_correct[region] += correct[region_mask].sum().item()
                        region_top5_correct[region] += top5_hit[region_mask].sum().item()
                        region_masked[region] += region_mask.sum().item()
                        region_nll[region] += per_token_nll[region_mask].sum().item()

        def _safe_div(a: float, b: float) -> float:
            return a / b if b > 0 else 0.0

        def _perplexity(nll_sum: float, count: int) -> float:
            if count == 0:
                return 0.0
            return math.exp(nll_sum / count)

        metrics: dict[str, Any] = {
            "mlm_accuracy": _safe_div(total_correct, total_masked),
            "mlm_top5_accuracy": _safe_div(total_top5_correct, total_masked),
            "total_masked_tokens": total_masked,
            "perplexity_overall": _perplexity(total_nll, total_masked),
        }

        if has_regions:
            for region in _REGION_FILTERS:
                metrics[f"mlm_accuracy_{region}"] = _safe_div(
                    region_correct[region], region_masked[region]
                )
                metrics[f"mlm_top5_accuracy_{region}"] = _safe_div(
                    region_top5_correct[region], region_masked[region]
                )
                metrics[f"masked_tokens_{region}"] = region_masked[region]
                metrics[f"perplexity_{region}"] = _perplexity(
                    region_nll[region], region_masked[region]
                )

        logger.info("MLM metrics: %s", metrics)
        return metrics
