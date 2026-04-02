"""CDR infilling and sequence restoration evaluation.

Three zero-shot tasks that measure a model's ability to reconstruct
masked regions using parallel argmax decoding (one forward pass):

  A. CDR span infilling  -- mask entire CDR, predict all positions
  B. N-terminus restoration -- mask first N residues, predict
  C. Random scattered restoration -- mask k random positions, predict
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from evaluation.base import BaseEvaluator

logger = logging.getLogger(__name__)

CDR_NAMES = {1: "cdr1", 2: "cdr2", 3: "cdr3"}

CDR3_LENGTH_BUCKETS = {
    "short": (0, 10),
    "medium": (11, 15),
    "long": (16, float("inf")),
}


def _levenshtein(a: list[int], b: list[int]) -> int:
    """Compute Levenshtein edit distance between two integer sequences."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j], prev = min(dp[j] + 1, dp[j - 1] + 1, prev + cost), dp[j]
    return dp[m]


class InfillingEvaluator(BaseEvaluator):
    """Zero-shot infilling evaluation for antibody masked LMs.

    Runs CDR span infilling, N-terminus restoration, and random scattered
    restoration on a held-out dataset. Uses parallel argmax decoding.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
        nterm_length: int = 10,
        scattered_k_values: tuple[int, ...] = (1, 5, 10),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.nterm_length = nterm_length
        self.scattered_k_values = scattered_k_values

    def _get_maskable_positions(self, input_ids: torch.Tensor) -> list[int]:
        """Return indices of non-special tokens (positions eligible for masking)."""
        special = set(self.tokenizer.all_special_ids)
        return [i for i in range(len(input_ids)) if input_ids[i].item() not in special]

    def _predict_masked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_positions: list[int],
    ) -> torch.Tensor:
        """Mask the given positions, forward pass, return argmax predictions."""
        masked = input_ids.clone()
        for pos in mask_positions:
            masked[pos] = self.tokenizer.mask_token_id

        with torch.no_grad():
            logits = self.model(
                input_ids=masked.unsqueeze(0).to(self.device),
                attention_mask=attention_mask.unsqueeze(0).to(self.device),
            ).logits[0]

        return logits.argmax(dim=-1).cpu()

    def _eval_cdr_infilling(self, sample: dict) -> dict[str, list[float]]:
        """Task A: mask each CDR entirely and predict."""
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)
        cdr_mask = sample.get("cdr_mask")
        if cdr_mask is None:
            return {}

        cdr_mask_t = torch.tensor(cdr_mask, dtype=torch.long)
        results: dict[str, list[float]] = defaultdict(list)

        for region_id, region_name in CDR_NAMES.items():
            positions = (cdr_mask_t == region_id).nonzero(as_tuple=True)[0].tolist()
            if not positions:
                continue

            true_tokens = [input_ids[p].item() for p in positions]
            predictions = self._predict_masked(input_ids, attention_mask, positions)
            pred_tokens = [predictions[p].item() for p in positions]

            correct = sum(1 for t, p in zip(true_tokens, pred_tokens) if t == p)
            cdr_len = len(positions)
            accuracy = correct / cdr_len
            exact_match = 1.0 if correct == cdr_len else 0.0
            edit_dist = _levenshtein(true_tokens, pred_tokens)

            results[f"infill_{region_name}_accuracy"].append(accuracy)
            results[f"infill_{region_name}_exact_match"].append(exact_match)
            results[f"infill_{region_name}_edit_distance"].append(edit_dist)

            if region_id == 3:
                for bucket_name, (lo, hi) in CDR3_LENGTH_BUCKETS.items():
                    if lo <= cdr_len <= hi:
                        results[f"infill_cdr3_{bucket_name}_accuracy"].append(accuracy)
                        results[f"infill_cdr3_{bucket_name}_exact_match"].append(exact_match)
                        break

        return results

    def _eval_nterm_restoration(self, sample: dict) -> dict[str, list[float]]:
        """Task B: mask first N amino acid positions and predict."""
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)
        maskable = self._get_maskable_positions(input_ids)

        n = min(self.nterm_length, len(maskable))
        if n == 0:
            return {}

        positions = maskable[:n]
        true_tokens = [input_ids[p].item() for p in positions]
        predictions = self._predict_masked(input_ids, attention_mask, positions)
        pred_tokens = [predictions[p].item() for p in positions]

        correct = sum(1 for t, p in zip(true_tokens, pred_tokens) if t == p)
        accuracy = correct / len(positions)
        exact_match = 1.0 if correct == len(positions) else 0.0
        edit_dist = _levenshtein(true_tokens, pred_tokens)

        return {
            "nterm_accuracy": [accuracy],
            "nterm_exact_match": [exact_match],
            "nterm_edit_distance": [edit_dist],
        }

    def _eval_scattered_restoration(self, sample: dict) -> dict[str, list[float]]:
        """Task C: mask k random positions and predict."""
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)
        maskable = self._get_maskable_positions(input_ids)
        results: dict[str, list[float]] = {}

        for k in self.scattered_k_values:
            if k > len(maskable):
                continue
            positions = random.sample(maskable, k)
            true_tokens = [input_ids[p].item() for p in positions]
            predictions = self._predict_masked(input_ids, attention_mask, positions)
            pred_tokens = [predictions[p].item() for p in positions]

            correct = sum(1 for t, p in zip(true_tokens, pred_tokens) if t == p)
            results[f"scattered_accuracy_k{k}"] = [correct / k]

        return results

    def evaluate(
        self,
        dataset: torch.utils.data.Dataset,
        max_samples: int = 1000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run all infilling tasks on the dataset.

        Args:
            dataset: AntibodyDataset (or Subset) with tokenized samples.
            max_samples: Maximum number of samples to evaluate.

        Returns:
            Dict of aggregated metrics (means across all evaluated samples).
        """
        self.model.eval()
        self.model.to(self.device)

        n = min(max_samples, len(dataset))
        indices = list(range(n))

        all_metrics: dict[str, list[float]] = defaultdict(list)

        for idx in tqdm(indices, desc="Infilling evaluation"):
            sample = dataset[idx]

            for task_results in [
                self._eval_cdr_infilling(sample),
                self._eval_nterm_restoration(sample),
                self._eval_scattered_restoration(sample),
            ]:
                for key, values in task_results.items():
                    all_metrics[key].extend(values)

        aggregated: dict[str, Any] = {}
        for key, values in sorted(all_metrics.items()):
            aggregated[key] = sum(values) / len(values) if values else 0.0
            aggregated[f"{key}_count"] = len(values)

        logger.info("Infilling metrics: %d keys computed over %d samples", len(aggregated), n)
        return aggregated
