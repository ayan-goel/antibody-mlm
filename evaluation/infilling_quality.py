"""Infilling quality analysis: distributional comparison of model-generated vs natural CDRs.

Complements the exact-match / edit-distance metrics in infilling.py with a
softer check: are the amino acid frequencies and CDR lengths produced by the
model realistic compared to the ground-truth distribution?

Metrics computed per CDR type (CDR1, CDR2, CDR3):
  - Per-AA frequency (fraction of each of the 20 standard AAs)
  - Jensen-Shannon divergence between predicted and true AA distributions
  - CDR3 length distribution statistics (mean, std, histogram)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
CDR_NAMES = {1: "cdr1", 2: "cdr2", 3: "cdr3"}


def _count_to_distribution(counter: Counter, vocabulary: list[str]) -> np.ndarray:
    """Convert a Counter of AA characters to a normalized distribution over vocabulary."""
    total = sum(counter.values())
    if total == 0:
        return np.ones(len(vocabulary)) / len(vocabulary)
    return np.array([counter.get(aa, 0) / total for aa in vocabulary], dtype=np.float64)


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two probability distributions."""
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


class InfillingQualityAnalyzer:
    """Analyze distributional properties of model-infilled CDRs vs ground truth.

    For each sample, masks each CDR entirely, predicts via argmax, then
    collects AA frequencies and lengths for comparison.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _predict_masked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_positions: list[int],
    ) -> torch.Tensor:
        """Mask the given positions, run forward pass, return argmax predictions."""
        masked = input_ids.clone()
        for pos in mask_positions:
            masked[pos] = self.tokenizer.mask_token_id

        with torch.no_grad():
            logits = self.model(
                input_ids=masked.unsqueeze(0).to(self.device),
                attention_mask=attention_mask.unsqueeze(0).to(self.device),
            ).logits[0]

        return logits.argmax(dim=-1).cpu()

    def _tokens_to_aa_string(self, token_ids: list[int]) -> str:
        """Convert a list of token IDs to a string of amino acid characters."""
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        special = set(self.tokenizer.all_special_tokens)
        return "".join(t for t in tokens if t not in special and len(t) == 1 and t.isalpha())

    def analyze(
        self,
        dataset: torch.utils.data.Dataset,
        max_samples: int = 1000,
    ) -> dict[str, Any]:
        """Run infilling quality analysis.

        Args:
            dataset: AntibodyDataset (or Subset) with tokenized samples and cdr_mask.
            max_samples: Maximum number of samples to evaluate.

        Returns:
            Dict with per-CDR AA frequency tables, JSD values, and CDR3 length stats.
        """
        self.model.eval()
        self.model.to(self.device)

        n = min(max_samples, len(dataset))

        true_aa_counts: dict[str, Counter] = defaultdict(Counter)
        pred_aa_counts: dict[str, Counter] = defaultdict(Counter)
        true_cdr3_lengths: list[int] = []
        pred_cdr3_lengths: list[int] = []

        samples_with_cdr = 0
        for idx in tqdm(range(n), desc="Infilling quality analysis"):
            sample = dataset[idx]
            cdr_mask = sample.get("cdr_mask")
            if cdr_mask is None:
                continue
            samples_with_cdr += 1

            input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)
            cdr_mask_t = torch.tensor(cdr_mask, dtype=torch.long)
            chain_type_ids = sample.get("chain_type_ids")
            chain_type_t = (
                torch.tensor(chain_type_ids, dtype=torch.long)
                if chain_type_ids is not None else None
            )

            for region_id, region_name in CDR_NAMES.items():
                region_positions = (cdr_mask_t == region_id).nonzero(as_tuple=True)[0].tolist()
                if not region_positions:
                    continue

                # Split heavy / light positions for paired samples so the
                # `cdr*_jsd` metric is comparable to single-chain models
                # (which only have heavy CDRs). chain_type_ids convention:
                # 1 = heavy, 2 = light, 0 = special.
                if chain_type_t is None:
                    chain_buckets = [(region_name, region_positions)]
                else:
                    heavy = [p for p in region_positions if chain_type_t[p].item() == 1]
                    light = [p for p in region_positions if chain_type_t[p].item() == 2]
                    chain_buckets = [
                        (region_name, heavy),
                        (f"{region_name}_light", light),
                    ]

                for bucket_name, positions in chain_buckets:
                    if not positions:
                        continue
                    true_tokens = [input_ids[p].item() for p in positions]
                    true_aa_str = self._tokens_to_aa_string(true_tokens)
                    for aa in true_aa_str:
                        true_aa_counts[bucket_name][aa] += 1

                    predictions = self._predict_masked(input_ids, attention_mask, positions)
                    pred_tokens = [predictions[p].item() for p in positions]
                    pred_aa_str = self._tokens_to_aa_string(pred_tokens)
                    for aa in pred_aa_str:
                        pred_aa_counts[bucket_name][aa] += 1

                    if region_id == 3 and bucket_name == "cdr3":
                        true_cdr3_lengths.append(len(true_aa_str))
                        pred_cdr3_lengths.append(len(pred_aa_str))

        results: dict[str, Any] = {
            "num_samples": n,
            "num_samples_with_cdr": samples_with_cdr,
        }

        # Iterate over every bucket we accumulated, not just heavy CDR1/2/3.
        # Paired samples populate `<region>_light` buckets in addition.
        for bucket_name in sorted(true_aa_counts):
            true_dist = _count_to_distribution(true_aa_counts[bucket_name], STANDARD_AAS)
            pred_dist = _count_to_distribution(pred_aa_counts[bucket_name], STANDARD_AAS)

            jsd = _jensen_shannon_divergence(true_dist, pred_dist)

            results[f"{bucket_name}_jsd"] = float(jsd)
            results[f"{bucket_name}_true_aa_freq"] = {
                aa: float(f) for aa, f in zip(STANDARD_AAS, true_dist)
            }
            results[f"{bucket_name}_pred_aa_freq"] = {
                aa: float(f) for aa, f in zip(STANDARD_AAS, pred_dist)
            }
            results[f"{bucket_name}_true_total_residues"] = sum(true_aa_counts[bucket_name].values())
            results[f"{bucket_name}_pred_total_residues"] = sum(pred_aa_counts[bucket_name].values())

        if true_cdr3_lengths:
            results["cdr3_length_true_mean"] = float(np.mean(true_cdr3_lengths))
            results["cdr3_length_true_std"] = float(np.std(true_cdr3_lengths))
            results["cdr3_length_pred_mean"] = float(np.mean(pred_cdr3_lengths))
            results["cdr3_length_pred_std"] = float(np.std(pred_cdr3_lengths))

            bins = list(range(0, max(max(true_cdr3_lengths), max(pred_cdr3_lengths)) + 5, 2))
            true_hist, _ = np.histogram(true_cdr3_lengths, bins=bins)
            pred_hist, _ = np.histogram(pred_cdr3_lengths, bins=bins)
            results["cdr3_length_histogram_bins"] = bins
            results["cdr3_length_histogram_true"] = true_hist.tolist()
            results["cdr3_length_histogram_pred"] = pred_hist.tolist()

        return results
