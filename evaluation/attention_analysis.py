"""Attention perturbation analysis for antibody masked language models.

Zero-shot analysis (no training required).  Three sub-analyses:
  1. Attention entropy — measures how diffuse or focused each head is.
  2. Head importance — ablation-based accuracy drop per head.
  3. Attention–contact correlation — Spearman between attention weights
     and structural contact matrices (when coords data is available).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from evaluation.base import BaseEvaluator

logger = logging.getLogger(__name__)


class AttentionAnalyzer(BaseEvaluator):
    """Zero-shot attention analysis for antibody MLMs.

    Computes per-layer/head entropy, ablation-based importance, and
    (optionally) attention–contact correlation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str = "cuda",
        coords_data: list | None = None,
        max_samples: int = 200,
        sabdab_real_coords: list | None = None,
    ) -> None:
        """
        Args:
            model, tokenizer, device, max_samples: standard.
            coords_data: legacy ESM-2 kNN data aligned by index with the
                eval ``dataset``. Each entry is ``{"knn_indices": tensor}``
                or None. If provided (and ``sabdab_real_coords`` is not),
                the attention–contact correlation runs against
                ESM-2-derived contact maps. Deprecated.
            sabdab_real_coords: list of dicts from
                ``data/structures/sabdab_liberis_coords.pt``. Each entry
                has ``{"sequence": str, "coords_ca": tensor(L, 3)}``.
                When provided, the attention–contact correlation iterates
                over these real-X-ray chains directly, tokenizing each
                sequence on the fly. Bypasses the eval ``dataset`` for
                that sub-analysis. Strongly preferred for paper-quality
                results.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.coords_data = coords_data
        self.sabdab_real_coords = sabdab_real_coords
        self.max_samples = max_samples

    def evaluate(self, dataset: Any = None, **kwargs: Any) -> dict[str, Any]:
        self.model.eval()
        self.model.to(self.device)

        n = min(self.max_samples, len(dataset))
        samples = [dataset[i] for i in range(n)]

        metrics: dict[str, Any] = {}
        metrics.update(self._attention_entropy(samples))
        metrics.update(self._head_importance(samples))
        if self.sabdab_real_coords is not None:
            metrics.update(self._attention_contact_correlation_real(
                self.sabdab_real_coords[: self.max_samples]
            ))
        elif self.coords_data is not None:
            metrics.update(self._attention_contact_correlation(samples))
        else:
            logger.info("  No coords data provided — skipping attention–contact correlation")
        return metrics

    def _get_attentions(self, sample: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning attention weights and attention mask."""
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        # attentions: tuple of (1, num_heads, L, L) per layer
        attentions = torch.stack([a.squeeze(0) for a in outputs.attentions])  # (n_layers, n_heads, L, L)
        return attentions, attention_mask.squeeze(0)

    def _attention_entropy(self, samples: list[dict]) -> dict[str, float]:
        """Compute Shannon entropy of attention distributions per layer/head."""
        logger.info("  [Attention] Computing attention entropy...")
        layer_head_entropies: dict[tuple[int, int], list[float]] = defaultdict(list)

        for sample in tqdm(samples, desc="Attention entropy"):
            attentions, mask = self._get_attentions(sample)
            n_layers, n_heads, L, _ = attentions.shape
            valid_len = mask.sum().item()

            for layer in range(n_layers):
                for head in range(n_heads):
                    attn = attentions[layer, head, :valid_len, :valid_len]
                    # Shannon entropy: -sum(p * log(p)) per query, averaged
                    attn_clamped = attn.clamp(min=1e-12)
                    entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)
                    layer_head_entropies[(layer, head)].append(entropy.mean().item())

        metrics: dict[str, float] = {}
        all_entropies: list[float] = []
        n_layers = max(k[0] for k in layer_head_entropies) + 1 if layer_head_entropies else 0
        n_heads = max(k[1] for k in layer_head_entropies) + 1 if layer_head_entropies else 0

        for layer in range(n_layers):
            layer_vals: list[float] = []
            for head in range(n_heads):
                mean_ent = float(np.mean(layer_head_entropies[(layer, head)]))
                metrics[f"attn_entropy_L{layer}_H{head}"] = mean_ent
                layer_vals.append(mean_ent)
            metrics[f"attn_entropy_layer{layer}"] = float(np.mean(layer_vals))
            all_entropies.extend(layer_vals)

        if all_entropies:
            metrics["attn_entropy_mean"] = float(np.mean(all_entropies))

        return metrics

    def _head_importance(self, samples: list[dict]) -> dict[str, float]:
        """Measure accuracy drop when zeroing each attention head's output."""
        logger.info("  [Attention] Computing head importance via ablation...")
        special_ids = set(self.tokenizer.all_special_ids)
        mask_token_id = self.tokenizer.mask_token_id

        # Get model config
        config = self.model.config
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        head_dim = config.hidden_size // n_heads

        # Baseline accuracy (no ablation)
        baseline_correct, baseline_total = 0, 0
        for sample in samples:
            c, t = self._mlm_accuracy_single(sample, special_ids, mask_token_id)
            baseline_correct += c
            baseline_total += t
        baseline_acc = baseline_correct / max(baseline_total, 1)

        metrics: dict[str, float] = {}
        max_drop = -1.0
        most_important = ""

        for layer in tqdm(range(n_layers), desc="Head ablation"):
            # Access the attention output projection in RoFormer
            attn_module = self.model.roformer.encoder.layer[layer].attention.self

            for head in range(n_heads):
                handle = _register_zero_head_hook(attn_module, head, head_dim)
                try:
                    ablated_correct, ablated_total = 0, 0
                    for sample in samples:
                        c, t = self._mlm_accuracy_single(sample, special_ids, mask_token_id)
                        ablated_correct += c
                        ablated_total += t
                    ablated_acc = ablated_correct / max(ablated_total, 1)
                    drop = baseline_acc - ablated_acc
                    metrics[f"head_importance_L{layer}_H{head}"] = float(drop)
                    if drop > max_drop:
                        max_drop = drop
                        most_important = f"L{layer}_H{head}"
                finally:
                    handle.remove()

        metrics["head_importance_baseline_acc"] = baseline_acc
        if most_important:
            metrics["most_important_head"] = most_important
            metrics["most_important_head_drop"] = max_drop

        return metrics

    def _mlm_accuracy_single(
        self, sample: dict, special_ids: set, mask_token_id: int,
    ) -> tuple[int, int]:
        """Compute MLM accuracy for a single sample using 15% random masking."""
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)

        maskable = [
            i for i in range(len(input_ids))
            if input_ids[i].item() not in special_ids
        ]
        if not maskable:
            return 0, 0

        n_mask = max(1, int(0.15 * len(maskable)))
        # Use deterministic subset based on content hash for reproducibility
        positions = maskable[:n_mask]

        masked_ids = input_ids.clone()
        true_tokens = []
        for pos in positions:
            true_tokens.append(input_ids[pos].item())
            masked_ids[pos] = mask_token_id

        with torch.no_grad():
            logits = self.model(
                input_ids=masked_ids.unsqueeze(0).to(self.device),
                attention_mask=attention_mask.unsqueeze(0).to(self.device),
            ).logits[0]

        preds = logits.argmax(dim=-1).cpu()
        correct = sum(1 for pos, true in zip(positions, true_tokens) if preds[pos].item() == true)
        return correct, len(positions)

    def _attention_contact_correlation_real(
        self, sabdab_entries: list[dict],
    ) -> dict[str, float]:
        """Correlate attention with REAL X-ray contact maps (SAbDab).

        For each SAbDab antibody chain:
          1. Tokenize its PDB-derived sequence
          2. Run a forward pass to get attention weights
          3. Build the contact map from real Calpha distances (8 Å threshold)
          4. Compute Spearman correlation between attention and contact
             over the upper triangle (sequence-aware indexing — only
             non-special token positions)

        Aggregates per-layer/head means across all chains.
        """
        from utils.tokenizer import tokenize_single_chain

        logger.info(
            "  [Attention] Computing attention-contact correlation against REAL "
            "X-ray crystal contacts (SAbDab Liberis, 8 Å threshold)..."
        )
        if not sabdab_entries:
            return {}

        layer_head_corrs: dict[tuple[int, int], list[float]] = defaultdict(list)
        n_used = 0
        threshold_sq = 8.0 ** 2

        for entry in tqdm(sabdab_entries, desc="Attn-contact corr (real)"):
            sequence = entry["sequence"]
            coords = entry["coords_ca"].float()
            if len(sequence) < 10 or coords.size(0) < 10:
                continue

            # Tokenize on the fly. tokenize_single_chain handles both
            # standard and multispecific (paired) tokenizers.
            try:
                encoding = tokenize_single_chain(self.tokenizer, sequence, max_length=256)
            except Exception as e:  # noqa: BLE001
                logger.debug("Tokenization failed for %s: %s", entry.get("id", "?"), e)
                continue

            input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long).unsqueeze(0).to(self.device)
            special_tokens_mask = encoding["special_tokens_mask"]

            # Identify AA-only token positions (non-special, in attention mask).
            aa_token_positions = [
                i for i, sm in enumerate(special_tokens_mask) if sm == 0
            ]
            n_aa = min(len(aa_token_positions), coords.size(0))
            if n_aa < 5:
                continue
            aa_token_positions = aa_token_positions[:n_aa]
            coords_used = coords[:n_aa]

            # Real contact matrix at 8 Å (squared distance < 64).
            diff = coords_used.unsqueeze(0) - coords_used.unsqueeze(1)
            dist_sq = (diff ** 2).sum(dim=-1)
            contact = (dist_sq < threshold_sq).float()
            contact.fill_diagonal_(0.0)

            tri_i, tri_j = torch.triu_indices(n_aa, n_aa, offset=1)
            contact_flat = contact[tri_i, tri_j].cpu().numpy()
            if contact_flat.std() == 0:
                continue

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
            # attentions: tuple of (1, n_heads, L, L) per layer
            attentions = torch.stack([a.squeeze(0) for a in outputs.attentions])  # (n_layers, n_heads, L, L)
            n_layers, n_heads = attentions.shape[0], attentions.shape[1]

            # Index into the AA-only positions in token space.
            aa_idx = torch.tensor(aa_token_positions, dtype=torch.long)
            for layer in range(n_layers):
                for head in range(n_heads):
                    attn = attentions[layer, head][aa_idx][:, aa_idx].cpu()
                    attn_flat = attn[tri_i, tri_j].numpy()
                    if attn_flat.std() == 0:
                        continue
                    rho, _ = spearmanr(attn_flat, contact_flat)
                    if not np.isnan(rho):
                        layer_head_corrs[(layer, head)].append(float(rho))

            n_used += 1

        logger.info("  Real-contact correlation computed over %d chains", n_used)

        metrics: dict[str, float] = {}
        best_corr = -1.0
        best_head = ""
        for (layer, head), corrs in sorted(layer_head_corrs.items()):
            mean_corr = float(np.mean(corrs))
            metrics[f"attn_contact_corr_L{layer}_H{head}"] = mean_corr
            if mean_corr > best_corr:
                best_corr = mean_corr
                best_head = f"L{layer}_H{head}"

        if best_head:
            metrics["best_structural_head"] = best_head
            metrics["best_structural_head_corr"] = best_corr
        metrics["attn_contact_n_samples"] = n_used
        return metrics

    def _attention_contact_correlation(self, samples: list[dict]) -> dict[str, float]:
        """[DEPRECATED] Correlate attention with ESM-2 kNN contact maps.

        Kept for backwards compatibility. Prefer
        :meth:`_attention_contact_correlation_real` which uses REAL
        X-ray crystal contacts instead of ESM-2's predictions.
        """
        logger.info("  [Attention] Computing attention–contact correlation...")
        if self.coords_data is None:
            return {}

        layer_head_corrs: dict[tuple[int, int], list[float]] = defaultdict(list)
        n_used = 0

        for i, sample in enumerate(tqdm(samples, desc="Attn-contact corr")):
            if i >= len(self.coords_data) or self.coords_data[i] is None:
                continue
            knn_entry = self.coords_data[i]
            if "knn_indices" not in knn_entry:
                continue

            knn = knn_entry["knn_indices"]
            seq_len = knn.size(0)

            # Build binary contact matrix
            contact = torch.zeros(seq_len, seq_len)
            for ri in range(seq_len):
                for rj in knn[ri]:
                    j = rj.item()
                    if 0 <= j < seq_len:
                        contact[ri, j] = 1.0
                        contact[j, ri] = 1.0
            contact.fill_diagonal_(0.0)

            # Extract upper triangle
            tri_i, tri_j = torch.triu_indices(seq_len, seq_len, offset=1)
            contact_flat = contact[tri_i, tri_j].numpy()

            if contact_flat.std() == 0:
                continue

            attentions, mask = self._get_attentions(sample)
            n_layers, n_heads, L, _ = attentions.shape

            # Attention is over token positions; map to AA-only positions
            # (skip [CLS] at 0, take next seq_len positions)
            aa_start = 1
            aa_end = min(aa_start + seq_len, L)
            actual_len = aa_end - aa_start
            if actual_len < seq_len:
                tri_i_a, tri_j_a = torch.triu_indices(actual_len, actual_len, offset=1)
                contact_sub = contact[:actual_len, :actual_len]
                contact_flat = contact_sub[tri_i_a, tri_j_a].numpy()
                if contact_flat.std() == 0:
                    continue
            else:
                tri_i_a = tri_i
                tri_j_a = tri_j

            for layer in range(n_layers):
                for head in range(n_heads):
                    attn = attentions[layer, head, aa_start:aa_end, aa_start:aa_end].cpu()
                    attn_flat = attn[tri_i_a, tri_j_a].numpy()
                    if attn_flat.std() == 0:
                        continue
                    rho, _ = spearmanr(attn_flat, contact_flat)
                    if not np.isnan(rho):
                        layer_head_corrs[(layer, head)].append(float(rho))

            n_used += 1

        logger.info("  Attention–contact correlation computed over %d samples", n_used)

        metrics: dict[str, float] = {}
        best_corr = -1.0
        best_head = ""

        for (layer, head), corrs in sorted(layer_head_corrs.items()):
            mean_corr = float(np.mean(corrs))
            metrics[f"attn_contact_corr_L{layer}_H{head}"] = mean_corr
            if mean_corr > best_corr:
                best_corr = mean_corr
                best_head = f"L{layer}_H{head}"

        if best_head:
            metrics["best_structural_head"] = best_head
            metrics["best_structural_head_corr"] = best_corr
        metrics["attn_contact_n_samples"] = n_used

        return metrics


def _register_zero_head_hook(
    attn_module: torch.nn.Module, head_idx: int, head_dim: int,
) -> torch.utils.hooks.RemovableHook:
    """Register a forward hook that zeros out a specific attention head's output."""

    def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> Any:
        # RoFormer self-attention returns (context_layer, attention_probs)
        # or just context_layer depending on output_attentions
        if isinstance(output, tuple):
            context = output[0]
        else:
            context = output
        start = head_idx * head_dim
        end = start + head_dim
        context[:, :, start:end] = 0.0
        if isinstance(output, tuple):
            return (context,) + output[1:]
        return context

    return attn_module.register_forward_hook(hook_fn)
