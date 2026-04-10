"""Embedding cache for probe-mode downstream training.

Extracts per-token hidden states from a frozen encoder once and saves
them to disk so the probe head can train without re-running the encoder
every epoch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from masking.base import BaseMaskingStrategy
from masking.collator import MLMDataCollator

logger = logging.getLogger(__name__)


# Bumped from v1 (hidden_states + attention_mask only) to v2 when we
# started saving special_tokens_mask so downstream pooling can exclude
# framing tokens. Caches without `cache_version >= CACHE_VERSION` are
# rejected and regenerated.
CACHE_VERSION = 2


def _save_cache_meta(
    cache_path: Path, checkpoint_path: str, tokenizer_type: str = "standard",
) -> None:
    """Write a sidecar metadata file alongside the embedding cache."""
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta")
    meta_path.write_text(json.dumps({
        "checkpoint_path": checkpoint_path,
        "tokenizer_type": tokenizer_type,
        "cache_version": CACHE_VERSION,
    }))


def cache_is_valid(
    cache_path: str | Path, checkpoint_path: str, tokenizer_type: str = "standard",
) -> bool:
    """Check if a cache file exists and matches the given checkpoint + tokenizer.

    A cache is considered invalid if any of the following are true:
      - the cache file or its meta sidecar is missing
      - the checkpoint path differs
      - the tokenizer type (standard vs paired) differs
      - ``cache_version`` is older than the current ``CACHE_VERSION``
        (i.e. it lacks new fields like ``special_tokens_mask``)
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta")
    if not meta_path.exists():
        return False  # legacy cache without meta — treat as invalid
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("checkpoint_path", "") != checkpoint_path:
            return False
        if meta.get("tokenizer_type", "standard") != tokenizer_type:
            return False
        if meta.get("cache_version", 1) < CACHE_VERSION:
            return False
        return True
    except Exception:
        return False


def extract_and_cache(
    encoder: torch.nn.Module,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    cache_path: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
    checkpoint_path: str = "",
    tokenizer_type: str = "standard",
) -> Path:
    """Extract per-token embeddings and save to disk.

    Args:
        encoder: EncoderWrapper (frozen, in eval mode).
        dataset: AntibodyDataset or similar returning tokenized dicts.
        tokenizer: Tokenizer for the collator.
        cache_path: Where to save the .pt file.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        device: Torch device.
        checkpoint_path: Path to the checkpoint used for encoding.
            Saved alongside the cache for invalidation on re-run.

    Returns:
        Path to the saved cache file.
    """

    class _NoOpStrategy(BaseMaskingStrategy):
        def select_mask_positions(self, input_ids, special_tokens_mask, metadata=None):
            return torch.zeros_like(input_ids, dtype=torch.bool)

    collator = MLMDataCollator(
        tokenizer=tokenizer,
        strategy=_NoOpStrategy(tokenizer=tokenizer, mask_prob=0.0),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    all_hidden: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_specials: list[torch.Tensor] = []

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Caching embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # The collator does not produce special_tokens_mask (it's only
            # in each example dict). Re-derive from input_ids using the
            # tokenizer's special-id set so we don't have to plumb the
            # field through the collator.
            special_ids_tensor = torch.tensor(
                list(tokenizer.all_special_ids), device=input_ids.device,
            )
            special_mask = torch.isin(input_ids, special_ids_tensor).long()
            hidden_states = encoder(input_ids, attention_mask)
            all_hidden.append(hidden_states.cpu())
            all_masks.append(attention_mask.cpu())
            all_specials.append(special_mask.cpu())

    global_max_len = max(h.size(1) for h in all_hidden)
    for i in range(len(all_hidden)):
        pad = global_max_len - all_hidden[i].size(1)
        if pad > 0:
            all_hidden[i] = F.pad(all_hidden[i], (0, 0, 0, pad))
            all_masks[i] = F.pad(all_masks[i], (0, pad))
            # Pad positions are "special" (they should never be pooled).
            all_specials[i] = F.pad(all_specials[i], (0, pad), value=1)

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_states": torch.cat(all_hidden, dim=0),
            "attention_mask": torch.cat(all_masks, dim=0),
            "special_tokens_mask": torch.cat(all_specials, dim=0),
        },
        cache_path,
    )
    if checkpoint_path:
        _save_cache_meta(cache_path, checkpoint_path, tokenizer_type)
    logger.info("Cached %d samples to %s", sum(m.size(0) for m in all_masks), cache_path)
    return cache_path


class CachedEmbeddingDataset(Dataset):
    """Dataset backed by pre-extracted per-token embeddings.

    Each item returns a dict with 'hidden_states', 'attention_mask',
    and 'labels' (supplied at init).

    When ``device`` is provided, the entire cache is moved to that device
    at construction time, and labels are stacked into a single tensor.
    Probe-mode training can then iterate via direct indexing without
    paying CPU→GPU transfer cost on every batch.
    """

    def __init__(
        self,
        cache_path: str | Path,
        labels: list[Any],
        device: str = "cpu",
    ) -> None:
        cache = torch.load(cache_path, weights_only=True)
        self.hidden_states: torch.Tensor = cache["hidden_states"]
        self.attention_mask: torch.Tensor = cache["attention_mask"]
        # cache_version >= 2 includes special_tokens_mask. Older caches
        # won't reach this code path because cache_is_valid rejects them.
        if "special_tokens_mask" in cache:
            self.special_tokens_mask: torch.Tensor = cache["special_tokens_mask"]
        else:
            # Defensive fallback (shouldn't happen given cache_is_valid checks).
            self.special_tokens_mask = torch.zeros_like(self.attention_mask)
        if len(labels) != self.hidden_states.size(0):
            raise ValueError(
                f"Label count ({len(labels)}) != cached sample count "
                f"({self.hidden_states.size(0)})"
            )
        seq_len = self.hidden_states.size(1)
        self.labels = self._align_labels(labels, seq_len)

        # Stack labels into a single tensor (all per-dataset labels share shape).
        self.labels_tensor: torch.Tensor = torch.stack(
            [t if t.dim() > 0 else t.unsqueeze(0).squeeze(0) for t in self.labels]
        ) if len(self.labels) > 0 else torch.empty(0)

        self.device = device
        if device != "cpu":
            self.hidden_states = self.hidden_states.to(device)
            self.attention_mask = self.attention_mask.to(device)
            self.special_tokens_mask = self.special_tokens_mask.to(device)
            self.labels_tensor = self.labels_tensor.to(device)

    @staticmethod
    def _align_labels(labels: list[Any], seq_len: int) -> list[torch.Tensor]:
        """Convert labels to tensors, padding token-level labels to seq_len.

        Only pads 1D labels whose size is already close to seq_len (i.e.
        per-token labels that just need padding alignment). Short vectors
        like multi-target regression labels (e.g. size 5) are left as-is.
        """
        aligned: list[torch.Tensor] = []
        for lab in labels:
            t = lab if isinstance(lab, torch.Tensor) else torch.tensor(lab)
            is_token_level = (
                t.dim() >= 1
                and t.size(0) != seq_len
                and seq_len // 4 < t.size(0) <= seq_len * 2
            )
            if is_token_level:
                pad = seq_len - t.size(0)
                if pad > 0:
                    t = F.pad(t, (0, pad), value=-100)
                else:
                    t = t[:seq_len]
            aligned.append(t)
        return aligned

    def __len__(self) -> int:
        return self.hidden_states.size(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "hidden_states": self.hidden_states[idx],
            "attention_mask": self.attention_mask[idx],
            "special_tokens_mask": self.special_tokens_mask[idx],
            "labels": self.labels_tensor[idx],
        }
