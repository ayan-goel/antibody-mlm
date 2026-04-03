"""PyTorch Dataset for antibody sequences."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils.io import load_jsonl

CDR_FIELDS = {
    "cdr1_aa": 1,
    "cdr2_aa": 2,
    "cdr3_aa": 3,
}


def _build_cdr_region_labels(record: dict, seq_len: int) -> list[int]:
    """Build per-residue CDR region labels from OAS CDR amino acid fields.

    Returns a list of length seq_len where 0=framework, 1=CDR1, 2=CDR2, 3=CDR3.
    Searches for each CDR substring sequentially so later CDRs are found after
    earlier ones (avoids false matches from repeated subsequences).
    """
    sequence = record["sequence"]
    labels = [0] * seq_len
    search_start = 0
    for field, region_id in CDR_FIELDS.items():
        cdr_seq = record.get(field, "")
        if not cdr_seq:
            continue
        idx = sequence.find(cdr_seq, search_start)
        if idx == -1:
            continue
        for i in range(idx, idx + len(cdr_seq)):
            labels[i] = region_id
        search_start = idx + len(cdr_seq)
    return labels


class AntibodyDataset(Dataset):
    """Dataset that tokenizes antibody sequences on-the-fly.

    Each item returns a dict with 'input_ids', 'attention_mask',
    'special_tokens_mask', and optionally 'cdr_mask' (per-token CDR
    region labels) ready to be passed to an MLM data collator.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 160,
        coords_path: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = load_jsonl(data_path)
        self.coords: list | None = None
        if coords_path and Path(coords_path).exists():
            self.coords = torch.load(coords_path, weights_only=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        sequence = record["sequence"]
        spaced = " ".join(list(sequence))
        encoding = self.tokenizer(
            spaced,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_special_tokens_mask=True,
        )

        result = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
        }

        has_cdrs = any(record.get(f) for f in CDR_FIELDS)
        if has_cdrs:
            aa_labels = _build_cdr_region_labels(record, len(sequence))
            num_tokens = len(encoding["input_ids"])
            token_labels = [0] * num_tokens
            for i, label in enumerate(aa_labels):
                token_pos = i + 1  # offset for [CLS]
                if token_pos < num_tokens - 1:  # don't overwrite [SEP]
                    token_labels[token_pos] = label
            result["cdr_mask"] = token_labels

        if self.coords is not None and idx < len(self.coords) and self.coords[idx] is not None:
            aa_coords = self.coords[idx]["coords_ca"]
            num_tokens = len(encoding["input_ids"])
            token_coords = torch.zeros(num_tokens, 3)
            n_aa = min(len(aa_coords), num_tokens - 2)
            token_coords[1 : 1 + n_aa] = aa_coords[:n_aa].float()
            result["coords_ca"] = token_coords.tolist()

        return result
