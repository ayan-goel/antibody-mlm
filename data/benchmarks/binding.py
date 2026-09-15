"""Binding specificity dataset from CoV-AbDab.

Binary classification: given a VH sequence, predict whether the antibody
neutralizes SARS-CoV-2.  Label 1 = neutralizing, label 0 = non-neutralizing.

Source: https://opig.stats.ox.ac.uk/webapps/covabdab/
"""

from __future__ import annotations

import logging
import random
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from utils.tokenizer import tokenize_single_chain

logger = logging.getLogger(__name__)

_COVABDAB_URL = (
    "https://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/"
    "CoV-AbDab_080224.csv"
)
_MIN_VH_LEN = 20


def download_covabdab(data_dir: str | Path) -> Path:
    """Download CoV-AbDab CSV if not already cached."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "CoV-AbDab.csv"
    if csv_path.exists():
        logger.info("CoV-AbDab already cached at %s", csv_path)
        return csv_path
    logger.info("Downloading CoV-AbDab from %s ...", _COVABDAB_URL)
    req = urllib.request.Request(_COVABDAB_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        csv_path.write_bytes(resp.read())
    logger.info("Saved CoV-AbDab to %s", csv_path)
    return csv_path


def _load_and_filter(csv_path: Path) -> pd.DataFrame:
    """Load CoV-AbDab CSV, filter to usable SARS-CoV-2 neutralization samples."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    df = df.rename(columns=lambda c: c.strip())

    vh_col = "VHorVHH"
    df = df[df[vh_col].notna()]
    df = df[df[vh_col].str.strip() != "ND"]
    df = df[df[vh_col].str.len() >= _MIN_VH_LEN]

    df = df[df["Binds to"].str.contains("SARS-CoV2", na=False)]

    has_neut = df["Neutralising Vs"].fillna("").str.strip() != ""
    has_not_neut = df["Not Neutralising Vs"].fillna("").str.strip() != ""
    df = df[has_neut | has_not_neut].copy()

    df["label"] = df["Neutralising Vs"].fillna("").apply(
        lambda s: 1 if "SARS-CoV2" in s else 0
    )
    df["vh_sequence"] = df[vh_col].str.strip()

    # Clonotype key for grouped splitting. SARS-CoV-2 responses are dominated
    # by public clonotypes (IGHV3-53/3-66 and friends), so clonal relatives are
    # near-duplicates: a random split scatters them across train and test and
    # inflates every metric. Standard approximate clonotype = same V gene,
    # same CDRH3 length, same CDRH3 prefix.
    v_gene = df["Heavy V Gene"].fillna("NA").str.split("*").str[0].str.strip()
    cdrh3 = df["CDRH3"].fillna("NA").astype(str)
    df["clonotype"] = (
        v_gene + "|" + cdrh3.str.len().astype(str) + "|" + cdrh3.str[:4]
    )

    logger.info(
        "CoV-AbDab filtered: %d samples (pos=%d, neg=%d, clonotypes=%d)",
        len(df),
        (df["label"] == 1).sum(),
        (df["label"] == 0).sum(),
        df["clonotype"].nunique(),
    )
    return df[["vh_sequence", "label", "clonotype"]].reset_index(drop=True)


class BindingDataset(Dataset):
    """Sequence-level binary classification dataset for binding specificity.

    Each item returns input_ids, attention_mask, special_tokens_mask,
    and a scalar label (0 or 1).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 160,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences: list[str] = df["vh_sequence"].tolist()
        self.labels: list[int] = df["label"].tolist()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sequence = self.sequences[idx]
        encoding = tokenize_single_chain(self.tokenizer, sequence, self.max_length)
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
            "labels": self.labels[idx],
        }


def load_binding_splits(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 160,
    data_dir: str | Path = "data/covabdab",
    seed: int = 42,
) -> tuple[BindingDataset, BindingDataset, BindingDataset]:
    """Download CoV-AbDab and return clonotype-grouped (train, val, test) datasets.

    Splits are grouped by approximate clonotype so no clonal family straddles
    train and test. A label-stratified *random* split would place near-identical
    public-clonotype relatives on both sides and overstate performance; the
    grouping trades a little label balance for a split that measures
    generalisation to unseen lineages.
    """
    csv_path = download_covabdab(data_dir)
    df = _load_and_filter(csv_path)

    # Assign whole clonotypes to splits, largest first, to whichever split is
    # furthest below its target share (same bin-packing idea as AB-Bind).
    sizes = df.groupby("clonotype").size().to_dict()
    targets = {"train": 0.70 * len(df), "val": 0.15 * len(df), "test": 0.15 * len(df)}
    current = {"train": 0, "val": 0, "test": 0}
    assignment: dict[str, str] = {}

    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    for clono in sorted(order, key=lambda c: -sizes[c]):
        split = max(targets, key=lambda s: targets[s] - current[s])
        assignment[clono] = split
        current[split] += sizes[clono]

    split_col = df["clonotype"].map(assignment)
    train_df = df[split_col == "train"]
    val_df = df[split_col == "val"]
    test_df = df[split_col == "test"]

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_pos = (split_df["label"] == 1).sum()
        n_neg = (split_df["label"] == 0).sum()
        logger.info(
            "  %s: %d samples (pos=%d, neg=%d, clonotypes=%d)",
            name, len(split_df), n_pos, n_neg, split_df["clonotype"].nunique(),
        )

    return (
        BindingDataset(train_df.reset_index(drop=True), tokenizer, max_length),
        BindingDataset(val_df.reset_index(drop=True), tokenizer, max_length),
        BindingDataset(test_df.reset_index(drop=True), tokenizer, max_length),
    )


def compute_class_weights(dataset: BindingDataset) -> list[float]:
    """Compute per-class weights (inverse frequency) for CrossEntropyLoss."""
    n_pos = sum(1 for lab in dataset.labels if lab == 1)
    n_neg = sum(1 for lab in dataset.labels if lab == 0)
    total = n_pos + n_neg
    w_neg = total / (2.0 * max(n_neg, 1))
    w_pos = total / (2.0 * max(n_pos, 1))
    logger.info("Binding class weights: neg=%.3f, pos=%.3f", w_neg, w_pos)
    return [w_neg, w_pos]
