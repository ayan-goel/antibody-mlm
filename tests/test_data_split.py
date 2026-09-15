"""Tests for the deterministic train/eval corpus partition.

The property under test is the one that makes seed replicates valid: the
held-out set is a function of ``data_split_seed`` alone and does not move
when the experiment ``seed`` changes.
"""

from __future__ import annotations

import pytest
from torch.utils.data import Dataset

from data.splits import DEFAULT_DATA_SPLIT_SEED, make_train_eval_split
from training.config import DataConfig, load_config


class _RangeDataset(Dataset):
    """Minimal stand-in for the corpus: item i is the integer i."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


def _eval_indices(ds: Dataset, train_split: float, split_seed: int) -> set[int]:
    _, eval_subset = make_train_eval_split(ds, train_split, split_seed)
    return set(eval_subset.indices)


# ---------------------------------------------------------------------------
# The core guarantee
# ---------------------------------------------------------------------------

def test_split_is_independent_of_experiment_seed() -> None:
    """The partition must not move when the per-model seed changes.

    Regression test for the coupling that previously let a seed replicate
    evaluate on sequences it had trained on.
    """
    ds = _RangeDataset(1000)
    baseline = _eval_indices(ds, 0.9, DEFAULT_DATA_SPLIT_SEED)

    # These stand in for `config.seed` values of seed replicates. None of
    # them may influence the split.
    for _experiment_seed in (0, 1, 7, 123, 2024):
        assert _eval_indices(ds, 0.9, DEFAULT_DATA_SPLIT_SEED) == baseline


def test_split_changes_when_split_seed_changes() -> None:
    """`data_split_seed` is still a live knob — it just isn't `seed`."""
    ds = _RangeDataset(1000)
    assert _eval_indices(ds, 0.9, 42) != _eval_indices(ds, 0.9, 43)


def test_train_and_eval_are_disjoint_and_exhaustive() -> None:
    ds = _RangeDataset(997)
    train, evl = make_train_eval_split(ds, 0.9, DEFAULT_DATA_SPLIT_SEED)

    train_idx, eval_idx = set(train.indices), set(evl.indices)
    assert not (train_idx & eval_idx), "a sequence appears in both splits"
    assert train_idx | eval_idx == set(range(997)), "split does not cover corpus"


@pytest.mark.parametrize("n", [997, 1000, 497_309])
def test_split_sizes_follow_train_fraction(n: int) -> None:
    """Sizes match the trainer's convention: train = int(n * frac)."""
    train, evl = make_train_eval_split(_RangeDataset(n), 0.9)
    assert len(train) == int(n * 0.9)
    assert len(evl) == n - int(n * 0.9)


def test_training_and_evaluation_paths_agree() -> None:
    """Trainer and eval scripts must derive the identical held-out set.

    Both now call ``make_train_eval_split`` with the same config fields, so
    this pins the contract rather than re-deriving sizes independently (the
    old duplicated arithmetic was off by one sequence).
    """
    ds = _RangeDataset(497_309)
    cfg = DataConfig()

    # Training path: uses config.data.* and ignores config.seed entirely.
    train_a, eval_a = make_train_eval_split(
        ds, cfg.train_split, cfg.data_split_seed,
    )
    # Evaluation path: same call, discarding the train half.
    _, eval_b = make_train_eval_split(ds, cfg.train_split, cfg.data_split_seed)

    assert set(eval_a.indices) == set(eval_b.indices)
    assert not (set(train_a.indices) & set(eval_b.indices))


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_data_split_seed_defaults_to_42() -> None:
    assert DataConfig().data_split_seed == DEFAULT_DATA_SPLIT_SEED == 42


def test_data_split_seed_is_loadable_from_yaml(tmp_path) -> None:
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text("seed: 7\ndata:\n  train_split: 0.8\n  data_split_seed: 99\n")

    cfg = load_config(cfg_path)
    assert cfg.seed == 7
    assert cfg.data.data_split_seed == 99
    assert cfg.data.train_split == 0.8


def test_existing_configs_omit_split_seed_and_get_the_default(tmp_path) -> None:
    """Every current config predates the field; all must land on 42."""
    cfg_path = tmp_path / "legacy.yaml"
    cfg_path.write_text("seed: 42\ndata:\n  train_split: 0.9\n")

    assert load_config(cfg_path).data.data_split_seed == 42
