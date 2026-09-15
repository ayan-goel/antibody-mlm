"""Tests for seed-replicate aggregation.

Covers name parsing, marginal mean/sd, and the paired-contrast arithmetic
that the rebuttal numbers will be quoted from.
"""

from __future__ import annotations

import csv

import pytest

from scripts.aggregate_seeds import (
    load_rows,
    split_experiment_name,
    summarize,
    write_contrasts,
    write_marginals,
)

_METRIC = "ds_paratope_auprc_mean"


def _write_table(path, rows: list[dict]) -> None:
    cols = ["experiment", "strategy", *sorted({k for r in rows for k in r} - {"experiment", "strategy"})]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({"strategy": "", **r})


@pytest.fixture()
def table(tmp_path):
    """uniform at 0.80/0.82/0.84, interface uniformly +0.05 on every seed."""
    path = tmp_path / "comparison_table.csv"
    _write_table(path, [
        {"experiment": "uniform_medium",      _METRIC: 0.80},
        {"experiment": "uniform_medium_s1",   _METRIC: 0.82},
        {"experiment": "uniform_medium_s2",   _METRIC: 0.84},
        {"experiment": "interface_medium",    _METRIC: 0.85},
        {"experiment": "interface_medium_s1", _METRIC: 0.87},
        {"experiment": "interface_medium_s2", _METRIC: 0.89},
        {"experiment": "cdr_medium",          _METRIC: 0.79},  # unreplicated
    ])
    return path


# ---------------------------------------------------------------------------
# Name parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("name", "expected"), [
    ("interface_medium", ("interface_medium", 42)),
    ("interface_medium_s1", ("interface_medium", 1)),
    ("interface_medium_s2", ("interface_medium", 2)),
    ("structure_longrange_medium_s13", ("structure_longrange_medium", 13)),
    ("untrained_medium", ("untrained_medium", 42)),
])
def test_split_experiment_name(name: str, expected: tuple[str, int]) -> None:
    assert split_experiment_name(name) == expected


def test_bare_name_is_seed_42_not_a_replicate() -> None:
    """Seed 42 keeps its original directory name; it must not be dropped."""
    strategy, seed = split_experiment_name("span_medium")
    assert (strategy, seed) == ("span_medium", 42)


# ---------------------------------------------------------------------------
# Grouping and marginals
# ---------------------------------------------------------------------------

def test_load_rows_groups_replicates(table) -> None:
    grouped = load_rows(table)
    assert set(grouped) == {"uniform_medium", "interface_medium", "cdr_medium"}
    assert sorted(grouped["uniform_medium"]) == [1, 2, 42]
    assert sorted(grouped["cdr_medium"]) == [42]


def test_summarize_reports_sample_sd() -> None:
    mean, sd, n = summarize([0.80, 0.82, 0.84])
    assert mean == pytest.approx(0.82)
    assert sd == pytest.approx(0.02)  # sample sd, ddof=1
    assert n == 3


def test_summarize_single_value_has_no_sd() -> None:
    assert summarize([0.79]) == (0.79, None, 1)


def test_marginals_csv(tmp_path, table) -> None:
    out = tmp_path / "agg.csv"
    write_marginals(load_rows(table), out, [_METRIC])
    rows = {r["strategy"]: r for r in csv.DictReader(out.open())}

    assert float(rows["uniform_medium"]["mean"]) == pytest.approx(0.82)
    assert float(rows["uniform_medium"]["sd"]) == pytest.approx(0.02)
    assert rows["uniform_medium"]["n"] == "3"
    # Unreplicated strategies survive with a blank sd rather than being dropped.
    assert rows["cdr_medium"]["sd"] == ""
    assert rows["cdr_medium"]["n"] == "1"


# ---------------------------------------------------------------------------
# Paired contrasts — the numbers that get quoted
# ---------------------------------------------------------------------------

def test_paired_contrast_is_tighter_than_marginals(tmp_path, table) -> None:
    """interface is +0.05 on every seed: paired sd is 0, marginal sd is not.

    This is the whole reason to pair — the marginals both carry the same
    seed-to-seed drift, which cancels within seed.
    """
    out = tmp_path / "contrasts.csv"
    write_contrasts(load_rows(table), "uniform_medium", out, [_METRIC])
    rows = {r["strategy"]: r for r in csv.DictReader(out.open())}

    interface = rows["interface_medium"]
    assert float(interface["mean_delta"]) == pytest.approx(0.05)
    assert float(interface["sd_delta"]) == pytest.approx(0.0, abs=1e-12)
    assert interface["n_paired"] == "3"
    assert interface["per_seed_deltas"] == "+0.0500;+0.0500;+0.0500"


def test_contrast_pairs_only_shared_seeds(tmp_path, table) -> None:
    """cdr has only seed 42, so it contributes exactly one paired delta."""
    out = tmp_path / "contrasts.csv"
    write_contrasts(load_rows(table), "uniform_medium", out, [_METRIC])
    cdr = {r["strategy"]: r for r in csv.DictReader(out.open())}["cdr_medium"]

    assert cdr["n_paired"] == "1"
    assert float(cdr["mean_delta"]) == pytest.approx(-0.01)  # 0.79 - 0.80
    assert cdr["sd_delta"] == ""
    assert cdr["effect_size_sd_units"] == ""


def test_reference_strategy_excluded_from_contrasts(tmp_path, table) -> None:
    out = tmp_path / "contrasts.csv"
    write_contrasts(load_rows(table), "uniform_medium", out, [_METRIC])
    assert "uniform_medium" not in {r["strategy"] for r in csv.DictReader(out.open())}


def test_unknown_reference_raises(tmp_path, table) -> None:
    with pytest.raises(KeyError):
        write_contrasts(load_rows(table), "nope_medium", tmp_path / "c.csv", [_METRIC])


def test_effect_size_is_delta_over_paired_sd(tmp_path) -> None:
    path = tmp_path / "t.csv"
    _write_table(path, [
        {"experiment": "uniform_medium",    _METRIC: 0.80},
        {"experiment": "uniform_medium_s1", _METRIC: 0.80},
        {"experiment": "interface_medium",    _METRIC: 0.84},  # +0.04
        {"experiment": "interface_medium_s1", _METRIC: 0.82},  # +0.02
    ])
    out = tmp_path / "c.csv"
    write_contrasts(load_rows(path), "uniform_medium", out, [_METRIC])
    row = {r["strategy"]: r for r in csv.DictReader(out.open())}["interface_medium"]

    # deltas +0.04, +0.02 -> mean 0.03, sample sd 0.01414 -> 2.12 sd units
    assert float(row["mean_delta"]) == pytest.approx(0.03)
    assert float(row["effect_size_sd_units"]) == pytest.approx(2.12, abs=0.01)
