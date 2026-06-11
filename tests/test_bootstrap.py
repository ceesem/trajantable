"""Tests for cell-bootstrap CI.

Pins the contract of ``bootstrap_over_cells`` and ``cell_bootstrap_iter``:

- shape: bin keys + p (point) + p_lo / p_hi (percentile CI)
- bounds ordered (p_lo ≤ p_hi) and bracketed in [0, 1] for percentile of [0,1] values
- reproducibility under fixed seed
- different seeds → different (Monte-Carlo) results
- agreement with an in-test reference implementation for a tiny universe
- validation: bad alpha / n_resamples / missing keys raise
"""

import numpy as np
import polars as pl
import pytest

from trajan import (
    SynapseTable,
    bootstrap_over_cells,
    cell_bootstrap_iter,
    possible_pairs,
    with_distance,
)
from trajan.spatial import pack_position, radial_distance

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def pu():
    syn = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "pre_pt_root_id": [10, 10, 20, 20, 30, 30, 40, 40],
            "post_pt_root_id": [20, 30, 10, 40, 10, 20, 10, 30],
        }
    )
    cells = pl.DataFrame(
        {
            "root_id": [10, 20, 30, 40, 50, 60],
            "cell_type": ["L23", "L23", "L4", "L4", "L5", "L5"],
            "soma_x": [0.0, 50.0, 200.0, 250.0, 500.0, 550.0],
            "soma_y": [0.0] * 6,
            "soma_z": [0.0] * 6,
        }
    )
    packed = pack_position(cells, "soma", x="soma_x", y="soma_y", z="soma_z")
    st = SynapseTable(syn)
    st.add_cell_annotation(
        "cells",
        packed,
        cell_id_col="root_id",
        is_universe=True,
        position_col="soma",
    )
    st = with_distance(st, "d_radial", radial_distance)
    return possible_pairs(st)


# ── shape and bounds ─────────────────────────────────────────────────────────


def test_returns_point_plus_ci(pu):
    df = bootstrap_over_cells(pu, group_by="cell_type_pre", n_resamples=50, seed=0)
    for col in ("cell_type_pre", "k_observed", "n_possible", "p", "p_lo", "p_hi"):
        assert col in df.columns


def test_bounds_ordered_and_in_unit_interval(pu):
    df = bootstrap_over_cells(
        pu,
        bin_by={"d_radial": [0, 100, 300, 1000]},
        n_resamples=200,
        seed=42,
    )
    assert (df["p_lo"] <= df["p_hi"]).all()
    # p is bounded in [0, 1] (it's a proportion), and so are its quantiles.
    assert (df["p_lo"] >= 0.0).all()
    assert (df["p_hi"] <= 1.0).all()


def test_bin_by_and_group_by_compose(pu):
    df = bootstrap_over_cells(
        pu,
        bin_by={"d_radial": [0, 200]},
        group_by="cell_type_post",
        n_resamples=50,
        seed=1,
    )
    assert {"cell_type_post", "d_radial_bin", "p_lo", "p_hi"} <= set(df.columns)


def test_point_estimate_matches_connection_probability(pu):
    """The `p` column equals the observed MLE; the bootstrap only adds
    p_lo / p_hi. (Library convention: estimator never overrides the MLE.)"""
    from trajan import connection_probability

    cp = connection_probability(pu, group_by="cell_type_pre").sort("cell_type_pre")
    bo = bootstrap_over_cells(
        pu, group_by="cell_type_pre", n_resamples=10, seed=0
    ).sort("cell_type_pre")
    assert cp["p"].to_list() == bo["p"].to_list()


# ── reproducibility ─────────────────────────────────────────────────────────


def test_seed_reproducibility(pu):
    a = bootstrap_over_cells(
        pu, group_by="cell_type_pre", n_resamples=100, seed=7
    ).sort("cell_type_pre")
    b = bootstrap_over_cells(
        pu, group_by="cell_type_pre", n_resamples=100, seed=7
    ).sort("cell_type_pre")
    assert a["p_lo"].to_list() == b["p_lo"].to_list()
    assert a["p_hi"].to_list() == b["p_hi"].to_list()


def test_different_seeds_produce_different_cis(pu):
    a = bootstrap_over_cells(
        pu, group_by="cell_type_pre", n_resamples=100, seed=1
    ).sort("cell_type_pre")
    b = bootstrap_over_cells(
        pu, group_by="cell_type_pre", n_resamples=100, seed=2
    ).sort("cell_type_pre")
    # Some bound must differ across seeds (Monte-Carlo error). All-equal on
    # BOTH bounds would mean the seeds collapsed somehow. Don't assert on
    # p_lo alone: sparse bins with mostly 0/0 connection rates can legitimately
    # have p_lo = 0 across many seeds (the lower quantile floors out).
    bounds_a = (a["p_lo"].to_list(), a["p_hi"].to_list())
    bounds_b = (b["p_lo"].to_list(), b["p_hi"].to_list())
    assert bounds_a != bounds_b


# ── cell_bootstrap_iter ──────────────────────────────────────────────────────


def test_iter_yields_n_resamples(pu):
    frames = list(
        cell_bootstrap_iter(pu, group_by="cell_type_pre", n_resamples=7, seed=0)
    )
    assert len(frames) == 7
    for df in frames:
        # Per-resample columns: keys + k_observed, n_possible, p
        # (names match counts() so estimators compose on either output)
        assert {"cell_type_pre", "k_observed", "n_possible", "p"} <= set(df.columns)


def test_iter_reproducible_with_seed(pu):
    a = list(cell_bootstrap_iter(pu, group_by="cell_type_pre", n_resamples=3, seed=99))
    b = list(cell_bootstrap_iter(pu, group_by="cell_type_pre", n_resamples=3, seed=99))
    for x, y in zip(a, b):
        assert x.sort("cell_type_pre").equals(y.sort("cell_type_pre"))


# ── reference-implementation cross-check ─────────────────────────────────────


def test_matches_reference_implementation(pu):
    """Run a slow Python reference implementation alongside the polars
    version and confirm they produce identical CIs on a tiny fixture.

    Reference algorithm: for each resample r, draw N multiplicities;
    for each pair (a, b) in the universe, contribute m_a * m_b to
    n_possible (always) and to k_observed (if n_syn > 0); aggregate per
    bin; collect p across resamples; take percentile.
    """
    # Use a coarse grouping (one bin) to keep the reference cheap and
    # numerically stable.
    n_resamples = 50
    seed = 123
    alpha = 0.1

    # Library output.
    lib = bootstrap_over_cells(
        pu,
        group_by="cell_type_pre",
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    ).sort("cell_type_pre")

    # Reference: pull universe cells and pair data eagerly.
    pair_df = (
        pu.build_lazy()
        .select(["pre_pt_root_id", "post_pt_root_id", "n_syn", "cell_type_pre"])
        .collect()
    )
    universe = sorted(
        set(pair_df["pre_pt_root_id"].to_list())
        | set(pair_df["post_pt_root_id"].to_list())
    )
    # The library uses the universe annotation directly, which includes
    # cells with zero connections (here cells 50 and 60 from the fixture).
    universe = [10, 20, 30, 40, 50, 60]
    N = len(universe)
    idx_of = {c: i for i, c in enumerate(universe)}

    rows = pair_df.to_dicts()
    rng = np.random.default_rng(seed)
    p_by_type: dict[str, list[float]] = {}
    for _ in range(n_resamples):
        m = rng.multinomial(N, [1.0 / N] * N)
        bin_counts: dict[str, dict[str, float]] = {}  # cell_type → {k, n}
        for row in rows:
            a = row["pre_pt_root_id"]
            b = row["post_pt_root_id"]
            w = float(m[idx_of[a]] * m[idx_of[b]])
            if w == 0:
                continue
            ct = row["cell_type_pre"]
            slot = bin_counts.setdefault(ct, {"k": 0.0, "n": 0.0})
            slot["n"] += w
            if row["n_syn"] > 0:
                slot["k"] += w
        for ct, kn in bin_counts.items():
            p_by_type.setdefault(ct, []).append(kn["k"] / kn["n"])

    # Compare per type — must agree on every common bin.
    for row in lib.iter_rows(named=True):
        ct = row["cell_type_pre"]
        ref = np.quantile(p_by_type[ct], [alpha / 2, 1 - alpha / 2])
        assert abs(row["p_lo"] - ref[0]) < 1e-9
        assert abs(row["p_hi"] - ref[1]) < 1e-9


# ── table inputs (normalized via possible_pairs) ─────────────────────────────


def test_bootstrap_over_cells_accepts_table():
    """bootstrap_over_cells accepts a SynapseTable directly (normalized via
    possible_pairs), matching the pre-built PairUniverse path under a fixed seed."""
    syn = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "pre_pt_root_id": [10, 10, 20, 20, 30, 30, 40, 40],
            "post_pt_root_id": [20, 30, 10, 40, 10, 20, 10, 30],
        }
    )
    cells = pl.DataFrame(
        {
            "root_id": [10, 20, 30, 40, 50, 60],
            "cell_type": ["L23", "L23", "L4", "L4", "L5", "L5"],
        }
    )
    st = SynapseTable(syn)
    st.add_cell_annotation("cells", cells, cell_id_col="root_id", is_universe=True)

    from_table = bootstrap_over_cells(
        st, group_by="cell_type_pre", n_resamples=50, seed=5
    ).sort("cell_type_pre")
    from_pu = bootstrap_over_cells(
        possible_pairs(st), group_by="cell_type_pre", n_resamples=50, seed=5
    ).sort("cell_type_pre")
    assert from_table["p_lo"].to_list() == from_pu["p_lo"].to_list()
    assert from_table["p_hi"].to_list() == from_pu["p_hi"].to_list()


def test_cell_bootstrap_iter_accepts_table():
    """cell_bootstrap_iter also normalizes a table input to a PairUniverse."""
    syn = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "pre_pt_root_id": [10, 10, 20],
            "post_pt_root_id": [20, 30, 10],
        }
    )
    cells = pl.DataFrame({"root_id": [10, 20, 30], "cell_type": ["A", "A", "B"]})
    st = SynapseTable(syn)
    st.add_cell_annotation("cells", cells, cell_id_col="root_id", is_universe=True)
    frames = list(
        cell_bootstrap_iter(st, group_by="cell_type_pre", n_resamples=4, seed=0)
    )
    assert len(frames) == 4


# ── validation ──────────────────────────────────────────────────────────────


def test_invalid_alpha_raises(pu):
    with pytest.raises(ValueError, match="alpha must be in"):
        bootstrap_over_cells(pu, group_by="cell_type_pre", n_resamples=10, alpha=0.0)


def test_invalid_n_resamples_raises(pu):
    with pytest.raises(ValueError, match="n_resamples must be >= 1"):
        bootstrap_over_cells(pu, group_by="cell_type_pre", n_resamples=0)


def test_no_keys_raises(pu):
    with pytest.raises(ValueError, match="at least one bin_by or group_by"):
        bootstrap_over_cells(pu, n_resamples=10)
