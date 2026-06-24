"""Tests for the binning / counts primitives and connection_probability.

These pin the contract that downstream spatial-stats / null-model code
will rely on:

- counts() requires at least one bin_by or group_by
- 1D and 2D binning compose naturally with categorical group_by
- n_possible covers ALL pairs in the bin (including n_syn == 0)
- k_observed only counts pairs with n_syn > 0
- registered weights are summed as sum_<weight>
- connection_probability adds p = k/n and optionally estimator columns
"""

import polars as pl
import pytest

from trajan import (
    PairUniverse,
    SynapseTable,
    agresti_coull_ci,
    connection_density,
    connection_probability,
    counts,
    possible_pairs,
    wilson_ci,
    with_distance,
)
from trajan.spatial import euclidean_distance, pack_position, radial_distance

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def universe_cells():
    return pl.DataFrame(
        {
            "root_id": [10, 20, 30, 40, 50],
            "cell_type": ["L23", "L23", "L4", "L4", "L5"],
            "soma_x": [0.0, 10.0, 100.0, 200.0, 500.0],
            "soma_y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "soma_z": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


@pytest.fixture
def synapses():
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "pre_pt_root_id": [10, 10, 20, 20, 30, 30],
            "post_pt_root_id": [20, 30, 10, 30, 10, 20],
            "size": [100, 50, 200, 150, 80, 60],
        }
    )


@pytest.fixture
def pu(synapses, universe_cells):
    packed = pack_position(universe_cells, "soma", x="soma_x", y="soma_y", z="soma_z")
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", packed, cell_id_col="root_id", is_universe=True, position_col="soma"
    )
    st.add_weight("size")
    st.add_spatial_features(prefix="d")  # adds d_euclidean, d_rho, d_phi, ...
    return possible_pairs(st)


# ── counts: contract ─────────────────────────────────────────────────────────


def test_counts_requires_keys(pu):
    """Without bin_by or group_by, counts() would collapse to one row —
    refuse and tell the user to use pu.build_lazy().select(...) instead."""
    with pytest.raises(ValueError, match="at least one bin_by or group_by"):
        counts(pu)


def test_counts_categorical_group_by(pu):
    """group_by on cell-type columns produces one row per type pair that has
    at least one possible pair. (Type pairs with n_possible == 0 don't appear
    — the L5×L5 case here is absent because there's only one L5 cell and
    self-pairs are excluded.)"""
    df = counts(pu, group_by=["cell_type_pre", "cell_type_post"])
    # 3 types, 9 cross-product cells, minus L5×L5 (one L5 cell + no autapses).
    assert len(df) == 8
    # k_observed sums to total observed pairs (6 synapses → 6 distinct
    # ordered pairs in this fixture).
    assert df["k_observed"].sum() == 6
    # n_possible sums to total possible pairs (5²−5 = 20).
    assert df["n_possible"].sum() == 20


def test_counts_1d_binning(pu):
    """A single continuous bin_by spec produces a {name}_bin column."""
    df = counts(pu, bin_by={"d_rho": [0, 50, 200, 600]})
    assert "d_rho_bin" in df.columns
    assert df["n_possible"].sum() == 20  # denominator is preserved


def test_counts_joint_2d_binning(pu):
    """Joint binning groups by the cross product of bin columns. Here
    lateral distance (d_rho) and signed depth (d_depth_diff) are the
    cortical-decomposition pair flagged in
    ``project_connection_probability.md`` as the case that scalar-distance
    binning collapses out."""
    df = counts(pu, bin_by={"d_rho": [0, 50, 200], "d_depth_diff": [-10, 0, 10]})
    assert "d_rho_bin" in df.columns
    assert "d_depth_diff_bin" in df.columns
    # Denominator preserved regardless of grid shape.
    assert df["n_possible"].sum() == 20


def test_counts_mixed_continuous_and_categorical(pu):
    """Mixing a continuous bin with a categorical pass-through (None) works."""
    df = counts(pu, bin_by={"d_rho": [0, 50, 200], "cell_type_post": None})
    assert "d_rho_bin" in df.columns
    assert "cell_type_post" in df.columns  # pass-through keeps the raw name
    assert df["n_possible"].sum() == 20


def test_counts_group_by_and_bin_by_compose(pu):
    """group_by extends keys alongside bin_by; both contribute to grouping."""
    df = counts(pu, bin_by={"d_rho": [0, 200]}, group_by="cell_type_pre")
    assert "cell_type_pre" in df.columns
    assert "d_rho_bin" in df.columns


def test_counts_registered_weights_summed(pu):
    """Each registered weight on the PairUniverse appears as sum_<weight>."""
    df = counts(pu, group_by=["cell_type_pre"])
    assert "sum_n_syn" in df.columns
    assert "sum_size" in df.columns
    # sum_n_syn across all bins == total observed synapses
    assert df["sum_n_syn"].sum() == 6
    # sum_size across all bins == total size across observed synapses
    assert df["sum_size"].sum() == 100 + 50 + 200 + 150 + 80 + 60


def test_counts_k_and_n_distinguish_observed_from_possible(pu):
    """A pair with n_syn == 0 contributes to n_possible but not k_observed."""
    df = counts(pu, group_by=["cell_type_pre", "cell_type_post"])
    # At least one bin must have n_possible > k_observed (unobserved pairs).
    assert (df["n_possible"] > df["k_observed"]).any()


# ── input normalization: counts is strict, estimators are forgiving ──────────


def _universe_st(synapses, universe_cells):
    """SynapseTable with a universe annotation (positions not needed here)."""
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", universe_cells, cell_id_col="root_id", is_universe=True
    )
    return st


def test_counts_rejects_edgelist(synapses, universe_cells):
    """counts() refuses observed-only data — an EdgeList would make every
    bin's denominator the observed count (p ~ 1) silently. Guard it."""
    el = _universe_st(synapses, universe_cells).edgelist()
    with pytest.raises(TypeError, match="requires a PairUniverse"):
        counts(el, group_by=["cell_type_pre", "cell_type_post"])


def test_counts_rejects_pair_universe_without_n_syn():
    """A hand-built PairUniverse missing the n_syn observed-predicate fails
    with a clear message rather than a raw ColumnNotFoundError."""
    cross = pl.DataFrame({"pre": [1, 2], "post": [2, 1], "w": [1, 0]})
    pu = PairUniverse(cross.lazy(), pre_col="pre", post_col="post", weights=["w"])
    with pytest.raises(ValueError, match="n_syn"):
        counts(pu, group_by="pre")


def test_connection_probability_accepts_synapse_table(synapses, universe_cells):
    """A SynapseTable is normalized to its possible-pairs denominator, so the
    result matches the explicit possible_pairs() path exactly (zeros and all)."""
    st = _universe_st(synapses, universe_cells)
    from_table = connection_probability(
        st, group_by=["cell_type_pre", "cell_type_post"]
    ).sort(["cell_type_pre", "cell_type_post"])
    from_pu = connection_probability(
        possible_pairs(st), group_by=["cell_type_pre", "cell_type_post"]
    ).sort(["cell_type_pre", "cell_type_post"])
    assert from_table.equals(from_pu)
    # Unobserved type-pairs are present with p == 0 — not collapsed away.
    assert (from_table["p"] == 0).any()


def test_connection_probability_accepts_edgelist(synapses, universe_cells):
    """An EdgeList is also normalized via possible_pairs — the denominator
    spans the full universe (5²−5 = 20 ordered pairs), not the observed subset."""
    el = _universe_st(synapses, universe_cells).edgelist()
    df = connection_probability(el, group_by=["cell_type_pre", "cell_type_post"])
    assert df["n_possible"].sum() == 20


def test_connection_density_is_probability_twin(synapses, universe_cells):
    """connection_density is a name-only twin: identical output to
    connection_probability for the same inputs."""
    st = _universe_st(synapses, universe_cells)
    a = connection_density(st, group_by=["cell_type_pre"]).sort("cell_type_pre")
    b = connection_probability(st, group_by=["cell_type_pre"]).sort("cell_type_pre")
    assert a.equals(b)


def test_connection_probability_include_self(synapses, universe_cells):
    """include_self adds the n diagonal pairs to the denominator."""
    st = _universe_st(synapses, universe_cells)
    without = connection_probability(st, group_by=["cell_type_pre"])
    with_self = connection_probability(
        st, group_by=["cell_type_pre"], include_self=True
    )
    assert without["n_possible"].sum() == 20  # 5² − 5
    assert with_self["n_possible"].sum() == 25  # 5²


# ── connection_probability ───────────────────────────────────────────────────


def test_connection_probability_adds_p(pu):
    """p column = k_observed / n_possible per bin."""
    df = connection_probability(pu, group_by=["cell_type_pre"])
    assert "p" in df.columns
    # Per-row sanity check
    for row in df.iter_rows(named=True):
        assert abs(row["p"] - row["k_observed"] / row["n_possible"]) < 1e-9


def test_connection_probability_no_estimator_no_ci_columns(pu):
    """estimator=None → no extra columns beyond counts() + p."""
    df = connection_probability(pu, group_by=["cell_type_pre"])
    for col in ("p_lo", "p_hi"):
        assert col not in df.columns


def test_connection_probability_user_estimator(pu):
    """User-supplied estimator (any callable matching the protocol) is
    invoked and its columns appear on the output."""

    def trivial_band(k: pl.Expr, n: pl.Expr) -> dict[str, pl.Expr]:
        # Toy estimator: ±1/sqrt(n) symmetric band, clipped to [0,1].
        p = k / n
        margin = 1.0 / n.sqrt()
        return {
            "p_lo": (p - margin).clip(lower_bound=0.0),
            "p_hi": (p + margin).clip(upper_bound=1.0),
        }

    df = connection_probability(pu, group_by=["cell_type_pre"], estimator=trivial_band)
    assert "p_lo" in df.columns
    assert "p_hi" in df.columns
    # bounds bracket the point estimate
    assert (df["p_lo"] <= df["p"]).all()
    assert (df["p"] <= df["p_hi"]).all()


def test_connection_probability_with_joint_binning(pu):
    """Joint binning composes with the p estimate."""
    df = connection_probability(
        pu, bin_by={"d_rho": [0, 50, 200, 600], "cell_type_post": None}
    )
    assert {"d_rho_bin", "cell_type_post", "p"} <= set(df.columns)


# ── invalid spec ─────────────────────────────────────────────────────────────


def test_bin_spec_rejects_unknown_type(pu):
    with pytest.raises(TypeError, match="must be a list of edges or None"):
        counts(pu, bin_by={"d_rho": "quantile:10"})  # not yet supported


# ── CI estimators: Wilson ────────────────────────────────────────────────────


def test_wilson_ci_returns_p_lo_and_p_hi(pu):
    """Wilson estimator plugs into connection_probability and adds p_lo/p_hi
    while leaving p = k/n (the MLE)."""
    df = connection_probability(
        pu, group_by=["cell_type_pre"], estimator=wilson_ci(alpha=0.05)
    )
    assert "p" in df.columns
    assert "p_lo" in df.columns
    assert "p_hi" in df.columns
    # connection_probability is responsible for p; estimator only adds the
    # interval bounds. So p stays the raw MLE regardless of estimator.
    for row in df.iter_rows(named=True):
        assert abs(row["p"] - row["k_observed"] / row["n_possible"]) < 1e-12


def test_wilson_ci_brackets_mle(pu):
    """The Wilson interval always contains the MLE p̂ = k/n. (It's an
    asymmetric interval around a shrunken center, but it provably contains
    the MLE for any 0 ≤ k ≤ n.)"""
    df = connection_probability(
        pu, group_by=["cell_type_pre", "cell_type_post"], estimator=wilson_ci()
    )
    # Allow tiny float slack.
    assert ((df["p_lo"] - df["p"]) <= 1e-12).all()
    assert ((df["p"] - df["p_hi"]) <= 1e-12).all()


def test_wilson_ci_bounded_at_extremes(pu):
    """At k=0 the Wilson lower bound is exactly 0; at k=n the upper bound
    is exactly 1. (This is the Wilson formula's natural behavior and one of
    the reasons it's preferred over normal approximation.)"""
    df = connection_probability(
        pu, group_by=["cell_type_pre", "cell_type_post"], estimator=wilson_ci()
    )
    for row in df.iter_rows(named=True):
        if row["k_observed"] == 0:
            assert abs(row["p_lo"]) < 1e-12
        if row["k_observed"] == row["n_possible"]:
            assert abs(row["p_hi"] - 1.0) < 1e-12


def test_wilson_ci_smaller_alpha_widens_interval(pu):
    """A smaller alpha → wider confidence interval (more conservative)."""
    df95 = connection_probability(
        pu, group_by="cell_type_pre", estimator=wilson_ci(alpha=0.05)
    ).sort("cell_type_pre")
    df99 = connection_probability(
        pu, group_by="cell_type_pre", estimator=wilson_ci(alpha=0.01)
    ).sort("cell_type_pre")
    widths_95 = (df95["p_hi"] - df95["p_lo"]).to_list()
    widths_99 = (df99["p_hi"] - df99["p_lo"]).to_list()
    for w95, w99 in zip(widths_95, widths_99):
        assert w99 >= w95


def test_wilson_ci_matches_known_value():
    """Spot-check against a hand-computed Wilson value. For k=5, n=10,
    α=0.05: center=0.5, half-width≈0.2629, so interval ≈ [0.2366, 0.7634].
    Computed against R's binom.test / scipy.stats.binomtest('wilson')."""
    # Apply the estimator to a literal-counts frame directly so the test
    # doesn't depend on PairUniverse construction.
    import polars as pl

    df = pl.DataFrame({"k": [5], "n": [10]})
    est = wilson_ci(alpha=0.05)
    cols = est(pl.col("k"), pl.col("n"))
    out = df.with_columns(*[expr.alias(name) for name, expr in cols.items()])
    assert abs(out["p_lo"].item() - 0.2366) < 1e-3
    assert abs(out["p_hi"].item() - 0.7634) < 1e-3


# ── CI estimators: Agresti-Coull ─────────────────────────────────────────────


def test_agresti_coull_ci_returns_interval(pu):
    df = connection_probability(
        pu, group_by="cell_type_pre", estimator=agresti_coull_ci(alpha=0.05)
    )
    assert "p_lo" in df.columns
    assert "p_hi" in df.columns
    # Bounds are ordered.
    assert (df["p_lo"] <= df["p_hi"]).all()


def test_agresti_coull_unclipped_at_extremes():
    """Agresti-Coull is not clipped to [0, 1] — at k=0 the lower bound can
    go slightly negative. Documented behavior; the library exposes the raw
    formula and leaves clipping to the caller. Pin the contract so it
    can't drift accidentally."""
    import polars as pl

    df = pl.DataFrame({"k": [0], "n": [10]})
    est = agresti_coull_ci(alpha=0.05)
    cols = est(pl.col("k"), pl.col("n"))
    out = df.with_columns(*[expr.alias(name) for name, expr in cols.items()])
    # k=0, n=10, α=0.05 → p̃≈0.139, half-width≈0.183 → lo ≈ -0.044
    assert out["p_lo"].item() < 0


def test_agresti_coull_wider_than_wilson(pu):
    """Agresti-Coull is generally wider than Wilson — pin the well-known
    relationship as a sanity check. (Strict inequality holds for n > z²
    away from the extremes; using <= for slack at the boundaries.)"""
    df_w = connection_probability(
        pu, group_by="cell_type_pre", estimator=wilson_ci()
    ).sort("cell_type_pre")
    df_a = connection_probability(
        pu, group_by="cell_type_pre", estimator=agresti_coull_ci()
    ).sort("cell_type_pre")
    width_w = (df_w["p_hi"] - df_w["p_lo"]).to_list()
    width_a = (df_a["p_hi"] - df_a["p_lo"]).to_list()
    for ww, wa in zip(width_w, width_a):
        assert wa >= ww - 1e-9


# ── CI estimators: validation ────────────────────────────────────────────────


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be in"):
        wilson_ci(alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        agresti_coull_ci(alpha=1.5)


# ── with_distance helper ─────────────────────────────────────────────────────


def test_with_distance_on_synapse_table(synapses, universe_cells):
    """with_distance on a SynapseTable registers a cell-side expression that
    propagates through possible_pairs() to the PairUniverse cross-product."""
    packed = pack_position(universe_cells, "soma", x="soma_x", y="soma_y", z="soma_z")
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", packed, cell_id_col="root_id", is_universe=True, position_col="soma"
    )
    st = with_distance(st, "d_eucl", euclidean_distance)
    # The expression is classified as "both" (references both _pre and _post
    # position columns) so it propagates to el.edgelist().
    assert "d_eucl" in st.expression_names
    assert st.expression_sides["d_eucl"] == "both"
    # And ultimately reaches the PairUniverse, where binning works directly.
    pu = possible_pairs(st)
    df = connection_probability(pu, bin_by={"d_eucl": [0, 100, 500]})
    assert "d_eucl_bin" in df.columns


def test_with_distance_on_pair_universe(synapses, universe_cells):
    """with_distance on a PairUniverse works without going through the
    SynapseTable — useful when you've already constructed the cross-product
    and want to add a distance feature without rebuilding."""
    packed = pack_position(universe_cells, "soma", x="soma_x", y="soma_y", z="soma_z")
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", packed, cell_id_col="root_id", is_universe=True, position_col="soma"
    )
    pu = possible_pairs(st)
    pu = with_distance(pu, "d_radial", radial_distance)
    df = connection_probability(pu, bin_by={"d_radial": [0, 100, 500]})
    assert "d_radial_bin" in df.columns
    assert df["n_possible"].sum() == 20  # cross-product still 5²−5


def test_with_distance_custom_callable(synapses, universe_cells):
    """A user-defined distance_fn (here, x-axis-only distance — the
    stand-in for a cortical-curvature-corrected metric) composes with
    binning exactly like the shipped ones."""

    def x_only_distance(a: str, b: str) -> pl.Expr:
        # Toy "corrected" distance: just the x-component magnitude.
        # In a real workflow this is where curvature unwarping would go.
        return (pl.col(a).struct.field("x") - pl.col(b).struct.field("x")).abs()

    packed = pack_position(universe_cells, "soma", x="soma_x", y="soma_y", z="soma_z")
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", packed, cell_id_col="root_id", is_universe=True, position_col="soma"
    )
    pu = with_distance(possible_pairs(st), "d_custom", x_only_distance)
    df = connection_probability(
        pu, bin_by={"d_custom": [0, 50, 200, 1000]}, estimator=wilson_ci()
    )
    assert "d_custom_bin" in df.columns
    assert "p_lo" in df.columns


def test_with_distance_ambiguous_annotation_raises(synapses):
    """When more than one position-bearing annotation is registered,
    with_distance forces an explicit pick — same error contract as the
    underlying _resolve_position_annotation."""
    base_a = pl.DataFrame(
        {
            "root_id": [10, 20, 30, 40, 50],
            "a_x": [0.0, 10.0, 100.0, 200.0, 500.0],
            "a_y": [0.0] * 5,
            "a_z": [0.0] * 5,
        }
    )
    base_b = pl.DataFrame(
        {
            "root_id": [10, 20, 30, 40, 50],
            "b_x": [0.0, 5.0, 50.0, 100.0, 250.0],
            "b_y": [0.0] * 5,
            "b_z": [0.0] * 5,
        }
    )
    packed_a = pack_position(base_a, "a", x="a_x", y="a_y", z="a_z")
    packed_b = pack_position(base_b, "b", x="b_x", y="b_y", z="b_z")
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "a", packed_a, cell_id_col="root_id", is_universe=True, position_col="a"
    )
    st.add_cell_annotation("b", packed_b, cell_id_col="root_id", position_col="b")
    with pytest.raises(ValueError, match="Multiple cell annotations"):
        with_distance(st, "d", euclidean_distance)
    # Explicit annotation= disambiguates.
    st_b = with_distance(st, "d_b", euclidean_distance, annotation="b")
    assert "d_b" in st_b.expression_names


# ── numeric bins sort numerically (ordered Enum) ─────────────────────────────


def test_numeric_bin_column_is_ordered_enum(pu):
    """Continuous bins are emitted as an ordered Enum so they sort by bin
    order, not lexically. d_rho here = |dx| (y=z=0)."""
    df = counts(pu, bin_by={"d_rho": [0, 50, 100, 200]})
    assert isinstance(df.schema["d_rho_bin"], pl.Enum)
    bins = df["d_rho_bin"].to_list()
    # counts() sorts by keys; with the Enum that is ascending bin order.
    # Lexical order would put "(100, 200]" before "(50, 100]".
    assert bins == ["(0, 50]", "(50, 100]", "(100, 200]", "(200, inf]"]


def test_numeric_bin_sorts_numerically_after_shuffle(pu):
    """Even after a shuffle, sorting on the bin column respects numeric order."""
    df = counts(pu, bin_by={"d_rho": [0, 50, 100, 200]})
    reshuffled = df.sort("k_observed")  # scramble row order
    assert reshuffled.sort("d_rho_bin")["d_rho_bin"].to_list() == [
        "(0, 50]",
        "(50, 100]",
        "(100, 200]",
        "(200, inf]",
    ]


def test_categorical_passthrough_bin_not_enum(pu):
    """Categorical pass-through (spec=None) is user-controlled and left as-is —
    the ordered-Enum treatment is only for numeric bins."""
    df = counts(pu, bin_by={"cell_type_post": None})
    assert not isinstance(df.schema["cell_type_post"], pl.Enum)


# ── bootstrap progress flag ──────────────────────────────────────────────────


def test_bootstrap_progress_runs(pu):
    """progress=True produces the same result as progress=False (the bar is
    cosmetic)."""
    kw = dict(bin_by={"d_rho": [0, 100, 200]}, n_resamples=20, seed=0)
    from trajan import bootstrap_over_cells

    a = bootstrap_over_cells(pu, progress=False, **kw)
    b = bootstrap_over_cells(pu, progress=True, **kw)
    assert a.sort("d_rho_bin").equals(b.sort("d_rho_bin"))


def test_bootstrap_progress_without_tqdm_warns(pu, monkeypatch):
    """progress=True degrades gracefully (warns, no bar) when tqdm is absent."""
    import builtins

    real_import = builtins.__import__

    def no_tqdm(name, *args, **kwargs):
        if name.startswith("tqdm"):
            raise ImportError("no tqdm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tqdm)
    from trajan import bootstrap_over_cells

    with pytest.warns(UserWarning, match="tqdm"):
        out = bootstrap_over_cells(
            pu, bin_by={"d_rho": [0, 100, 200]}, n_resamples=5, seed=0, progress=True
        )
    assert "p_lo" in out.columns


def test_bootstrap_numpy_path_matches_manual_resample(pu):
    """The vectorized bincount resample equals an independent join/group-by
    recomputation of the same multinomial draw (pins the numpy rewrite)."""
    import numpy as np

    from trajan.stats import cell_bootstrap_iter

    seed = 12345
    frame = next(
        iter(
            cell_bootstrap_iter(pu, group_by="cell_type_post", n_resamples=1, seed=seed)
        )
    )

    # Independent recomputation of the SAME (first) draw.
    cell_ids = [10, 20, 30, 40, 50]  # universe annotation order
    N = len(cell_ids)
    idx = {c: i for i, c in enumerate(cell_ids)}
    m = np.random.default_rng(seed).multinomial(N, [1.0 / N] * N)

    pairs = pu.collect()
    w = np.array(
        [
            m[idx[a]] * m[idx[b]]
            for a, b in zip(
                pairs["pre_pt_root_id"].to_list(), pairs["post_pt_root_id"].to_list()
            )
        ],
        dtype=float,
    )
    manual = (
        pairs.with_columns(pl.Series("w", w), (pl.col("n_syn") > 0).alias("c"))
        .group_by("cell_type_post")
        .agg(
            pl.col("w").filter(pl.col("c")).sum().alias("k_observed"),
            pl.col("w").sum().alias("n_possible"),
        )
        .filter(pl.col("n_possible") > 0)
    )

    a = frame.sort("cell_type_post")
    b = manual.sort("cell_type_post")
    assert a["cell_type_post"].to_list() == b["cell_type_post"].to_list()
    assert np.allclose(a["k_observed"].to_numpy(), b["k_observed"].to_numpy())
    assert np.allclose(a["n_possible"].to_numpy(), b["n_possible"].to_numpy())
