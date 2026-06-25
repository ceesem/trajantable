"""Tests for trajan.pair_universe: PairUniverse and possible_pairs.

Pressure-tests the contract in DESIGN-universe.md §4:

- possible_pairs enumerates universe × universe and overlays observed counts
- Self-pairs excluded by default, included on opt-in
- Cell-level filters from the source project onto the cross-product
- Synapse-level filters bake into observed counts via .edgelist()
- PairUniverse exposes filter/group_by/collect but not .df
- Spatial filters work via the registered position-bearing annotation
"""

import polars as pl
import pytest

from trajan import EdgeList, PairUniverse, SynapseTable, possible_pairs
from trajan.spatial import pack_position

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def universe_cells():
    """5 cells, with cell_type and is_proofread metadata."""
    return pl.DataFrame(
        {
            "root_id": [10, 20, 30, 40, 50],
            "cell_type": ["L23", "L23", "L4", "L5", "L5"],
            "is_proofread": [True, True, False, True, False],
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
def st(synapses, universe_cells):
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", universe_cells, cell_id_col="root_id", is_universe=True
    )
    return st


# ── enumeration ──────────────────────────────────────────────────────────────


def test_returns_pair_universe(st):
    pu = possible_pairs(st)
    assert isinstance(pu, PairUniverse)


def test_excludes_self_pairs_by_default(st):
    pu = possible_pairs(st)
    df = pu.collect()
    # 5 cells, 5² − 5 = 20 pairs.
    assert len(df) == 20
    assert (df["pre_pt_root_id"] != df["post_pt_root_id"]).all()


def test_include_self_keeps_autapses(st):
    pu = possible_pairs(st, include_self=True)
    df = pu.collect()
    # 5² = 25 pairs including self.
    assert len(df) == 25


def test_overlays_observed_n_syn(st):
    pu = possible_pairs(st)
    df = pu.collect()
    # 6 synapses across 5 distinct ordered pairs (counts):
    #   (10,20)=1, (10,30)=1, (20,10)=1, (20,30)=1, (30,10)=1, (30,20)=1
    observed = df.filter(pl.col("n_syn") > 0)
    assert len(observed) == 6
    # Unobserved pairs get 0.
    unobserved = df.filter(pl.col("n_syn") == 0)
    assert len(unobserved) == 14


def test_no_universe_raises(synapses):
    """possible_pairs needs a universe annotation."""
    st = SynapseTable(synapses)
    with pytest.raises(ValueError, match="is_universe"):
        possible_pairs(st)


def test_rejects_non_table_input():
    with pytest.raises(TypeError, match="accepts SynapseTable or EdgeList"):
        possible_pairs(pl.DataFrame({"x": [1]}))


# ── annotation propagation ───────────────────────────────────────────────────


def test_universe_annotation_propagates(st):
    pu = possible_pairs(st)
    assert "cells" in pu.annotation_names
    # Sanity: pre/post annotation columns appear on the lazy plan.
    schema = pu.build_lazy().collect_schema().names()
    assert "cell_type_pre" in schema
    assert "cell_type_post" in schema


# ── cell-level filter projection ─────────────────────────────────────────────


def test_pre_only_filter_constrains_pre_side_only(st):
    """A pre-only filter (`cell_type_pre == 'L23'`) restricts pre to {10, 20}
    but post still spans the universe."""
    filtered = st.filter(pl.col("cell_type_pre") == "L23")
    pu = possible_pairs(filtered)
    df = pu.collect()
    # pre ∈ {10, 20}, post ∈ universe (5 cells), exclude self.
    #   pre=10: post ∈ {20, 30, 40, 50} → 4
    #   pre=20: post ∈ {10, 30, 40, 50} → 4
    assert len(df) == 8
    assert set(df["pre_pt_root_id"].to_list()) == {10, 20}


def test_two_separate_filters_decompose(st):
    """Pre L23, post L4 — separate filters decompose cleanly."""
    filtered = st.filter(pl.col("cell_type_pre") == "L23").filter(
        pl.col("cell_type_post") == "L4"
    )
    pu = possible_pairs(filtered)
    df = pu.collect()
    # pre ∈ {10, 20}, post ∈ {30} → 2 pairs (no self-collision since 30 ∉ pre).
    assert len(df) == 2
    assert set(df["post_pt_root_id"].to_list()) == {30}


def test_both_filter_warns_and_skips_projection(st):
    """Both-sided AND in one expression warns and does not project on the
    cross-product (still constrains observed n_syn baked in upstream)."""
    filtered = st.filter(
        (pl.col("cell_type_pre") == "L23") & (pl.col("cell_type_post") == "L4")
    )
    with pytest.warns(UserWarning, match="both"):
        pu = possible_pairs(filtered)
    df = pu.collect()
    # Cross-product is universe × universe (20 pairs), not narrowed.
    assert len(df) == 20


# ── synapse-level filter bakes in only on observed counts ────────────────────


def test_synapse_filter_constrains_observed_not_cross_product(st):
    """A synapse-level filter restricts observed counts but the cross-product
    still spans the full universe — unobserved pairs get 0."""
    filtered = st.filter(pl.col("size") >= 150)  # keeps synapses with size>=150
    pu = possible_pairs(filtered)
    df = pu.collect()
    assert len(df) == 20  # cross-product unchanged
    # Observed (n_syn > 0) is narrowed:
    #   size>=150 keeps (20,10):200 and (20,30):150 → 2 observed pairs.
    observed = df.filter(pl.col("n_syn") > 0)
    assert len(observed) == 2


# ── filter / spatial filters on PairUniverse itself ──────────────────────────


def test_filter_by_ids_pre(st):
    pu = possible_pairs(st).filter_by_ids(pre_ids=[10, 20])
    df = pu.collect()
    assert set(df["pre_pt_root_id"].to_list()) == {10, 20}


# ── mixed signed/unsigned id dtypes (regression) ─────────────────────────────


@pytest.fixture
def st_mismatched_id_dtype():
    """Synapse ids are UInt64 (as parquet root ids load), but the universe
    annotation's root_id was inferred Int64. Joining the two used to panic
    in the streaming engine ('cannot get ref Int64 from UInt64')."""
    synapses = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "pre_pt_root_id": pl.Series([10, 10, 20, 30], dtype=pl.UInt64),
            "post_pt_root_id": pl.Series([20, 30, 30, 10], dtype=pl.UInt64),
            "size": [100, 50, 200, 80],
        }
    )
    cells = pl.DataFrame(
        {
            "root_id": pl.Series([10, 20, 30, 40], dtype=pl.Int64),
            "cell_type": ["L23", "L23", "L4", "L5"],
        }
    )
    st = SynapseTable(synapses)
    st.add_cell_annotation("cells", cells, cell_id_col="root_id", is_universe=True)
    return st


def test_possible_pairs_mismatched_id_dtype(st_mismatched_id_dtype):
    """possible_pairs aligns the cross-product id dtype to the observed
    (source) dtype, so the weight-overlay join doesn't panic on the
    signed/unsigned mismatch."""
    pu = possible_pairs(st_mismatched_id_dtype)
    df = pu.collect()
    assert len(df) == 12  # 4 cells, 4² − 4
    # observed ordered pairs (10,20),(10,30),(20,30),(30,10) → n_syn > 0
    assert df.filter(pl.col("n_syn") > 0).height == 4


def test_filter_by_ids_mismatched_id_dtype(st_mismatched_id_dtype):
    """filter_by_ids with a python-int list against an unsigned id column
    matches the column dtype rather than panicking on the lowered join."""
    pu = possible_pairs(st_mismatched_id_dtype).filter_by_ids(pre_ids=[10, 20])
    df = pu.collect()
    assert set(df["pre_pt_root_id"].to_list()) == {10, 20}


def test_connection_probability_mismatched_id_dtype(st_mismatched_id_dtype):
    """End-to-end: the original failing call (multi-id filter feeding a
    binned connection_probability) completes."""
    from trajan.stats import connection_probability

    pu = possible_pairs(st_mismatched_id_dtype).filter_by_ids(pre_ids=[10, 20, 30])
    out = connection_probability(pu, group_by="pre_pt_root_id")
    assert set(out["pre_pt_root_id"].to_list()) == {10, 20, 30}


def test_build_time_pre_ids_prunes_universe(st):
    """``possible_pairs(pre_ids=...)`` restricts the pre side before the cross
    product, matching a post-hoc filter_by_ids on the result."""
    pruned = possible_pairs(st, pre_ids=[10, 20]).collect()
    posthoc = possible_pairs(st).filter_by_ids(pre_ids=[10, 20]).collect()
    assert set(pruned["pre_pt_root_id"].to_list()) == {10, 20}
    assert pruned.sort(["pre_pt_root_id", "post_pt_root_id"]).equals(
        posthoc.sort(["pre_pt_root_id", "post_pt_root_id"])
    )


def test_build_time_pre_post_ids_bound_cross_product(st):
    """With both sides pruned the product is exactly |pre| × |post| (minus any
    self-pairs), never the full universe square."""
    df = possible_pairs(st, pre_ids=[10, 20], post_ids=[30, 40]).collect()
    assert set(df["pre_pt_root_id"].to_list()) == {10, 20}
    assert set(df["post_pt_root_id"].to_list()) == {30, 40}
    assert df.height == 4  # 2 × 2, no overlap so no self-pairs dropped


def test_build_time_pruning_does_not_enumerate_full_universe(st_mismatched_id_dtype):
    """The pruned id set is applied to the universe scan(s) feeding the cross
    join, so the plan does not depend on post-cross pushdown to stay bounded —
    no id predicate is left stranded above the cross join."""
    pu = possible_pairs(st_mismatched_id_dtype, pre_ids=[10])
    plan = pu.build_lazy().explain(optimized=True)
    cross_at = plan.find("NESTED LOOP JOIN")
    assert cross_at != -1
    assert "is_in" not in plan[:cross_at], (
        f"build-time pruning left an id filter above the cross join:\n{plan}"
    )


def test_filter_by_ids_pushes_below_cross_join(st_mismatched_id_dtype):
    """A single-sided ``filter_by_ids`` must prune the cross-product *before*
    it is enumerated, not after. The id-dtype alignment is done at the per-side
    select feeding ``join(how="cross")`` rather than as a ``with_columns`` cast
    placed above the cross join — a post-cross ``strict_cast`` redefines the id
    column and so is an opaque predicate-pushdown barrier, stranding the
    ``is_in`` above the join and forcing the full |U|² product to materialize
    before the filter drops it (the cause of a single-pre-id OOM on a large
    universe). Assert the optimizer pushes ``is_in`` below the cross join.
    """
    pu = possible_pairs(st_mismatched_id_dtype).filter_by_ids(pre_ids=[10])
    plan = pu.build_lazy().explain(optimized=True)
    # The cross join shows as a NESTED LOOP JOIN (the pre != post condition).
    cross_at = plan.find("NESTED LOOP JOIN")
    assert cross_at != -1, f"expected a cross join in the plan:\n{plan}"
    # If pushdown worked, no id predicate sits above the cross join; it has been
    # lowered onto the scan(s) feeding it.
    assert "is_in" not in plan[:cross_at], (
        "filter_by_ids predicate was not pushed below the cross join — the "
        f"full cross-product will materialize:\n{plan}"
    )


@pytest.fixture
def st_with_positions(synapses, universe_cells):
    cells_with_pos = universe_cells.with_columns(
        pl.Series("soma_x", [0.0, 100.0, 0.0, 0.0, 500.0]),
        pl.Series("soma_y", [0.0, 0.0, 0.0, 0.0, 0.0]),
        pl.Series("soma_z", [0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    cells_packed = pack_position(
        cells_with_pos, "soma", x="soma_x", y="soma_y", z="soma_z"
    )
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells",
        cells_packed,
        cell_id_col="root_id",
        is_universe=True,
        position_col="soma",
    )
    return st


def test_filter_by_euclidean_distance(st_with_positions):
    pu = possible_pairs(st_with_positions).filter_by_euclidean_distance(150.0)
    df = pu.collect()
    # Distances under 150:
    #   10-20, 20-10  (=100)
    #   10-30, 30-10, 20-30, 30-20  (=0 or 100)
    #   10-40, 40-10, 20-40, 40-20, 30-40, 40-30 (=0 or 100)
    # All cells {10,30,40} share position (0,0,0); 20 at (100,0,0); 50 at (500,0,0).
    # Pairs within 150: any pair among {10,20,30,40} (since 10/30/40 share origin
    # and 20 is 100 away). Cell 50 is 500 away, excluded.
    # That's 4 cells × 3 partners = 12 pairs.
    assert len(df) == 12
    assert 50 not in df["pre_pt_root_id"].to_list()
    assert 50 not in df["post_pt_root_id"].to_list()


def test_filter_by_radial_distance(st_with_positions):
    # positions all share y=z=0, so radial (xz) == euclidean here → same 12 pairs.
    pu = possible_pairs(st_with_positions).filter_by_radial_distance(150.0)
    assert len(pu.collect()) == 12


# ── from EdgeList directly ───────────────────────────────────────────────────


def test_possible_pairs_from_edgelist(st):
    """EdgeList input skips re-aggregation."""
    el = st.edgelist()
    pu = possible_pairs(el)
    df = pu.collect()
    assert len(df) == 20


# ── PairUniverse surface ─────────────────────────────────────────────────────


def test_no_df_property(st):
    """PairUniverse intentionally has no .df — forces explicit .collect()."""
    pu = possible_pairs(st)
    assert not hasattr(pu, "df")


def test_group_by_aggregates_without_materializing(st):
    """group_by(...).agg(...) is the canonical reduction path."""
    pu = possible_pairs(st)
    by_pre = pu.group_by("pre_pt_root_id").agg(pl.sum("n_syn").alias("total")).collect()
    assert len(by_pre) == 5  # one row per pre cell


def test_inherits_lazy_surface_from_base(st):
    """PairUniverse gains .lazy / .select / .count / len() from _LazyBacked."""
    pu = possible_pairs(st)
    assert isinstance(pu.lazy, pl.LazyFrame)
    assert isinstance(pu.select(["pre_pt_root_id"]), pl.LazyFrame)
    # 5 cells, 5² − 5 = 20 ordered non-self pairs
    assert pu.count() == len(pu) == 20


def test_pair_universe_has_no_df_cache(st):
    """PairUniverse extends _LazyBacked only — no caching .df surface."""
    pu = possible_pairs(st)
    assert not hasattr(pu, "df")
    assert not hasattr(pu, "clear_cache")


def test_to_edgelist_returns_observed_only(st):
    """to_edgelist() honors the EdgeList contract (observed connections only).

    The fixture has 6 distinct ordered observed pairs across 8 synapses;
    the full cross-product would be 20. Returning the cross-product as an
    EdgeList would mis-type the result: an EdgeList row carries the claim
    'this connection was observed'."""
    pu = possible_pairs(st)
    el = pu.to_edgelist()
    assert isinstance(el, EdgeList)
    assert len(el.df) == 6  # 6 observed pairs, not 20 cross-product
    assert (el.df["n_syn"] > 0).all()


def test_to_pair_frame_returns_cross_product(st):
    """to_pair_frame() returns the full universe × universe DataFrame
    including unobserved pairs (n_syn = 0). Not an EdgeList — explicitly
    a DataFrame so callers can't accidentally pass it to consumers that
    assume observed-only data."""
    pu = possible_pairs(st)
    df = pu.to_pair_frame()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 20  # 5² − 5 self-pairs
    # The zeros are present and load-bearing for denominators.
    assert (df["n_syn"] == 0).any()


def test_filter_returns_pair_universe(st):
    pu = possible_pairs(st).filter(pl.col("n_syn") > 0)
    assert isinstance(pu, PairUniverse)
    df = pu.collect()
    assert (df["n_syn"] > 0).all()


def test_truthy_without_collect(st):
    """bool(pu) is always True — never implicitly collects."""
    pu = possible_pairs(st)
    assert bool(pu) is True


# ── end-to-end: the asymmetric "pre proofread, post universe" analysis ───────


def test_sibling_annotation_filter_resolves(st, universe_cells):
    """A filter referencing a sibling cell annotation column (registered
    after the universe) must resolve when projected onto the cross-product.
    This was a regression: sibling annotations were re-registered on the
    PairUniverse but the filter machinery worked because of the join. The
    test pins the contract so future refactors can't drift cells() and
    possible_pairs() apart on what a given filter selects."""
    sibling = pl.DataFrame({"root_id": [10, 20, 30, 40, 50], "tag": list("ABCDE")})
    st.add_cell_annotation("tags", sibling, cell_id_col="root_id")
    filtered = st.filter(pl.col("tag_pre") == "A")
    pu = possible_pairs(filtered)
    df = pu.collect()
    assert set(df["pre_pt_root_id"].to_list()) == {10}


def test_cell_side_expression_filter_resolves(st):
    """A filter referencing a registered cell-side expression must resolve
    when projected onto the cross-product. Without expression propagation
    from EdgeList → PairUniverse, this previously crashed at collect with
    ColumnNotFoundError."""
    st.add_expression("ct_upper", pl.col("cell_type_pre").str.to_uppercase())
    filtered = st.filter(pl.col("ct_upper") == "L23")
    pu = possible_pairs(filtered)
    df = pu.collect()
    # ct_upper == "L23" picks pre cells of type L23: {10, 20}.
    assert set(df["pre_pt_root_id"].to_list()) == {10, 20}


def test_asymmetric_pre_proofread_density(st):
    """Analysis-1 shape from DESIGN-universe.md §6: pre side restricted to a
    subset, post side spans the universe; density computed via group_by on
    the lazy PairUniverse without ever materializing the cross-product.
    """
    proofread = st.filter(pl.col("is_proofread_pre"))
    pu = possible_pairs(proofread)

    # 3 proofread pre cells × 5 post cells, minus 3 self-pairs.
    assert len(pu.collect()) == 12

    # Connection probability per pre cell: |observed partners| / |possible partners|.
    prob = (
        pu.group_by("pre_pt_root_id")
        .agg(
            (pl.col("n_syn") > 0).sum().alias("k"),
            pl.len().alias("n_possible"),
        )
        .with_columns((pl.col("k") / pl.col("n_possible")).alias("p"))
        .collect()
    )
    assert set(prob["pre_pt_root_id"].to_list()) == {10, 20, 40}
    # Cell 40 is proofread but has no observed outgoing synapses → p=0.
    p40 = prob.filter(pl.col("pre_pt_root_id") == 40)["p"].item()
    assert p40 == 0


# ── collect(cols) / preview ──────────────────────────────────────────────────


def test_collect_narrow_projects_columns(st):
    pu = possible_pairs(st)
    out = pu.collect(cols=[pu.pre_col, pu.post_col])
    assert out.columns == [pu.pre_col, pu.post_col]


def test_collect_accepts_single_string(st):
    pu = possible_pairs(st)
    out = pu.collect(cols=pu.pre_col)
    assert out.columns == [pu.pre_col]


def test_collect_unknown_column_raises(st):
    pu = possible_pairs(st)
    with pytest.raises(ValueError, match="not found in pair universe"):
        pu.collect(cols=["bogus"])


def test_collect_none_returns_full_frame(st):
    pu = possible_pairs(st)
    full = pu.collect()
    # universe has 5 cells → 5*5 - 5 self-pairs = 20 rows
    assert len(full) == 20


def test_preview_limits_rows(st):
    pu = possible_pairs(st)
    out = pu.preview(3)
    assert len(out) == 3
    assert pu.pre_col in out.columns


# ── sample_pairs ─────────────────────────────────────────────────────────────


def test_sample_pairs_basic(st):
    pu = possible_pairs(st)
    out = pu.sample_pairs(50, seed=0)
    assert out.height == 50  # with replacement, n may exceed the 20 pairs
    assert "connected" in out.columns
    # connected is exactly n_syn > 0
    assert out["connected"].to_list() == (out["n_syn"] > 0).to_list()
    assert pu.pre_col in out.columns and pu.post_col in out.columns


def test_sample_pairs_seed_reproducible(st):
    pu = possible_pairs(st)
    a = pu.sample_pairs(30, seed=7)
    b = pu.sample_pairs(30, seed=7)
    assert a.equals(b)


def test_sample_pairs_without_replacement_distinct(st):
    pu = possible_pairs(st)
    out = pu.sample_pairs(20, replace=False, seed=1)
    pairs = set(zip(out[pu.pre_col].to_list(), out[pu.post_col].to_list()))
    assert len(pairs) == 20  # all 20 distinct pairs, each once


def test_sample_pairs_without_replacement_too_many_raises(st):
    pu = possible_pairs(st)
    with pytest.raises(ValueError, match="without replacement"):
        pu.sample_pairs(21, replace=False)


def test_sample_pairs_weights_bias_draw(st):
    """A weight that zeroes out every post cell except 30 must only draw
    pairs whose post is 30."""
    pu = possible_pairs(st)
    w = (pl.col("cell_type_post") == "L4").cast(pl.Float64)  # only cell 30 is L4
    out = pu.sample_pairs(40, weights=w, seed=2)
    assert set(out[pu.post_col].to_list()) == {30}


def test_sample_pairs_weights_by_column_name(st):
    pu = possible_pairs(st).add_expression(
        "w", (pl.col("cell_type_post") == "L4").cast(pl.Float64)
    )
    out = pu.sample_pairs(40, weights="w", seed=3)
    assert set(out[pu.post_col].to_list()) == {30}


def test_sample_pairs_negative_weights_raise(st):
    pu = possible_pairs(st)
    with pytest.raises(ValueError, match="finite and non-negative"):
        pu.sample_pairs(10, weights=pl.lit(-1.0))


def test_sample_pairs_zero_weight_sum_raises(st):
    pu = possible_pairs(st)
    with pytest.raises(ValueError, match="sum to zero"):
        pu.sample_pairs(10, weights=pl.lit(0.0))


def test_sample_pairs_nonpositive_n_raises(st):
    pu = possible_pairs(st)
    with pytest.raises(ValueError, match="n must be positive"):
        pu.sample_pairs(0)


def test_sample_pairs_empty_universe_raises(st):
    pu = possible_pairs(st).filter_by_ids(pre_ids=[999999])  # no such cell
    with pytest.raises(ValueError, match="empty pair universe"):
        pu.sample_pairs(5)
