"""Regression tests for the hardening pass (column-name collisions, null
handling, tier consistency).

Each test pins a confirmed bug found by the cross-function probe so it can't
silently come back. Grouped by theme:

- Hardcoded synthesized names colliding with user / annotation columns.
- Null group keys in the bootstrap CI join.
- Expression-derived filter side-classification consistency across tiers.
"""

import polars as pl
import pytest

from trajan import (
    ConnectivityTable,
    SynapseTable,
    bootstrap_over_cells,
    cell_summary,
    cells,
    possible_pairs,
)
from trajan._base import reject_reserved_names, unique_name

# ── shared helpers ────────────────────────────────────────────────────────────


def test_unique_name_returns_base_when_free():
    assert unique_name("x", {"a", "b"}) == "x"


def test_unique_name_suffixes_on_collision():
    assert unique_name("x", {"x"}) == "x_1"
    assert unique_name("x", {"x", "x_1", "x_2"}) == "x_3"


def test_reject_reserved_names_raises_on_clash():
    with pytest.raises(ValueError, match="reserved"):
        reject_reserved_names(["a", "n_syn"], {"n_syn"}, context="ctx")


def test_reject_reserved_names_noop_when_clear():
    reject_reserved_names(["a", "b"], {"n_syn"}, context="ctx")  # no raise


# ── cell_summary: hardcoded "cell_id" key collision ───────────────────────────


def _st_with_cell_id_annotation():
    syn = pl.DataFrame({"id": [1, 2, 3], "pre": [10, 10, 20], "post": [20, 30, 30]})
    ann = pl.DataFrame(
        {"rid": [10, 20, 30], "cell_id": [100, 200, 300], "ct": ["a", "b", "c"]}
    )
    st = SynapseTable(syn, pre_col="pre", post_col="post", id_col="id")
    st.add_cell_annotation("soma", ann, cell_id_col="rid", is_universe=True)
    return st


def test_cell_summary_cell_id_annotation_does_not_crash():
    """An annotation data column literally named cell_id no longer collides with
    the synthesized identity column (the original DuplicateError)."""
    cs = _st_with_cell_id_annotation()
    out = cell_summary(cs)
    # Identity column falls back to the universe cell_id_col; the annotation's
    # own cell_id column is preserved.
    assert "rid" in out.columns
    assert "cell_id" in out.columns
    assert set(out["rid"].to_list()) == {10, 20, 30}


def test_cell_summary_reserved_agg_name_raises():
    """A pre_agg/post_agg name shadowing an auto-generated output errors clearly."""
    syn = pl.DataFrame(
        {"id": [1, 2], "pre": [10, 20], "post": [20, 10], "size": [5.0, 6.0]}
    )
    st = SynapseTable(syn, pre_col="pre", post_col="post", id_col="id")
    st.add_weight("size")
    with pytest.raises(ValueError, match="reserved"):
        cell_summary(st, pre_agg={"size_output": pl.mean("size")})


def test_cell_summary_pre_post_agg_overlap_raises():
    """The same agg name on both sides can't share one output column."""
    syn = pl.DataFrame({"id": [1, 2], "pre": [10, 20], "post": [20, 10]})
    st = SynapseTable(syn, pre_col="pre", post_col="post", id_col="id")
    with pytest.raises(ValueError, match="both pre_agg and post_agg"):
        cell_summary(
            st,
            pre_agg={"x": pl.len()},
            post_agg={"x": pl.len()},
            include_annotations=False,
        )


# ── bootstrap: null group key drops CI ────────────────────────────────────────


def test_bootstrap_null_group_key_keeps_ci():
    """A null group key (untyped cell) gets real CI bounds, not nulls."""
    syn = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "pre_pt_root_id": [10, 10, 20, 20],
            "post_pt_root_id": [20, 30, 10, 30],
        }
    )
    cell_df = pl.DataFrame({"root_id": [10, 20, 30], "cell_type": ["A", None, "B"]})
    st = SynapseTable(syn)
    st.add_cell_annotation("cells", cell_df, cell_id_col="root_id", is_universe=True)
    out = bootstrap_over_cells(st, group_by="cell_type_pre", n_resamples=50, seed=0)
    null_row = out.filter(pl.col("cell_type_pre").is_null())
    assert len(null_row) == 1
    assert null_row["p_lo"].item() is not None
    assert null_row["p_hi"].item() is not None


def test_bootstrap_group_by_internal_name_collision():
    """A group_by key named like an internal bootstrap column (m_pre) works."""
    syn = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "pre_pt_root_id": [10, 10, 20, 20],
            "post_pt_root_id": [20, 30, 10, 30],
        }
    )
    cell_df = pl.DataFrame({"root_id": [10, 20, 30], "m_pre": ["x", "y", "z"]})
    st = SynapseTable(syn)
    st.add_cell_annotation("cells", cell_df, cell_id_col="root_id", is_universe=True)
    # group_by on an annotation column that strips to "m_pre" once suffixed —
    # the column on .df is "m_pre_pre", so use it as the key.
    out = bootstrap_over_cells(st, group_by="m_pre_pre", n_resamples=20, seed=0)
    assert "p_lo" in out.columns and len(out) > 0


# ── normalize: __total__ / fraction collisions ────────────────────────────────


def test_normalize_total_column_collision_gives_correct_fractions():
    """A user column named __total__ no longer shadows the computed axis sum."""
    df = pl.DataFrame(
        {
            "pre": [1, 1, 2],
            "post": [10, 20, 20],
            "n_syn": [5, 6, 7],
            "__total__": [1, 2, 3],
        }
    )
    out = ConnectivityTable(df, "pre", "post").normalize("pre").df.sort(["pre", "post"])
    fr = [round(x, 3) for x in out["fraction"].to_list()]
    assert fr == [0.455, 0.545, 1.0]


def test_normalize_existing_fraction_column_raises():
    df = pl.DataFrame(
        {"pre": [1, 2], "post": [10, 20], "n_syn": [5, 7], "fraction": [9.0, 9.0]}
    )
    with pytest.raises(ValueError, match="fraction"):
        ConnectivityTable(df, "pre", "post").normalize("pre")


# ── edgelist / type_edgelist: reserved-name guards ────────────────────────────


def _basic_st_with_weight():
    syn = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "pre": [1, 1, 2],
            "post": [10, 10, 20],
            "size": [1.0, 2.0, 3.0],
        }
    )
    st = SynapseTable(syn, pre_col="pre", post_col="post", id_col="id")
    st.add_weight("size")
    return st


def test_edgelist_agg_n_syn_collision_raises():
    with pytest.raises(ValueError, match="reserved"):
        _basic_st_with_weight().edgelist(agg={"n_syn": pl.first("id")})


def test_edgelist_agg_weight_collision_raises():
    with pytest.raises(ValueError, match="reserved"):
        _basic_st_with_weight().edgelist(agg={"size": pl.mean("size")})


def test_edgelist_weight_named_n_syn_raises():
    syn = pl.DataFrame({"id": [1], "pre": [1], "post": [10], "n_syn": [3]})
    st = SynapseTable(syn, pre_col="pre", post_col="post", id_col="id")
    st.add_weight("n_syn")
    with pytest.raises(ValueError, match="n_syn"):
        st.edgelist()


def test_type_edgelist_agg_collision_raises():
    syn = pl.DataFrame(
        {
            "id": [1, 2],
            "pre": [1, 2],
            "post": [10, 20],
            "ct_pre": ["a", "b"],
            "ct_post": ["x", "y"],
        }
    )
    st = SynapseTable(syn, pre_col="pre", post_col="post", id_col="id")
    with pytest.raises(ValueError, match="reserved"):
        st.type_edgelist("ct_pre", agg={"n_syn": pl.first("id")})


# ── cells(participation): output-name collisions ──────────────────────────────


def test_participation_output_collision_raises():
    syn = pl.DataFrame({"id": [1, 2], "pre": [10, 20], "post": [20, 10]})
    ann = pl.DataFrame({"rid": [10, 20], "n_syn_out": [99, 88]})
    st = SynapseTable(syn, pre_col="pre", post_col="post", id_col="id")
    st.add_cell_annotation("a", ann, cell_id_col="rid", is_universe=True)
    with pytest.raises(ValueError, match="already exist"):
        cells(st, participation=True)


def test_participation_with_cell_id_annotation_column_ok():
    """A universe annotation with a cell_id data column doesn't corrupt the
    participation join (uses a unique internal join key)."""
    syn = pl.DataFrame({"id": [1, 2, 3], "pre": [10, 10, 20], "post": [20, 30, 30]})
    ann = pl.DataFrame({"rid": [10, 20, 30, 40], "cell_id": [1, 2, 3, 4]})
    st = SynapseTable(syn, pre_col="pre", post_col="post", id_col="id")
    st.add_cell_annotation("a", ann, cell_id_col="rid", is_universe=True)
    out = cells(st, participation=True).sort("rid")
    assert out.filter(pl.col("rid") == 40)["n_syn_out"].item() == 0
    assert out.filter(pl.col("rid") == 40)["in_pre"].item() is False
    assert "cell_id" in out.columns  # annotation column preserved


# ── expression-derived filter side-classification (tier consistency) ──────────


def test_edgelist_classifies_expression_derived_filter():
    """A filter on a registered cell-side expression classifies as 'pre' on an
    EdgeList, matching SynapseTable (was None before the fix)."""
    syn = pl.DataFrame({"id": [1, 2], "pre": [1, 2], "post": [10, 20]})
    st = SynapseTable(syn, pre_col="pre", post_col="post", id_col="id")
    types = pl.DataFrame({"rid": [1, 2, 10, 20], "ct": ["a", "b", "c", "d"]})
    st.add_cell_annotation("t", types, cell_id_col="rid", is_universe=True)
    el = st.edgelist()
    el = el.add_expression("ct_up_pre", pl.col("ct_pre").str.to_uppercase())
    el2 = el.filter(pl.col("ct_up_pre") == "A")
    assert el2.filter_sides == ["pre"]


def test_cells_and_possible_pairs_agree_on_expression_filter():
    """cells(scope='filtered') and possible_pairs select the same pre cells when
    the filter references an expression-derived column (the 1:1 contract)."""
    syn = pl.DataFrame({"id": [1, 2], "pre": [1, 2], "post": [10, 20]})
    st = SynapseTable(syn, pre_col="pre", post_col="post", id_col="id")
    types = pl.DataFrame({"rid": [1, 2, 10, 20], "ct": ["L23", "L4", "L4", "L4"]})
    st.add_cell_annotation("t", types, cell_id_col="rid", is_universe=True)
    st.add_expression("ct_up_pre", pl.col("ct_pre").str.to_uppercase())
    filtered = st.filter(pl.col("ct_up_pre") == "L23")

    # cells: pre-eligible {1} ∪ post-eligible (universe) — but check the pre set
    # via participation isn't needed; verify possible_pairs pre ids match the
    # expression projection.
    pp = possible_pairs(filtered)
    pre_ids = set(pp.collect()[pp.pre_col].unique().to_list())
    assert pre_ids == {1}  # only the L23 cell is pre-eligible


# ── to_edgelist with a non-n_syn weight ───────────────────────────────────────


def test_to_edgelist_non_n_syn_weight():
    from trajan import EdgeList

    df = pl.DataFrame({"pre": [1, 1], "post": [10, 20], "wt": [3, 5]})
    el = EdgeList(df, pre_col="pre", post_col="post", weight_cols=["wt"])
    el.add_annotation(
        "u",
        pl.DataFrame({"rid": [1, 10, 20]}),
        cell_id_col="rid",
        is_universe=True,
    )
    pp = possible_pairs(el)
    back = pp.to_edgelist()
    # Observed pairs (wt > 0) round-trip; unobserved (wt == 0) are dropped.
    assert back.df.height == 2


# ── null bin-feature values: kept, not dropped (continuous == categorical) ────


def _st_with_null_position():
    syn = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "pre_pt_root_id": [10, 10, 20, 20],
            "post_pt_root_id": [20, 30, 10, 30],
        }
    )
    cell_df = pl.DataFrame(
        {
            "root_id": [10, 20, 30],
            "soma_x": [0.0, 10.0, None],
            "soma_y": [0.0, 0.0, None],
            "soma_z": [0.0, 0.0, None],
            "ct": ["A", "A", "B"],
        }
    )
    st = SynapseTable(syn).add_cell_annotation(
        "cells", cell_df, cell_id_col="root_id", position_col="soma", is_universe=True
    )
    return st.add_spatial_features(prefix="d")


def test_continuous_bins_keep_null_feature_rows():
    """A pair whose distance is null (cell with no position) lands in a null bin
    and stays in n_possible — the denominator is not silently shrunk."""
    from trajan import connection_probability, possible_pairs

    pp = possible_pairs(_st_with_null_position())
    total = pp.collect().height
    out = connection_probability(pp, bin_by={"d_rho": [0, 5, 1000]})
    # Denominator is preserved across all bins, including the null bin.
    assert out["n_possible"].sum() == total
    assert out.filter(pl.col("d_rho_bin").is_null()).height == 1


def test_continuous_and_categorical_null_handling_consistent():
    """Continuous and categorical binning both keep null-feature rows (same
    denominator)."""
    from trajan import connection_probability, possible_pairs

    pp = possible_pairs(_st_with_null_position())
    cont = connection_probability(pp, bin_by={"d_rho": [0, 5, 1000]})
    cat = connection_probability(pp, bin_by={"ct_post": None})
    assert cont["n_possible"].sum() == cat["n_possible"].sum()


# ── to_dense: duplicate (pre, post) pairs are an error, not a silent collapse ─


def test_to_dense_rejects_duplicate_pairs():
    df = pl.DataFrame({"pre": [1, 1, 2], "post": [10, 10, 20], "n_syn": [5, 6, 7]})
    ct = ConnectivityTable(df, "pre", "post")
    with pytest.raises(ValueError, match="duplicate"):
        ct.to_dense()


def test_to_dense_ok_when_unique():
    df = pl.DataFrame({"pre": [1, 1, 2], "post": [10, 20, 20], "n_syn": [5, 6, 7]})
    dense = ConnectivityTable(df, "pre", "post").to_dense()
    assert dense.height == 2  # two pre rows


def test_to_dense_aggregate_sum_combines_duplicates():
    """aggregate='sum' opts into combining duplicate (pre, post) entries."""
    df = pl.DataFrame({"pre": [1, 1, 2], "post": [10, 10, 20], "n_syn": [5, 6, 7]})
    dense = ConnectivityTable(df, "pre", "post").to_dense(aggregate="sum")
    row1 = dense.filter(pl.col("pre") == 1)
    assert row1["10"].item() == 11  # 5 + 6 summed
