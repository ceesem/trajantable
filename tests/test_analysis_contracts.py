"""Guardrails across spatial eligibility, tier transitions, and matrix export."""

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from trajan import (
    ConnectivityTable,
    EdgeList,
    SynapseTable,
    connection_density,
    depth_component,
    euclidean_distance,
    position_is_valid,
    possible_pairs,
    radial_distance,
    transform_point,
)
from trajan.spatial import bbox_predicate, spatial_feature_exprs


def test_dense_float_fill_preserves_large_integer_axis_ids():
    cell_id = 2**60 + 1
    ct = ConnectivityTable(
        pl.DataFrame(
            {"pre": [cell_id, cell_id + 1], "post": ["A", "B"], "size": [2.0, 4.0]}
        ),
        pre_col="pre",
        post_col="post",
        weight_cols="size",
    )
    dense = ct.to_dense(fill_value=float("nan"))
    assert dense["pre"].dtype == pl.Int64
    assert dense["pre"].to_list() == [cell_id, cell_id + 1]
    assert dense["B"][0] != dense["B"][0]  # Missing measurement remains NaN.


@pytest.fixture
def st():
    table = SynapseTable(
        pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "pre": [10, 10, 20, 10],
                "post": [20, 20, 10, 30],
                "size": [2.0, 4.0, 8.0, 100.0],
            }
        ),
        pre_col="pre",
        post_col="post",
    )
    table.add_cell_annotation(
        "cells",
        pl.DataFrame(
            {
                "root": [10, 20, 30, 40, 50],
                "type": ["E", "I", "I", "I", "I"],
                "soma_x": [0.0, 10.0, 100.0, 5.0, None],
                "soma_y": [0.0, 0.0, 0.0, 0.0, 0.0],
                "soma_z": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
        cell_id_col="root",
        position_col="soma",
        is_universe=True,
    )
    table.metadata = {
        "synapse_source": {"version": 7},
        "coordinates": {"unit": "nm", "frame": "dataset-native"},
        "annotation_versions": {"cells": 3},
    }
    return table


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_invalid_positions_never_make_valid_spatial_measurements(bad, axis):
    point = {"x": 0.0, "y": 0.0, "z": 0.0}
    point[axis] = bad
    schema = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64})
    df = pl.DataFrame(
        {"a": [point], "b": [{"x": 10.0, "y": 0.0, "z": 0.0}]},
        schema={"a": schema, "b": schema},
    )
    assert df.select(position_is_valid("a")).item() is False
    measurements = {
        "distance": euclidean_distance("a", "b"),
        "radial": radial_distance("a", "b"),
        "depth": depth_component("a"),
        **spatial_feature_exprs("a", "b"),
    }
    result = df.select(expr.alias(name) for name, expr in measurements.items())
    assert all(np.isnan(v) for v in result.row(0))
    assert df.filter(
        bbox_predicate("a", ((-100.0, -100.0, -100.0), (100.0, 100.0, 100.0)))
    ).is_empty()
    assert (
        df.select(transform_point("a", lambda pts: pts).alias("out"))[
            "out"
        ].null_count()
        == 1
    )
    assert (
        df.select(
            transform_point("a", lambda x, y, z: (x, y, z), vectorized=True).alias(
                "out"
            )
        )["out"].null_count()
        == 1
    )


def test_transforms_reject_nonfinite_output():
    df = pl.DataFrame({"p": [{"x": 1.0, "y": 0.0, "z": 0.0}]})
    arr = df.select(
        transform_point("p", lambda pts: np.full_like(pts, np.inf)).alias("p")
    )
    vec = df.select(
        transform_point("p", lambda x, y, z: (x / 0, y, z), vectorized=True).alias("p")
    )
    assert arr["p"].null_count() == vec["p"].null_count() == 1


@pytest.mark.parametrize(
    "bbox", [((1, 0, 0), (0, 1, 1)), ((0, 0, 0), (float("inf"), 1, 1))]
)
def test_invalid_bbox_rejected(bbox):
    with pytest.raises(ValueError, match="Bounding-box"):
        bbox_predicate("p", bbox)


def test_invalid_depth_axis_rejected():
    with pytest.raises(ValueError, match="depth_axis"):
        spatial_feature_exprs("a", "b", depth_axis="depth")


@pytest.mark.parametrize("maximum", [-1.0, float("inf"), float("nan")])
def test_invalid_distance_cutoff_rejected(st, maximum):
    with pytest.raises(ValueError, match="max_distance"):
        st.filter_by_euclidean_distance(maximum)


def test_spatial_eligibility_survives_tier_transition(st):
    selected = st.filter_by_euclidean_distance(15)
    source = possible_pairs(selected)
    transitioned = possible_pairs(selected.edgelist())
    assert_frame_equal(
        source.collect().sort("pre", "post"), transitioned.collect().sort("pre", "post")
    )
    result = connection_density(source, group_by="type_pre")
    assert result["n_possible"].sum() == 6
    assert result["k_observed"].sum() == 2
    assert result["sum_n_syn"].sum() == 3
    # A synapse measurement filter changes only the observed numerator.
    measured = connection_density(
        selected.filter(pl.col("size") > 5), group_by="type_pre"
    )
    assert measured["n_possible"].sum() == 6
    assert measured["k_observed"].sum() == 1


def test_explicit_id_eligibility_includes_unconnected_pairs(st):
    selected = st.filter_pairs((pl.col("pre") == 10) & (pl.col("post") == 40))
    df = possible_pairs(selected).collect()
    assert df.height == 1
    assert df["n_syn"].item() == 0
    assert_frame_equal(
        df, possible_pairs(st.filter_by_ids(pre_ids=[10], post_ids=[40])).collect()
    )


def test_pair_filter_validates_measurement_dependencies(st):
    for table, measurement in [(st, "size"), (st.edgelist(), "n_syn")]:
        with pytest.raises(ValueError, match="Pair eligibility"):
            table.filter_pairs(pl.col(measurement) > 1)
    st.add_expression("scaled", pl.col("size") * 2)
    with pytest.raises(ValueError, match="Pair eligibility"):
        st.filter_pairs(pl.col("scaled") > 1)


def test_pair_filter_supports_id_derived_expression_and_persistence(st, tmp_path):
    st.add_expression("id_gap", pl.col("post") - pl.col("pre"))
    selected = st.filter_pairs(pl.col("id_gap") > 0)
    direct = possible_pairs(selected).collect().sort("pre", "post")
    assert direct.height == 10
    el = selected.edgelist()
    assert_frame_equal(direct, possible_pairs(el).collect().sort("pre", "post"))
    el.save(tmp_path / "edges")
    assert_frame_equal(
        direct,
        possible_pairs(EdgeList.load(tmp_path / "edges")).collect().sort("pre", "post"),
    )


def test_candidate_to_edge_transition_retains_universe_and_eligibility(st):
    pu = possible_pairs(st).filter_by_euclidean_distance(15).filter_by_ids(pre_ids=[10])
    returned = possible_pairs(pu.to_edgelist())
    assert_frame_equal(
        pu.collect().sort("pre", "post"), returned.collect().sort("pre", "post")
    )


def test_tier_transitions_are_lazy_without_annotation_rescans(st):
    with patch.object(
        pl.LazyFrame, "collect", side_effect=AssertionError("eager scan")
    ):
        el = st.edgelist(agg={"mean_size": pl.mean("size")})
        ct = el.aggregate_to_type(
            post="type_post", agg={"mean_size": pl.mean("mean_size")}
        )
        typed = st.type_edgelist("type_pre", "type_post")
        pu = possible_pairs(el)
        assert "uncached" in repr(el)
        assert "uncached" in repr(ct)
        assert isinstance(pu.to_edgelist(), EdgeList)
    assert el.pairs["n_syn"].sum() == typed.pairs["n_syn"].sum() == 4


def test_dense_guard_fires_before_wide_collection_or_pivot():
    ct = ConnectivityTable(
        pl.DataFrame(
            {"pre": list(range(4000)), "post": list(range(4000)), "n_syn": [1] * 4000}
        ).lazy(),
        pre_col="pre",
        post_col="post",
    )
    with patch.object(
        pl.DataFrame, "pivot", side_effect=AssertionError("allocated dense")
    ):
        with pytest.raises(ValueError, match="to_sparse"):
            ct.to_dense()


def test_dense_guard_can_be_overridden(st):
    ct = st.edgelist()
    with pytest.raises(ValueError, match="max_cells"):
        ct.to_dense(max_cells=1)
    assert ct.to_dense(max_cells=None).height == 2


def test_sparse_preserves_axis_order_and_zero_cells(st):
    el = st.filter_by_euclidean_distance(15).edgelist()
    with patch.object(pl.DataFrame, "pivot", side_effect=AssertionError("dense pivot")):
        matrix, pre, post = el.to_sparse(pre_ids=[20, 10, 40], post_ids=[10, 20, 40])
    assert pre.to_list() == [20, 10, 40]
    assert post.to_list() == [10, 20, 40]
    np.testing.assert_array_equal(matrix.toarray(), [[1, 0, 0], [0, 2, 0], [0, 0, 0]])
    huge, _, _ = el.to_sparse(pre_ids=range(60000), post_ids=range(60000))
    assert huge.shape == (60000, 60000)
    assert huge.nnz == 2


def test_sparse_duplicate_and_axis_validation():
    ct = ConnectivityTable(
        pl.DataFrame({"pre": [1, 1], "post": [2, 2], "n_syn": [2, 4]}),
        pre_col="pre",
        post_col="post",
    )
    with pytest.raises(ValueError, match="duplicate"):
        ct.to_sparse()
    matrix, _, _ = ct.to_sparse(aggregate="mean")
    assert matrix[0, 0] == 3
    with pytest.raises(ValueError, match="unique"):
        ct.to_sparse(aggregate="sum", pre_ids=[1, 1])
    with pytest.raises(ValueError, match="every observed"):
        ct.to_sparse(aggregate="sum", pre_ids=[3])


def test_sparse_empty_axes_and_null_values():
    ct = ConnectivityTable(
        pl.DataFrame(schema={"pre": pl.Int64, "post": pl.Int64, "n_syn": pl.Int64}),
        pre_col="pre",
        post_col="post",
    )
    matrix, pre, post = ct.to_sparse(pre_ids=[1], post_ids=[2, 3])
    assert matrix.shape == (1, 2)
    assert matrix.nnz == 0
    bad = ConnectivityTable(
        pl.DataFrame(
            {"pre": [1], "post": [2], "n_syn": [None]},
            schema_overrides={"n_syn": pl.Int64},
        ),
        pre_col="pre",
        post_col="post",
    )
    with pytest.raises(ValueError, match="non-null"):
        bad.to_sparse()


def test_metadata_is_independent_through_transitions_and_save(st, tmp_path):
    el = st.edgelist()
    ct = el.aggregate_to_type(post="type_post")
    typed = st.type_edgelist("type_pre", "type_post")
    pu = possible_pairs(el)
    tables = [
        st.view(),
        el,
        el.filter(pl.col("n_syn") > 0),
        ct,
        typed,
        pu,
        pu.filter(pl.col("pre") == 10),
        pu.to_edgelist(),
        ct.normalize(by="pre"),
    ]
    for table in tables:
        assert table.metadata == st.metadata
        table.metadata["synapse_source"]["version"] = 99
        assert st.metadata["synapse_source"]["version"] == 7
        table.metadata["synapse_source"]["version"] = 7
    for name, table in [("edges", el), ("types", ct)]:
        table.save(tmp_path / name)
        restored = ConnectivityTable.load(tmp_path / name)
        assert restored.metadata == st.metadata
        assert_frame_equal(restored.df, table.df, check_row_order=False)
