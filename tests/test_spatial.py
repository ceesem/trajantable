import math

import polars as pl
import pytest

from trajan.spatial import (
    depth_component,
    euclidean_distance,
    radial_distance,
    spatial_feature_exprs,
)


@pytest.fixture
def pos_df():
    return pl.DataFrame(
        {
            "a": [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 1.0, "y": 2.0, "z": 2.0}, None],
            "b": [{"x": 3.0, "y": 0.0, "z": 4.0}, None, {"x": 1.0, "y": 1.0, "z": 1.0}],
        }
    )


def test_euclidean_distance_normal(pos_df):
    result = pos_df.select(euclidean_distance("a", "b").alias("d"))["d"]
    assert math.isclose(result[0], 5.0)


def test_euclidean_distance_null_b(pos_df):
    result = pos_df.select(euclidean_distance("a", "b").alias("d"))["d"]
    assert math.isnan(result[1])


def test_euclidean_distance_null_a(pos_df):
    result = pos_df.select(euclidean_distance("a", "b").alias("d"))["d"]
    assert math.isnan(result[2])


def test_radial_distance_normal(pos_df):
    result = pos_df.select(radial_distance("a", "b").alias("d"))["d"]
    assert math.isclose(result[0], 5.0)


def test_radial_distance_null_b(pos_df):
    result = pos_df.select(radial_distance("a", "b").alias("d"))["d"]
    assert math.isnan(result[1])


def test_radial_distance_null_a(pos_df):
    result = pos_df.select(radial_distance("a", "b").alias("d"))["d"]
    assert math.isnan(result[2])


# ── depth_component ────────────────────────────────────────────────────────────


def test_depth_component_default_axis(pos_df):
    result = pos_df.select(depth_component("a").alias("d"))["d"]
    assert math.isclose(result[0], 0.0)  # a[0].y == 0
    assert math.isclose(result[1], 2.0)  # a[1].y == 2


def test_depth_component_reversed(pos_df):
    result = pos_df.select(depth_component("a", depth_axis="y_r").alias("d"))["d"]
    assert math.isclose(result[1], -2.0)  # a[1].y == 2, reversed


def test_depth_component_other_axis(pos_df):
    result = pos_df.select(depth_component("b", depth_axis="x").alias("d"))["d"]
    assert math.isclose(result[0], 3.0)  # b[0].x == 3


def test_depth_component_null_gives_nan(pos_df):
    result = pos_df.select(depth_component("a").alias("d"))["d"]
    assert math.isnan(result[2])  # a[2] is null


# ── spatial_feature_exprs ──────────────────────────────────────────────────────


@pytest.fixture
def vec_df():
    """Two position structs: lateral vector (dx=3, dz=4, dy=0) and pure depth (dy=5)."""
    return pl.DataFrame(
        {
            "a": [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 0.0, "y": 0.0, "z": 0.0},
                None,
                {"x": 0.0, "y": 0.0, "z": 0.0},
            ],
            "b": [
                {"x": 3.0, "y": 0.0, "z": 4.0},  # lateral: dx=3, dz=4, dy=0
                {"x": 0.0, "y": 5.0, "z": 0.0},  # pure depth: dy=5
                {"x": 1.0, "y": 1.0, "z": 1.0},  # null a
                None,  # null b
            ],
        }
    )


def _feats(vec_df, **kwargs):
    exprs = spatial_feature_exprs("a", "b", **kwargs)
    return vec_df.select([e.alias(k) for k, e in exprs.items()])


def test_spatial_lateral_vector(vec_df):
    df = _feats(vec_df)
    row = df.row(0, named=True)
    assert math.isclose(row["euclidean"], 5.0)
    assert math.isclose(row["rho"], 5.0)
    assert math.isclose(row["depth_diff"], 0.0)
    assert math.isclose(row["dy"], 0.0)
    assert math.isclose(row["theta"], math.pi / 2, rel_tol=1e-6)
    assert math.isclose(row["r"], 5.0)


def test_spatial_depth_vector(vec_df):
    df = _feats(vec_df)
    row = df.row(1, named=True)
    assert math.isclose(row["euclidean"], 5.0)
    assert math.isclose(row["rho"], 0.0, abs_tol=1e-10)
    assert math.isclose(row["depth_diff"], 5.0)
    assert math.isclose(row["theta"], 0.0, abs_tol=1e-6)


def test_spatial_null_a_gives_nan(vec_df):
    df = _feats(vec_df)
    row = df.row(2, named=True)
    assert all(math.isnan(v) for v in row.values())


def test_spatial_null_b_gives_nan(vec_df):
    df = _feats(vec_df)
    row = df.row(3, named=True)
    assert all(math.isnan(v) for v in row.values())


def test_spatial_depth_axis_reversed(vec_df):
    df = _feats(vec_df, depth_axis="y_r")
    # Pure depth vector (dy=5 from a to b), reversed: depth_diff should be -5
    row = df.row(1, named=True)
    assert math.isclose(row["depth_diff"], -5.0)
    assert math.isclose(row["dy"], -5.0)
    # theta = arccos(-5/5) = pi (pointing "up" from depth axis)
    assert math.isclose(row["theta"], math.pi, rel_tol=1e-6)


def test_spatial_phi_shared_between_spherical_and_cylindrical(vec_df):
    df = _feats(vec_df)
    # phi appears exactly once even with both spherical and cylindrical enabled
    assert df.columns.count("phi") == 1


def test_spatial_toggles_euclidean_off(vec_df):
    df = _feats(vec_df, euclidean=False)
    assert "euclidean" not in df.columns


def test_spatial_toggles_spherical_off(vec_df):
    df = _feats(vec_df, spherical=False)
    assert "r" not in df.columns
    assert "theta" not in df.columns
    # phi may still appear if cylindrical=True (default)
    assert "phi" in df.columns


def test_spatial_depth_axis_x(vec_df):
    """With depth_axis='x', depth component is dx and lateral is (y, z)."""
    # Use row 0: a=(0,0,0), b=(3,0,4) → vector (dx=3, dy=0, dz=4)
    # With depth_axis='x': dd=3, lateral=(y,z) → da=0, db=4 → rho=4
    df = _feats(vec_df, depth_axis="x")
    row = df.row(0, named=True)
    assert math.isclose(row["depth_diff"], 3.0)
    assert math.isclose(row["rho"], 4.0)
    assert math.isclose(row["euclidean"], 5.0)
