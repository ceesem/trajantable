import math

import polars as pl
import pytest

from trajan.spatial import euclidean_distance, radial_distance


@pytest.fixture
def pos_df():
    return pl.DataFrame(
        {
            "a": [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 1.0, "y": 2.0, "z": 2.0}, None],
            "b": [{"x": 3.0, "y": 4.0, "z": 0.0}, None, {"x": 1.0, "y": 1.0, "z": 1.0}],
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
