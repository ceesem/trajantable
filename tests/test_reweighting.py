"""Population aggregation and opportunity-adjusted allocation contracts."""

import polars as pl
import pytest

from trajan import ConnectivityTable, SynapseTable


@pytest.fixture
def st():
    table = SynapseTable(
        pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "pre": [10] * 4,
                "post": [1, 1, 2, 3],
                "size": [2.0, 4.0, None, 6.0],
            }
        ),
        pre_col="pre",
        post_col="post",
    )
    table.add_cell_annotation(
        "cells",
        pl.DataFrame(
            {
                "root": [1, 2, 3, 4, 5],
                "subclass": ["A", "A", "B", "C", None],
                "length": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ),
        cell_id_col="root",
        is_universe=True,
    )
    return table


def test_population_and_allocation_workflow(st):
    population = dict(st.cell_agg("subclass").iter_rows())
    assert population == {"A": 2, "B": 1, "C": 1, None: 1}
    assert (
        dict(st.filter(pl.col("size") > 5).cell_agg("subclass").iter_rows())
        == population
    )
    ct = st.type_edgelist("pre", "subclass_post", agg={"size_sum": pl.sum("size")})
    ct.metadata = {"source": {"version": 3}}
    result = ct.normalize(by="pre", values="n_syn", reweight_by=population)
    got = dict(result.df.select("subclass_post", "fraction").iter_rows())
    assert got == pytest.approx({"A": 0.6, "B": 0.4})
    assert "n_syn" in ct.df.columns
    assert "n_syn" not in result.df.columns
    assert result.weights == []
    assert result.metadata == ct.metadata
    sizes = ct.normalize(by="pre", values="size_sum", reweight_by=population)
    assert dict(
        sizes.df.select("subclass_post", "fraction").iter_rows()
    ) == pytest.approx({"A": 1 / 3, "B": 2 / 3})
    ct.add_annotation(
        "population", st.cell_agg("subclass"), cell_id_col="subclass", side="post"
    )
    column = ct.normalize(by="pre", reweight_by="n_cells_post")
    assert dict(
        column.df.select("subclass_post", "fraction").iter_rows()
    ) == pytest.approx(got)


def test_cell_aggregations_and_scopes(st):
    lengths = st.cell_agg("subclass", agg={"total_length": pl.sum("length")})
    assert lengths.filter(pl.col("subclass") == "A")["total_length"].item() == 3
    expr = st.cell_agg(["subclass"], pl.mean("length").alias("mean_length"))
    assert expr.filter(pl.col("subclass") == "A")["mean_length"].item() == 1.5
    multiple = st.cell_agg(
        "subclass", [pl.min("length"), pl.max("length").alias("max_length")]
    )
    assert multiple.height == 4
    assert st.cell_agg("subclass", scope="observed")["n_cells"].sum() == 3
    filtered = st.filter(pl.col("subclass_post") == "A").filter(
        pl.col("subclass_pre") == "A"
    )
    assert dict(filtered.cell_agg("subclass", scope="filtered").iter_rows()) == {"A": 2}


def simple(axis="post", categories=None):
    return ConnectivityTable(
        pl.DataFrame(
            {
                "pre": ["x", "y"] if axis == "pre" else ["x", "x"],
                "post": categories
                if categories is not None
                else (["a", "a"] if axis == "pre" else ["a", "b"]),
                "n_syn": [2, 4],
            }
        ),
        pre_col="pre",
        post_col="post",
    )


def test_opposite_axis_enum_null_and_empty():
    result = simple("pre").normalize(by="post", reweight_by={"x": 1, "y": 2})
    assert result.df["fraction"].to_list() == [0.5, 0.5]
    ct = simple(categories=pl.Series(["a", "b"], dtype=pl.Enum(["b", "a"])))
    assert ct.normalize(by="pre", reweight_by={"a": 1, "b": 2, "unused": 7}).df[
        "fraction"
    ].to_list() == [0.5, 0.5]


def test_dataframe_population_workflow(st):
    ct = st.type_edgelist("pre", "subclass_post")
    result = ct.normalize(by="pre", reweight_by=st.cell_agg("subclass"))
    assert dict(
        result.df.select("subclass_post", "fraction").iter_rows()
    ) == pytest.approx({"A": 0.6, "B": 0.4})
    frame = pl.DataFrame(
        {"category": ["x", "y", "extra"], "population": [1.0, 2.0, 3.0]}
    )
    assert simple("pre").normalize(by="post", reweight_by=frame).df[
        "fraction"
    ].to_list() == [0.5, 0.5]
    ct = simple(categories=pl.Series(["a", "b"], dtype=pl.Enum(["b", "a"])))
    frame = pl.DataFrame({"key": ["a", "b", "extra"], "size": [1, 2, 3]})
    assert ct.normalize(by="pre", reweight_by=frame).df["fraction"].to_list() == [
        0.5,
        0.5,
    ]
    frame = pl.DataFrame({"key": ["a", None], "size": [1.0, 2.0]})
    assert simple(categories=["a", None]).normalize(by="pre", reweight_by=frame).df[
        "fraction"
    ].to_list() == [0.5, 0.5]


@pytest.mark.parametrize(
    "frame, message",
    [
        (pl.DataFrame({"key": ["a"]}), "exactly two"),
        (pl.DataFrame({"key": ["a", "a"], "n": [1, 2]}), "unique"),
        (pl.DataFrame({"key": ["a", "b"], "n": [1, 0]}), "finite and positive"),
        (pl.DataFrame({"key": ["a", "b"], "n": [1, None]}), "finite and positive"),
        (
            pl.DataFrame({"key": ["a", "b"], "n": [1.0, float("inf")]}),
            "finite and positive",
        ),
        (pl.DataFrame({"key": ["a", "b"], "n": ["1", "2"]}), "numeric"),
        (pl.DataFrame({"key": ["a"], "n": [1]}), "missing categories"),
    ],
)
def test_invalid_dataframe_population(frame, message):
    with pytest.raises((ValueError, TypeError), match=message):
        simple().normalize(by="pre", reweight_by=frame)


def test_enrichment_unit_baseline_and_full_population(st):
    ct = st.type_edgelist("pre", "subclass_post", agg={"size_sum": pl.sum("size")})
    result = ct.normalize(
        by="pre", reweight_by=st.cell_agg("subclass"), mode="enrichment"
    )
    got = dict(result.df.select("subclass_post", "enrichment").iter_rows())
    assert got == pytest.approx({"A": 1.875, "B": 1.25})
    assert result.weights == []
    assert "n_syn" in ct.df.columns
    # Allocation proportional to the population gives exactly one.
    balanced = simple().normalize(
        by="pre", reweight_by={"a": 1, "b": 2}, mode="enrichment"
    )
    assert balanced.df["enrichment"].to_list() == [1.0, 1.0]
    # Unobserved categories change the expected total, unlike fraction mode.
    extra = simple().normalize(
        by="pre", reweight_by={"a": 1, "b": 2, "c": 3}, mode="enrichment"
    )
    assert extra.df["enrichment"].to_list() == [2.0, 2.0]
    assert "c" not in extra.to_dense("enrichment").columns
    sizes = ct.normalize(
        by="pre", values="size_sum", reweight_by={"A": 2, "B": 1}, mode="enrichment"
    )
    assert dict(
        sizes.df.select("subclass_post", "enrichment").iter_rows()
    ) == pytest.approx({"A": 0.75, "B": 1.5})


def test_enrichment_both_axes_zero_groups_and_persistence(tmp_path):
    population = pl.DataFrame({"category": ["x", "y"], "n_cells": [2, 4]})
    result = simple("pre").normalize(
        by="post", reweight_by=population, mode="enrichment"
    )
    assert result.df["enrichment"].to_list() == [1.0, 1.0]
    result.save(tmp_path / "enrichment")
    assert ConnectivityTable.load(tmp_path / "enrichment").df[
        "enrichment"
    ].to_list() == [1.0, 1.0]
    ct = ConnectivityTable(
        pl.DataFrame({"pre": ["x", "x"], "post": ["a", "b"], "n_syn": [0, 0]}),
        pre_col="pre",
        post_col="post",
    )
    assert (
        ct.normalize(by="pre", reweight_by={"a": 1, "b": 2}, mode="enrichment")
        .df["enrichment"]
        .null_count()
        == 2
    )
    ct = ConnectivityTable(
        pl.DataFrame({"pre": ["x", "x"], "post": ["a", "b"], "n_syn": [2, 0]}),
        pre_col="pre",
        post_col="post",
    )
    assert ct.normalize(by="pre", reweight_by={"a": 1, "b": 2}, mode="enrichment").df[
        "enrichment"
    ].to_list() == [3.0, 0.0]


def test_enrichment_invalid_mode_population_and_output():
    with pytest.raises(ValueError, match="mode must"):
        simple().normalize(mode="wrong")
    for denominator in (None, "n_syn"):
        with pytest.raises(ValueError, match="full eligible population"):
            simple().normalize(mode="enrichment", reweight_by=denominator)
    with pytest.raises(ValueError, match="combined"):
        simple().normalize(
            mode="enrichment", reweight_by={"a": 1, "b": 2}, total_col="n_syn"
        )
    with pytest.raises(ValueError, match="finite population total"):
        simple().normalize(mode="enrichment", reweight_by={"a": 1e308, "b": 1e308})
    ct = simple()
    ct.add_expression("enrichment", pl.lit(0.0))
    with pytest.raises(ValueError, match="already exists"):
        ct.normalize(by="pre", mode="enrichment", reweight_by={"a": 1, "b": 2})
    assert simple(categories=["a", None]).normalize(
        by="pre", reweight_by={"a": 1, None: 2}
    ).df["fraction"].to_list() == [0.5, 0.5]
    ct = ConnectivityTable(
        pl.DataFrame(schema={"pre": pl.String, "post": pl.String, "n_syn": pl.Int64}),
        pre_col="pre",
        post_col="post",
    )
    assert ct.normalize(by="pre", reweight_by={}).df.height == 0


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), None, "2", True])
def test_invalid_mapping(bad):
    with pytest.raises((ValueError, TypeError), match="mapping values"):
        simple().normalize(by="pre", reweight_by={"a": bad, "b": 1})


def test_invalid_arguments():
    with pytest.raises(ValueError, match="missing categories"):
        simple().normalize(by="pre", reweight_by={"a": 1})
    with pytest.raises(ValueError, match="combined"):
        simple().normalize(total_col="n_syn", reweight_by={"a": 1})
    with pytest.raises(ValueError, match="not found"):
        simple().normalize(reweight_by="missing")
    with pytest.raises(TypeError, match="numeric"):
        simple().normalize(reweight_by="post")
    with pytest.raises(TypeError, match="column name or a mapping"):
        simple().normalize(reweight_by=[1, 2])


@pytest.mark.parametrize("bad", [0.0, -1.0, None, float("nan"), float("inf")])
def test_invalid_column_denominators_and_zero_totals(bad):
    ct = simple()
    ct.add_expression(
        "population",
        pl.when(pl.col("post") == "a")
        .then(pl.lit(bad, dtype=pl.Float64))
        .otherwise(2.0),
    )
    got = dict(
        ct.normalize(by="pre", reweight_by="population")
        .df.select("post", "fraction")
        .iter_rows()
    )
    assert got == {"a": None, "b": 1.0}
    ct.add_expression("zero", pl.lit(0.0))
    assert (
        ct.normalize(by="pre", values="zero", reweight_by="population")
        .df["fraction"]
        .null_count()
        == 2
    )


def test_normalize_existing_fraction():
    ct = simple().normalize(by="pre")
    assert ct.normalize(by="pre", values="fraction").df[
        "fraction"
    ].sum() == pytest.approx(1)
    assert ct.normalize(values="fraction", total_col="fraction").df[
        "fraction"
    ].to_list() == [1.0, 1.0]
    assert ct.normalize(by="pre", values="fraction", reweight_by={"a": 1, "b": 2}).df[
        "fraction"
    ].to_list() == [0.5, 0.5]
