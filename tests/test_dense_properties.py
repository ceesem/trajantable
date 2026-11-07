"""Wide feature exports retain cell properties and declared taxonomy order."""

import polars as pl
import pytest

from trajan import ConnectivityTable, SynapseTable


def test_enum_order_properties_and_null_preservation():
    ct = ConnectivityTable(
        pl.DataFrame(
            {
                "pre": [2**60 + 1, 2**60 + 1, 2**60 + 2],
                "post": pl.Series(
                    ["B", "A", None], dtype=pl.Enum(["A", "B", "unused"])
                ),
                "n_syn": [2, 1, 3],
                "cell_type": ["exc", "exc", None],
                "depth": [None, None, 50.0],
            }
        ),
        pre_col="pre",
        post_col="post",
    )
    dense = ct.to_dense(fill_value=float("nan"), keep=["cell_type", "depth"])
    assert dense.columns == ["pre", "cell_type", "depth", "A", "B", "null"]
    assert dense["pre"].dtype == pl.Int64
    assert dense["pre"].to_list() == [2**60 + 1, 2**60 + 2]
    assert dense["cell_type"].to_list() == ["exc", None]
    assert dense["depth"].to_list() == [None, 50.0]
    assert dense["B"][1] != dense["B"][1]
    assert ct.to_dense(keep="cell_type").columns[:2] == ["pre", "cell_type"]
    assert ct.to_dense().columns == ["pre", "A", "B", "null"]


def test_type_transition_carries_pre_cell_properties_through_normalize(tmp_path):
    st = SynapseTable(
        pl.DataFrame(
            {
                "id": [1, 2, 3],
                "pre": [10, 10, 20],
                "post": [20, 30, 10],
                "size": [2.0, 4.0, 8.0],
            }
        ),
        pre_col="pre",
        post_col="post",
    )
    st.add_cell_annotation(
        "cells",
        pl.DataFrame(
            {
                "root": [10, 20, 30],
                "class": pl.Series(
                    ["exc", "inh", "exc"], dtype=pl.Enum(["inh", "exc"])
                ),
                "depth": [5.0, 10.0, None],
            }
        ),
        cell_id_col="root",
        is_universe=True,
    )
    ct = st.type_edgelist("pre", "class_post", agg={"mean_size": pl.mean("size")})
    assert "class_pre" in ct.df.columns
    assert "depth_pre" in ct.df.columns
    assert "depth_post" not in ct.df.columns
    dense = ct.to_dense("mean_size", keep=["class_pre", "depth_pre"])
    assert dense.columns == ["pre", "class_pre", "depth_pre", "inh", "exc"]
    assert dense.filter(pl.col("pre") == 10)["depth_pre"].item() == 5
    result = ct.normalize(by="pre").to_dense(
        "fraction", keep=["class_pre", "depth_pre"]
    )
    assert result.columns == dense.columns
    ct.save(tmp_path / "with_properties")
    restored = ConnectivityTable.load(tmp_path / "with_properties")
    assert restored.to_dense("mean_size", keep="depth_pre").columns == [
        "pre",
        "depth_pre",
        "inh",
        "exc",
    ]
    # Keeping only the post ID axis propagates post properties, and keeping
    # both original ID axes propagates both annotation sides.
    assert "depth_post" in st.type_edgelist("class_pre", "post").df.columns
    assert "depth_pre" in st.type_edgelist("pre", "post").df.columns
    assert "depth_post" in st.type_edgelist("pre", "post").df.columns


@pytest.mark.parametrize(
    "keep, message",
    [
        (["missing"], "not found"),
        (["pre"], "pair axes"),
        (["post"], "pair axes"),
        (["n_syn"], "pivot value"),
        (["prop", "prop"], "unique"),
    ],
)
def test_invalid_keep(keep, message):
    ct = ConnectivityTable(
        pl.DataFrame({"pre": [1], "post": ["a"], "n_syn": [1], "prop": [2]}),
        pre_col="pre",
        post_col="post",
    )
    with pytest.raises(ValueError, match=message):
        ct.to_dense(keep=keep)


@pytest.mark.parametrize("properties", [[1, 2], [None, 2]])
def test_conflicting_properties_rejected(properties):
    ct = ConnectivityTable(
        pl.DataFrame(
            {"pre": [1, 1], "post": ["a", "b"], "n_syn": [1, 2], "prop": properties}
        ),
        pre_col="pre",
        post_col="post",
    )
    with pytest.raises(ValueError, match="single value"):
        ct.to_dense(keep="prop")


def test_duplicate_reducer_and_collision():
    ct = ConnectivityTable(
        pl.DataFrame(
            {
                "pre": [1, 1],
                "post": ["a", "a"],
                "n_syn": [1, 2],
                "prop": [7, 7],
                "a": [0, 0],
            }
        ),
        pre_col="pre",
        post_col="post",
    )
    dense = ct.to_dense(keep="prop", aggregate="sum")
    assert dense.row(0) == (1, 7, 3)
    with pytest.raises(ValueError, match="collide"):
        ct.to_dense(keep="a", aggregate="sum")


def test_empty_and_null_id_property_export():
    ct = ConnectivityTable(
        pl.DataFrame(
            schema={
                "pre": pl.Int64,
                "post": pl.Enum(["a", "b"]),
                "n_syn": pl.Int64,
                "prop": pl.String,
            }
        ),
        pre_col="pre",
        post_col="post",
    )
    dense = ct.to_dense(keep="prop")
    assert dense.columns == ["pre", "prop"]
    assert dense.height == 0
    ct = ConnectivityTable(
        pl.DataFrame(
            {
                "pre": pl.Series([None], dtype=pl.Int64),
                "post": ["a"],
                "n_syn": [1],
                "prop": ["x"],
            }
        ),
        pre_col="pre",
        post_col="post",
    )
    assert ct.to_dense(keep="prop")["prop"].item() == "x"
