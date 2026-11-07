"""Type grouping excludes unresolved axes consistently with cell grouping."""

import polars as pl
import pytest

from trajan import SynapseTable


@pytest.fixture
def st():
    table = SynapseTable(
        pl.DataFrame(
            {
                "id": list(range(6)),
                "pre": [1, 1, 2, 2, 3, 3],
                "post": [2, 3, 3, 1, 1, 2],
                "type_pre": ["A", "A", None, "B", None, "A"],
                "type_post": ["B", None, "A", "A", None, "B"],
                "size": [1.0, 2.0, 3.0, 4.0, 5.0, None],
                "tag": ["shaft", "soma", "spine", None, "shaft", "spine"],
            }
        ),
        pre_col="pre",
        post_col="post",
    )
    table.add_weight("size")
    return table


def test_type_grouping_drops_both_null_keys_by_default(st):
    ct = st.type_edgelist(
        "type_pre", agg={"mean_size": pl.mean("size")}, fraction_by="tag"
    )
    df = ct.df.sort("type_pre", "type_post")
    assert df.select("type_pre", "type_post").rows() == [("A", "B"), ("B", "A")]
    assert df["n_syn"].to_list() == [2, 1]
    assert df["size"].to_list() == [1.0, 4.0]
    assert df["mean_size"].to_list() == [1.0, 4.0]
    assert df["tag"].struct.field("spine").to_list() == [0.5, 0.0]
    assert df["tag"].struct.field("__null__").to_list() == [0.0, 1.0]
    assert st.df.height == 6
    assert ct.normalize(by="pre").to_dense("fraction").height == 2


def test_opt_out_retains_unresolved_type_groups(st):
    df = st.type_edgelist("type_pre", "type_post", drop_nulls=False).df
    assert df["n_syn"].sum() == 6
    assert df["type_pre"].null_count() == 2
    assert df["type_post"].null_count() == 2
    assert st.type_edgelist("pre", "type_post").df["n_syn"].sum() == 4
    assert st.type_edgelist("pre", "type_post", drop_nulls=False).df["n_syn"].sum() == 6


def test_all_null_keys_return_empty_typed_table(st):
    ct = st.filter(pl.col("type_pre").is_null()).type_edgelist("type_pre", "type_post")
    assert ct.df.height == 0
    assert ct.df.schema["type_pre"] == pl.String
    assert ct.to_dense().height == 0
