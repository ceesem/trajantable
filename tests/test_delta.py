"""Delta storage contracts and synapse-to-spatial-analysis integration."""

import sys
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from trajan import SynapseTable, connection_density, possible_pairs


@pytest.fixture
def synapses():
    return pl.DataFrame(
        {
            "sid": [1, 2, 3],
            "pre": [10, 10, 20],
            "post": [20, 20, 10],
            "pos_x": [1.0, 2.0, 3.0],
            "pos_y": [0.0, 0.0, 0.0],
            "pos_z": [0.0, 0.0, 0.0],
            "size": [2.0, 3.0, 4.0],
            "region": ["A", "A", "B"],
        }
    )


def table(df):
    return SynapseTable(
        df, pre_col="pre", post_col="post", id_col="sid", synapse_position_col="pos"
    )


def open_table(path, **kwargs):
    return SynapseTable.from_delta(
        path,
        pre_col="pre",
        post_col="post",
        id_col="sid",
        synapse_position_col="pos",
        **kwargs,
    )


@pytest.fixture
def delta_path(tmp_path, synapses):
    pytest.importorskip("deltalake")
    pytest.importorskip("pyarrow")
    path = tmp_path / "synapses"
    table(synapses).write_synapses_delta(path)
    return path


def test_roundtrip_positions_weights_and_projection(delta_path, synapses):
    st = open_table(delta_path)
    assert_frame_equal(st.synapses, table(synapses).synapses)
    st.add_weight("size")
    pairs = st.edgelist().pairs.sort("pre")
    assert pairs["n_syn"].to_list() == [2, 1]
    assert pairs["size"].to_list() == [5.0, 4.0]
    assert st.collect("sid").columns == ["sid"]
    assert st._cache is None


def test_open_and_repr_do_not_collect_synapses(delta_path):
    with patch.object(pl.LazyFrame, "collect", side_effect=AssertionError("row scan")):
        st = open_table(delta_path)
        assert "n_syn=uncached" in repr(st)
        assert st.columns
        assert st.filter(pl.col("pre") == 10).view()
    assert len(st) == 3
    assert "n_syn=3" in repr(st)
    assert len(st.filter(pl.col("pre") == 10)) == 2
    assert "3 synapses" in st.info()


def test_lazy_frame_constructor_does_not_collect(synapses):
    with patch.object(pl.LazyFrame, "collect", side_effect=AssertionError("row scan")):
        st = table(synapses.lazy())
        assert "uncached" in repr(st)
    assert len(st) == 3
    assert len(table(synapses.head(0).lazy())) == 0


def test_snapshot_isolation_and_time_travel(delta_path, synapses):
    original = open_table(delta_path)
    table(synapses.head(1)).write_synapses_delta(delta_path, mode="append")
    latest = open_table(delta_path)
    assert len(original) == 3  # first count occurs after the new commit
    assert len(original.filter(pl.col("pre") == 10)) == 2
    assert len(latest) == 4
    assert len(open_table(delta_path, version=0)) == 3
    assert original.metadata["synapse_source"] == {
        "format": "delta",
        "uri": str(delta_path.resolve()),
        "version": 0,
    }
    assert latest.metadata["synapse_source"]["version"] == 1
    assert original.view().metadata == original.metadata


def test_export_is_base_rows_and_does_not_replace_source(delta_path, tmp_path):
    st = open_table(delta_path).filter(pl.col("pre") == 10)
    st.add_expression("derived", pl.col("size") * 2)
    st.add_synapse_annotation("quality", pl.DataFrame({"sid": [1], "quality": [True]}))
    target = tmp_path / "export"
    st.write_synapses_delta(target)
    exported = open_table(target)
    assert len(exported) == 3
    assert "derived" not in exported.columns
    assert "quality" not in exported.columns
    assert st.metadata["synapse_source"]["uri"] == str(delta_path.resolve())


def test_existing_table_requires_explicit_mode(delta_path, synapses):
    from deltalake.exceptions import DeltaError

    with pytest.raises(DeltaError):
        table(synapses).write_synapses_delta(delta_path)
    assert len(open_table(delta_path)) == 3
    table(synapses.head(1)).write_synapses_delta(delta_path, mode="overwrite")
    assert len(open_table(delta_path)) == 1
    assert len(open_table(delta_path, version=0)) == 3


def test_partitioned_table(delta_path, tmp_path, synapses):
    target = tmp_path / "partitioned"
    table(synapses).write_synapses_delta(target, partition_by=["region"])
    result = open_table(target).filter(pl.col("region") == "B").collect()
    assert result["sid"].to_list() == [3]


def test_uint64_ids_preserve_values_and_reject_overflow(delta_path, tmp_path, synapses):
    df = synapses.with_columns(pl.col("sid", "pre", "post").cast(pl.UInt64))
    df = df.with_columns(pl.lit(2**60 + 1, dtype=pl.UInt64).alias("pre"))
    target = tmp_path / "unsigned"
    table(df).write_synapses_delta(target)
    result = open_table(target).synapses.sort("sid")
    assert result["pre"].to_list() == df["pre"].to_list()
    assert result.schema["pre"] == pl.Int64
    large = df.with_columns(pl.lit(2**63, dtype=pl.UInt64).alias("pre"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        table(large).write_synapses_delta(tmp_path / "overflow")
    assert not (tmp_path / "overflow" / "_delta_log").exists()


def test_delta_through_cell_types_space_and_denominator(delta_path):
    st = open_table(delta_path)
    cells = pl.DataFrame(
        {
            "root": [10, 20, 30],
            "type": ["E", "I", "I"],
            "soma_x": [0.0, 10.0, 100.0],
            "soma_y": [0.0, 0.0, 0.0],
            "soma_z": [0.0, 0.0, 0.0],
        }
    )
    st.add_cell_annotation(
        "cells", cells, cell_id_col="root", position_col="soma", is_universe=True
    )
    spatial = st.filter_by_euclidean_distance(20)
    pairs = possible_pairs(spatial)
    density = connection_density(pairs, group_by="type_pre")
    assert density["n_possible"].sum() == 2
    assert density["k_observed"].sum() == 2
    assert density["p"].to_list() == [1.0, 1.0]
    typed = spatial.type_edgelist("type_pre", "type_post").pairs
    assert typed["n_syn"].sum() == 3
    # Unconnected cells remain in the denominator when no spatial mask is used.
    assert connection_density(st, group_by="type_pre")["n_possible"].sum() == 6


def test_datafolio_preserves_analysis_and_source_provenance(delta_path, tmp_path):
    st = open_table(delta_path).filter(pl.col("pre") == 10)
    folio = tmp_path / "analysis"
    st.save(folio)
    restored = SynapseTable.load(folio)
    assert_frame_equal(st.df, restored.df)
    assert restored.metadata == st.metadata


def test_missing_dependency_is_actionable(synapses):
    with patch.dict(sys.modules, {"deltalake": None}):
        with pytest.raises(ImportError, match=r"trajan\[delta\]"):
            open_table("unused")
        with pytest.raises(ImportError, match=r"trajan\[delta\]"):
            table(synapses).write_synapses_delta("unused")


def test_invalid_mode_rejected_before_write(synapses, tmp_path):
    target = tmp_path / "unused"
    with pytest.raises(ValueError, match="mode must"):
        table(synapses).write_synapses_delta(target, mode="ignore")
    assert not target.exists()
