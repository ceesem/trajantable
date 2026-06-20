"""Tests for ConnectivityTable (Tier 2).

Covers direct construction from a pair-frame (first-class entry point — no
upstream SynapseTable required), annotation registration, normalization in
both internal and external modes, binarize, log1p, and dense pivot. Rejection
of synapse- and vertex-level annotations is verified as a construction-time
invariant.
"""

import math

import polars as pl
import pytest

from trajan import ConnectivityTable


@pytest.fixture
def pair_frame():
    """Small hand-built pair frame.

    pre -> post edges (with n_syn):
      1 -> 10 (3), 1 -> 11 (1), 2 -> 10 (2), 2 -> 11 (4), 3 -> 11 (5)
    Post totals: post=10 sums to 5, post=11 sums to 10.
    Pre totals: pre=1 sums to 4, pre=2 sums to 6, pre=3 sums to 5.
    """
    return pl.DataFrame(
        {
            "pre": [1, 1, 2, 2, 3],
            "post": [10, 11, 10, 11, 11],
            "n_syn": [3, 1, 2, 4, 5],
        }
    )


@pytest.fixture
def ct(pair_frame):
    return ConnectivityTable(pair_frame, pre_col="pre", post_col="post")


# ── construction ─────────────────────────────────────────────────────────────


def test_construct_from_dataframe(pair_frame):
    ct = ConnectivityTable(pair_frame, pre_col="pre", post_col="post")
    assert ct.pre_col == "pre"
    assert ct.post_col == "post"
    assert ct.weights == ["n_syn"]  # auto-detected
    assert len(ct.df) == 5


def test_construct_explicit_weight(pair_frame):
    df = pair_frame.with_columns((pl.col("n_syn") * 10).alias("area"))
    ct = ConnectivityTable(df, pre_col="pre", post_col="post", weight_cols=["area"])
    assert ct.weights == ["area"]


def test_construct_no_weight_when_n_syn_absent():
    df = pl.DataFrame({"pre": [1], "post": [2], "strength": [0.5]})
    ct = ConnectivityTable(df, pre_col="pre", post_col="post")
    assert ct.weights == []


def test_construct_missing_pre_col_raises(pair_frame):
    with pytest.raises(ValueError, match="not found"):
        ConnectivityTable(pair_frame, pre_col="bogus", post_col="post")


def test_construct_missing_weight_raises(pair_frame):
    with pytest.raises(ValueError, match="not found"):
        ConnectivityTable(
            pair_frame, pre_col="pre", post_col="post", weight_cols=["bogus"]
        )


# ── repr ────────────────────────────────────────────────────────────────────


def test_repr_pair_count(ct):
    r = repr(ct)
    assert "n_pairs=5" in r
    assert "pre_col='pre'" in r


# ── annotations ─────────────────────────────────────────────────────────────


def test_add_annotation_symmetric_join(ct):
    ann = pl.DataFrame(
        {
            "entity_id": [1, 2, 3, 10, 11],
            "cell_type": ["exc", "exc", "inh", "exc", "inh"],
        }
    )
    ct.add_annotation("types", ann, cell_id_col="entity_id")
    result = ct.df
    assert "cell_type_pre" in result.columns
    assert "cell_type_post" in result.columns
    assert result.filter(pl.col("pre") == 3)["cell_type_pre"].item() == "inh"
    assert result.filter(pl.col("post") == 11).head(1)["cell_type_post"].item() == "inh"


def test_add_annotation_side_pre_only(ct):
    """side='pre' joins only the pre axis: no _post column is produced, and
    no collision arises if another annotation later claims the post side."""
    ann = pl.DataFrame(
        {
            "entity_id": [1, 2, 3, 10, 11],
            "cell_type": ["exc", "exc", "inh", "exc", "inh"],
        }
    )
    ct.add_annotation("types", ann, cell_id_col="entity_id", side="pre")
    result = ct.df
    assert "cell_type_pre" in result.columns
    assert "cell_type_post" not in result.columns
    # _current_columns reflects only the produced side.
    assert "cell_type_pre" in ct._current_columns()
    assert "cell_type_post" not in ct._current_columns()


def test_add_annotation_side_post_only(ct):
    ann = pl.DataFrame(
        {"entity_id": [1, 2, 3, 10, 11], "cell_type": ["a", "b", "c", "d", "e"]}
    )
    ct.add_annotation("types", ann, cell_id_col="entity_id", side="post")
    cols = ct.df.columns
    assert "cell_type_post" in cols
    assert "cell_type_pre" not in cols


def test_add_annotation_one_sided_avoids_collision(ct):
    """Two pre/post-disjoint one-sided annotations on the same column name do
    not collide (one owns _pre, the other _post)."""
    ann_pre = pl.DataFrame(
        {"eid": [1, 2, 3, 10, 11], "cell_type": ["a", "b", "c", "d", "e"]}
    )
    ann_post = pl.DataFrame(
        {"eid": [1, 2, 3, 10, 11], "cell_type": ["v", "w", "x", "y", "z"]}
    )
    ct.add_annotation("pre_types", ann_pre, cell_id_col="eid", side="pre")
    ct.add_annotation("post_types", ann_post, cell_id_col="eid", side="post")
    cols = ct.df.columns
    assert "cell_type_pre" in cols and "cell_type_post" in cols


def test_add_annotation_invalid_side_raises(ct):
    ann = pl.DataFrame({"eid": [1, 2, 3, 10, 11], "x": ["a", "b", "c", "d", "e"]})
    with pytest.raises(ValueError, match="side must be"):
        ct.add_annotation("bad", ann, cell_id_col="eid", side="left")


def test_add_annotation_duplicate_key_raises(ct):
    ann = pl.DataFrame({"eid": [1, 1, 2], "x": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="duplicate"):
        ct.add_annotation("bad", ann, cell_id_col="eid")


def test_add_annotation_bad_entity_col_raises(ct):
    ann = pl.DataFrame({"eid": [1, 2], "x": ["a", "b"]})
    with pytest.raises(ValueError, match="cell_id_col"):
        ct.add_annotation("bad", ann, cell_id_col="nonexistent")


def test_add_annotation_column_collision_raises(ct):
    """Registering two annotations that both produce the same _pre/_post
    column name raises on the second registration."""
    ann_first = pl.DataFrame({"eid": [1, 2, 3], "cell_type": ["a", "b", "c"]})
    ann_second = pl.DataFrame({"eid": [1, 2, 3], "cell_type": ["x", "y", "z"]})
    ct.add_annotation("first", ann_first, cell_id_col="eid")
    with pytest.raises(ValueError, match="already exist"):
        ct.add_annotation("second", ann_second, cell_id_col="eid")


def test_reject_synapse_annotation(ct):
    with pytest.raises(TypeError, match="synapse-level"):
        ct.add_synapse_annotation("x")


def test_reject_vertex_annotation(ct):
    with pytest.raises(TypeError, match="vertex-level"):
        ct.add_vertex_annotation("x")


def test_remove_annotation(ct):
    ann = pl.DataFrame({"eid": [1, 2, 3, 10, 11], "t": ["a", "b", "c", "d", "e"]})
    ct.add_annotation("foo", ann, cell_id_col="eid")
    assert "foo" in ct.annotation_names
    ct.remove_annotation("foo")
    assert "foo" not in ct.annotation_names


# ── filter ──────────────────────────────────────────────────────────────────


def test_filter_returns_new_table(ct):
    filtered = ct.filter(pl.col("n_syn") >= 3)
    assert len(filtered.df) == 3
    # original is unchanged
    assert len(ct.df) == 5


# ── normalize — internal axis sum ───────────────────────────────────────────


def test_normalize_by_post_internal(ct):
    """Each post column should sum to 1 after normalize(by='post')."""
    out = ct.normalize(by="post")
    df = out.df
    assert "fraction" in df.columns
    assert "n_syn" not in df.columns  # replaced
    # per-post totals of fraction
    totals = df.group_by("post").agg(pl.sum("fraction").alias("s")).sort("post")
    for s in totals["s"].to_list():
        assert math.isclose(s, 1.0, rel_tol=1e-9)


def test_normalize_by_pre_internal(ct):
    """Each pre row should sum to 1 after normalize(by='pre')."""
    out = ct.normalize(by="pre")
    df = out.df
    totals = df.group_by("pre").agg(pl.sum("fraction").alias("s")).sort("pre")
    for s in totals["s"].to_list():
        assert math.isclose(s, 1.0, rel_tol=1e-9)


def test_normalize_drops_from_weight_list(ct):
    out = ct.normalize(by="post")
    assert "n_syn" not in out.weights
    assert out.weights == []


def test_normalize_bad_by(ct):
    with pytest.raises(ValueError, match="'pre' or 'post'"):
        ct.normalize(by="diagonal")


# ── normalize — external total_col ──────────────────────────────────────────


def test_normalize_external_total_col():
    """External mode divides by the per-row value of a user-supplied column,
    without interpreting its meaning. This exercises the Drosophila-style
    input-fraction pattern where the true cell total lives elsewhere."""
    df = pl.DataFrame(
        {
            "pre": [1, 1, 2],
            "post": [10, 11, 10],
            "n_syn": [3, 1, 2],
            # user-supplied "true total input" per row — trajan does not
            # interpret what this means, only divides by it
            "post_total_input": [100, 50, 100],
        }
    )
    ct = ConnectivityTable(df, pre_col="pre", post_col="post")
    out = ct.normalize(by="post", total_col="post_total_input")
    result = out.df.sort("pre", "post")
    # row (pre=1, post=10): 3 / 100 = 0.03
    # row (pre=1, post=11): 1 / 50  = 0.02
    # row (pre=2, post=10): 2 / 100 = 0.02
    expected = [0.03, 0.02, 0.02]
    for got, want in zip(result["fraction"].to_list(), expected):
        assert math.isclose(got, want, rel_tol=1e-9)


def test_normalize_external_missing_total_col_raises(ct):
    with pytest.raises(ValueError, match="total_col"):
        ct.normalize(by="post", total_col="missing")


# ── binarize ────────────────────────────────────────────────────────────────


def test_binarize_default(ct):
    out = ct.binarize()  # threshold=0, all weights > 0 → all 1
    assert set(out.df["n_syn"].to_list()) == {1}


def test_binarize_threshold(ct):
    out = ct.binarize(threshold=2)  # weights 3,1,2,4,5 → > 2: 3,4,5 → 1,0,0,1,1
    vals = out.df.sort("pre", "post")["n_syn"].to_list()
    assert vals == [1, 0, 0, 1, 1]


# ── log1p ───────────────────────────────────────────────────────────────────


def test_log1p(ct):
    out = ct.log1p()
    df = out.df
    assert "log1p_n_syn" in df.columns
    # n_syn=3 → log1p(3) = ln(4)
    row = df.filter((pl.col("pre") == 1) & (pl.col("post") == 10))
    assert math.isclose(row["log1p_n_syn"].item(), math.log1p(3), rel_tol=1e-9)
    # log1p output is NOT a weight (summing log1p isn't meaningful)
    assert "log1p_n_syn" not in out.weights
    assert out.weights == []


# ── to_dense ────────────────────────────────────────────────────────────────


def test_to_dense_shape_and_values(ct):
    mat = ct.to_dense()
    # should have 3 rows (pre=1,2,3) and 1 + 2 columns (pre_col + 2 post ids)
    assert mat.shape == (3, 3)
    assert "pre" in mat.columns
    # value for (pre=1, post=10) should be 3
    row = mat.filter(pl.col("pre") == 1)
    # post columns are stringified by pl.pivot
    post_cols = [c for c in mat.columns if c != "pre"]
    assert len(post_cols) == 2
    # Pick the first post column and confirm at least one numeric value
    assert sum(int(v) for v in row.select(post_cols).row(0)) == 4  # 3 + 1


def test_to_dense_fill_value(ct):
    mat = ct.to_dense(fill_value=-1)
    # the (pre=3, post=10) pair is absent → should be fill_value
    row = mat.filter(pl.col("pre") == 3)
    post_cols = [c for c in mat.columns if c != "pre"]
    vals = list(row.select(post_cols).row(0))
    assert -1 in vals  # one missing entry


# ── weight-list management ─────────────────────────────────────────────────


def test_add_remove_weight(ct):
    ct.add_expression("ten_x", pl.col("n_syn") * 10)
    ct.add_weight("ten_x")
    assert "ten_x" in ct.weights
    ct.remove_weight("ten_x")
    assert "ten_x" not in ct.weights


def test_add_weight_missing_column(ct):
    with pytest.raises(ValueError, match="not found"):
        ct.add_weight("bogus")


# ── persistence ──────────────────────────────────────────────────────────────


def test_save_load_basic_roundtrip(ct, tmp_path):
    """Round-trip a ConnectivityTable with no annotations / filters / expressions."""
    import datafolio

    folio = datafolio.DataFolio(str(tmp_path / "folio"))
    ct.save(folio)

    loaded = ConnectivityTable.load(folio)
    assert isinstance(loaded, ConnectivityTable)
    assert loaded.pre_col == ct.pre_col
    assert loaded.post_col == ct.post_col
    assert loaded.weights == ct.weights
    # pair data equal after sort
    left = ct.df.sort("pre", "post")
    right = loaded.df.sort("pre", "post")
    assert left.equals(right)


def test_save_load_with_annotations(ct, tmp_path):
    """Registered annotations round-trip and produce the same merged pairs."""
    import datafolio

    ann = pl.DataFrame(
        {
            "entity_id": [1, 2, 3, 10, 11],
            "cell_type": ["exc", "exc", "inh", "exc", "inh"],
        }
    )
    ct.add_annotation("types", ann, cell_id_col="entity_id")

    folio = datafolio.DataFolio(str(tmp_path / "folio"))
    ct.save(folio)
    loaded = ConnectivityTable.load(folio)

    assert "types" in loaded.annotation_names
    assert (
        loaded.df.sort("pre", "post")["cell_type_pre"].to_list()
        == ct.df.sort("pre", "post")["cell_type_pre"].to_list()
    )


def test_save_load_with_filter(ct, tmp_path):
    """A registered filter survives save/load."""
    import datafolio

    filtered = ct.filter(pl.col("n_syn") >= 3)
    folio = datafolio.DataFolio(str(tmp_path / "folio"))
    filtered.save(folio)

    loaded = ConnectivityTable.load(folio)
    assert len(loaded.df) == 3
    assert (loaded.df["n_syn"] >= 3).all()


def test_save_load_with_expression(ct, tmp_path):
    """Named expressions survive save/load (binary-format round-trip)."""
    import datafolio

    ct.add_expression("double", pl.col("n_syn") * 2)
    folio = datafolio.DataFolio(str(tmp_path / "folio"))
    ct.save(folio)

    loaded = ConnectivityTable.load(folio)
    result = loaded.df.sort("pre", "post")
    expected = (ct.df.sort("pre", "post")["n_syn"] * 2).to_list()
    assert result["double"].to_list() == expected


def test_save_load_accepts_path(ct, tmp_path):
    """save()/load() accept str/Path in addition to DataFolio instances."""
    folio_path = tmp_path / "folio"
    ct.save(str(folio_path))
    loaded = ConnectivityTable.load(str(folio_path))
    assert loaded.df.sort("pre", "post").equals(ct.df.sort("pre", "post"))


def test_save_load_overwrite(ct, tmp_path):
    """save(overwrite=True) replaces an existing folio."""
    import datafolio

    folio = datafolio.DataFolio(str(tmp_path / "folio"))
    ct.save(folio)
    # Modify and re-save — must pass overwrite=True
    ct2 = ct.filter(pl.col("n_syn") >= 3)
    ct2.save(folio, overwrite=True)
    loaded = ConnectivityTable.load(folio)
    assert len(loaded.df) == 3


# ── is_universe role ─────────────────────────────────────────────────────────


def test_is_universe_default_false(ct):
    types = pl.DataFrame({"cid": [1, 2, 3, 10, 11], "type": ["a"] * 5})
    ct.add_annotation("types", types, cell_id_col="cid")
    assert ct._cell_annotations["types"].is_universe is False


def test_resolve_universe_annotation_single(ct):
    cells = pl.DataFrame({"cid": [1, 2, 3, 10, 11], "type": ["a"] * 5})
    ct.add_annotation("cells", cells, cell_id_col="cid", is_universe=True)
    assert ct._resolve_universe_annotation() == "cells"


def test_resolve_universe_annotation_zero_raises(ct):
    with pytest.raises(ValueError, match="No annotation is marked is_universe"):
        ct._resolve_universe_annotation()


def test_resolve_universe_annotation_ambiguous_raises(ct):
    a = pl.DataFrame({"cid": [1, 2, 3], "ta": ["x"] * 3})
    b = pl.DataFrame({"cid": [10, 11], "tb": ["y"] * 2})
    ct.add_annotation("a", a, cell_id_col="cid", is_universe=True)
    ct.add_annotation("b", b, cell_id_col="cid", is_universe=True)
    with pytest.raises(ValueError, match="Multiple annotations are marked is_universe"):
        ct._resolve_universe_annotation()
    assert ct._resolve_universe_annotation("a") == "a"


def test_resolve_universe_annotation_named_not_universe_raises(ct):
    cells = pl.DataFrame({"cid": [1, 2], "t": ["x"] * 2})
    ct.add_annotation("cells", cells, cell_id_col="cid")  # is_universe=False
    with pytest.raises(ValueError, match="not marked is_universe=True"):
        ct._resolve_universe_annotation("cells")


def test_is_universe_persists_through_save_load(ct, tmp_path):
    cells = pl.DataFrame({"cid": [1, 2, 3, 10, 11], "type": ["a"] * 5})
    ct.add_annotation("cells", cells, cell_id_col="cid", is_universe=True)
    folio_path = tmp_path / "folio"
    ct.save(str(folio_path))
    loaded = ConnectivityTable.load(str(folio_path))
    assert loaded._cell_annotations["cells"].is_universe is True
    assert loaded._resolve_universe_annotation() == "cells"


# ── filter side-classification (cross-tier consistency with SynapseTable) ────


def test_filter_sides_classify_at_filter_time(ct):
    """ConnectivityTable.filter() classifies each filter via the shared
    classify_by_cell_sides helper, mirroring SynapseTable's behavior."""
    types = pl.DataFrame({"cid": [1, 2, 3, 10, 11], "kind": list("abcde")})
    ct.add_annotation("types", types, cell_id_col="cid")
    pre = ct.filter(pl.col("kind_pre") == "a")
    assert pre.filter_sides == ["pre"]
    both = pre.filter((pl.col("kind_pre") == "a") & (pl.col("kind_post") == "b"))
    assert both.filter_sides == ["pre", "both"]
    pair = both.filter(pl.col("n_syn") > 2)  # non-cell-level → None
    assert pair.filter_sides == ["pre", "both", None]


def test_filter_sides_persist_through_save_load(ct, tmp_path):
    types = pl.DataFrame({"cid": [1, 2, 3, 10, 11], "kind": list("abcde")})
    ct.add_annotation("types", types, cell_id_col="cid")
    filtered = ct.filter(pl.col("kind_pre") == "a").filter(pl.col("n_syn") >= 1)
    folio = tmp_path / "folio"
    filtered.save(str(folio))
    loaded = ConnectivityTable.load(str(folio))
    assert loaded.filter_sides == filtered.filter_sides == ["pre", None]


# ── info() smoke test ────────────────────────────────────────────────────────


def test_info_returns_summary_string(ct):
    """info() mirrors SynapseTable.info(): runs without error, returns the
    summary string, and includes the core column names."""
    cells = pl.DataFrame({"cid": [1, 2, 3, 10, 11], "type": ["A"] * 5})
    ct.add_annotation("cells", cells, cell_id_col="cid", is_universe=True)
    out = ct.info()
    assert "pre_col" in out
    assert "post_col" in out
    assert "cells" in out
    assert "universe" in out


def test_annotation_data_cols_returns_dict(ct):
    """annotation_data_cols() returns {name: [data_cols]} parallel to
    SynapseTable.cell_annotation_data_cols()."""
    cells = pl.DataFrame(
        {"cid": [1, 2, 3, 10, 11], "type": ["A"] * 5, "layer": [2] * 5}
    )
    ct.add_annotation("cells", cells, cell_id_col="cid")
    data_cols = ct.annotation_data_cols()
    assert data_cols == {"cells": ["type", "layer"]}


# ── clear_cache / preview / collect ──────────────────────────────────────────


def test_clear_cache_drops_and_repopulates(ct):
    _ = ct.df
    assert ct._cache is not None
    assert ct.clear_cache() is ct
    assert ct._cache is None
    assert len(ct.df) == 5
    assert ct._cache is not None


def test_preview_limits_rows_without_caching(ct):
    out = ct.preview(2)
    assert len(out) == 2
    assert ct._cache is None


def test_collect_none_returns_cached_df(ct):
    assert ct.collect().equals(ct.df)


def test_collect_narrow_projects_without_caching(ct):
    out = ct.collect(["pre", "n_syn"])
    assert out.columns == ["pre", "n_syn"]
    assert ct._cache is None


def test_collect_accepts_single_string(ct):
    assert ct.collect("pre").columns == ["pre"]


def test_collect_unknown_column_raises(ct):
    with pytest.raises(ValueError, match="not found in table"):
        ct.collect(["bogus"])
