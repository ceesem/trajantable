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
    assert len(ct.pairs) == 5


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
    ct.add_annotation("types", ann, entity_id_col="entity_id")
    result = ct.pairs
    assert "cell_type_pre" in result.columns
    assert "cell_type_post" in result.columns
    assert result.filter(pl.col("pre") == 3)["cell_type_pre"].item() == "inh"
    assert result.filter(pl.col("post") == 11).head(1)["cell_type_post"].item() == "inh"


def test_add_annotation_duplicate_key_raises(ct):
    ann = pl.DataFrame({"eid": [1, 1, 2], "x": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="duplicate"):
        ct.add_annotation("bad", ann, entity_id_col="eid")


def test_add_annotation_bad_entity_col_raises(ct):
    ann = pl.DataFrame({"eid": [1, 2], "x": ["a", "b"]})
    with pytest.raises(ValueError, match="entity_id_col"):
        ct.add_annotation("bad", ann, entity_id_col="nonexistent")


def test_add_annotation_column_collision_raises(ct):
    """Registering two annotations that both produce the same _pre/_post
    column name raises on the second registration."""
    ann_first = pl.DataFrame({"eid": [1, 2, 3], "cell_type": ["a", "b", "c"]})
    ann_second = pl.DataFrame({"eid": [1, 2, 3], "cell_type": ["x", "y", "z"]})
    ct.add_annotation("first", ann_first, entity_id_col="eid")
    with pytest.raises(ValueError, match="already exist"):
        ct.add_annotation("second", ann_second, entity_id_col="eid")


def test_reject_synapse_annotation(ct):
    with pytest.raises(TypeError, match="synapse-level"):
        ct.add_synapse_annotation("x")


def test_reject_vertex_annotation(ct):
    with pytest.raises(TypeError, match="vertex-level"):
        ct.add_vertex_annotation("x")


def test_remove_annotation(ct):
    ann = pl.DataFrame({"eid": [1, 2, 3, 10, 11], "t": ["a", "b", "c", "d", "e"]})
    ct.add_annotation("foo", ann, entity_id_col="eid")
    assert "foo" in ct.annotation_names
    ct.remove_annotation("foo")
    assert "foo" not in ct.annotation_names


# ── filter ──────────────────────────────────────────────────────────────────


def test_filter_returns_new_table(ct):
    filtered = ct.filter(pl.col("n_syn") >= 3)
    assert len(filtered.pairs) == 3
    # original is unchanged
    assert len(ct.pairs) == 5


# ── normalize — internal axis sum ───────────────────────────────────────────


def test_normalize_by_post_internal(ct):
    """Each post column should sum to 1 after normalize(by='post')."""
    out = ct.normalize(by="post")
    df = out.pairs
    assert "fraction" in df.columns
    assert "n_syn" not in df.columns  # replaced
    # per-post totals of fraction
    totals = df.group_by("post").agg(pl.sum("fraction").alias("s")).sort("post")
    for s in totals["s"].to_list():
        assert math.isclose(s, 1.0, rel_tol=1e-9)


def test_normalize_by_pre_internal(ct):
    """Each pre row should sum to 1 after normalize(by='pre')."""
    out = ct.normalize(by="pre")
    df = out.pairs
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
    result = out.pairs.sort("pre", "post")
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
    assert set(out.pairs["n_syn"].to_list()) == {1}


def test_binarize_threshold(ct):
    out = ct.binarize(threshold=2)  # weights 3,1,2,4,5 → > 2: 3,4,5 → 1,0,0,1,1
    vals = out.pairs.sort("pre", "post")["n_syn"].to_list()
    assert vals == [1, 0, 0, 1, 1]


# ── log1p ───────────────────────────────────────────────────────────────────


def test_log1p(ct):
    out = ct.log1p()
    df = out.pairs
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
