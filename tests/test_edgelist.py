"""Tests for EdgeList (Tier 1, subclass of ConnectivityTable).

Covers direct construction, inheritance of ConnectivityTable behavior
(Liskov check via a representative sample of parent tests), and the
cell-specific additions: ``filter_by_ids``, ``filter_by_soma_distance``,
``filter_by_bbox``, and ``aggregate_to_type`` (tier promotion).
"""

import math

import polars as pl
import pytest

from trajan import ConnectivityTable, EdgeList
from trajan.spatial import pack_position


@pytest.fixture
def pair_frame():
    return pl.DataFrame(
        {
            "pre": [1, 1, 2, 2, 3],
            "post": [10, 11, 10, 11, 11],
            "n_syn": [3, 1, 2, 4, 5],
        }
    )


@pytest.fixture
def el(pair_frame):
    return EdgeList(pair_frame, pre_col="pre", post_col="post")


# ── inheritance / Liskov sanity ───────────────────────────────────────────────


def test_is_a_connectivity_table(el):
    assert isinstance(el, ConnectivityTable)


def test_inherits_pairs_property(el):
    assert len(el.df) == 5


def test_inherits_normalize(el):
    out = el.normalize(by="post")
    totals = out.df.group_by("post").agg(pl.sum("fraction").alias("s"))
    for s in totals["s"].to_list():
        assert math.isclose(s, 1.0, rel_tol=1e-9)


def test_inherits_binarize(el):
    out = el.binarize(threshold=2)
    vals = out.df.sort("pre", "post")["n_syn"].to_list()
    assert vals == [1, 0, 0, 1, 1]


def test_inherits_to_dense(el):
    mat = el.to_dense()
    assert mat.shape == (3, 3)  # 3 pre, 1 pre_col + 2 post columns


# ── shared lazy/query surface (from _CachedTable / _LazyBacked) ────────────────


def test_inherits_lazy_surface(el):
    """The lazy escape hatches added to the base are available on EdgeList."""
    assert isinstance(el.lazy, pl.LazyFrame)
    assert isinstance(el.select(["pre"]), pl.LazyFrame)
    from polars.lazyframe.group_by import LazyGroupBy

    assert isinstance(el.group_by("pre"), LazyGroupBy)


def test_inherits_count_and_len(el):
    assert el.count() == len(el) == 5
    assert el.filter(pl.col("pre") == 1).count() == 2


def test_lazy_group_by_does_not_cache(el):
    out = el.group_by("pre").agg(pl.col("n_syn").sum()).sort("pre").collect()
    assert out["n_syn"].to_list() == [4, 6, 5]
    assert el._cache is None  # escape hatch never materialized .df


# ── filter preserves EdgeList type ───────────────────────────────────────────


def test_filter_returns_edgelist(el):
    """filter() returning an EdgeList preserves the cell-axis invariant."""
    filtered = el.filter(pl.col("n_syn") >= 3)
    assert isinstance(filtered, EdgeList)
    assert len(filtered.df) == 3


# ── filter_by_ids ────────────────────────────────────────────────────────────


def test_filter_by_ids_pre_only(el):
    out = el.filter_by_ids(pre_ids=[1, 2])
    assert isinstance(out, EdgeList)
    assert out.df["pre"].to_list() == [1, 1, 2, 2]


def test_filter_by_ids_post_only(el):
    out = el.filter_by_ids(post_ids=[10])
    assert out.df["post"].to_list() == [10, 10]


def test_filter_by_ids_both(el):
    out = el.filter_by_ids(pre_ids=[1], post_ids=[11])
    assert len(out.df) == 1
    assert out.df["pre"].item() == 1
    assert out.df["post"].item() == 11


def test_filter_by_ids_neither_returns_copy(el):
    out = el.filter_by_ids()
    assert isinstance(out, EdgeList)
    assert len(out.df) == 5


# ── filter_by_soma_distance ──────────────────────────────────────────────────


@pytest.fixture
def el_with_positions(pair_frame):
    """EdgeList with soma positions registered via an annotation, producing
    position struct columns on both pre and post sides."""
    pos = pl.DataFrame(
        {
            "cid": [1, 2, 3, 10, 11],
            "soma_x": [0.0, 100.0, 0.0, 0.0, 500.0],
            "soma_y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "soma_z": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    pos_packed = pack_position(pos, "soma", x="soma_x", y="soma_y", z="soma_z")
    el = EdgeList(pair_frame, pre_col="pre", post_col="post")
    el.add_annotation("pos", pos_packed, cell_id_col="cid", position_col="soma")
    return el


def test_filter_by_soma_distance(el_with_positions):
    """Only pairs with soma-soma distance <= 200 should remain.

    Distances (pre->post):
      1->10 = 0, 1->11 = 500, 2->10 = 100, 2->11 = 400, 3->11 = 500
    With max=200: keep (1,10), (2,10). Two rows.
    """
    out = el_with_positions.filter_by_soma_distance(200.0)
    assert isinstance(out, EdgeList)
    assert len(out.df) == 2


def test_filter_by_soma_distance_no_position_annotation_raises(pair_frame):
    el = EdgeList(pair_frame, pre_col="pre", post_col="post")
    with pytest.raises(ValueError, match="No annotation with a position_col"):
        el.filter_by_soma_distance(100.0)


def test_filter_by_soma_distance_ambiguous_raises(pair_frame):
    pos1 = pl.DataFrame(
        {"cid": [1, 2, 3, 10, 11], "a_x": [0.0] * 5, "a_y": [0.0] * 5, "a_z": [0.0] * 5}
    )
    pos2 = pl.DataFrame(
        {"cid": [1, 2, 3, 10, 11], "b_x": [0.0] * 5, "b_y": [0.0] * 5, "b_z": [0.0] * 5}
    )
    a_packed = pack_position(pos1, "a", x="a_x", y="a_y", z="a_z")
    b_packed = pack_position(pos2, "b", x="b_x", y="b_y", z="b_z")
    el = EdgeList(pair_frame, pre_col="pre", post_col="post")
    el.add_annotation("a", a_packed, cell_id_col="cid", position_col="a")
    el.add_annotation("b", b_packed, cell_id_col="cid", position_col="b")
    with pytest.raises(ValueError, match="Multiple annotations carry positions"):
        el.filter_by_soma_distance(100.0)
    # disambiguating works
    out = el.filter_by_soma_distance(100.0, annotation="a")
    assert isinstance(out, EdgeList)


# ── filter_by_bbox ───────────────────────────────────────────────────────────


def test_filter_by_bbox(el_with_positions):
    """Bbox keeps pairs where BOTH somas are inside.

    With the positions above and bbox ((-10,-10,-10), (150,10,10)):
      - cells 1 (x=0), 2 (x=100), 3 (x=0), 10 (x=0) are inside;
      - cell 11 (x=500) is outside.
    So pairs missing cell 11 survive: (1,10), (2,10). Two rows.
    """
    bbox = ((-10.0, -10.0, -10.0), (150.0, 10.0, 10.0))
    out = el_with_positions.filter_by_bbox(bbox)
    assert isinstance(out, EdgeList)
    assert len(out.df) == 2


# ── aggregate_to_type ────────────────────────────────────────────────────────


def test_aggregate_to_type_both_sides(el):
    """Collapse both axes to a label column. Result is a ConnectivityTable
    (not an EdgeList — at least one axis is no longer a cell id)."""
    # attach a cell-type label for each cell via an annotation
    types = pl.DataFrame(
        {
            "cid": [1, 2, 3, 10, 11],
            "type": ["exc", "exc", "inh", "exc", "inh"],
        }
    )
    el.add_annotation("types", types, cell_id_col="cid")
    ct = el.aggregate_to_type(pre="type_pre", post="type_post")
    assert isinstance(ct, ConnectivityTable)
    assert not isinstance(ct, EdgeList)
    # four type pairs possible; three appear in the data:
    # exc -> exc: (1,10)=3 → sum 3
    # exc -> inh: (1,11), (2,11) → 1+4 = 5
    # inh -> inh: (3,11) → 5
    # exc -> inh overlap from (2,10) = exc->exc → add 2 more. So exc->exc = 3+2 = 5
    df = ct.df.sort("type_pre", "type_post")
    sums = {
        (row["type_pre"], row["type_post"]): row["n_syn"]
        for row in df.iter_rows(named=True)
    }
    assert sums[("exc", "exc")] == 5
    assert sums[("exc", "inh")] == 5
    assert sums[("inh", "inh")] == 5


def test_aggregate_to_type_keeps_surviving_pre_annotation(el):
    """Collapsing only the post axis leaves pre as a cell axis, so a cell
    annotation on the pre side survives — re-registered one-sided."""
    props = pl.DataFrame(
        {
            "cid": [1, 2, 3, 10, 11],
            "region": ["A", "B", "A", "X", "Y"],
            "type": ["exc", "exc", "inh", "exc", "inh"],
        }
    )
    el.add_annotation("props", props, cell_id_col="cid")
    ct = el.aggregate_to_type(post="type_post")

    # pre axis is still the cell id; post axis is now a type label.
    assert ct.pre_col == "pre"
    assert ct.post_col == "type_post"
    # The surviving annotation came across, now one-sided on pre.
    assert "props" in ct.annotations
    assert ct._cell_annotations["props"].side == "pre"
    cols = ct.df.columns
    assert "region_pre" in cols and "type_pre" in cols
    # No post-side annotation columns: the post axis is a label, not a cell.
    assert "region_post" not in cols and "type_post_post" not in cols
    # Values are correct: pre cell 1 is region A.
    row = ct.df.filter(pl.col("pre") == 1).row(0, named=True)
    assert row["region_pre"] == "A"


def test_aggregate_to_type_keeps_surviving_post_annotation(el):
    """Symmetric case: collapsing only the pre axis keeps post-side annotations.

    ``region`` is used as the collapse axis; ``tag`` is the tracked payload —
    kept distinct so we can tell the annotation's post column apart from the
    new axis column itself.
    """
    props = pl.DataFrame(
        {
            "cid": [1, 2, 3, 10, 11],
            "region": ["A", "B", "A", "X", "Y"],
            "tag": ["p", "q", "r", "s", "t"],
        }
    )
    el.add_annotation("props", props, cell_id_col="cid")
    ct = el.aggregate_to_type(pre="region_pre")

    assert ct.pre_col == "region_pre"
    assert ct.post_col == "post"
    assert ct._cell_annotations["props"].side == "post"
    cols = ct.df.columns
    assert "tag_post" in cols
    assert "tag_pre" not in cols


def test_aggregate_to_type_both_collapsed_drops_annotations(el):
    """When both axes collapse to labels, no cell axis survives, so cell
    annotations are dropped (the old behavior, now scoped to this case)."""
    props = pl.DataFrame(
        {"cid": [1, 2, 3, 10, 11], "type": ["exc", "exc", "inh", "exc", "inh"]}
    )
    el.add_annotation("props", props, cell_id_col="cid")
    ct = el.aggregate_to_type(pre="type_pre", post="type_post")
    assert ct.annotations == {} or "props" not in ct._cell_annotations


def test_aggregate_to_type_materializes_aliased_annotation(el):
    """Aliased annotations on a surviving axis are materialized (baked into
    root-keyed specs) rather than dropped, so their columns come across."""
    types = pl.DataFrame(
        {"cid": [1, 2, 3, 10, 11], "type": ["exc", "exc", "inh", "exc", "inh"]}
    )
    el.add_annotation("types", types, cell_id_col="cid")
    el.set_cell_alias("types", "type", alias_name="ct_alias")
    extra = pl.DataFrame({"type": ["exc", "inh"], "score": [0.1, 0.9]})
    el.add_annotation("scored", extra, cell_id_col="type", join_on_alias="ct_alias")

    ct = el.aggregate_to_type(post="type_post")
    # Both annotations survive on the pre axis; the aliased one is now root-keyed.
    assert ct._cell_annotations["types"].side == "pre"
    scored = ct._cell_annotations["scored"]
    assert scored.side == "pre"
    assert scored.join_on_alias is None
    # The score rides through: pre cell 3 is 'inh' -> score 0.9.
    row = ct.df.filter(pl.col("pre") == 3).row(0, named=True)
    assert row["score_pre"] == 0.9
    assert "score_post" not in ct.df.columns


def test_materialize_aliases_equivalent_df(el):
    """materialize_aliases() bakes aliases into root-keyed annotations while
    leaving the materialized .df identical to the original."""
    types = pl.DataFrame(
        {"cid": [1, 2, 3, 10, 11], "type": ["exc", "exc", "inh", "exc", "inh"]}
    )
    el.add_annotation("types", types, cell_id_col="cid")
    el.set_cell_alias("types", "type", alias_name="ct_alias")
    extra = pl.DataFrame({"type": ["exc", "inh"], "score": [0.1, 0.9]})
    el.add_annotation("scored", extra, cell_id_col="type", join_on_alias="ct_alias")

    baked = el.materialize_aliases()
    assert baked.cell_aliases == {}
    assert baked._cell_annotations["scored"].join_on_alias is None
    # Same data, both sides still present (default side="both").
    before = el.df.sort("pre", "post")
    after = baked.df.sort("pre", "post")
    for col in ("score_pre", "score_post", "type_pre", "type_post"):
        assert before[col].to_list() == after[col].to_list()


def test_materialize_aliases_noop_without_aliases(el):
    types = pl.DataFrame({"cid": [1, 2, 3, 10, 11], "type": ["a", "b", "c", "d", "e"]})
    el.add_annotation("types", types, cell_id_col="cid")
    baked = el.materialize_aliases()
    assert baked._cell_annotations["types"].join_on_alias is None
    assert (
        baked.df.sort("pre", "post")["type_pre"].to_list()
        == el.df.sort("pre", "post")["type_pre"].to_list()
    )


def test_aggregate_to_type_carry_forward_round_trips(el, tmp_path):
    """A carried-forward one-sided annotation survives save/load with its side."""
    props = pl.DataFrame(
        {
            "cid": [1, 2, 3, 10, 11],
            "region": ["A", "B", "A", "X", "Y"],
            "tag": ["p", "q", "r", "s", "t"],
        }
    )
    el.add_annotation("props", props, cell_id_col="cid")
    ct = el.aggregate_to_type(post="region_post")

    folio = tmp_path / "ct_folio"
    ct.save(folio)
    loaded = ConnectivityTable.load(folio)
    assert loaded._cell_annotations["props"].side == "pre"
    assert "tag_pre" in loaded.df.columns
    assert "tag_post" not in loaded.df.columns


def test_aggregate_to_type_requires_axis(el):
    with pytest.raises(ValueError, match="at least one of pre/post"):
        el.aggregate_to_type()


def test_aggregate_to_type_requires_weight(pair_frame):
    """If no weights are registered and none passed, aggregate_to_type fails."""
    df = pl.DataFrame({"pre": [1], "post": [2], "strength": [0.5]})
    el = EdgeList(df, pre_col="pre", post_col="post")  # no n_syn → no auto weight
    types = pl.DataFrame({"cid": [1, 2], "t": ["a", "b"]})
    el.add_annotation("types", types, cell_id_col="cid")
    with pytest.raises(ValueError, match="weight"):
        el.aggregate_to_type(pre="t_pre", post="t_post")


# ── persistence: type is preserved across save/load ─────────────────────────


def test_edgelist_saves_and_loads_as_edgelist(el, tmp_path):
    """EdgeList round-trips as EdgeList, not demoted to ConnectivityTable."""
    import datafolio

    folio = datafolio.DataFolio(str(tmp_path / "folio"))
    el.save(folio)

    via_edgelist = EdgeList.load(folio)
    assert isinstance(via_edgelist, EdgeList)
    assert via_edgelist.df.sort("pre", "post").equals(el.df.sort("pre", "post"))


def test_connectivitytable_load_dispatches_to_edgelist(el, tmp_path):
    """ConnectivityTable.load() on a folio saved as EdgeList returns an EdgeList."""
    import datafolio

    folio = datafolio.DataFolio(str(tmp_path / "folio"))
    el.save(folio)

    via_parent = ConnectivityTable.load(folio)
    assert isinstance(via_parent, EdgeList)  # dispatched, not demoted


def test_edgelist_load_on_plain_connectivity_table_raises(tmp_path, pair_frame):
    """Loading a plain ConnectivityTable as EdgeList raises — the cell-axis
    invariant can't be guaranteed for arbitrary ConnectivityTable data."""
    import datafolio

    ct = ConnectivityTable(pair_frame, pre_col="pre", post_col="post")
    folio = datafolio.DataFolio(str(tmp_path / "folio"))
    ct.save(folio)

    with pytest.raises(TypeError, match="ConnectivityTable"):
        EdgeList.load(folio)


def test_edgelist_save_load_preserves_annotations_and_filter(el, tmp_path):
    """Annotations, weights, and filters survive the EdgeList round-trip."""
    import datafolio

    types = pl.DataFrame(
        {"cid": [1, 2, 3, 10, 11], "cell_type": ["a", "b", "c", "d", "e"]}
    )
    el.add_annotation("types", types, cell_id_col="cid")
    filtered = el.filter(pl.col("n_syn") >= 3)

    folio = datafolio.DataFolio(str(tmp_path / "folio"))
    filtered.save(folio)
    loaded = EdgeList.load(folio)

    assert isinstance(loaded, EdgeList)
    assert "types" in loaded.annotation_names
    assert len(loaded.df) == 3
    assert "cell_type_pre" in loaded.df.columns


def test_edgelist_cell_specific_ops_after_load(el, tmp_path):
    """After loading, cell-specific operations (filter_by_ids) still work —
    confirming the restored object is a real EdgeList, not a ConnectivityTable
    with a class-tag sticker."""
    import datafolio

    folio = datafolio.DataFolio(str(tmp_path / "folio"))
    el.save(folio)
    loaded = EdgeList.load(folio)

    out = loaded.filter_by_ids(pre_ids=[1])
    assert isinstance(out, EdgeList)
    assert set(out.df["pre"].to_list()) == {1}


# ── filter side-classification (Liskov: behavior inherited from CT) ──────────


def test_filter_sides_on_edgelist(el):
    """EdgeList inherits filter side-classification from ConnectivityTable;
    cell-side filters are tracked, weight filters classify as None."""
    types = pl.DataFrame({"cid": [1, 2, 3, 10, 11], "kind": list("abcde")})
    el.add_annotation("types", types, cell_id_col="cid")
    f = el.filter(pl.col("kind_pre") == "a").filter(pl.col("n_syn") >= 2)
    assert f.filter_sides == ["pre", None]


def test_filter_sides_round_trip_edgelist(el, tmp_path):
    """Saved EdgeList preserves filter_sides through load."""
    types = pl.DataFrame({"cid": [1, 2, 3, 10, 11], "kind": list("abcde")})
    el.add_annotation("types", types, cell_id_col="cid")
    filtered = el.filter(pl.col("kind_post") == "b").filter(pl.col("n_syn") > 0)
    folio = tmp_path / "folio"
    filtered.save(str(folio))
    loaded = EdgeList.load(str(folio))
    assert loaded.filter_sides == filtered.filter_sides == ["post", None]


# ── inherited clear_cache / preview / collect ─────────────────────────────────


def test_inherits_clear_cache(el):
    _ = el.df
    assert el._cache is not None
    assert el.clear_cache() is el
    assert el._cache is None


def test_inherits_preview_without_caching(el):
    out = el.preview(2)
    assert len(out) == 2
    assert el._cache is None


def test_inherits_narrow_collect(el):
    out = el.collect(["pre", "n_syn"])
    assert out.columns == ["pre", "n_syn"]
    assert el._cache is None
