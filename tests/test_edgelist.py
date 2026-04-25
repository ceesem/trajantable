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
    assert len(el.pairs) == 5


def test_inherits_normalize(el):
    out = el.normalize(by="post")
    totals = out.pairs.group_by("post").agg(pl.sum("fraction").alias("s"))
    for s in totals["s"].to_list():
        assert math.isclose(s, 1.0, rel_tol=1e-9)


def test_inherits_binarize(el):
    out = el.binarize(threshold=2)
    vals = out.pairs.sort("pre", "post")["n_syn"].to_list()
    assert vals == [1, 0, 0, 1, 1]


def test_inherits_to_dense(el):
    mat = el.to_dense()
    assert mat.shape == (3, 3)  # 3 pre, 1 pre_col + 2 post columns


# ── filter preserves EdgeList type ───────────────────────────────────────────


def test_filter_returns_edgelist(el):
    """filter() returning an EdgeList preserves the cell-axis invariant."""
    filtered = el.filter(pl.col("n_syn") >= 3)
    assert isinstance(filtered, EdgeList)
    assert len(filtered.pairs) == 3


# ── filter_by_ids ────────────────────────────────────────────────────────────


def test_filter_by_ids_pre_only(el):
    out = el.filter_by_ids(pre_ids=[1, 2])
    assert isinstance(out, EdgeList)
    assert out.pairs["pre"].to_list() == [1, 1, 2, 2]


def test_filter_by_ids_post_only(el):
    out = el.filter_by_ids(post_ids=[10])
    assert out.pairs["post"].to_list() == [10, 10]


def test_filter_by_ids_both(el):
    out = el.filter_by_ids(pre_ids=[1], post_ids=[11])
    assert len(out.pairs) == 1
    assert out.pairs["pre"].item() == 1
    assert out.pairs["post"].item() == 11


def test_filter_by_ids_neither_returns_copy(el):
    out = el.filter_by_ids()
    assert isinstance(out, EdgeList)
    assert len(out.pairs) == 5


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
    el.add_annotation("pos", pos_packed, entity_id_col="cid")
    return el


def test_filter_by_soma_distance(el_with_positions):
    """Only pairs with soma-soma distance <= 200 should remain.

    Distances (pre->post):
      1->10 = 0, 1->11 = 500, 2->10 = 100, 2->11 = 400, 3->11 = 500
    With max=200: keep (1,10), (2,10). Two rows.
    """
    out = el_with_positions.filter_by_soma_distance(
        200.0, pre_position_col="soma_pre", post_position_col="soma_post"
    )
    assert isinstance(out, EdgeList)
    assert len(out.pairs) == 2


# ── filter_by_bbox ───────────────────────────────────────────────────────────


def test_filter_by_bbox(el_with_positions):
    """Bbox keeps pairs where BOTH somas are inside.

    With the positions above and bbox ((-10,-10,-10), (150,10,10)):
      - cells 1 (x=0), 2 (x=100), 3 (x=0), 10 (x=0) are inside;
      - cell 11 (x=500) is outside.
    So pairs missing cell 11 survive: (1,10), (2,10). Two rows.
    """
    bbox = ((-10.0, -10.0, -10.0), (150.0, 10.0, 10.0))
    out = el_with_positions.filter_by_bbox(
        bbox, pre_position_col="soma_pre", post_position_col="soma_post"
    )
    assert isinstance(out, EdgeList)
    assert len(out.pairs) == 2


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
    el.add_annotation("types", types, entity_id_col="cid")
    ct = el.aggregate_to_type(pre="type_pre", post="type_post")
    assert isinstance(ct, ConnectivityTable)
    assert not isinstance(ct, EdgeList)
    # four type pairs possible; three appear in the data:
    # exc -> exc: (1,10)=3 → sum 3
    # exc -> inh: (1,11), (2,11) → 1+4 = 5
    # inh -> inh: (3,11) → 5
    # exc -> inh overlap from (2,10) = exc->exc → add 2 more. So exc->exc = 3+2 = 5
    df = ct.pairs.sort("type_pre", "type_post")
    sums = {
        (row["type_pre"], row["type_post"]): row["n_syn"]
        for row in df.iter_rows(named=True)
    }
    assert sums[("exc", "exc")] == 5
    assert sums[("exc", "inh")] == 5
    assert sums[("inh", "inh")] == 5


def test_aggregate_to_type_requires_axis(el):
    with pytest.raises(ValueError, match="at least one of pre/post"):
        el.aggregate_to_type()


def test_aggregate_to_type_requires_weight(pair_frame):
    """If no weights are registered and none passed, aggregate_to_type fails."""
    df = pl.DataFrame({"pre": [1], "post": [2], "strength": [0.5]})
    el = EdgeList(df, pre_col="pre", post_col="post")  # no n_syn → no auto weight
    types = pl.DataFrame({"cid": [1, 2], "t": ["a", "b"]})
    el.add_annotation("types", types, entity_id_col="cid")
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
    assert via_edgelist.pairs.sort("pre", "post").equals(el.pairs.sort("pre", "post"))


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
    el.add_annotation("types", types, entity_id_col="cid")
    filtered = el.filter(pl.col("n_syn") >= 3)

    folio = datafolio.DataFolio(str(tmp_path / "folio"))
    filtered.save(folio)
    loaded = EdgeList.load(folio)

    assert isinstance(loaded, EdgeList)
    assert "types" in loaded.annotation_names
    assert len(loaded.pairs) == 3
    assert "cell_type_pre" in loaded.pairs.columns


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
    assert set(out.pairs["pre"].to_list()) == {1}
