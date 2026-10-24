"""Universe completeness: the share of proofread output onto universe cells by edge distance."""

import polars as pl
import pytest

from trajan import LateralBox, interior_share, universe_share_by_edge_distance
from trajan.spatial import pack_position

BOX = LateralBox(axes=("x", "z"), lo=(0.0, 0.0), hi=(1000.0, 1000.0))


def test_curve_and_interior_share():
    cells = pl.DataFrame(
        {
            "root_id": [1, 2, 3],
            "is_proofread": [True, True, False],
            "class": ["exc", "inh", "exc"],
        }
    )
    # Cell 1 (exc): two interior synapses, one onto universe cell 3 and one onto orphan 99;
    # two edge synapses (x = 10), both onto orphans. Cell 2 (inh): one interior onto 3.
    syn = pl.DataFrame(
        {
            "pre": [1, 1, 1, 1, 2],
            "post": [3, 99, 99, 99, 3],
            "x": [500.0, 400.0, 10.0, 10.0, 600.0],
            "y": [0.0] * 5,
            "z": [500.0, 500.0, 500.0, 600.0, 500.0],
        }
    )
    syn = pack_position(syn, "pos", x="x", y="y", z="z")
    curve = universe_share_by_edge_distance(
        syn, cells, BOX, pre_col="pre", post_col="post", position_col="pos"
    )
    assert curve.columns == ["class", "d_bin", "n", "share"]
    exc = curve.filter(pl.col("class") == "exc")
    assert (
        exc.filter(pl.col("d_bin") == 0.0)["share"].item() == 0.0
    )  # the two edge synapses
    # 400 µm from the edge: the orphan synapse; 500 µm: the one onto cell 3.
    assert exc.filter(pl.col("d_bin") == 400.0)["share"].item() == 0.0
    assert exc.filter(pl.col("d_bin") == 500.0)["share"].item() == 1.0

    s = interior_share(curve, 150).sort("class")
    assert s["class"].to_list() == ["exc", "inh"]
    assert s["interior_share"].to_list() == [0.5, 1.0]
    assert s["n_interior"].to_list() == [2, 1]
    assert s.filter(pl.col("class") == "exc")["share_all"].item() == pytest.approx(0.25)


def test_missing_positions_are_dropped():
    cells = pl.DataFrame(
        {"root_id": [1, 3], "is_proofread": [True, False], "class": ["exc", "exc"]}
    )
    syn = pl.DataFrame(
        {
            "pre": [1, 1],
            "post": [3, 3],
            "pos": [{"x": 500.0, "y": 0.0, "z": 500.0}, None],
        }
    )
    curve = universe_share_by_edge_distance(
        syn, cells, BOX, pre_col="pre", post_col="post", position_col="pos"
    )
    assert curve["n"].sum() == 1
