"""Volume edge correction: ring fractions, the lateral box and the pair-universe weight."""

import numpy as np
import polars as pl
import pytest

from trajan import (
    SynapseTable,
    counts,
    edge_weight,
    lateral_box,
    possible_pairs,
    ring_inside_fraction,
    with_distance,
)
from trajan.edge_correction import LateralBox
from trajan.spatial import pack_position, radial_distance

BOX = LateralBox(axes=("x", "z"), lo=(0.0, 0.0), hi=(1000.0, 1000.0))


def test_ring_fraction_centre_edge_and_outside():
    assert ring_inside_fraction(500, 500, 100, BOX) == 1.0
    # On the x edge: half of every ring is outside.
    assert ring_inside_fraction(0, 500, 100, BOX) == pytest.approx(0.5)
    # In a corner: a quarter.
    assert ring_inside_fraction(0, 0, 100, BOX) == pytest.approx(0.25, abs=0.03)
    # Far outside: nothing.
    assert ring_inside_fraction(-500, 500, 100, BOX) == 0.0
    # Broadcasts over radii.
    f = ring_inside_fraction(100, 500, np.array([50.0, 200.0]), BOX)
    assert f[0] == 1.0 and 0.5 < f[1] < 1.0


def test_lateral_box_uses_percentiles():
    rng = np.random.default_rng(0)
    cells = pl.DataFrame(
        {
            "x": rng.uniform(0, 1000, 10_000),
            "y": np.zeros(10_000),
            "z": rng.uniform(0, 500, 10_000),
        }
    )
    box = lateral_box(pack_position(cells, "soma", x="x", y="y", z="z"), "soma")
    assert box.axes == ("x", "z")
    assert box.lo[0] == pytest.approx(10, abs=5) and box.hi[0] == pytest.approx(
        990, abs=5
    )
    assert box.lo[1] == pytest.approx(5, abs=3) and box.hi[1] == pytest.approx(
        495, abs=3
    )


@pytest.fixture
def universe():
    """20k cells uniform in a 1000 × 1000 box, plus two anchors: one at the x edge, one central."""
    rng = np.random.default_rng(1)
    n = 20_000
    cells = pl.DataFrame(
        {
            "root_id": np.arange(1, n + 3),
            "soma_x": np.concatenate([rng.uniform(0, 1000, n), [20.0, 500.0]]),
            "soma_y": np.zeros(n + 2),
            "soma_z": np.concatenate([rng.uniform(0, 1000, n), [500.0, 500.0]]),
        }
    )
    synapses = pl.DataFrame(
        {"id": [1, 2], "pre_pt_root_id": [1, 2], "post_pt_root_id": [n + 1, n + 2]}
    )
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells",
        pack_position(cells, "soma", x="soma_x", y="soma_y", z="soma_z"),
        cell_id_col="root_id",
        is_universe=True,
        position_col="soma",
    )
    return st, cells, n


def test_edge_weight_restores_uniform_density(universe):
    st, cells, n = universe
    edge_anchor, centre_anchor = n + 1, n + 2
    box = LateralBox(axes=("x", "z"), lo=(0.0, 0.0), hi=(1000.0, 1000.0))
    pu = (
        possible_pairs(st)
        .filter_by_ids(post_ids=[edge_anchor, centre_anchor])
        .filter_by_radial_distance(400)
    )
    pu = with_distance(pu, "d_rho", radial_distance)
    pu = edge_weight(pu, box, anchor="post")
    assert "edge_weight" in pu.weights
    edges = [0, 100, 200, 300, 400]
    out = counts(pu, bin_by={"d_rho": edges}, group_by="post_pt_root_id")
    density = n / 1e6  # cells per µm²
    for row in out.iter_rows(named=True):
        lo, hi = [float(v) for v in str(row["d_rho_bin"]).strip("(]").split(", ")]
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        expected = density * np.pi * (hi**2 - lo**2)
        if row["post_pt_root_id"] == centre_anchor:
            # Rings fully inside: no correction, and the count matches the density.
            assert row["sum_edge_weight"] == pytest.approx(row["n_possible"])
            assert row["n_possible"] == pytest.approx(expected, rel=0.1)
        else:
            # Rings at the x edge are about half outside: observed is ~half, corrected ~full.
            assert row["n_possible"] < 0.7 * expected
            assert row["sum_edge_weight"] == pytest.approx(expected, rel=0.12)


def test_edge_weight_caps_and_validates(universe):
    st, cells, n = universe
    box = LateralBox(axes=("x", "z"), lo=(0.0, 0.0), hi=(1000.0, 1000.0))
    pu = (
        possible_pairs(st)
        .filter_by_ids(post_ids=[n + 1])
        .filter_by_radial_distance(400)
    )
    with pytest.raises(ValueError):
        edge_weight(pu, box, anchor="left")  # type: ignore[arg-type]
    pu = edge_weight(pu, box, cap=3.0)
    w = pu.collect(["edge_weight"])["edge_weight"]
    assert w.max() <= 3.0 and w.min() >= 1.0
