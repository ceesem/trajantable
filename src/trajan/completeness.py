"""Universe completeness: how much of the proofread cells' output lands on universe cells.

A proofread axon's synapses can land on segments that are not cells of the universe for
three reasons: the postsynaptic soma lies outside the imaged volume, the cell is inside but
missing from the universe (segmentation errors), or the synapse sits on a detached spine
head of a universe cell. The first is a property of the volume's shape and is handled by
the :mod:`edge_correction`; the other two are losses of cells that *should* have been
measured, and they differ between datasets.

Far from the lateral edge of the volume (farther than a dendritic tree reaches) no
postsynaptic dendrite can belong to an outside cell, so the share of synapses landing on
universe cells measured there is the loss to the second and third causes alone. It is
measured per presynaptic class, because where a class's synapses land (spines vs shafts)
sets how many are lost to detached spines. Two datasets' interior shares give a
ratiometric calibration (``reference ÷ own``) for estimates that count universe cells.

Typical use::

    box = lateral_box(synapse_positions, "ctr_pt_position")   # in the synapse frame
    curve = universe_share_by_edge_distance(synapses, cells, box, pre_col=..., post_col=...,
                                            position_col="ctr_pt_position")
    interior_share(curve, edge_distance=150_000)              # same units as the frame
"""

from __future__ import annotations

import polars as pl

from .edge_correction import LateralBox
from .spatial import position_is_valid


def edge_distance_expr(position_col: str, box: LateralBox) -> pl.Expr:
    """Distance from a position struct to the nearest lateral edge of ``box``.

    Negative outside the box. Null where the position is missing. Same units as the
    positions.
    """
    p = pl.col(position_col)
    a, b = p.struct.field(box.axes[0]), p.struct.field(box.axes[1])
    d = pl.min_horizontal(a - box.lo[0], box.hi[0] - a, b - box.lo[1], box.hi[1] - b)
    return pl.when(position_is_valid(position_col)).then(d).otherwise(None)


def universe_share_by_edge_distance(
    synapses: pl.LazyFrame | pl.DataFrame,
    cells: pl.DataFrame | pl.LazyFrame,
    box: LateralBox,
    *,
    pre_col: str,
    post_col: str,
    position_col: str,
    cell_id_col: str = "root_id",
    proofread_col: str = "is_proofread",
    by: str = "class",
    bin: float = 50.0,
) -> pl.DataFrame:
    """The share of proofread cells' synapses landing on universe cells, by distance from the edge.

    Parameters
    ----------
    synapses : LazyFrame or DataFrame
        One row per synapse with ``pre_col``, ``post_col`` (cell ids) and ``position_col``,
        a position struct (see :func:`pack_position`) in the same frame as ``box``.
    cells : DataFrame or LazyFrame
        The cell universe: ``cell_id_col``, a boolean ``proofread_col`` and the class
        column ``by``.
    box : LateralBox
        The volume's lateral footprint in the synapse frame (:func:`lateral_box` over the
        synapse positions themselves works well).
    pre_col, post_col, position_col : str
        Synapse column names.
    cell_id_col, proofread_col, by : str, optional
        Cell column names. ``by`` is the presynaptic class the curve is split by.
    bin : float, optional
        Width of the distance bins, in the frame's units.

    Returns
    -------
    DataFrame
        Columns ``by`` (the presynaptic class), ``d_bin`` (lower edge of the distance bin,
        negative for synapses outside the box), ``n`` (synapses) and ``share`` (fraction
        landing on universe cells). Sorted by class and distance.
    """
    syn = synapses.lazy() if isinstance(synapses, pl.DataFrame) else synapses
    cl = cells.lazy() if isinstance(cells, pl.DataFrame) else cells
    universe = cl.select(
        pl.col(cell_id_col).alias("__post"), pl.lit(True).alias("__in_universe")
    )
    proofread = cl.filter(pl.col(proofread_col)).select(
        pl.col(cell_id_col).alias("__pre"), pl.col(by).alias(by)
    )
    return (
        syn.select(
            pl.col(pre_col).alias("__pre"),
            pl.col(post_col).alias("__post"),
            edge_distance_expr(position_col, box).alias("__d"),
        )
        .join(proofread, on="__pre", how="inner")
        .join(universe, on="__post", how="left")
        .filter(pl.col("__d").is_not_null())
        .with_columns(
            pl.col("__in_universe").fill_null(False),
            ((pl.col("__d") / bin).floor() * bin).alias("d_bin"),
        )
        .group_by(by, "d_bin")
        .agg(pl.len().alias("n"), pl.col("__in_universe").mean().alias("share"))
        .sort(by, "d_bin")
        .collect()
    )


def interior_share(
    curve: pl.DataFrame, edge_distance: float, *, by: str = "class"
) -> pl.DataFrame:
    """Summarise a curve from :func:`universe_share_by_edge_distance`.

    ``interior_share`` is the synapse-weighted share over bins starting at or beyond
    ``edge_distance`` (so, far enough from the edge that no postsynaptic soma can lie
    outside); ``share_all`` is over every bin. ``n_interior`` and ``n_total`` are the
    synapse counts behind them. One row per class.
    """
    weighted = (pl.col("share") * pl.col("n")).sum() / pl.col("n").sum()
    inner = (
        curve.filter(pl.col("d_bin") >= edge_distance)
        .group_by(by)
        .agg(weighted.alias("interior_share"), pl.col("n").sum().alias("n_interior"))
    )
    total = curve.group_by(by).agg(
        weighted.alias("share_all"), pl.col("n").sum().alias("n_total")
    )
    return total.join(inner, on=by, how="left").sort(by)
