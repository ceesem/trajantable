"""Volume edge correction for candidate counts.

A dataset only holds the cells inside its imaged volume. Around a cell near the lateral
edge, part of every ring of radius ``rho`` lies outside the volume, so counting the
candidate partners in that ring undercounts it: the cells there were never imaged, they
are not absent. Weighting each observed candidate by ``1 / (fraction of its ring inside
the volume)`` restores the count a complete volume would have shown, under the assumption
that cells beyond the edge sit at the same density as those inside at the same distance
(Horvitz–Thompson weighting). Connection probabilities need no such correction: they are
ratios among pairs that are all inside.

Typical use, on a pair universe with soma positions registered::

    box = lateral_box(cells, "soma")                 # the volume's footprint
    pu = with_distance(pu, "d_rho", radial_distance)
    pu = edge_weight(pu, box, anchor="post")        # rings around the postsynaptic cell
    counts(pu, bin_by={"d_rho": [0, 50, 100, ...]})
    # → n_possible (observed candidates) and sum_edge_weight (edge-corrected candidates)

The depth axis is not corrected: a cortical volume spans pia to white matter, so the
fraction depends on the lateral ring alone.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import polars as pl

from .spatial import _parse_depth_axis, position_is_valid

if TYPE_CHECKING:
    from .pair_universe import PairUniverse

#: Weights beyond this are capped: a ring almost entirely outside the volume is guesswork.
DEFAULT_CAP = 10.0
#: Angles stepped around a ring when measuring the fraction inside the box.
DEFAULT_ANGLES = 36


class LateralBox(NamedTuple):
    """The volume's lateral footprint: a box over the two non-depth axes.

    ``axes`` names the lateral position fields (e.g. ``("x", "z")`` for depth along
    ``y``); ``lo`` / ``hi`` are the box bounds on each, in the position units.
    """

    axes: tuple[str, str]
    lo: tuple[float, float]
    hi: tuple[float, float]

    @property
    def size(self) -> tuple[float, float]:
        """Box extent along each lateral axis."""
        return (self.hi[0] - self.lo[0], self.hi[1] - self.lo[1])


def lateral_box(
    cells: pl.DataFrame | pl.LazyFrame,
    position_col: str,
    *,
    depth_axis: str = "y",
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> LateralBox:
    """The volume's lateral footprint from the cells' soma positions.

    The box spans the given percentiles of the positions on each lateral axis.
    Percentiles rather than min / max, because the outermost cells sit in the thinly
    segmented margins of a volume and would stretch the box past where cells are dense.

    Parameters
    ----------
    cells : DataFrame or LazyFrame
        One row per cell with a position struct column (see :func:`pack_position`).
    position_col : str
        The position struct column.
    depth_axis : str, optional
        The depth axis (``"x"``, ``"y"`` or ``"z"``, optionally ``"_r"``); the other two
        are lateral. Defaults to ``"y"``.
    percentiles : (float, float), optional
        Lower and upper percentiles defining the box. Defaults to ``(1, 99)``.
    """
    axis, _ = _parse_depth_axis(depth_axis)
    lat = tuple(sorted({"x", "y", "z"} - {axis}))
    lf = cells.lazy() if isinstance(cells, pl.DataFrame) else cells
    pos = pl.col(position_col)
    vals = (
        lf.filter(position_is_valid(position_col))
        .select([pos.struct.field(a).alias(a) for a in lat])
        .collect()
    )
    if vals.is_empty():
        raise ValueError("lateral_box: no cells with a valid position.")
    lo_q, hi_q = percentiles[0] / 100.0, percentiles[1] / 100.0
    lo = tuple(float(vals[a].quantile(lo_q, interpolation="linear")) for a in lat)
    hi = tuple(float(vals[a].quantile(hi_q, interpolation="linear")) for a in lat)
    return LateralBox(axes=(lat[0], lat[1]), lo=(lo[0], lo[1]), hi=(hi[0], hi[1]))


def ring_inside_fraction(
    a: float | np.ndarray,
    b: float | np.ndarray,
    rho: float | np.ndarray,
    box: LateralBox,
    *,
    n_angles: int = DEFAULT_ANGLES,
) -> np.ndarray:
    """Fraction of the ring of radius ``rho`` around lateral point ``(a, b)`` inside ``box``.

    Steps ``n_angles`` points around the ring and tests each against the box. Inputs
    broadcast, so one anchor with many radii or many anchors with one radius both work.
    A ring of radius 0 is the point itself (fraction 0 or 1).
    """
    a, b, rho = np.broadcast_arrays(
        np.asarray(a, float), np.asarray(b, float), np.asarray(rho, float)
    )
    theta = np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)
    pa = a[..., None] + rho[..., None] * np.cos(theta)
    pb = b[..., None] + rho[..., None] * np.sin(theta)
    inside = (
        (pa >= box.lo[0]) & (pa <= box.hi[0]) & (pb >= box.lo[1]) & (pb <= box.hi[1])
    )
    return inside.mean(axis=-1)


def ring_inside_fraction_expr(
    position_col: str,
    rho: pl.Expr,
    box: LateralBox,
    *,
    n_angles: int = DEFAULT_ANGLES,
) -> pl.Expr:
    """:func:`ring_inside_fraction` as a lazy Polars expression.

    ``position_col`` is the anchor's position struct column in the frame (for a pair
    frame, the per-side column such as ``"soma_post"``); ``rho`` is the ring radius
    expression. Null where the position is missing.
    """
    p = pl.col(position_col)
    pa, pb = p.struct.field(box.axes[0]), p.struct.field(box.axes[1])
    hits = []
    for k in range(n_angles):
        t = 2.0 * math.pi * k / n_angles
        xa = pa + rho * math.cos(t)
        xb = pb + rho * math.sin(t)
        hits.append(
            (
                (xa >= box.lo[0])
                & (xa <= box.hi[0])
                & (xb >= box.lo[1])
                & (xb <= box.hi[1])
            ).cast(pl.Float64)
        )
    frac = pl.sum_horizontal(hits) / float(n_angles)
    return (
        pl.when(position_is_valid(position_col) & rho.is_finite())
        .then(frac)
        .otherwise(None)
    )


def edge_weight(
    pu: "PairUniverse",
    box: LateralBox,
    *,
    anchor: Literal["pre", "post"] = "post",
    cap: float = DEFAULT_CAP,
    n_angles: int = DEFAULT_ANGLES,
    annotation: str | None = None,
    name: str = "edge_weight",
) -> "PairUniverse":
    """Register the volume edge-correction weight on a pair universe.

    For each pair, the ring of radius ``rho`` (the pair's lateral soma distance) around
    the ``anchor`` cell's soma is partly outside ``box``; the weight is
    ``1 / (fraction inside)``, capped at ``cap``. Registered both as a named expression
    and as a weight, so :func:`counts` reports ``sum_<name>``: the edge-corrected number
    of candidate pairs per bin, next to the observed ``n_possible``.

    Parameters
    ----------
    pu : PairUniverse
        Pair frame with a position-bearing cell annotation (``position_col=``).
    box : LateralBox
        The volume's lateral footprint, from :func:`lateral_box`.
    anchor : {"pre", "post"}, optional
        Which cell the ring is drawn around: the one whose surroundings are being
        counted. ``"post"`` for inputs to a target (candidates are presynaptic),
        ``"pre"`` for outputs of a source. Defaults to ``"post"``.
    cap : float, optional
        Maximum weight. Defaults to 10.
    n_angles : int, optional
        Angles stepped around each ring. Defaults to 36.
    annotation : str or None, optional
        Which cell annotation's position to use; auto-resolved when unique.
    name : str, optional
        Column name of the weight. Defaults to ``"edge_weight"``.

    Notes
    -----
    Pairs with a missing anchor position or distance get weight 1 (no correction), so
    they still count once in ``sum_<name>`` as they do in ``n_possible``.
    """
    if anchor not in ("pre", "post"):
        raise ValueError("anchor must be 'pre' or 'post'.")
    if not cap >= 1:
        raise ValueError("cap must be at least 1.")
    ann = pu._resolve_position_annotation(annotation)
    pos_col = pu._cell_annotations[ann].position_col
    a_col, o_col = (
        f"{pos_col}_{anchor}",
        f"{pos_col}_{'pre' if anchor == 'post' else 'post'}",
    )
    pa, po = pl.col(a_col), pl.col(o_col)
    rho = pl.sum_horizontal(
        [(pa.struct.field(ax) - po.struct.field(ax)).pow(2) for ax in box.axes]
    ).sqrt()
    frac = ring_inside_fraction_expr(a_col, rho, box, n_angles=n_angles)
    weight = (
        (1.0 / pl.max_horizontal(frac, pl.lit(1.0 / cap)))
        .clip(upper_bound=cap)
        .fill_null(1.0)
    )
    new = pu.add_expression(name, weight)
    return new.add_weight(name)
