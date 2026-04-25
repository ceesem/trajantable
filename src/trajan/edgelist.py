"""EdgeList — Tier 1 in the trajan architecture.

An EdgeList is a ConnectivityTable whose pre/post entities are specifically
*cells*, not labels. The class inherits all the pair-level machinery
(annotation registration, filter accumulation, normalize / binarize / log1p /
to_dense, weight-list management) from ConnectivityTable, and adds the
operations that only make sense when entities are cells-as-points-in-space:
bounding-box / soma-distance filtering, id-set filtering, and tier promotion
to a ConnectivityTable via label aggregation.

See ``project_edgelist_abstraction.md`` and ``project_unified_blessed_columns.md``
for the design rationale. In particular: an ``EdgeList`` "strengthens" a
ConnectivityTable's contract (cells, not labels); operations that preserve
that invariant return an EdgeList, operations that collapse an axis to labels
return a ConnectivityTable.
"""

from __future__ import annotations

from typing import Callable, Iterable

import polars as pl

from .connectivity_table import ConnectivityTable
from .spatial import euclidean_distance


class EdgeList(ConnectivityTable):
    """A pair-level connectivity table where both axes are cells.

    Constructed either by direct frame passing (for published edgelists etc.)
    or as the output of ``SynapseTable.edgelist()``.

    The cell invariant is documented rather than statically enforced: a
    ConnectivityTable has no distinguished "kind" on its axes, so EdgeList's
    promise is semantic, not type-checked. Operations that require points-in-
    space (spatial filters) take the position-column names explicitly at call
    time; the user is responsible for having registered them.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
        One row per (pre, post) cell pair.
    pre_col : str
        Column holding pre-synaptic cell ids.
    post_col : str
        Column holding post-synaptic cell ids.
    weight_cols : list[str] or str or None, optional
        Weight columns (defaults to ``["n_syn"]`` when present).
    """

    # Overrides ConnectivityTable._TYPE_TAG so save() writes "EdgeList" and
    # ConnectivityTable.load() can dispatch back to this class when the
    # saved folio contains an EdgeList.
    _TYPE_TAG: str = "EdgeList"

    # ── cell-specific filters ──────────────────────────────────────────────

    def filter_by_ids(
        self,
        pre_ids: Iterable | None = None,
        post_ids: Iterable | None = None,
    ) -> EdgeList:
        """Keep only pairs whose pre / post cell id is in the given set(s).

        Either or both arguments may be None to leave that side unfiltered.
        Returns a new EdgeList.
        """
        if pre_ids is None and post_ids is None:
            return self._copy()
        new = self
        if pre_ids is not None:
            new = new.filter(pl.col(self._pre_col).is_in(list(pre_ids)))
        if post_ids is not None:
            new = new.filter(pl.col(self._post_col).is_in(list(post_ids)))
        return new

    def filter_by_soma_distance(
        self,
        max_distance: float,
        *,
        pre_position_col: str,
        post_position_col: str,
        distance_fn: Callable[[str, str], pl.Expr] = euclidean_distance,
    ) -> EdgeList:
        """Keep pairs whose pre / post soma-soma distance is ``<= max_distance``.

        The user supplies the position-column names explicitly — EdgeList does
        not sniff or infer them. Positions are typically brought into the plan
        via a registered cell annotation whose data columns include a position
        struct; after registration the columns appear as e.g.
        ``soma_pt_position_pre`` / ``soma_pt_position_post``.

        Parameters
        ----------
        max_distance : float
            Maximum distance to retain. Units match the position columns.
        pre_position_col : str
            Name of the pre-side position struct column (post-annotation join).
        post_position_col : str
            Name of the post-side position struct column.
        distance_fn : callable
            Takes two position-column names and returns a ``pl.Expr`` for the
            distance. Defaults to 3-D Euclidean. Use ``radial_distance`` for
            lateral-only.
        """
        return self.filter(
            distance_fn(pre_position_col, post_position_col) <= max_distance
        )

    def filter_by_bbox(
        self,
        bbox,
        *,
        pre_position_col: str,
        post_position_col: str,
    ) -> EdgeList:
        """Keep pairs where both pre and post soma positions fall inside a bbox.

        Unlike ``SynapseTable.filter_by_bbox`` (which filters on a single
        synapse position), EdgeList's bbox filter checks both cells — a pair
        is kept only if both somas lie within the box. The user supplies the
        pre / post position column names explicitly.

        Parameters
        ----------
        bbox : Sequence
            ((xmin, ymin, zmin), (xmax, ymax, zmax)).
        """
        (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox
        pre = pl.col(pre_position_col)
        post = pl.col(post_position_col)

        def _inside(p):
            return (
                (p.struct.field("x") >= xmin)
                & (p.struct.field("x") <= xmax)
                & (p.struct.field("y") >= ymin)
                & (p.struct.field("y") <= ymax)
                & (p.struct.field("z") >= zmin)
                & (p.struct.field("z") <= zmax)
            )

        return self.filter(_inside(pre) & _inside(post))

    # ── tier promotion ─────────────────────────────────────────────────────

    def aggregate_to_type(
        self,
        pre: str | None = None,
        post: str | None = None,
        weight_cols: list[str] | str | None = None,
    ) -> ConnectivityTable:
        """Collapse one or both axes from cells to a label column.

        At least one of ``pre`` / ``post`` must be given. The returned
        ConnectivityTable has the supplied label columns as its pre/post
        entity columns, and the registered weights summed per type pair
        (per the weight contract: sum of a weight is a weight).

        Non-weight columns are NOT carried across — this is a semantic
        reduction and arbitrary user columns don't have a canonical
        aggregation. The resulting type-level table is a new object; the
        source EdgeList is unchanged.

        Returns a ConnectivityTable (not an EdgeList) because at least one
        axis is now a label, violating the cell-axis invariant.

        Parameters
        ----------
        pre : str or None
            Column in the current merged plan to use as the new pre axis.
            ``None`` keeps the pre cell id (so the result is cell x label).
        post : str or None
            Column for the new post axis. ``None`` keeps the post cell id.
        weight_cols : list[str] or str or None, optional
            Weights to carry forward (summed). Defaults to all currently
            registered weights.
        """
        if pre is None and post is None:
            raise ValueError(
                "aggregate_to_type requires at least one of pre/post to be given; "
                "passing neither would return an unchanged EdgeList."
            )
        new_pre = pre if pre is not None else self._pre_col
        new_post = post if post is not None else self._post_col

        if isinstance(weight_cols, str):
            weight_cols = [weight_cols]
        if weight_cols is None:
            weight_cols = list(self._weights)
        if not weight_cols:
            raise ValueError(
                "aggregate_to_type requires at least one weight column. "
                "Pass weight_cols=... or register one via add_weight()."
            )

        lf = self.build_lazy()
        missing = [c for c in weight_cols if c not in lf.collect_schema().names()]
        if missing:
            raise ValueError(f"Weight column(s) not found in plan: {missing}")
        for c in (new_pre, new_post):
            if c not in lf.collect_schema().names():
                raise ValueError(f"Axis column {c!r} not found in plan.")

        agg = (
            lf.group_by([new_pre, new_post])
            .agg([pl.sum(w).alias(w) for w in weight_cols])
            .collect()
        )
        return ConnectivityTable(
            agg, pre_col=new_pre, post_col=new_post, weight_cols=weight_cols
        )

    # ── copy: preserve the EdgeList type through filter etc. ───────────────

    def _copy(self) -> EdgeList:
        """Produce a copy of the same type (EdgeList, not ConnectivityTable).

        ``ConnectivityTable._copy`` hardcodes ``ConnectivityTable``; overriding
        here ensures operations like ``filter`` on an EdgeList return an
        EdgeList, preserving the cell-axis invariant when it is preserved.
        """
        new = EdgeList.__new__(EdgeList)
        new._pair_lf = self._pair_lf
        new._pair_col_names = self._pair_col_names
        new._pre_col = self._pre_col
        new._post_col = self._post_col
        new._weights = self._weights.copy()
        new._annotations = self._annotations.copy()
        new._filters = self._filters.copy()
        new._expressions = self._expressions.copy()
        new._cache = None
        return new

    # Note on the shape-preserving ops inherited from ConnectivityTable:
    # ``normalize`` / ``binarize`` / ``log1p`` currently call
    # ``ConnectivityTable._replace_base`` which always returns a
    # ConnectivityTable. That is *correct* — those operations drop
    # annotations (they're baked into the lazy frame) and replace weight
    # semantics, which weakens rather than preserves the EdgeList contract.
    # If per-op EdgeList-preservation is desired later (e.g. binarize should
    # stay an EdgeList because entities are unchanged), override _replace_base
    # to preserve type for the right ops. Not done here.
