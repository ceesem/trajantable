"""PairUniverse and the ``possible_pairs`` denominator primitive.

A ``PairUniverse`` is a lazy ``universe × universe`` pair frame with observed
counts overlaid (0 for unobserved pairs). It is the denominator primitive for
connection density, connection probability, and null-model analyses.

Unlike ``EdgeList``, ``PairUniverse`` deliberately omits a ``.df`` property:
the full cross-product can be huge (``|universe|² = 4 × 10⁹`` for a 60k-cell
universe), and accidental materialization would dwarf observed-edgelist
analyses by orders of magnitude. The intended workflow is: build via
``possible_pairs``, compose filters / aggregations to bring the row count
down, then call ``.collect()`` or ``.group_by().agg()`` to materialize the
reduced result.

See ``DESIGN-universe.md`` §4 for the full design.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterable, Union

import polars as pl

from ._base import (
    CellAnnotationSpec,
    _LazyBacked,
    build_cell_annotation_spec,
    build_pair_plan,
    classify_by_cell_sides,
    filter_by_id_sets,
    filter_by_soma_distance_impl,
    resolve_position_annotation,
    resolve_universe_annotation,
)
from .spatial import bbox_predicate, euclidean_distance, radial_distance

if TYPE_CHECKING:
    from .edgelist import EdgeList
    from .synapse_table import SynapseTable


class PairUniverse(_LazyBacked):
    """Lazy ``universe × universe`` pair frame with observed counts overlaid.

    Carries the same kinds of state as ``EdgeList`` — blessed pre / post id
    columns, a weight list, a registry of cell annotations, accumulated
    filters and named expressions, and the cell-alias registry — but does
    NOT expose ``.df``. Materialization is explicit via ``.collect()`` after
    enough filters / aggregations have been composed to bring the row count
    down.

    Construction is via :func:`possible_pairs`; direct instantiation is
    supported but rare (you generally want the observed-counts overlay).

    Parameters
    ----------
    lf : pl.LazyFrame
        Lazy plan for the base pair frame. Must already contain ``pre_col``,
        ``post_col``, and any ``weights`` columns.
    pre_col, post_col : str
        Pre / post cell id columns.
    weights : list[str] or None
        Weight columns carrying the weight contract (auto-sum semantics).
        Defaults to ``[]``.
    cell_aliases : dict[str, tuple[str, str]] or None
        Pre-populated alias registry inherited from the source table.

    Notes
    -----
    No ``.df`` and no automatic caching: every observation requires an
    explicit ``.collect()`` or ``.group_by().agg().collect()``. Density and
    probability statistics aggregate over this without ever materializing
    the cross-product.
    """

    def __init__(
        self,
        lf: pl.LazyFrame,
        *,
        pre_col: str,
        post_col: str,
        weights: list[str] | None = None,
        cell_aliases: dict[str, tuple[str, str]] | None = None,
    ):
        self._pair_lf = lf
        self._pre_col = pre_col
        self._post_col = post_col
        self._weights: list[str] = list(weights or [])
        self._cell_annotations: dict[str, CellAnnotationSpec] = {}
        self._filters: list[pl.Expr] = []
        self._filter_sides: list[str | None] = []
        self._expressions: dict[str, pl.Expr] = {}
        self._cell_aliases: dict[str, tuple[str, str]] = dict(cell_aliases or {})

    # ── read-only accessors ────────────────────────────────────────────────

    @property
    def pre_col(self) -> str:
        return self._pre_col

    @property
    def post_col(self) -> str:
        return self._post_col

    @property
    def weights(self) -> list[str]:
        return list(self._weights)

    @property
    def annotation_names(self) -> list[str]:
        return list(self._cell_annotations)

    @property
    def cell_aliases(self) -> dict[str, tuple[str, str]]:
        return dict(self._cell_aliases)

    @property
    def filter_sides(self) -> list[str | None]:
        return list(self._filter_sides)

    def __repr__(self) -> str:
        return (
            f"PairUniverse(pre_col={self._pre_col!r}, post_col={self._post_col!r}, "
            f"weights={self._weights}, annotations={list(self._cell_annotations)})"
        )

    def __bool__(self) -> bool:
        # Always truthy — avoid implicit collects. Use len(pu.collect()) to
        # count rows.
        return True

    # ── annotation registration ────────────────────────────────────────────

    def add_annotation(
        self,
        name: str,
        df,
        cell_id_col: str,
        position_col: str | None = None,
        is_universe: bool = False,
        join_on_alias: str | None = None,
    ) -> "PairUniverse":
        """Register a per-cell annotation joined symmetrically on pre and post.

        Same semantics as ``EdgeList.add_annotation`` — each data column
        produces ``{col}_pre`` and ``{col}_post`` outputs at plan-build time.
        Useful when an analysis needs an annotation that wasn't on the source
        table.
        """
        if join_on_alias is not None and join_on_alias not in self._cell_aliases:
            raise ValueError(
                f"No cell alias named {join_on_alias!r}. Register the source "
                f"annotation and call set_cell_alias() first. Registered "
                f"aliases: {list(self._cell_aliases)}"
            )
        self._cell_annotations[name] = build_cell_annotation_spec(
            df,
            cell_id_col=cell_id_col,
            position_col=position_col,
            is_universe=is_universe,
            join_on_alias=join_on_alias,
            current_columns=self._current_columns(),
        )
        return self

    def _resolve_universe_annotation(self, annotation: str | None = None) -> str:
        """Return the name of the annotation that defines the cell universe.

        Delegates to :func:`trajan._base.resolve_universe_annotation` so
        free functions (bootstrap, density / probability stats) resolve the
        universe on a PairUniverse the same way they do on the upstream tiers.
        """
        return resolve_universe_annotation(self._cell_annotations, annotation)

    def _resolve_position_annotation(self, annotation: str | None = None) -> str:
        """Return the name of the annotation whose ``position_col`` to use.

        Delegates to :func:`trajan._base.resolve_position_annotation`.
        """
        return resolve_position_annotation(self._cell_annotations, annotation)

    # ── filter / expression accumulation ──────────────────────────────────

    def _classify_expression(self, expr: pl.Expr) -> str | None:
        return classify_by_cell_sides(expr, self._cell_annotations)

    def filter(self, expr: pl.Expr) -> "PairUniverse":
        """Return a new PairUniverse with ``expr`` accumulated."""
        new = self._copy()
        new._filters = self._filters + [expr]
        new._filter_sides = self._filter_sides + [self._classify_expression(expr)]
        return new

    def add_expression(self, name: str, expr: pl.Expr) -> "PairUniverse":
        """Register a named computed column applied after joins, before filters."""
        if name in self._current_columns():
            raise ValueError(f"Column {name!r} already exists in pair universe")
        self._expressions[name] = expr.alias(name)
        return self

    def filter_by_ids(
        self,
        pre_ids: Iterable | None = None,
        post_ids: Iterable | None = None,
    ) -> "PairUniverse":
        """Restrict pre / post id sets to the given iterables."""
        return filter_by_id_sets(self, pre_ids, post_ids)

    def filter_by_radial_distance(
        self, max_distance: float, *, annotation: str | None = None
    ) -> "PairUniverse":
        """Keep pairs whose pre/post **lateral** soma distance is ``<= max_distance``.

        Lateral (radial, depth-free) distance via :func:`trajan.radial_distance`
        (``sqrt(dx² + dz²)``), ignoring the depth axis — the metric for cortical
        column / lateral-reach analyses, and the one that matches ``rho`` bins.
        Reads the position column from the annotation declared with
        ``position_col=`` (auto-resolved when unique). Prune BEFORE materializing
        — this is what keeps the cross-product tractable on whole-brain scales.
        """
        return filter_by_soma_distance_impl(
            self, max_distance, annotation, radial_distance
        )

    def filter_by_euclidean_distance(
        self, max_distance: float, *, annotation: str | None = None
    ) -> "PairUniverse":
        """Keep pairs whose pre/post **3-D euclidean** soma distance is ``<= max_distance``.

        Full 3-D distance via :func:`trajan.euclidean_distance`
        (``sqrt(dx² + dy² + dz²)``), depth included. Reads the position column
        from the annotation declared with ``position_col=`` (auto-resolved when
        unique). Prune BEFORE materializing.
        """
        return filter_by_soma_distance_impl(
            self, max_distance, annotation, euclidean_distance
        )

    def filter_by_bbox(self, bbox, *, annotation: str | None = None) -> "PairUniverse":
        """Keep pairs where both pre and post somas fall inside ``bbox``."""
        ann_name = self._resolve_position_annotation(annotation)
        pos_col = self._cell_annotations[ann_name].position_col
        return self.filter(
            bbox_predicate(f"{pos_col}_pre", bbox)
            & bbox_predicate(f"{pos_col}_post", bbox)
        )

    # ── plan construction ─────────────────────────────────────────────────

    def _current_columns(self) -> set[str]:
        cols = set(self._pair_lf.collect_schema().names())
        for spec in self._cell_annotations.values():
            cols |= {f"{c}_pre" for c in spec.data_cols}
            cols |= {f"{c}_post" for c in spec.data_cols}
        cols |= set(self._expressions)
        return cols

    def build_lazy(self) -> pl.LazyFrame:
        """Construct (without collecting) the annotated + filtered lazy plan."""
        return build_pair_plan(
            self._pair_lf,
            self._cell_annotations,
            pre_col=self._pre_col,
            post_col=self._post_col,
            aliases=self._cell_aliases,
            expressions=self._expressions,
            filters=self._filters,
        )

    # ── materialization ──────────────────────────────────────────────────

    def collect(self, cols: list[str] | str | None = None) -> pl.DataFrame:
        """Materialize the lazy plan into a DataFrame, optionally projecting.

        With ``cols=None`` the full pair frame is collected. With an explicit
        column list, those columns are selected *before* collecting, so Polars'
        projection pushdown avoids materializing every annotation column — the
        memory-cheap path when you only need a few columns of the cross-product.

        Warns when the result row count exceeds ~10M — at that point you
        almost certainly want to compose more filters / aggregations
        upstream rather than realize the cross-product in memory.

        Parameters
        ----------
        cols : list[str] or str or None, optional
            Columns to project. ``None`` (default) collects the full frame.
            A single string is treated as a one-element list.

        Raises
        ------
        ValueError
            If any requested column is not present in the pair frame.
        """
        lf = self.build_lazy()
        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
            schema = lf.collect_schema().names()
            missing = [c for c in cols if c not in schema]
            if missing:
                raise ValueError(
                    f"Column(s) {missing} not found in pair universe. "
                    f"Available: {schema}"
                )
            lf = lf.select(cols)
        df = lf.collect()
        if len(df) > 10_000_000:
            warnings.warn(
                f"PairUniverse.collect() materialized {len(df):,} rows. "
                "Consider composing more filters (spatial pruning, id restriction) "
                "or aggregating via group_by(...).agg(...) before collecting.",
                stacklevel=2,
            )
        return df

    # .lazy / .select / .group_by / .preview / .count / __len__ / __bool__ come
    # from _LazyBacked (_base.py). PairUniverse intentionally has no .df cache
    # (the cross-product can be enormous), so it overrides collect() above with
    # a row-count warning instead of inheriting the caching _CachedTable surface.

    def to_edgelist(self) -> "EdgeList":
        """Collect the observed sub-frame as an ``EdgeList``.

        Returns the subset of *observed* pairs (any weight ``> 0``), wrapped as
        an ``EdgeList``. This is the EdgeList contract: rows are observed
        connections, not the full cross-product. Unobserved pairs (all weights
        ``0`` after the overlay) are dropped, so the returned table is the same
        data the upstream ``el.edgelist()`` produced — the round-trip is
        lossless on observed data.

        The observed predicate is ``n_syn > 0`` when an ``n_syn`` weight is
        present (the usual case from ``possible_pairs``); otherwise it is "any
        registered weight > 0", so a ``PairUniverse`` carrying a non-``n_syn``
        weight still round-trips. With no weights at all, every pair is kept.

        Use this when a downstream consumer requires an EdgeList explicitly.
        For denominator-bearing analyses (connection probability, density,
        null models) keep the ``PairUniverse`` — the zeros are the whole point.

        Use :meth:`to_pair_frame` if you actually want the raw cross-product
        DataFrame (observed + unobserved with weight ``0``); that is *not* an
        EdgeList.
        """
        from .edgelist import EdgeList

        lf = self.build_lazy()
        if "n_syn" in self._weights:
            observed = pl.col("n_syn") > 0
        elif self._weights:
            observed = pl.any_horizontal(*[pl.col(w) > 0 for w in self._weights])
        else:
            observed = pl.lit(True)
        df = lf.filter(observed).collect()
        return EdgeList(
            df,
            pre_col=self._pre_col,
            post_col=self._post_col,
            weight_cols=list(self._weights),
            cell_aliases=dict(self._cell_aliases),
        )

    def to_pair_frame(self) -> pl.DataFrame:
        """Materialize the full cross-product (observed + unobserved) as a DataFrame.

        Returns every pair in the universe, including those with ``n_syn = 0``
        (no observed synapses). This is *not* an EdgeList — an EdgeList by
        contract holds observed connections only. Returning a raw
        ``pl.DataFrame`` makes that explicit at the type level.

        Warns when the resulting row count is large (≥10M); for whole-brain
        universes this is almost never what you want — compose more filters
        or use ``group_by().agg()`` upstream.
        """
        df = self.build_lazy().collect()
        if len(df) > 10_000_000:
            warnings.warn(
                f"to_pair_frame materialized {len(df):,} rows (full "
                "cross-product including unobserved pairs). Consider composing "
                "more filters (spatial pruning, id restriction) or aggregating "
                "via group_by(...).agg(...) before collecting.",
                stacklevel=2,
            )
        return df

    def sample_pairs(
        self,
        n: int,
        *,
        weights: str | pl.Expr | None = None,
        replace: bool = True,
        seed: int | None = None,
    ) -> pl.DataFrame:
        """Draw ``n`` random pairs from the filtered universe, with connectivity.

        The pair-draw primitive: a uniform (or weight-biased) sample of pairs
        from the universe of *possible* connections, each carrying its observed
        connectivity. Where the cell bootstrap (:func:`cell_bootstrap_iter`)
        re-draws whole cells — the right unit for dense reconstructions, where
        pairs sharing an endpoint co-vary — this re-draws **pairs**, the right
        unit for designs that probe individual pairs and aggregate them (e.g.
        paired-recording electrophysiology). Build pair-level resampling /
        null-model summaries on top of repeated calls.

        Materializes the filtered cross-product once, then samples from it, so
        every accumulated filter (distance, bounding box, cell type, id
        restriction) is honored exactly. Compose those filters first to keep
        the materialized frame small — sampling does not avoid the collect, it
        just draws rows from it. The same ~10M-row warning as :meth:`collect`
        fires if the frame is large.

        ``weights`` biases the draw, the hook for matching an experimental
        sampling distribution. Pass a column name or a polars expression that
        evaluates to a non-negative per-pair weight; pairs are drawn with
        probability proportional to it. For example, to mimic "80 recorded
        pairs drawn according to a target distance distribution", register a
        kernel of the pair distance and pass it here::

            kernel = (-((pl.col("rho") - 50) ** 2) / 200).exp()  # any non-neg expr
            pu.filter_by_ids(pre_ids=type_a, post_ids=type_b)
              .sample_pairs(80, weights=kernel, seed=0)

        Parameters
        ----------
        n : int
            Number of pairs to draw.
        weights : str or polars.Expr or None, optional
            Per-pair sampling weight. ``None`` (default) draws uniformly. A
            string is read as a column name; an expression is evaluated against
            the materialized frame. Weights must be finite and non-negative and
            sum to a positive value.
        replace : bool, optional
            Draw with replacement (default ``True``, the bootstrap-friendly
            choice and required when ``n`` exceeds the number of eligible
            pairs). ``False`` draws ``n`` distinct pairs.
        seed : int or None, optional
            Numpy seed for reproducibility.

        Returns
        -------
        pl.DataFrame
            ``n`` sampled rows with the full pair-frame columns (ids, weights,
            registered annotations / expressions) plus a ``connected`` boolean
            (``n_syn > 0``) when an ``n_syn`` weight is present. ``n_syn`` and
            any other weights carry the per-pair connectivity for non-binary
            summaries.

        Raises
        ------
        ValueError
            If ``n`` is not positive, the universe is empty, ``replace=False``
            and ``n`` exceeds the eligible-pair count, or the weights are
            invalid (negative, non-finite, or summing to zero).

        See Also
        --------
        cell_bootstrap_iter : cell-draw resampling; the right unit for dense
            reconstructions.
        connection_probability : closed-form binomial CI — the analytic twin of
            an unweighted pair draw of binary connectivity.
        """
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "sample_pairs requires numpy. Install with `pip install numpy` "
                "or `uv add numpy`."
            ) from e

        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        df = self.collect()  # full filtered frame; reuses the ~10M-row warning
        pop = df.height
        if pop == 0:
            raise ValueError("cannot sample from an empty pair universe")
        if not replace and n > pop:
            raise ValueError(
                f"cannot draw {n} distinct pairs without replacement from "
                f"{pop} eligible pairs; pass replace=True or lower n"
            )

        rng = np.random.default_rng(seed)
        if weights is None:
            idx = rng.choice(pop, size=n, replace=replace)
        else:
            w_expr = pl.col(weights) if isinstance(weights, str) else weights
            w = df.select(w_expr.alias("__w__"))["__w__"].to_numpy().astype(float)
            if not np.isfinite(w).all() or (w < 0).any():
                raise ValueError("sample weights must be finite and non-negative")
            total = w.sum()
            if total <= 0:
                raise ValueError("sample weights sum to zero; nothing to draw")
            idx = rng.choice(pop, size=n, replace=replace, p=w / total)

        sampled = df[idx]
        if "n_syn" in self._weights and "connected" not in sampled.columns:
            sampled = sampled.with_columns((pl.col("n_syn") > 0).alias("connected"))
        return sampled

    # ── copy ─────────────────────────────────────────────────────────────

    def _copy(self) -> "PairUniverse":
        new = PairUniverse.__new__(PairUniverse)
        new._pair_lf = self._pair_lf
        new._pre_col = self._pre_col
        new._post_col = self._post_col
        new._weights = self._weights.copy()
        new._cell_annotations = self._cell_annotations.copy()
        new._filters = self._filters.copy()
        new._filter_sides = self._filter_sides.copy()
        new._expressions = self._expressions.copy()
        new._cell_aliases = self._cell_aliases.copy()
        return new


# ── public free function ──────────────────────────────────────────────────────


def possible_pairs(
    table: Union["SynapseTable", "EdgeList"],
    *,
    universe: str | None = None,
    include_self: bool = False,
    pre_ids: Iterable | None = None,
    post_ids: Iterable | None = None,
) -> PairUniverse:
    """Enumerate the ``universe × universe`` pair frame with observed counts.

    The denominator primitive: every possible ``(pre, post)`` pair drawn from
    the registered universe annotation. The ``n_syn`` weight (and any other
    weight registered on the source) is overlaid from the observed
    edgelist; pairs absent from the observed data get 0.

    Cell annotations registered on the source (including the universe) are
    re-registered on the returned ``PairUniverse``, so spatial filters and
    cell-type filters compose on the result. Cell-level filters that were
    accumulated on the source are projected onto the cross-product
    (single-sided filters constrain only their side; ``"both"`` filters that
    cannot be cleanly decomposed are skipped with a warning, matching the
    contract of ``cells()``).

    Synapse-level filters on a ``SynapseTable`` source are baked into the
    observed-side counts via ``.edgelist()`` — they do NOT prune the
    cross-product itself (the universe of *possible* connections doesn't
    shrink because a particular synapse was excluded).

    Pair-level weight filters on an ``EdgeList`` source (``n_syn > k``) are
    similarly baked into observed counts only; unobserved pairs still
    appear with weight 0. To restrict the denominator to high-confidence
    cells, apply a cell-level filter on the source (or on the returned
    ``PairUniverse``) instead.

    Parameters
    ----------
    table : SynapseTable or EdgeList
        Source with a registered universe annotation. SynapseTable inputs
        are first aggregated via ``.edgelist()``.
    universe : str or None
        Name of the universe annotation; auto-resolved when unambiguous.
    include_self : bool
        If False (default), self-pairs (``pre == post``) are excluded.
    pre_ids, post_ids : iterable of cell id or None
        Restrict the pre / post universe to these ids *before* forming the
        cross-product, so the plan never enumerates more than
        ``|pre_ids| × |post_ids|`` pairs. Prefer this over a post-hoc
        ``.filter_by_ids(...)`` when you already know the id set at
        construction time: filtering the cross-product relies on the query
        optimizer pushing the predicate below the cross join, which a large
        whole-brain universe can defeat (e.g. an id-dtype cast barrier), in
        which case the full ``|U|²`` product materializes first. Pruning the
        universe here is deterministic — the ``|U|²`` node is never built.
        Both default to ``None`` (use the whole universe on that side).

    Returns
    -------
    PairUniverse
        Lazy row plan with observed weights overlaid — ``|pre| × |post|`` minus
        self-pairs, where each side is the full universe unless restricted by
        ``pre_ids`` / ``post_ids``. Compose filters / aggregations on this; call
        ``.collect()`` or ``.group_by(...).agg(...)`` at the end.

    See Also
    --------
    PairUniverse.filter_by_ids : restrict an already-built cross-product
        (relies on predicate pushdown; use ``pre_ids`` / ``post_ids`` here to
        prune before the cross-product is formed instead).
    """
    # Normalize to EdgeList — synapse-level filters bake in during aggregation.
    # Capture cell-level filters from the source BEFORE aggregation, because
    # SynapseTable.edgelist() bakes all filters into the observed pair frame
    # and drops them from el._filters. We still want to project the cell-level
    # ones onto the cross-product below.
    from .edgelist import EdgeList
    from .synapse_table import SynapseTable

    if isinstance(table, SynapseTable):
        cell_filters = [
            (f, side)
            for f, side in zip(table._filters, table._filter_sides)
            if side in ("pre", "post", "both")
        ]
        el = table.edgelist()
    elif isinstance(table, EdgeList):
        cell_filters = [
            (f, side)
            for f, side in zip(table._filters, table._filter_sides)
            if side in ("pre", "post", "both")
        ]
        el = table
    else:
        raise TypeError(
            f"possible_pairs accepts SynapseTable or EdgeList, got {type(table).__name__}"
        )

    universe_name = el._resolve_universe_annotation(universe)
    universe_spec = el._cell_annotations[universe_name]
    cell_id = universe_spec.cell_id_col
    pre_col = el.pre_col
    post_col = el.post_col

    # Overlay observed weights from the source's already-merged edgelist.
    # build_lazy() bakes in cell-level filters too — we'll also re-apply
    # them via projection on the cross-product below to constrain the
    # denominator universe consistently.
    weights = list(el.weights)
    observed = el.build_lazy().select([pre_col, post_col] + weights)
    obs_schema = observed.collect_schema()

    # Cross-product over universe cell ids only — annotations are re-attached
    # as registered annotations so the PairUniverse can manage them under the
    # same alias / projection rules as EdgeList.
    #
    # Align the id dtypes to the observed (source) ids BEFORE the cross join,
    # not after. The cross is keyed by the universe annotation's cell-id column,
    # which may differ from the observed ids' dtype (e.g. UInt64 root ids in
    # parquet vs Int64 inferred from a python-int frame). The dtypes must match
    # or the weight overlay / annotation joins panic in the streaming engine
    # ("cannot get ref Int64 from UInt64"). Casting here — at the per-side
    # ``select`` below the cross join — rather than with a ``with_columns`` cast
    # placed *above* the cross join is what keeps single-sided pruning cheap: a
    # post-cross ``strict_cast`` redefines pre_col/post_col and so is an opaque
    # predicate-pushdown barrier, leaving e.g. ``filter_by_ids(pre_ids=[x])``
    # stranded above the join — the full |U|² product then materializes before
    # the filter drops it to one row. Cast pre-cross and the optimizer pushes
    # the id predicate down onto the scan, so the cross is |selected| × |U|.
    pre_side = universe_spec.lf.select(
        pl.col(cell_id).cast(obs_schema[pre_col]).alias(pre_col)
    )
    post_side = universe_spec.lf.select(
        pl.col(cell_id).cast(obs_schema[post_col]).alias(post_col)
    )
    # Build-time universe pruning: filter each side's id list before the cross
    # join so the product is bounded by construction, independent of whether the
    # optimizer can push a post-cross predicate down. A plain python list (not a
    # dtype-matched Series) keeps is_in on the clean coercion path for
    # signed/unsigned id columns, matching filter_by_id_sets.
    if pre_ids is not None:
        pre_side = pre_side.filter(pl.col(pre_col).is_in(list(pre_ids)))
    if post_ids is not None:
        post_side = post_side.filter(pl.col(post_col).is_in(list(post_ids)))
    cross = pre_side.join(post_side, how="cross")
    if not include_self:
        cross = cross.filter(pl.col(pre_col) != pl.col(post_col))

    cross = cross.join(observed, on=[pre_col, post_col], how="left").with_columns(
        *[pl.col(w).fill_null(0) for w in weights]
    )

    pu = PairUniverse(
        cross,
        pre_col=pre_col,
        post_col=post_col,
        weights=weights,
        cell_aliases=dict(el.cell_aliases),
    )
    # Re-register the source's cell annotations so spatial / type filters
    # compose on the PairUniverse with the same syntax as on the EdgeList.
    for ann_name, spec in el._cell_annotations.items():
        pu.add_annotation(
            ann_name,
            spec.lf,
            cell_id_col=spec.cell_id_col,
            position_col=spec.position_col,
            is_universe=spec.is_universe,
            join_on_alias=spec.join_on_alias,
        )
    # Propagate cell-side computed expressions from the EdgeList so any
    # accumulated cell-level filter that references one (e.g. a log1p'd
    # cell column, an uppercased cell_type, etc.) resolves against the
    # cross-product side. Without this, projecting such a filter onto the
    # PairUniverse below would crash at collect with ColumnNotFoundError
    # because the cross-product plan has no expressions registered.
    # Registration order is preserved by dict iteration so dependents see
    # their dependencies.
    for expr_name, expr in el._expressions.items():
        pu.add_expression(expr_name, expr)
    # Project the source's accumulated cell-level filters onto the
    # cross-product. Same projection contract as cells(): single-sided
    # filters apply directly to that side (the annotation join exposes
    # both _pre and _post columns); 'both' filters skip with a warning.
    # Synapse / vertex / pair-level filters (side=None) are not propagated —
    # they bake into observed counts via .edgelist() but don't shrink the
    # universe of *possible* connections.
    for f_expr, side_class in cell_filters:
        if side_class == "both":
            warnings.warn(
                f"Cell-level filter classified as 'both' is not projected onto "
                f"the PairUniverse cross-product (it still constrains observed "
                f"counts). Split into separate .filter() calls for tight "
                f"projection. Filter: {f_expr!s}",
                stacklevel=2,
            )
            continue
        pu = pu.filter(f_expr)
    return pu
