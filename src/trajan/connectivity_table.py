"""ConnectivityTable — Tier 2 in the trajan architecture.

A ConnectivityTable has rows indexed by (pre_entity, post_entity) and carries
one or more weight columns sharing the weight contract (auto-sum under
aggregation, default-sum, log-scale compatible). It admits normalization
(``normalize(by=...)``), binarization, log transforms, and dense/sparse
materialization.

The pre/post entities can be cells or labels (cell types / categories) — the
table does not distinguish. Operations that treat entities as points in space
belong on the narrower ``EdgeList`` subclass, not here (see
``project_edgelist_abstraction.md`` and the blessed-columns memory).

This first landing is dense-only (``to_dense`` via Polars pivot). A sparse
backend is tracked as a follow-up (Phase 3b in REFACTOR.md).

Persistence via DataFolio is supported and preserves the concrete class —
a saved ``EdgeList`` round-trips as an ``EdgeList`` even when loaded via
``ConnectivityTable.load`` — through a ``__type__`` marker in the config.
"""

from __future__ import annotations

import base64
import warnings
from pathlib import Path
from typing import Union

import polars as pl

from ._base import (
    CellAnnotationSpec,
    _AnnotationProxy,
    _as_folio,
    _to_lazy,
    build_cell_annotation_spec,
    build_pair_plan,
    classify_by_cell_sides,
    resolve_position_annotation,
    resolve_universe_annotation,
    unique_name,
)


class ConnectivityTable:
    """A table of connectivity between pre / post entities with weight columns.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
        One row per (pre, post) pair. Must contain ``pre_col`` and ``post_col``.
    pre_col : str
        Name of the pre-entity id column.
    post_col : str
        Name of the post-entity id column.
    weight_cols : list[str] or str or None, optional
        Columns that carry the weight contract (auto-sum, log-scale compatible,
        ``normalize`` default). A single string is treated as a one-element
        list. Defaults to ``["n_syn"]`` if that column exists, else ``[]``.

    Notes
    -----
    Rejects synapse- and vertex-level annotations by construction — those kinds
    are per-synapse, not per-pair (see ``feedback_vertex_annotations.md``).
    ``add_annotation`` takes a per-entity table and joins it symmetrically on
    both pre and post entity columns, producing ``{col}_pre`` / ``{col}_post``
    outputs analogous to ``SynapseTable.add_cell_annotation``.

    How ``.df`` is assembled
    ------------------------
    ``.df`` is built lazily by ``build_lazy()`` and cached:

    1. Start from the base pair table (``pairs`` / ``pairs_lazy``).
    2. Join each registered annotation symmetrically on ``pre_col`` /
       ``post_col`` — each data column ``c`` produces ``c_pre`` and ``c_post``.
       (``EdgeList`` additionally supports alias-keyed joins; see its docs.)
    3. Apply each registered expression via ``with_columns`` (in order).
    4. Apply each accumulated filter via ``filter`` (in order).

    To inspect any single piece:

    - ``ct.pairs`` — base pair DataFrame (step 1 only)
    - ``ct.annotations[name]`` — a specific annotation frame
    - ``ct.build_lazy()`` — full plan as a ``pl.LazyFrame`` (no collect)
    """

    # ``__type__`` marker written into saved config so ``load`` can dispatch
    # to the correct concrete class (EdgeList overrides). See save/load.
    _TYPE_TAG: str = "ConnectivityTable"

    def __init__(
        self,
        df,
        pre_col: str,
        post_col: str,
        weight_cols: list[str] | str | None = None,
    ):
        self._pair_lf = _to_lazy(df)
        self._pre_col = pre_col
        self._post_col = post_col

        schema = self._pair_lf.collect_schema().names()
        for required in (pre_col, post_col):
            if required not in schema:
                raise ValueError(
                    f"Column {required!r} not found in input frame. Available: {schema}"
                )

        if isinstance(weight_cols, str):
            weight_cols = [weight_cols]
        if weight_cols is None:
            weight_cols = ["n_syn"] if "n_syn" in schema else []
        for w in weight_cols:
            if w not in schema:
                raise ValueError(
                    f"Weight column {w!r} not found in input frame. Available: {schema}"
                )
        self._weights: list[str] = list(weight_cols)

        self._pair_col_names: list[str] = schema

        # Per-entity annotations joined symmetrically onto pre_col / post_col.
        # The CellAnnotationSpec dataclass is reused — an "entity" on a
        # ConnectivityTable is a cell or a label, indistinguishable from the
        # table's perspective. join_on_alias is always None here (aliasing is a
        # SynapseTable-level feature).
        self._cell_annotations: dict[str, CellAnnotationSpec] = {}

        self._filters: list[pl.Expr] = []
        # Parallel to _filters: cell-side classification ('pre' / 'post' /
        # 'both' / None) captured at filter() time. Drives cells() and
        # possible_pairs() side-decomposed projection.
        self._filter_sides: list[str | None] = []
        self._expressions: dict[str, pl.Expr] = {}
        # Parallel to _expressions: cell-side classification of each named
        # expression, so a filter referencing an expression column inherits the
        # expression's side (matching SynapseTable). Drives the same
        # side-decomposed projection in cells() / possible_pairs().
        self._expression_sides: dict[str, str | None] = {}
        self._cache: pl.DataFrame | None = None

    # ── read-only accessors ────────────────────────────────────────────────

    @property
    def pre_col(self) -> str:
        return self._pre_col

    @property
    def post_col(self) -> str:
        return self._post_col

    @property
    def weights(self) -> list[str]:
        """Weight columns — share the contract (auto-sum, log-scale, normalize default)."""
        return list(self._weights)

    @property
    def annotation_names(self) -> list[str]:
        return list(self._cell_annotations)

    def annotation_data_cols(self) -> dict[str, list[str]]:
        """Data columns (non-key) for each registered annotation.

        Returns a fresh dict mapping annotation name to a fresh list of data
        column names. Mirrors ``SynapseTable.cell_annotation_data_cols`` so
        free-function consumers (stats, viz, exporters) can enumerate
        annotation columns without reaching into private storage.
        """
        return {
            name: list(spec.data_cols) for name, spec in self._cell_annotations.items()
        }

    def info(self) -> str:
        """Summarize the table structure: core columns, annotations, weights.

        Mirrors ``SynapseTable.info()`` for the pair-level tier. Prints and
        returns a human-readable string showing pre/post id columns, every
        registered annotation (with the universe / position role tags), any
        named expressions, weight columns, and the number of accumulated
        filters.
        """
        lines: list[str] = []
        if self._cache is not None:
            n = len(self._cache)
        elif not self._filters:
            n = self._pair_lf.select(pl.len()).collect().item()
        else:
            n = self.build_lazy().select(pl.len()).collect().item()
        lines.append(
            f"{type(self).__name__}  ({n:,} pairs, {len(self._filters)} filter(s))"
        )
        lines.append("")
        lines.append("Core columns")
        lines.append(f"  pre_col            : {self._pre_col}")
        lines.append(f"  post_col           : {self._post_col}")
        if self._cell_annotations:
            lines.append("")
            lines.append(f"Annotations ({len(self._cell_annotations)})")
            for name, spec in self._cell_annotations.items():
                role_tags = []
                if spec.is_universe:
                    role_tags.append("universe")
                if spec.position_col is not None:
                    role_tags.append(f"position={spec.position_col!r}")
                join_info = (
                    f"join on alias {spec.join_on_alias!r}"
                    if spec.join_on_alias
                    else f"join on {spec.cell_id_col!r}"
                )
                role_str = f"  [{', '.join(role_tags)}]" if role_tags else ""
                lines.append(
                    f"  {name!r} ({len(spec.data_cols)} col(s), {join_info}){role_str}"
                )
                for c in spec.data_cols:
                    lines.append(f"    {c}  ->  {c}_pre, {c}_post")
        if self._expressions:
            lines.append("")
            lines.append(f"Expressions ({len(self._expressions)})")
            for name in self._expressions:
                lines.append(f"  {name}")
        if self._weights:
            lines.append("")
            lines.append(f"Weights ({len(self._weights)})")
            for w in self._weights:
                lines.append(f"  {w}")
        result = "\n".join(lines)
        print(result)
        return result

    def __repr__(self) -> str:
        if self._cache is not None:
            n = len(self._cache)
        elif not self._filters:
            n = self._pair_lf.select(pl.len()).collect().item()
        else:
            n = "uncached"
        return (
            f"{type(self).__name__}(n_pairs={n}, "
            f"pre_col={self._pre_col!r}, post_col={self._post_col!r}, "
            f"weights={self._weights}, "
            f"annotations={list(self._cell_annotations)})"
        )

    def __len__(self) -> int:
        """Number of pairs after filters; collects lazily if needed."""
        if self._cache is not None:
            return len(self._cache)
        if not self._filters:
            return self._pair_lf.select(pl.len()).collect().item()
        return self.build_lazy().select(pl.len()).collect().item()

    def __bool__(self) -> bool:
        # Always truthy: prevents `if ct:` from implicitly forcing a collect
        # via __len__. Use `len(ct) > 0` for the row-count check.
        return True

    # ── registration: annotation / weight / expression / filter ────────────

    def add_annotation(
        self,
        name: str,
        df,
        cell_id_col: str,
        position_col: str | None = None,
        is_universe: bool = False,
    ) -> ConnectivityTable:
        """Register a per-entity annotation joined symmetrically on pre and post.

        Each data column in the annotation produces ``{col}_pre`` and
        ``{col}_post`` on the merged frame. Whether the entities are cells or
        labels (cell types) is up to the user — the join mechanics are the same.

        Parameters
        ----------
        name : str
            Identifier for this annotation.
        df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
            Annotation frame. Must contain ``cell_id_col`` plus one or more
            data columns. Each ``cell_id_col`` value must be unique.
        cell_id_col : str
            Column in ``df`` whose values are joined against both ``pre_col``
            and ``post_col``. Named ``cell_id_col`` rather than
            ``entity_id_col`` for consistency with ``SynapseTable`` and
            ``EdgeList`` — even on a plain ``ConnectivityTable`` where rows
            could be cell-type labels rather than cells, the kwarg name stays
            the same so user code is portable across tiers.
        position_col : str or None, optional
            Name of a data column carrying the entity's position as a struct
            with ``x`` / ``y`` / ``z`` fields. Declaring this role lets
            ``EdgeList`` spatial filters locate positions without explicit
            per-call args. ConnectivityTable itself does not consume the role
            (no built-in spatial ops on label-bearing tables), but the
            declaration is preserved through save/load.
        is_universe : bool, optional
            If True, this annotation's entity-id set defines the authoritative
            universe for denominator-bearing statistics. Auto-resolved when
            exactly one annotation is marked universe; pass the annotation
            name explicitly otherwise. See ``CellAnnotationSpec.is_universe``.
        """
        self._cell_annotations[name] = build_cell_annotation_spec(
            df,
            cell_id_col=cell_id_col,
            position_col=position_col,
            is_universe=is_universe,
            join_on_alias=None,
            current_columns=self._current_columns(),
        )
        self._cache = None
        return self

    # ── role resolvers (used by spatial filters and universe-aware stats) ──

    def _resolve_position_annotation(self, annotation: str | None = None) -> str:
        """Return the name of the annotation whose ``position_col`` to use.

        Delegates to :func:`trajan._base.resolve_position_annotation`; see
        there for the auto-resolution contract.
        """
        return resolve_position_annotation(self._cell_annotations, annotation)

    def _resolve_universe_annotation(self, annotation: str | None = None) -> str:
        """Return the name of the annotation that defines the entity universe.

        Delegates to :func:`trajan._base.resolve_universe_annotation`; see
        there for the auto-resolution contract.
        """
        return resolve_universe_annotation(self._cell_annotations, annotation)

    def remove_annotation(self, name: str) -> ConnectivityTable:
        if name not in self._cell_annotations:
            raise KeyError(f"No annotation named {name!r}")
        del self._cell_annotations[name]
        self._cache = None
        return self

    def add_synapse_annotation(self, *args, **kwargs):  # pragma: no cover
        raise TypeError(
            "ConnectivityTable does not accept synapse-level annotations. "
            "Synapse data varies per synapse and cannot be represented at the "
            "pair-level. Register it on a SynapseTable upstream instead."
        )

    def add_vertex_annotation(self, *args, **kwargs):  # pragma: no cover
        raise TypeError(
            "ConnectivityTable does not accept vertex-level annotations. "
            "Vertex data varies per synapse and cannot be represented at the "
            "pair-level. Register it on a SynapseTable upstream instead."
        )

    def add_weight(self, col: str) -> ConnectivityTable:
        """Mark an existing column as carrying the weight contract."""
        if col not in self._current_columns():
            raise ValueError(
                f"Column {col!r} not found in table. Add it first via "
                "add_annotation or add_expression."
            )
        if col in self._weights:
            raise ValueError(f"Column {col!r} is already registered as a weight.")
        self._weights.append(col)
        return self

    def remove_weight(self, col: str) -> ConnectivityTable:
        if col not in self._weights:
            raise KeyError(f"No weight registered for column {col!r}")
        self._weights.remove(col)
        return self

    def add_expression(self, name: str, expr: pl.Expr) -> ConnectivityTable:
        """Register a named computed column applied after joins and before filters."""
        if name in self._current_columns():
            raise ValueError(f"Column {name!r} already exists in the table")
        self._expressions[name] = expr.alias(name)
        self._expression_sides[name] = self._classify_expression(expr)
        self._cache = None
        return self

    def filter(self, expr: pl.Expr) -> ConnectivityTable:
        """Return a new ConnectivityTable with ``expr`` applied to the lazy plan."""
        new = self._copy()
        new._filters = self._filters + [expr]
        new._filter_sides = self._filter_sides + [self._classify_expression(expr)]
        return new

    def _classify_expression(self, expr: pl.Expr) -> str | None:
        """Classify ``expr`` as 'pre', 'post', 'both', or None.

        Thin wrapper over :func:`trajan._base.classify_by_cell_sides` that
        threads in this table's currently registered cell annotations and
        previously classified expressions, so a filter referencing an
        expression-derived cell column classifies on the same side it would on
        a ``SynapseTable``.
        """
        return classify_by_cell_sides(
            expr, self._cell_annotations, self._expression_sides
        )

    @property
    def filter_sides(self) -> list[str | None]:
        """Cell-side classification of each accumulated filter.

        See :attr:`SynapseTable.filter_sides`; same contract — one entry per
        filter in ``_filters``, captured at ``filter()`` time, and used by
        ``cells()`` / ``possible_pairs()`` for side-decomposed projection.
        """
        return list(self._filter_sides)

    # ── plan construction ──────────────────────────────────────────────────

    def _current_columns(self) -> set[str]:
        cols = set(self._pair_col_names)
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
            expressions=self._expressions,
            filters=self._filters,
        )

    @property
    def df(self) -> pl.DataFrame:
        """Full merged pair-level table with annotations joined. Cached."""
        if self._cache is None:
            self._cache = self.build_lazy().collect()
        return self._cache

    def clear_cache(self) -> ConnectivityTable:
        """Drop the materialized ``.df`` cache, releasing its memory.

        ``.df`` pins the merged pair-level frame for the object's lifetime once
        touched. Call this when you keep a reference to the table but no longer
        need it materialized — the next ``.df`` access rebuilds it lazily. A
        no-op when nothing is cached. Returns self for chaining.
        """
        self._cache = None
        return self

    def preview(self, n: int = 10) -> pl.DataFrame:
        """Collect the first ``n`` rows of the merged table without caching.

        Unlike ``.df.head(n)`` — which forces a full collect and pins the result
        on ``self._cache`` — this pushes a ``head(n)`` limit into the lazy plan,
        so only ``n`` rows are materialized and nothing is cached. Use it to peek
        at the schema / a few rows of a large table cheaply.

        Parameters
        ----------
        n : int, optional
            Number of rows to collect. Defaults to 10.
        """
        return self.build_lazy().head(n).collect()

    def collect(self, cols: list[str] | str | None = None) -> pl.DataFrame:
        """Materialize the merged table, optionally projecting to ``cols``.

        With ``cols=None`` this is just the cached ``.df``. With an explicit
        column list, it selects those columns *before* collecting, so Polars'
        projection pushdown skips materializing (and often skips joining) every
        other annotation column. This is the memory-cheap path for plotting:
        pull only the columns a figure needs instead of the whole frame. The
        narrow result is returned fresh and is **not** cached.

        Parameters
        ----------
        cols : list[str] or str or None, optional
            Columns to project. ``None`` (default) returns the full cached
            ``.df``. A single string is treated as a one-element list.

        Returns
        -------
        pl.DataFrame
            The full cached frame (``cols=None``) or a fresh narrow projection.

        Raises
        ------
        ValueError
            If any requested column is not present in the merged table.
        """
        if cols is None:
            return self.df
        if isinstance(cols, str):
            cols = [cols]
        schema = self.build_lazy().collect_schema().names()
        missing = [c for c in cols if c not in schema]
        if missing:
            raise ValueError(
                f"Column(s) {missing} not found in table. Available: {schema}"
            )
        return self.build_lazy().select(cols).collect()

    @property
    def pairs(self) -> pl.DataFrame:
        """Base pair table — no annotations joined, no filters, no expressions.

        For the fully merged + filtered + computed table use ``.df``. To
        inspect an individual annotation, use ``annotations[name]``.
        """
        return self._pair_lf.collect()

    @property
    def pairs_lazy(self) -> pl.LazyFrame:
        """Lazy view of the base pair table — for building custom plans."""
        return self._pair_lf

    @property
    def annotations(self) -> _AnnotationProxy:
        """Read-only view over registered per-entity annotations.

        ``ct.annotations["name"]`` returns the annotation as a collected
        ``pl.DataFrame``. Supports ``in``, iteration, and ``len``.
        """
        return _AnnotationProxy(self._cell_annotations)

    # ── copy ───────────────────────────────────────────────────────────────

    def _copy(self) -> ConnectivityTable:
        new = ConnectivityTable.__new__(ConnectivityTable)
        new._pair_lf = self._pair_lf
        new._pair_col_names = self._pair_col_names
        new._pre_col = self._pre_col
        new._post_col = self._post_col
        new._weights = self._weights.copy()
        new._cell_annotations = self._cell_annotations.copy()
        new._filters = self._filters.copy()
        new._filter_sides = self._filter_sides.copy()
        new._expressions = self._expressions.copy()
        new._expression_sides = self._expression_sides.copy()
        new._cache = None
        return new

    # ── shape-preserving operations ────────────────────────────────────────

    def _resolve_values(self, values: str | None) -> str:
        """Default to the first registered weight when values is omitted."""
        if values is not None:
            return values
        if not self._weights:
            raise ValueError(
                "No weight columns registered and values= not given. "
                "Pass values=<column> explicitly or register a weight via add_weight()."
            )
        return self._weights[0]

    def normalize(
        self,
        by: str = "pre",
        values: str | None = None,
        total_col: str | None = None,
    ) -> ConnectivityTable:
        """Normalize a weight column to a fraction.

        Two modes:

        - **Internal** (``total_col=None``): divide each row's weight by the
          sum of that weight across rows sharing the same ``by`` entity.
          Semantics are dynamic — if the table is filtered, the result is
          fraction-of-currently-visible-total.
        - **External** (``total_col=<col>``): divide each row's weight by the
          per-row value of ``total_col``. Trajan does not interpret what
          ``total_col`` means; the caller is responsible for supplying a
          column with the right semantics (e.g. the cell's true total input
          projected to the pair frame via a cell annotation).

        The result replaces ``values`` with ``fraction`` and drops that entry
        from the weight list (a fraction is not a weight under the type
        algebra — summing fractions across rows is not meaningful).

        Parameters
        ----------
        by : str
            ``"pre"`` — normalize by pre-entity totals (row-stochastic per pre).
            ``"post"`` — normalize by post-entity totals (column-stochastic per
            post). This is the Drosophila-style input-fraction convention.
        values : str or None, optional
            Weight column to normalize. Defaults to the first registered weight.
        total_col : str or None, optional
            If None, use internal axis sum. Otherwise the name of a column in
            the merged plan whose per-row value is the denominator.

        Returns
        -------
        ConnectivityTable
            New table with ``values`` replaced by a ``fraction`` column.
        """
        if by == "pre":
            group_col = self._pre_col
        elif by == "post":
            group_col = self._post_col
        else:
            raise ValueError(f"by must be 'pre' or 'post', got {by!r}")

        values = self._resolve_values(values)
        if values not in self._current_columns():
            raise ValueError(f"values={values!r} not found in table.")
        if total_col is not None and total_col not in self._current_columns():
            raise ValueError(f"total_col={total_col!r} not found in table.")
        if "fraction" in self._current_columns() and values != "fraction":
            raise ValueError(
                "normalize() writes a 'fraction' column, but one already exists. "
                "Rename the existing column first."
            )

        lf = self.build_lazy()

        if total_col is None:
            # Unique internal name so a user column literally named __total__
            # can't shadow the computed per-axis sum (which would silently
            # produce wrong fractions).
            total_name = unique_name("__total__", self._current_columns())
            totals = lf.group_by(group_col).agg(pl.sum(values).alias(total_name))
            normalized = (
                lf.join(totals, on=group_col, how="left")
                .with_columns((pl.col(values) / pl.col(total_name)).alias("fraction"))
                .drop([total_name, values])
            )
        else:
            normalized = lf.with_columns(
                (pl.col(values) / pl.col(total_col)).alias("fraction")
            ).drop(values)

        return self._replace_base(normalized, drop_weight=values)

    def binarize(
        self, threshold: float = 0, values: str | None = None
    ) -> ConnectivityTable:
        """Replace a weight column with 1 where value > threshold, else 0.

        The binarized column retains the original name and stays in the weight
        list. Useful for computing connectivity presence/absence.
        """
        values = self._resolve_values(values)
        if values not in self._current_columns():
            raise ValueError(f"values={values!r} not found in table.")
        lf = self.build_lazy().with_columns(
            (pl.col(values) > threshold).cast(pl.Int64).alias(values)
        )
        return self._replace_base(lf)

    def log1p(self, values: str | None = None) -> ConnectivityTable:
        """Apply log1p to a weight column.

        The log1p-transformed column replaces the original. Note: sum of
        log1p(x) is not meaningful as a tier-aggregated weight, so the output
        column is no longer registered in the weight list — it becomes a
        property column (per the type algebra in the blessed-columns memory).
        """
        values = self._resolve_values(values)
        if values not in self._current_columns():
            raise ValueError(f"values={values!r} not found in table.")
        out_name = f"log1p_{values}"
        if out_name in self._current_columns():
            raise ValueError(f"Output column {out_name!r} already exists in table.")
        lf = self.build_lazy().with_columns(pl.col(values).log1p().alias(out_name))
        return self._replace_base(lf, drop_weight=values)

    def _replace_base(
        self, new_lf: pl.LazyFrame, *, drop_weight: str | None = None
    ) -> ConnectivityTable:
        """Return a new ConnectivityTable backed by ``new_lf``, annotations
        already merged in. Shape-preserving ops use this to avoid re-joining.
        """
        new = ConnectivityTable.__new__(ConnectivityTable)
        new._pair_lf = new_lf
        new._pair_col_names = new_lf.collect_schema().names()
        new._pre_col = self._pre_col
        new._post_col = self._post_col
        new._weights = [w for w in self._weights if w != drop_weight]
        # Annotations and expressions are baked into new_lf; start empty.
        new._cell_annotations = {}
        new._filters = []
        new._filter_sides = []
        new._expressions = {}
        new._cache = None
        return new

    # ── materialization ────────────────────────────────────────────────────

    def to_dense(
        self,
        values: str | None = None,
        fill_value: float = 0,
        *,
        aggregate: str = "error",
    ) -> pl.DataFrame:
        """Pivot into a dense pre × post matrix DataFrame.

        Parameters
        ----------
        values : str or None, optional
            Column to use as matrix entries. Defaults to the first registered
            weight.
        fill_value : float, optional
            Value for (pre, post) pairs absent from the plan.
        aggregate : str, optional
            How to collapse duplicate ``(pre, post)`` entries. ``"error"``
            (default) raises if any duplicate pair exists — a well-formed
            ConnectivityTable has one row per entity pair (``type_edgelist`` /
            ``aggregate_to_type`` guarantee it), so a duplicate usually signals
            a malformed hand-built table. Any other value is passed through to
            polars as the pivot ``aggregate_function`` to combine duplicates
            with that reducer — ``"sum"`` (natural for additive weights),
            ``"mean"``, ``"max"``, ``"first"``, etc.

        Returns
        -------
        pl.DataFrame
            Columns: ``pre_col`` followed by one column per post entity.
        """
        values = self._resolve_values(values)
        if values not in self._current_columns():
            raise ValueError(f"values={values!r} not found in table.")
        df = self.build_lazy().select([self._pre_col, self._post_col, values]).collect()
        # By default, refuse duplicate (pre, post) rows: the pivot would
        # otherwise silently collapse them, producing a wrong matrix entry.
        # Pass aggregate="sum" (or another reducer) to combine them on purpose.
        if aggregate == "error":
            n_dup = df.height - df.select([self._pre_col, self._post_col]).n_unique()
            if n_dup:
                raise ValueError(
                    f"to_dense found {n_dup} duplicate (pre, post) pair(s); the "
                    f"matrix entry would be ambiguous. Pass aggregate='sum' (or "
                    f"'mean', 'max', ...) to combine them, or aggregate to one row "
                    f"per pair first (e.g. .aggregate_to_type(...))."
                )
            agg_fn = "first"  # no duplicates reach the pivot
        else:
            agg_fn = aggregate
        return df.pivot(
            on=self._post_col,
            index=self._pre_col,
            values=values,
            aggregate_function=agg_fn,
        ).fill_null(fill_value)

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, folio: Union[str, Path, object], overwrite: bool = False) -> None:
        """Save this ConnectivityTable (or EdgeList) to a DataFolio.

        Materializes the base pair frame and every registered annotation
        as Parquet tables; serializes filters as polars JSON expressions
        and named expressions as base64-encoded polars binary. The concrete
        class (``ConnectivityTable`` vs ``EdgeList``) is stored in a
        ``__type__`` marker so ``load`` can return the correct subclass.

        Parameters
        ----------
        folio : str, Path, or DataFolio
            Target folio.
        overwrite : bool, optional
            If True, overwrite existing items in the folio.
        """
        folio = _as_folio(folio)
        config = {
            "__type__": self._TYPE_TAG,
            "pre_col": self._pre_col,
            "post_col": self._post_col,
            "weights": list(self._weights),
            "filters": [f.meta.serialize(format="json") for f in self._filters],
            "filter_sides": list(self._filter_sides),
            "expressions": {
                name: base64.b64encode(expr.meta.serialize(format="binary")).decode(
                    "ascii"
                )
                for name, expr in self._expressions.items()
            },
            "annotations": {
                name: self._spec_to_config(spec)
                for name, spec in self._cell_annotations.items()
            },
            **self._extra_save_config(),
        }
        folio.add_json("config", config, overwrite=overwrite)
        folio.add_table("pairs", self._pair_lf.collect(), overwrite=overwrite)
        for name, spec in self._cell_annotations.items():
            folio.add_table(f"ann_{name}", spec.lf.collect(), overwrite=overwrite)

    def _spec_to_config(self, spec: CellAnnotationSpec) -> dict:
        """Serialize an annotation spec to a config dict.

        Hook for subclasses (notably ``EdgeList``) to extend with extra
        spec fields like ``join_on_alias``.
        """
        return {
            "cell_id_col": spec.cell_id_col,
            "data_cols": list(spec.data_cols),
            "position_col": spec.position_col,
            "is_universe": spec.is_universe,
        }

    def _extra_save_config(self) -> dict:
        """Extra top-level config fields beyond the base ConnectivityTable set.

        Hook for subclasses to add fields like ``cell_aliases``.
        """
        return {}

    @classmethod
    def load(cls, folio: Union[str, Path, object]) -> ConnectivityTable:
        """Load a ConnectivityTable (or EdgeList) from a DataFolio.

        When called on ``ConnectivityTable.load``, dispatches to
        ``EdgeList.load`` if the saved folio's ``__type__`` marker says
        the data was an EdgeList. When called on ``EdgeList.load``, the
        marker must match — loading a plain ConnectivityTable as an
        EdgeList raises ``TypeError`` because the cell-axis invariant
        isn't guaranteed.
        """
        folio = _as_folio(folio)
        config = folio.get_json("config")
        saved_type = config.get("__type__", "ConnectivityTable")

        # Dispatch from the base class to the concrete subclass if needed.
        if cls is ConnectivityTable and saved_type != "ConnectivityTable":
            if saved_type == "EdgeList":
                from .edgelist import EdgeList

                return EdgeList.load(folio)
            raise TypeError(
                f"Saved folio's __type__ is {saved_type!r}; unknown at "
                f"ConnectivityTable.load."
            )
        if cls is not ConnectivityTable and saved_type != cls._TYPE_TAG:
            raise TypeError(
                f"Saved folio is a {saved_type!r}; cannot load as "
                f"{cls.__name__}. Call the matching .load() — or "
                f"ConnectivityTable.load(folio), which dispatches."
            )

        def _lf(name: str) -> pl.LazyFrame:
            return pl.scan_parquet(folio.get_data_path(name))

        instance = cls(
            _lf("pairs"),
            pre_col=config["pre_col"],
            post_col=config["post_col"],
            weight_cols=config.get("weights", []),
        )

        for name, ann_meta in config.get("annotations", {}).items():
            instance.add_annotation(
                name,
                _lf(f"ann_{name}"),
                cell_id_col=ann_meta["cell_id_col"],
                position_col=ann_meta.get("position_col"),
                is_universe=ann_meta.get("is_universe", False),
            )

        # Named expressions — binary format first (round-trips literals
        # correctly), with a JSON fallback for older saved folios.
        for name, expr_val in config.get("expressions", {}).items():
            expr = None
            try:
                expr = pl.Expr.deserialize(base64.b64decode(expr_val), format="binary")
            except Exception:
                pass
            if expr is None:
                try:
                    expr = pl.Expr.deserialize(expr_val.encode(), format="json")
                except Exception:
                    warnings.warn(
                        f"Could not deserialize expression {name!r} from folio "
                        "(likely a Polars version incompatibility). Re-save to fix.",
                        stacklevel=2,
                    )
                    continue
            instance.add_expression(name, expr)

        # Filters accumulate via .filter() which returns a new instance.
        # Use saved filter_sides when present; fall back to live reclassification
        # for older folios. Both paths agree because annotations are restored
        # before filters above.
        saved_sides = config.get("filter_sides")
        for i, f_json in enumerate(config.get("filters", [])):
            instance = instance.filter(
                pl.Expr.deserialize(f_json.encode(), format="json")
            )
            if saved_sides is not None and i < len(saved_sides):
                instance._filter_sides[-1] = saved_sides[i]

        return instance
