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

Dense export uses a guarded Polars pivot. Sparse export returns a CSR array
and explicit axis identities without constructing a dense matrix.

Persistence via DataFolio is supported and preserves the concrete class —
a saved ``EdgeList`` round-trips as an ``EdgeList`` even when loaded via
``ConnectivityTable.load`` — through a ``__type__`` marker in the config.
"""

from __future__ import annotations

import base64
import math
import warnings
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Iterable, Literal, Union

import polars as pl

from ._base import (
    CellAnnotationSpec,
    _AnnotationProxy,
    _as_folio,
    _CachedTable,
    _folio_add,
    _folio_batch,
    _folio_read,
    _folio_scan,
    _to_lazy,
    build_cell_annotation_spec,
    build_pair_plan,
    classify_by_cell_sides,
    resolve_position_annotation,
    resolve_universe_annotation,
    unique_name,
)
from ._guard import check_collect


class ConnectivityTable(_CachedTable):
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
        self.metadata: dict = {}
        self._n_pair_base = len(df) if not isinstance(df, pl.LazyFrame) else None
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
                if spec.side != "both":
                    role_tags.append(f"side={spec.side!r}")
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
                    lines.append(
                        f"    {c}  ->  " + ", ".join(f"{c}_{s}" for s in spec.sides())
                    )
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
            n = self._n_pair_base if self._n_pair_base is not None else "uncached"
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

    # ── registration: annotation / weight / expression / filter ────────────

    def add_annotation(
        self,
        name: str,
        df,
        cell_id_col: str,
        position_col: str | None = None,
        is_universe: bool = False,
        side: str = "both",
    ) -> ConnectivityTable:
        """Register a per-entity annotation joined on pre and/or post.

        With the default ``side="both"`` each data column produces both
        ``{col}_pre`` and ``{col}_post`` on the merged frame — correct when both
        axes are the same kind of entity. On an asymmetric table (e.g. a
        cell × cell-type table from ``aggregate_to_type``) pass ``side="pre"``
        or ``"post"`` to annotate only the matching axis; the other side would
        join against labels and yield all-null columns. Whether the entities are
        cells or labels is up to the user — the join mechanics are the same.

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
        side : str, optional
            Which axis to join onto: ``"pre"``, ``"post"``, or ``"both"``
            (default). See the summary above.
        """
        self._cell_annotations[name] = build_cell_annotation_spec(
            df,
            cell_id_col=cell_id_col,
            position_col=position_col,
            is_universe=is_universe,
            join_on_alias=None,
            current_columns=self._current_columns(),
            side=side,
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
            cols |= set(spec.produced_columns())
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
    def pairs(self) -> pl.DataFrame:
        """Base pair table — no annotations joined, no filters, no expressions.

        For the fully merged + filtered + computed table use ``.df``. To
        inspect an individual annotation, use ``annotations[name]``.
        """
        check_collect(self._pair_lf, f"{type(self).__name__}.pairs")
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
        new.metadata = deepcopy(self.metadata)
        new._n_pair_base = self._n_pair_base
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
        *,
        reweight_by: str | Mapping[object, float] | pl.DataFrame | None = None,
        mode: Literal["fraction", "enrichment"] = "fraction",
    ) -> ConnectivityTable:
        """Normalize a weight column to fractions or population enrichment.

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
        reweight_by : str, Mapping, DataFrame, or None, optional
            Divide values by this opportunity denominator before computing
            internal fractions. A string names a numeric column in the merged
            plan. A mapping supplies category-to-denominator values for the
            opposite axis: post categories for ``by="pre"``, pre categories
            for ``by="post"``. Mapping values must be finite and positive,
            and every observed category must be covered; extra keys are fine.
            Invalid column denominators produce null fractions and are
            excluded from totals. Groups with no positive adjusted total
            produce null fractions. Cannot be combined with ``total_col``.
            A two-column DataFrame supplies categories in its first column and
            denominators in its second, for example ``st.cell_agg("subclass")``.
            Category keys must be unique and the mapping validation rules apply.
        mode : {"fraction", "enrichment"}, optional
            Default ``"fraction"`` normalizes adjusted rates to sum to one.
            ``"enrichment"`` divides the observed allocation fraction by the
            supplied population fraction, writing an ``enrichment`` column.
            One means allocation matches population abundance. Requires a
            mapping or two-column DataFrame; all supplied categories count
            toward the expected population total, including unobserved ones.
            Filter the population explicitly to the intended target set.
            Column-only denominators are insufficient to define a population
            that includes unobserved categories and are not accepted in this mode.

        Returns
        -------
        ConnectivityTable
            New table with ``values`` replaced by ``fraction`` or ``enrichment``.
        """
        if mode not in {"fraction", "enrichment"}:
            raise ValueError("mode must be 'fraction' or 'enrichment'.")
        if mode == "enrichment" and not isinstance(
            reweight_by, (Mapping, pl.DataFrame)
        ):
            raise ValueError(
                "Enrichment requires reweight_by as a mapping or two-column DataFrame defining the full eligible population."
            )
        output_col = "enrichment" if mode == "enrichment" else "fraction"
        if by == "pre":
            group_col = self._pre_col
        elif by == "post":
            group_col = self._post_col
        else:
            raise ValueError(f"by must be 'pre' or 'post', got {by!r}")

        values = self._resolve_values(values)
        if reweight_by is not None and total_col is not None:
            raise ValueError("reweight_by cannot be combined with total_col.")
        if values not in self._current_columns():
            raise ValueError(f"values={values!r} not found in table.")
        if total_col is not None and total_col not in self._current_columns():
            raise ValueError(f"total_col={total_col!r} not found in table.")
        if output_col in self._current_columns() and values != output_col:
            raise ValueError(
                f"normalize() writes a '{output_col}' column, but one already exists. "
                "Rename the existing column first."
            )

        lf = self.build_lazy()

        rate_name = None
        population_total = None
        if reweight_by is not None:
            schema = lf.collect_schema()
            if isinstance(reweight_by, str):
                if reweight_by not in schema:
                    raise ValueError(f"reweight_by column {reweight_by!r} not found.")
                if not schema[reweight_by].is_numeric():
                    raise TypeError("reweight_by column must be numeric.")
                denominator = pl.col(reweight_by).cast(pl.Float64)
            elif isinstance(reweight_by, Mapping):
                denominators = {}
                for key, value in reweight_by.items():
                    if not isinstance(value, Real) or isinstance(value, bool):
                        raise TypeError("reweight_by mapping values must be numeric.")
                    value = float(value)
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(
                            "reweight_by mapping values must be finite and positive."
                        )
                    denominators[key] = value
                axis = self._post_col if by == "pre" else self._pre_col
                observed = lf.select(axis).unique().collect()[axis].to_list()
                missing = [key for key in observed if key not in denominators]
                if missing:
                    raise ValueError(
                        f"reweight_by missing categories on {axis!r}: {missing!r}"
                    )
                category = pl.col(axis)
                if schema[axis] == pl.Categorical or isinstance(schema[axis], pl.Enum):
                    category = category.cast(pl.String)
                denominator = category.replace_strict(
                    denominators, default=None, return_dtype=pl.Float64
                )
                if mode == "enrichment":
                    population_total = sum(denominators.values())
            elif isinstance(reweight_by, pl.DataFrame):
                if reweight_by.width != 2:
                    raise ValueError(
                        "reweight_by DataFrame must have exactly two columns: category and denominator."
                    )
                keys, terms = reweight_by.get_columns()
                if keys.n_unique() != len(keys):
                    raise ValueError(
                        "reweight_by DataFrame category keys must be unique."
                    )
                if not terms.dtype.is_numeric():
                    raise TypeError(
                        "reweight_by DataFrame denominators must be numeric."
                    )
                terms = terms.cast(pl.Float64)
                if terms.null_count() or not (terms.is_finite() & (terms > 0)).all():
                    raise ValueError(
                        "reweight_by DataFrame denominators must be finite and positive."
                    )
                axis = self._post_col if by == "pre" else self._pre_col
                observed = lf.select(axis).unique().collect()[axis].to_list()
                supplied = set(keys.to_list())
                missing = [key for key in observed if key not in supplied]
                if missing:
                    raise ValueError(
                        f"reweight_by missing categories on {axis!r}: {missing!r}"
                    )
                category = pl.col(axis)
                if schema[axis] == pl.Categorical or isinstance(schema[axis], pl.Enum):
                    category = category.cast(pl.String)
                if keys.dtype == pl.Categorical or isinstance(keys.dtype, pl.Enum):
                    keys = keys.cast(pl.String)
                denominator = category.replace_strict(
                    keys, terms, default=None, return_dtype=pl.Float64
                )
                if mode == "enrichment":
                    population_total = terms.sum()
            else:
                raise TypeError(
                    "reweight_by must be a column name or a mapping or a two-column DataFrame."
                )
            if mode == "enrichment" and (
                not math.isfinite(population_total) or population_total <= 0
            ):
                raise ValueError(
                    "Enrichment requires a positive finite population total."
                )
            rate_name = unique_name("__rate__", self._current_columns())
            lf = lf.with_columns(
                pl.when(denominator.is_finite() & (denominator > 0))
                .then(pl.col(values) / denominator)
                .alias(rate_name)
            )

        if total_col is None:
            # Unique internal name so a user column literally named __total__
            # can't shadow the computed per-axis sum (which would silently
            # produce wrong fractions).
            total_name = unique_name("__total__", lf.collect_schema().names())
            numerator = rate_name or values
            total_values = values if mode == "enrichment" else numerator
            totals = lf.group_by(group_col).agg(pl.sum(total_values).alias(total_name))
            fraction = pl.col(numerator) / pl.col(total_name)
            if mode == "enrichment":
                fraction = fraction * population_total
            if rate_name is not None:
                fraction = pl.when(
                    pl.col(total_name).is_finite() & (pl.col(total_name) > 0)
                ).then(fraction)
            normalized = (
                lf.join(totals, on=group_col, how="left")
                .with_columns(fraction.alias(output_col))
                .drop(
                    [
                        total_name,
                        *([rate_name] if rate_name else []),
                        *([values] if values != output_col else []),
                    ]
                )
            )
        else:
            normalized = lf.with_columns(
                (pl.col(values) / pl.col(total_col)).alias("fraction")
            ).drop([values] if values != "fraction" else [])

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
        new.metadata = deepcopy(self.metadata)
        new._n_pair_base = None
        new._expression_sides = {}
        new._cache = None
        return new

    # ── materialization ────────────────────────────────────────────────────

    def to_dense(
        self,
        values: str | None = None,
        fill_value: float = 0,
        *,
        aggregate: str = "error",
        max_cells: int | None = 10_000_000,
        keep: str | list[str] | None = None,
        normalize: Literal["pre", "post"] | None = None,
    ) -> pl.DataFrame:
        """Pivot into a dense pre × post matrix DataFrame.

        Parameters
        ----------
        values : str or None, optional
            Column to use as matrix entries. Defaults to the first registered
            weight.
        normalize : {"pre", "post"} or None, optional
            Shorthand for ``.normalize(by=normalize, values=values)`` followed
            by pivoting the resulting ``fraction`` column, so the column that is
            normalized is guaranteed to be the one pivoted. ``"pre"`` makes rows
            sum to 1 (output budget); ``"post"`` makes columns sum to 1 (input
            budget). Fractions are of the currently visible table; for an
            external denominator, reweighting or enrichment, call
            :meth:`normalize` explicitly. Defaults to None (raw values).
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
        max_cells : int or None, optional
            Refuse a matrix exceeding this many entries before collecting edges
            or allocating the dense pivot. Defaults to 10 million. None disables
            the guard; use to_sparse for larger matrices.
        keep : str, list of str, or None, optional
            Additional columns to retain beside the pre entity ID. Each must
            have exactly one distinct value per pre entity, including nulls.
            Available pre-side cell annotations are typical examples. Kept
            columns retain their dtype and nulls and are not filled with
            ``fill_value``. Pair axes and the pivot value cannot be kept.

        Returns
        -------
        pl.DataFrame
            Columns: ``pre_col``, kept columns, then one column per observed
            post entity. Enum post categories follow their declared order;
            a null category, if observed, appears last. Unobserved Enum
            categories are not added automatically.
        """
        if normalize is not None:
            return self.normalize(by=normalize, values=values).to_dense(
                values="fraction",
                fill_value=fill_value,
                aggregate=aggregate,
                max_cells=max_cells,
                keep=keep,
            )
        values = self._resolve_values(values)
        if values not in self._current_columns():
            raise ValueError(f"values={values!r} not found in table.")
        kept = [keep] if isinstance(keep, str) else list(keep or [])
        if len(set(kept)) != len(kept):
            raise ValueError("keep columns must be unique.")
        reserved = {self._pre_col, self._post_col, values}
        if reserved.intersection(kept):
            raise ValueError("keep cannot contain pair axes or the pivot value column.")
        missing = set(kept).difference(self._current_columns())
        if missing:
            raise ValueError(
                f"keep columns not found in the current table: {sorted(missing)!r}"
            )
        lf = self.build_lazy().select([self._pre_col, self._post_col, values, *kept])
        if max_cells is not None:
            if max_cells < 0:
                raise ValueError("max_cells must be non-negative or None.")
            n_pre, n_post = (
                lf.select(
                    pl.col(self._pre_col).n_unique().alias("pre"),
                    pl.col(self._post_col).n_unique().alias("post"),
                )
                .collect()
                .row(0)
            )
            if n_pre * n_post > max_cells:
                raise ValueError(
                    f"Dense matrix would have {n_pre:,} × {n_post:,} "
                    f"= {n_pre * n_post:,} entries, exceeding max_cells={max_cells:,}. "
                    "Use to_sparse(), or explicitly increase/disable max_cells."
                )
        df = lf.collect()
        if kept:
            conflicts = (
                df.group_by(self._pre_col)
                .agg(pl.col(kept).n_unique())
                .filter(pl.any_horizontal(pl.col(kept) > 1))
            )
            if conflicts.height:
                raise ValueError(
                    "keep columns must have a single value per pre entity; conflicting values found."
                )
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
        dense = df.pivot(
            on=self._post_col,
            index=self._pre_col,
            values=values,
            aggregate_function=agg_fn,
        ).with_columns(pl.exclude(self._pre_col).fill_null(fill_value))
        post_dtype = df.schema[self._post_col]
        if isinstance(post_dtype, pl.Enum):
            ordered = [
                label
                for label in post_dtype.categories
                if label in dense.columns and label != self._pre_col
            ]
            trailing = [
                col
                for col in dense.columns
                if col != self._pre_col and col not in ordered
            ]
            dense = dense.select(self._pre_col, *ordered, *trailing)
        if kept:
            collisions = set(kept).intersection(dense.columns)
            if collisions:
                raise ValueError(
                    f"keep columns collide with pivot output columns: {sorted(collisions)!r}"
                )
            properties = df.select(self._pre_col, *kept).unique(subset=self._pre_col)
            dense = dense.join(
                properties, on=self._pre_col, how="left", nulls_equal=True
            )
            pivot_columns = [
                col for col in dense.columns if col != self._pre_col and col not in kept
            ]
            dense = dense.select(self._pre_col, *kept, *pivot_columns)
        return dense

    def to_sparse(
        self,
        values: str | None = None,
        *,
        pre_ids: Iterable | None = None,
        post_ids: Iterable | None = None,
        aggregate: str = "error",
        normalize: Literal["pre", "post"] | None = None,
    ) -> tuple:
        """Export a CSR array and its explicit row and column identities.

        Parameters
        ----------
        values : str or None, optional
            Numeric entry column, defaulting to the first registered weight.
        normalize : {"pre", "post"} or None, optional
            Normalize ``values`` by pre or post totals before export, exactly
            as in :meth:`to_dense`. Defaults to None (raw values).
        pre_ids, post_ids : iterable or None, optional
            Ordered axis identities. Supply universe IDs to include cells with
            zero observed connections. Defaults to sorted observed identities.
            Supplied axes must be unique and contain every observed identity.
        aggregate : {'error', 'sum', 'mean', 'min', 'max', 'first'}, optional
            Reducer for duplicate pairs. Defaults to rejecting duplicate entries.

        Returns
        -------
        tuple
            ``(scipy.sparse.csr_array, pre_entities, post_entities)``. The latter
            two are Polars Series whose order defines the matrix axes. Absent
            entries are zero. No dense matrix or full pair grid is constructed.

        Notes
        -----
        Requires scipy. Collects only observed IDs and the selected value column.
        Null entries and null axis identities are rejected; filter or fill them
        explicitly before export rather than silently interpreting them as zero.
        """
        try:
            from scipy.sparse import coo_array
        except ImportError as exc:
            raise ImportError(
                "Sparse export requires scipy. Install trajan[sparse]."
            ) from exc
        if normalize is not None:
            return self.normalize(by=normalize, values=values).to_sparse(
                values="fraction",
                pre_ids=pre_ids,
                post_ids=post_ids,
                aggregate=aggregate,
            )
        if aggregate not in {"error", "sum", "mean", "min", "max", "first"}:
            raise ValueError("Unsupported sparse duplicate reducer.")
        values = self._resolve_values(values)
        df = self.build_lazy().select(self._pre_col, self._post_col, values).collect()
        if not df.schema[values].is_numeric():
            raise ValueError("Sparse entries must be numeric.")
        if any(df[c].null_count() for c in df.columns):
            raise ValueError("Sparse export requires non-null axis IDs and values.")
        keys = [self._pre_col, self._post_col]
        if df.height != df.select(keys).n_unique():
            if aggregate == "error":
                raise ValueError(
                    "to_sparse found duplicate pairs; specify aggregate='sum' or another reducer."
                )
            df = df.group_by(keys).agg(
                getattr(pl.col(values), aggregate)().alias(values)
            )

        def _axis(col, ids):
            axis = (
                df[col].unique().sort()
                if ids is None
                else pl.Series(col, list(ids), dtype=df.schema[col])
            )
            if axis.null_count() or axis.n_unique() != len(axis):
                raise ValueError(
                    f"Axis {col!r} must contain unique, non-null identities."
                )
            return axis

        pre = _axis(self._pre_col, pre_ids)
        post = _axis(self._post_col, post_ids)
        import numpy as np

        try:
            row = (
                df[self._pre_col]
                .replace_strict(pre, pl.Series(np.arange(len(pre))))
                .to_numpy()
            )
            col = (
                df[self._post_col]
                .replace_strict(post, pl.Series(np.arange(len(post))))
                .to_numpy()
            )
        except pl.exceptions.InvalidOperationError as exc:
            raise ValueError(
                "Supplied axes must contain every observed identity."
            ) from exc
        matrix = coo_array(
            (df[values].to_numpy(), (row, col)), shape=(len(pre), len(post))
        ).tocsr()
        return matrix, pre, post

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
            "metadata": deepcopy(self.metadata),
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
        with _folio_batch(folio):
            folio.add("config", config, overwrite=overwrite)
            folio.add("pairs", self._pair_lf.collect(), overwrite=overwrite)
            for name, spec in self._cell_annotations.items():
                _folio_add(
                    folio,
                    f"ann_{name}",
                    spec.lf.collect(),
                    overwrite=overwrite,
                    described_as=f"annotation {name!r}",
                )

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
            "side": spec.side,
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
        with _folio_read(folio):
            config = folio.get("config")
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
                return _folio_scan(folio, name)

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
                    side=ann_meta.get("side", "both"),
                )

            # Named expressions — binary format first (round-trips literals
            # correctly), with a JSON fallback for older saved folios.
            for name, expr_val in config.get("expressions", {}).items():
                expr = None
                try:
                    expr = pl.Expr.deserialize(
                        base64.b64decode(expr_val), format="binary"
                    )
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

            instance.metadata = deepcopy(config.get("metadata", {}))
            return instance
