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

from ._base import CellAnnotationSpec, _as_folio, _to_lazy


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
        self._annotations: dict[str, CellAnnotationSpec] = {}

        self._filters: list[pl.Expr] = []
        self._expressions: dict[str, pl.Expr] = {}
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
        return list(self._annotations)

    def __repr__(self) -> str:
        if self._cache is not None:
            n = len(self._cache)
        elif not self._filters:
            n = self._pair_lf.select(pl.len()).collect().item()
        else:
            n = "uncached"
        return (
            f"ConnectivityTable(n_pairs={n}, "
            f"pre_col={self._pre_col!r}, post_col={self._post_col!r}, "
            f"weights={self._weights}, "
            f"annotations={list(self._annotations)})"
        )

    # ── registration: annotation / weight / expression / filter ────────────

    def add_annotation(
        self,
        name: str,
        df,
        entity_id_col: str,
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
            Annotation frame. Must contain ``entity_id_col`` plus one or more
            data columns. Each ``entity_id_col`` value must be unique.
        entity_id_col : str
            Column in ``df`` whose values are joined against both ``pre_col``
            and ``post_col``.
        """
        lf = _to_lazy(df)
        schema = lf.collect_schema().names()
        if entity_id_col not in schema:
            raise ValueError(
                f"entity_id_col {entity_id_col!r} not found in annotation. "
                f"Available: {schema}"
            )
        n_total = lf.select(pl.len()).collect().item()
        n_unique = lf.select(pl.col(entity_id_col).n_unique()).collect().item()
        if n_total != n_unique:
            raise ValueError(
                f"Annotation key {entity_id_col!r} has {n_total - n_unique} "
                "duplicate value(s); each entity id must appear at most once."
            )

        data_cols = [c for c in schema if c != entity_id_col]
        new_cols = {f"{c}_pre" for c in data_cols} | {f"{c}_post" for c in data_cols}
        collisions = new_cols & self._current_columns()
        if collisions:
            raise ValueError(f"Columns already exist in table: {sorted(collisions)}")

        self._annotations[name] = CellAnnotationSpec(
            lf=lf,
            cell_id_col=entity_id_col,
            join_on_alias=None,
            data_cols=data_cols,
        )
        self._cache = None
        return self

    def remove_annotation(self, name: str) -> ConnectivityTable:
        if name not in self._annotations:
            raise KeyError(f"No annotation named {name!r}")
        del self._annotations[name]
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
        self._cache = None
        return self

    def filter(self, expr: pl.Expr) -> ConnectivityTable:
        """Return a new ConnectivityTable with ``expr`` applied to the lazy plan."""
        new = self._copy()
        new._filters = self._filters + [expr]
        return new

    # ── plan construction ──────────────────────────────────────────────────

    def _current_columns(self) -> set[str]:
        cols = set(self._pair_col_names)
        for spec in self._annotations.values():
            cols |= {f"{c}_pre" for c in spec.data_cols}
            cols |= {f"{c}_post" for c in spec.data_cols}
        cols |= set(self._expressions)
        return cols

    def build_lazy(self) -> pl.LazyFrame:
        """Construct (without collecting) the annotated + filtered lazy plan."""
        lf = self._pair_lf
        for spec in self._annotations.values():
            pre_lf = spec.lf.rename({c: f"{c}_pre" for c in spec.data_cols})
            lf = lf.join(
                pre_lf,
                left_on=self._pre_col,
                right_on=spec.cell_id_col,
                how="left",
            )
            post_lf = spec.lf.rename({c: f"{c}_post" for c in spec.data_cols})
            lf = lf.join(
                post_lf,
                left_on=self._post_col,
                right_on=spec.cell_id_col,
                how="left",
            )
        for expr in self._expressions.values():
            lf = lf.with_columns(expr)
        for f in self._filters:
            lf = lf.filter(f)
        return lf

    @property
    def pairs(self) -> pl.DataFrame:
        """Full merged pair-level table with annotations joined. Cached."""
        if self._cache is None:
            self._cache = self.build_lazy().collect()
        return self._cache

    # ── copy ───────────────────────────────────────────────────────────────

    def _copy(self) -> ConnectivityTable:
        new = ConnectivityTable.__new__(ConnectivityTable)
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

        lf = self.build_lazy()

        if total_col is None:
            totals = lf.group_by(group_col).agg(pl.sum(values).alias("__total__"))
            normalized = (
                lf.join(totals, on=group_col, how="left")
                .with_columns((pl.col(values) / pl.col("__total__")).alias("fraction"))
                .drop(["__total__", values])
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
        new._annotations = {}
        new._filters = []
        new._expressions = {}
        new._cache = None
        return new

    # ── materialization ────────────────────────────────────────────────────

    def to_dense(
        self, values: str | None = None, fill_value: float = 0
    ) -> pl.DataFrame:
        """Pivot into a dense pre × post matrix DataFrame.

        Parameters
        ----------
        values : str or None, optional
            Column to use as matrix entries. Defaults to the first registered
            weight.
        fill_value : float, optional
            Value for (pre, post) pairs absent from the plan.

        Returns
        -------
        pl.DataFrame
            Columns: ``pre_col`` followed by one column per post entity.
        """
        values = self._resolve_values(values)
        if values not in self._current_columns():
            raise ValueError(f"values={values!r} not found in table.")
        lf = self.build_lazy().select([self._pre_col, self._post_col, values])
        return (
            lf.collect()
            .pivot(
                on=self._post_col,
                index=self._pre_col,
                values=values,
                aggregate_function="first",
            )
            .fill_null(fill_value)
        )

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
            "expressions": {
                name: base64.b64encode(expr.meta.serialize(format="binary")).decode(
                    "ascii"
                )
                for name, expr in self._expressions.items()
            },
            "annotations": {
                name: {
                    "entity_id_col": spec.cell_id_col,
                    "data_cols": list(spec.data_cols),
                }
                for name, spec in self._annotations.items()
            },
        }
        folio.add_json("config", config, overwrite=overwrite)
        folio.add_table("pairs", self._pair_lf.collect(), overwrite=overwrite)
        for name, spec in self._annotations.items():
            folio.add_table(f"ann_{name}", spec.lf.collect(), overwrite=overwrite)

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
                entity_id_col=ann_meta["entity_id_col"],
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
        for f_json in config.get("filters", []):
            instance = instance.filter(
                pl.Expr.deserialize(f_json.encode(), format="json")
            )

        return instance
