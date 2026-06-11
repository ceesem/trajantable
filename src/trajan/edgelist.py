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

from ._base import (
    CellAnnotationSpec,
    build_cell_annotation_spec,
    build_pair_plan,
    filter_by_id_sets,
)
from .connectivity_table import ConnectivityTable
from .spatial import bbox_predicate, euclidean_distance


class EdgeList(ConnectivityTable):
    """A pair-level connectivity table where both axes are cells.

    Constructed either by direct frame passing (for published edgelists etc.)
    or as the output of ``SynapseTable.edgelist()``.

    The cell invariant is documented rather than statically enforced: a
    ConnectivityTable has no distinguished "kind" on its axes, so EdgeList's
    promise is semantic, not type-checked. Spatial filters resolve to the
    position-bearing cell annotation declared via ``position_col=`` at
    registration.

    EdgeList strengthens ``ConnectivityTable.add_annotation`` to accept
    ``join_on_alias=``, mirroring ``SynapseTable.add_cell_annotation``. An
    alias is a (annotation_name, col) pair: the named column on the source
    annotation provides cell ids that downstream annotations can join on
    instead of root pre/post ids. Aliases survive ``SynapseTable.edgelist()``
    because they are 1:1 with root cell ids by contract.

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
    cell_aliases : dict[str, tuple[str, str]] or None, optional
        Pre-populated cell-alias registry: ``{alias_name: (source_annotation,
        col_in_source)}``. Typically passed by ``SynapseTable.edgelist()`` to
        carry forward aliases registered on the source SynapseTable. Direct
        users normally start empty and grow the registry via ``set_cell_alias``.
    """

    # Overrides ConnectivityTable._TYPE_TAG so save() writes "EdgeList" and
    # ConnectivityTable.load() can dispatch back to this class when the
    # saved folio contains an EdgeList.
    _TYPE_TAG: str = "EdgeList"

    def __init__(
        self,
        df,
        pre_col: str,
        post_col: str,
        weight_cols: list[str] | str | None = None,
        *,
        cell_aliases: dict[str, tuple[str, str]] | None = None,
    ):
        super().__init__(
            df, pre_col=pre_col, post_col=post_col, weight_cols=weight_cols
        )
        self._cell_aliases: dict[str, tuple[str, str]] = dict(cell_aliases or {})

    # ── alias registration ────────────────────────────────────────────────

    @property
    def cell_aliases(self) -> dict[str, tuple[str, str]]:
        """Registered cell aliases: ``{alias_name: (annotation_name, col)}``."""
        return dict(self._cell_aliases)

    def set_cell_alias(
        self,
        annotation_name: str,
        col: str,
        alias_name: str | None = None,
    ) -> EdgeList:
        """Declare a cell alias column produced by a registered annotation.

        Once registered, additional annotations can be added with
        ``join_on_alias=<alias_name>`` to key on the aliased ids rather than
        the table's root pre/post ids. Mirrors the SynapseTable method.

        Parameters
        ----------
        annotation_name : str
            Name of an already-registered annotation whose data columns
            include ``col``. The source annotation must itself join on root
            ids (``join_on_alias=None``).
        col : str
            Column within the source annotation that carries the aliased ids.
        alias_name : str or None, optional
            Name under which to register the alias. Defaults to
            ``annotation_name``.
        """
        if annotation_name not in self._cell_annotations:
            raise KeyError(f"No annotation named {annotation_name!r}")
        spec = self._cell_annotations[annotation_name]
        if spec.join_on_alias is not None:
            raise ValueError(
                f"Annotation {annotation_name!r} uses join_on_alias="
                f"{spec.join_on_alias!r} and cannot itself be an alias source. "
                f"The alias source must join on root id (join_on_alias=None)."
            )
        if col not in spec.data_cols:
            raise ValueError(
                f"Column {col!r} not found in annotation {annotation_name!r}. "
                f"Available columns: {spec.data_cols}"
            )
        key = alias_name if alias_name is not None else annotation_name
        self._cell_aliases[key] = (annotation_name, col)
        return self

    # ── annotation registration (overrides ConnectivityTable) ─────────────

    def add_annotation(
        self,
        name: str,
        df,
        cell_id_col: str,
        position_col: str | None = None,
        is_universe: bool = False,
        join_on_alias: str | None = None,
    ) -> EdgeList:
        """Register a per-cell annotation, optionally keyed on a cell alias.

        Extends :meth:`ConnectivityTable.add_annotation` with the
        ``join_on_alias`` kwarg. With ``join_on_alias=None`` (the default),
        the annotation is joined symmetrically on the EdgeList's
        ``pre_col`` / ``post_col``. With ``join_on_alias=<alias>``, the
        annotation joins on the alias-source's ``{alias_col}_pre`` /
        ``{alias_col}_post`` columns produced by the source annotation.

        See :class:`ConnectivityTable` for the full parameter list.
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
        self._cache = None
        return self

    # ── lazy plan (overrides to handle alias-keyed joins) ─────────────────

    def build_lazy(self) -> pl.LazyFrame:
        """Construct the annotated + filtered lazy plan, honoring aliases.

        Joins non-aliased annotations (which provide alias source columns)
        before aliased ones, in registration order. This matches the
        SynapseTable contract that an alias source must be registered before
        any annotation that joins on it.
        """
        return build_pair_plan(
            self._pair_lf,
            self._cell_annotations,
            pre_col=self._pre_col,
            post_col=self._post_col,
            aliases=self._cell_aliases,
            expressions=self._expressions,
            filters=self._filters,
        )

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
        return filter_by_id_sets(self, pre_ids, post_ids)

    def filter_by_soma_distance(
        self,
        max_distance: float,
        *,
        annotation: str | None = None,
        distance_fn: Callable[[str, str], pl.Expr] = euclidean_distance,
    ) -> EdgeList:
        """Keep pairs whose pre / post soma-soma distance is ``<= max_distance``.

        Positions are looked up from the registered cell annotation whose
        ``position_col`` was set at ``add_annotation`` time. The joined
        per-side columns are ``{position_col}_pre`` / ``{position_col}_post``.

        Parameters
        ----------
        max_distance : float
            Maximum distance to retain. Units match the position columns.
        annotation : str or None, optional
            Name of the annotation whose ``position_col`` to use. If ``None``
            (default), uses the unique position-bearing annotation; raises if
            zero or more than one are registered.
        distance_fn : callable
            Takes two position-column names and returns a ``pl.Expr`` for the
            distance. Defaults to 3-D Euclidean. Use ``radial_distance`` for
            lateral-only.
        """
        ann_name = self._resolve_position_annotation(annotation)
        pos_col = self._cell_annotations[ann_name].position_col
        return self.filter(
            distance_fn(f"{pos_col}_pre", f"{pos_col}_post") <= max_distance
        )

    def filter_by_bbox(
        self,
        bbox,
        *,
        annotation: str | None = None,
    ) -> EdgeList:
        """Keep pairs where both pre and post soma positions fall inside a bbox.

        Unlike ``SynapseTable.filter_by_bbox`` (which filters on a single
        synapse position), EdgeList's bbox filter checks both cells — a pair
        is kept only if both somas lie within the box. Positions are looked up
        from the position-bearing cell annotation (see
        ``filter_by_soma_distance`` for resolution rules).

        Parameters
        ----------
        bbox : Sequence
            ((xmin, ymin, zmin), (xmax, ymax, zmax)).
        annotation : str or None, optional
            Name of the annotation whose ``position_col`` to use; auto-resolved
            when exactly one position-bearing annotation is registered.
        """
        ann_name = self._resolve_position_annotation(annotation)
        pos_col = self._cell_annotations[ann_name].position_col
        return self.filter(
            bbox_predicate(f"{pos_col}_pre", bbox)
            & bbox_predicate(f"{pos_col}_post", bbox)
        )

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
        new._cell_annotations = self._cell_annotations.copy()
        new._filters = self._filters.copy()
        new._filter_sides = self._filter_sides.copy()
        new._expressions = self._expressions.copy()
        new._expression_sides = self._expression_sides.copy()
        new._cell_aliases = self._cell_aliases.copy()
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

    # ── persistence hooks ──────────────────────────────────────────────────

    def _spec_to_config(self, spec: CellAnnotationSpec) -> dict:
        config = super()._spec_to_config(spec)
        config["join_on_alias"] = spec.join_on_alias
        return config

    def _extra_save_config(self) -> dict:
        return {
            "cell_aliases": {
                alias_name: {"annotation_name": ann_name, "col": col}
                for alias_name, (ann_name, col) in self._cell_aliases.items()
            },
        }

    @classmethod
    def load(cls, folio) -> EdgeList:
        """Load an EdgeList from a DataFolio.

        Loads non-aliased annotations first (so their data columns are
        available as alias sources), then sets cell aliases, then loads
        aliased annotations. This avoids the join-key validation race that
        would arise from a single in-order loop.
        """
        import base64
        import warnings

        from ._base import _as_folio

        folio = _as_folio(folio)
        config = folio.get_json("config")
        saved_type = config.get("__type__", "ConnectivityTable")
        if saved_type != cls._TYPE_TAG:
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

        annotations = config.get("annotations", {})
        # Non-aliased pass — these may be alias sources.
        for name, ann_meta in annotations.items():
            if ann_meta.get("join_on_alias") is None:
                instance.add_annotation(
                    name,
                    _lf(f"ann_{name}"),
                    cell_id_col=ann_meta["cell_id_col"],
                    position_col=ann_meta.get("position_col"),
                    is_universe=ann_meta.get("is_universe", False),
                )
        # Cell aliases must be set after their source annotations are present.
        for alias_name, alias_meta in config.get("cell_aliases", {}).items():
            instance.set_cell_alias(
                alias_meta["annotation_name"],
                alias_meta["col"],
                alias_name=alias_name,
            )
        # Aliased annotations now find their alias.
        for name, ann_meta in annotations.items():
            if ann_meta.get("join_on_alias") is not None:
                instance.add_annotation(
                    name,
                    _lf(f"ann_{name}"),
                    cell_id_col=ann_meta["cell_id_col"],
                    position_col=ann_meta.get("position_col"),
                    is_universe=ann_meta.get("is_universe", False),
                    join_on_alias=ann_meta["join_on_alias"],
                )

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

        saved_sides = config.get("filter_sides")
        for i, f_json in enumerate(config.get("filters", [])):
            instance = instance.filter(
                pl.Expr.deserialize(f_json.encode(), format="json")
            )
            if saved_sides is not None and i < len(saved_sides):
                instance._filter_sides[-1] = saved_sides[i]

        return instance
