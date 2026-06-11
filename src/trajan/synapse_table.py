from __future__ import annotations

import base64
import math
import warnings
from pathlib import Path
from typing import Callable, Union

import polars as pl

from ._base import (
    CellAnnotationSpec,
    SynapseAnnotationSpec,
    VertexAnnotationSpec,
    _AnnotationProxy,
    _as_folio,
    _auto_pack,
    _to_lazy,
    apply_plan_tail,
    classify_by_cell_sides,
    filter_by_id_sets,
    join_cell_annotations_symmetric,
    reject_reserved_names,
    resolve_position_annotation,
    resolve_universe_annotation,
    validate_unique_key,
)
from .connectivity_table import ConnectivityTable
from .edgelist import EdgeList
from .spatial import bbox_predicate, euclidean_distance, spatial_feature_exprs

try:
    import pandas as pd  # noqa: F401  (still used elsewhere for duck-typing)

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def _spatial_col_name(prefix: str, feature: str) -> str:
    return f"{prefix}_{feature}" if prefix else feature


class SynapseTable:
    """Synapse list with automatic pre/post cell annotation merging.

    Owns the synapse list → edgelist → connectivity matrix → normalized
    connectivity pipeline. Cell annotations are merged symmetrically:
    adding an annotation named "cell_type" with a cell_type column produces
    cell_type_pre and cell_type_post on .df automatically.

    Built on Polars lazy frames; the merged table is cached and invalidated
    only when annotations are added or removed.

    Parameters
    ----------
    syn_df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
        Synapse table, one row per synapse.
    pre_col : str, optional
        Column name for the pre-synaptic cell id.
    post_col : str, optional
        Column name for the post-synaptic cell id.
    id_col : str, optional
        Column name for the synapse id (used to join synapse annotations).
    synapse_position_col : str or None, optional
        Column in syn_df holding synapse positions as a struct with x, y, z fields.
        Required for filter_by_bbox.

    Notes
    -----
    Soma-position support is now declared on the cell annotation itself, via
    ``add_cell_annotation(..., position_col=<col>)``. Spatial filters
    (``filter_by_soma_distance``, ``add_spatial_features``) auto-resolve the
    position-bearing annotation; pass ``annotation=<name>`` when more than one
    is registered.

    How ``.df`` is assembled
    ------------------------
    ``.df`` is built lazily by ``build_lazy()`` and cached. The plan is:

    1. Start from the base synapse table (``synapses`` / ``synapses_lazy``).
    2. Join each registered **synapse** annotation on ``id_col``
       (contributes its columns verbatim).
    3. Join each registered **cell** annotation symmetrically on
       ``pre_col`` / ``post_col`` (or an alias key) — each data column
       ``c`` produces ``c_pre`` and ``c_post``.
    4. Join each registered **vertex** annotation on the declared
       ``pre_vertex_col`` / ``post_vertex_col`` — same ``c_pre`` / ``c_post``
       output convention.
    5. Apply each registered **expression** in registration order via
       ``with_columns`` (so later expressions can reference earlier ones).
    6. Apply each accumulated **filter** in order via ``filter``.

    To inspect any single piece without building the whole thing:

    - ``st.synapses`` — base synapse DataFrame (step 1 only)
    - ``st.synapse_annotations[name]`` — a specific synapse annotation frame
    - ``st.cell_annotations[name]`` — a specific cell annotation frame
    - ``st.vertex_annotations[name]`` — a specific vertex annotation frame
    - ``st.build_lazy()`` — the full plan as a ``pl.LazyFrame`` (no collect)
    - ``st.info()`` — a printed summary of everything registered.
    """

    def __init__(
        self,
        syn_df,
        pre_col: str = "pre_pt_root_id",
        post_col: str = "post_pt_root_id",
        id_col: str = "id",
        synapse_position_col: str | None = None,
    ):
        self._syn_lf = _auto_pack(_to_lazy(syn_df), synapse_position_col)
        self._pre_col = pre_col
        self._post_col = post_col
        self._id_col = id_col
        self._synapse_position_col = synapse_position_col

        # Cache base schema names once — _syn_lf never changes
        self._syn_col_names: list[str] = self._syn_lf.collect_schema().names()

        # Registered annotations, keyed by user-chosen name.
        # See src/trajan/_base.py for the spec dataclass definitions.
        self._synapse_annotations: dict[str, SynapseAnnotationSpec] = {}
        self._cell_annotations: dict[str, CellAnnotationSpec] = {}
        self._vertex_annotations: dict[str, VertexAnnotationSpec] = {}

        self._filters: list[pl.Expr] = []
        # Parallel to _filters: cell-side classification ('pre' / 'post' / 'both' / None)
        # captured at filter() time against the annotations registered then.
        # Drives cells() and possible_pairs() side-decomposed projection.
        self._filter_sides: list[str | None] = []
        # name → aliased pl.Expr, applied in insertion order after joins and before filters
        self._expressions: dict[str, pl.Expr] = {}
        # free-form user metadata; persisted via folio.metadata on save/load
        self.metadata: dict = {}
        # name → 'pre', 'post', 'both', or None — classified at add_expression() time
        self._expression_sides: dict[str, str | None] = {}
        # columns to auto-sum in all synapse aggregations (edgelist, type_edgelist)
        self._weights: list[str] = []
        # alias_name → (annotation_name, col_in_annotation)
        # populated via set_cell_alias() or alias_col in add_cell_annotation()
        self._cell_aliases: dict[str, tuple[str, str]] = {}
        self._cache: pl.DataFrame | None = None
        self._n_syn_base: int = self._syn_lf.select(pl.len()).collect().item()

    # ── repr / info ───────────────────────────────────────────────────────

    def _is_position_col(self, lf: pl.LazyFrame, col: str) -> bool:
        """Check if col is a struct with x, y, z fields (a packed position)."""
        dtype = lf.collect_schema()[col]
        if dtype == pl.Struct:
            field_names = {f.name for f in dtype.fields}
            return {"x", "y", "z"} <= field_names
        return False

    def info(self) -> str:
        """Summarize the table structure: core columns, annotations, expressions, and weights.

        Prints and returns a human-readable string showing every "blessed" column
        in the base synapse table and each registered annotation, how those columns
        appear in ``.df``, cell aliases, expression dependencies, and weights.
        """
        lines: list[str] = []

        # ── header
        if self._cache is not None:
            n = len(self._cache)
        elif not self._filters:
            n = self._n_syn_base
        else:
            n = self.build_lazy().select(pl.len()).collect().item()
        lines.append(f"SynapseTable  ({n:,} synapses, {len(self._filters)} filter(s))")
        lines.append("")

        # ── core columns
        lines.append("Core columns")
        lines.append(f"  pre_col            : {self._pre_col}")
        lines.append(f"  post_col           : {self._post_col}")
        lines.append(f"  id_col             : {self._id_col}")
        if self._synapse_position_col:
            lines.append(f"  synapse_position   : {self._synapse_position_col}")
        for ann_name, spec in self._cell_annotations.items():
            if spec.position_col is not None:
                lines.append(
                    f"  soma_position      : {spec.position_col} "
                    f"(from annotation {ann_name!r})"
                )

        # ── base synapse columns
        other_cols = [
            c
            for c in self._syn_col_names
            if c
            not in {
                self._pre_col,
                self._post_col,
                self._id_col,
                self._synapse_position_col,
            }
        ]
        if other_cols:
            lines.append("")
            lines.append(f"Base synapse columns ({len(other_cols)})")
            for c in other_cols:
                lines.append(f"  {c}")

        # ── synapse annotations
        if self._synapse_annotations:
            lines.append("")
            lines.append(f"Synapse annotations ({len(self._synapse_annotations)})")
            for name, spec in self._synapse_annotations.items():
                lines.append(f"  {name!r} ({len(spec.data_cols)} col(s))")
                for c in spec.data_cols:
                    tag = "  [position]" if self._is_position_col(spec.lf, c) else ""
                    lines.append(f"    {c}{tag}")

        # ── cell annotations
        if self._cell_annotations:
            lines.append("")
            lines.append(f"Cell annotations ({len(self._cell_annotations)})")
            for name, spec in self._cell_annotations.items():
                join_info = (
                    f"join on alias {spec.join_on_alias!r}"
                    if spec.join_on_alias
                    else f"join on {spec.cell_id_col!r}"
                )
                role_tags = []
                if spec.is_universe:
                    role_tags.append("universe")
                if spec.position_col is not None:
                    role_tags.append(f"position={spec.position_col!r}")
                role_str = f"  [{', '.join(role_tags)}]" if role_tags else ""
                lines.append(
                    f"  {name!r} ({len(spec.data_cols)} col(s), {join_info}){role_str}"
                )
                for c in spec.data_cols:
                    tag = "  [position]" if self._is_position_col(spec.lf, c) else ""
                    lines.append(f"    {c}  ->  {c}_pre, {c}_post{tag}")

        # ── cell aliases
        if self._cell_aliases:
            lines.append("")
            lines.append(f"Cell aliases ({len(self._cell_aliases)})")
            for alias_name, (ann_name, col) in self._cell_aliases.items():
                lines.append(f"  {alias_name!r}  :  {col} from {ann_name!r}")

        # ── vertex annotations
        if self._vertex_annotations:
            lines.append("")
            lines.append(f"Vertex annotations ({len(self._vertex_annotations)})")
            for name, spec in self._vertex_annotations.items():
                sides = []
                if spec.pre_vertex_col:
                    sides.append(f"pre via {spec.pre_vertex_col!r}")
                if spec.post_vertex_col:
                    sides.append(f"post via {spec.post_vertex_col!r}")
                lines.append(
                    f"  {name!r} ({len(spec.data_cols)} col(s), {', '.join(sides)})"
                )
                for c in spec.data_cols:
                    suffixes = []
                    if spec.pre_vertex_col:
                        suffixes.append(f"{c}_pre")
                    if spec.post_vertex_col:
                        suffixes.append(f"{c}_post")
                    tag = "  [position]" if self._is_position_col(spec.lf, c) else ""
                    lines.append(f"    {c}  ->  {', '.join(suffixes)}{tag}")

        # ── expressions
        if self._expressions:
            lines.append("")
            lines.append(f"Expressions ({len(self._expressions)})")
            for name, expr in self._expressions.items():
                side = self._expression_sides.get(name)
                roots = expr.meta.root_names()
                parts = [name]
                if side:
                    parts.append(f"({side})")
                if roots:
                    parts.append(f"<-  {', '.join(roots)}")
                lines.append(f"  {' '.join(parts)}")

        # ── weights
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
            n = self._n_syn_base
        else:
            n = "uncached"
        return (
            f"SynapseTable("
            f"n_syn={n}, "
            f"synapse_annotations={list(self._synapse_annotations)}, "
            f"cell_annotations={list(self._cell_annotations)}, "
            f"vertex_annotations={list(self._vertex_annotations)}, "
            f"expressions={list(self._expressions)}, "
            f"weights={self._weights})"
        )

    def __len__(self) -> int:
        """Number of synapses after filters; collects lazily if needed."""
        if self._cache is not None:
            return len(self._cache)
        if not self._filters:
            return self._n_syn_base
        return self.build_lazy().select(pl.len()).collect().item()

    def __bool__(self) -> bool:
        # Always truthy: prevents `if st:` from implicitly forcing a collect
        # via __len__. Use `len(st) > 0` for the row-count check.
        return True

    # ── annotation name lists and accessors ───────────────────────────────

    @property
    def synapse_annotation_names(self) -> list[str]:
        return list(self._synapse_annotations)

    @property
    def cell_annotation_names(self) -> list[str]:
        return list(self._cell_annotations)

    @property
    def vertex_annotation_names(self) -> list[str]:
        return list(self._vertex_annotations)

    @property
    def synapse_annotations(self) -> _AnnotationProxy:
        """Read-only view over registered synapse annotations.

        Use ``st.synapse_annotations["name"]`` to retrieve a named annotation
        as a collected ``pl.DataFrame``. Supports ``in``, iteration, and ``len``.
        """
        return _AnnotationProxy(self._synapse_annotations)

    @property
    def cell_annotations(self) -> _AnnotationProxy:
        """Read-only view over registered cell annotations.

        Use ``st.cell_annotations["name"]`` to retrieve a named annotation
        as a collected ``pl.DataFrame``. Supports ``in``, iteration, and ``len``.
        """
        return _AnnotationProxy(self._cell_annotations)

    @property
    def vertex_annotations(self) -> _AnnotationProxy:
        """Read-only view over registered vertex annotations.

        Use ``st.vertex_annotations["name"]`` to retrieve a named annotation
        as a collected ``pl.DataFrame``. Supports ``in``, iteration, and ``len``.
        """
        return _AnnotationProxy(self._vertex_annotations)

    @property
    def expression_names(self) -> list[str]:
        return list(self._expressions)

    @property
    def filter_sides(self) -> list[str | None]:
        """Cell-side classification of each accumulated filter.

        Parallel to ``self._filters``. Each entry is ``'pre'``, ``'post'``,
        ``'both'``, or ``None`` — captured at ``filter()`` time against the
        cell annotations registered then. Classification does not retroactively
        change when annotations are added or removed later (matching the
        existing ``expression_sides`` caveat).

        Consumed by ``cells()`` and ``possible_pairs()`` to project cell-level
        filters onto per-cell and per-pair views without re-applying synapse-
        or vertex-level constraints that have no cell-level meaning.
        """
        return list(self._filter_sides)

    @property
    def expression_sides(self) -> dict[str, str | None]:
        """Cell-level classification of each expression: 'pre', 'post', 'both', or None.

        'pre' / 'post' / 'both' — depends only on cell-annotation columns on
        the named side(s). These ride along into ``edgelist()`` as per-pair
        values (aggregated via ``.first()`` since they're cell-invariant).
        ``None`` — depends on synapse- or vertex-level data, so it cannot be
        aggregated to the pair level and is excluded from ``edgelist()``.

        Classification is computed at ``add_expression()`` time based on the
        cell annotations registered at that moment.
        """
        return dict(self._expression_sides)

    @property
    def weights(self) -> list[str]:
        """Column names registered as synapse weights (auto-summed in aggregations)."""
        return list(self._weights)

    @property
    def cell_aliases(self) -> dict[str, tuple[str, str]]:
        """Registered cell aliases: {alias_name: (annotation_name, col)}."""
        return dict(self._cell_aliases)

    # ── role-declared blessed columns ──────────────────────────────────────

    @property
    def pre_col(self) -> str:
        """Column name for the pre-synaptic cell id."""
        return self._pre_col

    @property
    def post_col(self) -> str:
        """Column name for the post-synaptic cell id."""
        return self._post_col

    @property
    def id_col(self) -> str:
        """Column name for the synapse id (used to join synapse annotations)."""
        return self._id_col

    @property
    def synapse_position_col(self) -> str | None:
        """Column holding synapse positions as a struct with x, y, z fields, if declared."""
        return self._synapse_position_col

    # ── raw-table introspection ────────────────────────────────────────────

    @property
    def synapses(self) -> pl.DataFrame:
        """Base synapse table as a DataFrame — no joins, no filters, no expressions.

        For the fully merged + filtered + computed table use ``.df``. To inspect
        an individual registered annotation use ``synapse_annotations[name]``,
        ``cell_annotations[name]``, or ``vertex_annotations[name]``.
        """
        return self._syn_lf.collect()

    @property
    def synapses_lazy(self) -> pl.LazyFrame:
        """Lazy view of the base synapse table — for building custom plans."""
        return self._syn_lf

    def _resolve_position_annotation(self, annotation: str | None = None) -> str:
        """Return the name of the cell annotation whose ``position_col`` to use.

        Delegates to :func:`trajan._base.resolve_position_annotation`; see
        there for the auto-resolution contract.
        """
        return resolve_position_annotation(
            self._cell_annotations,
            annotation,
            noun="cell annotation",
            register_method="add_cell_annotation",
        )

    def _resolve_universe_annotation(self, annotation: str | None = None) -> str:
        """Return the name of the cell annotation that defines the cell universe.

        Delegates to :func:`trajan._base.resolve_universe_annotation`; see
        there for the auto-resolution contract.
        """
        return resolve_universe_annotation(
            self._cell_annotations,
            annotation,
            noun="cell annotation",
            register_method="add_cell_annotation",
        )

    def cell_annotation_data_cols(self) -> dict[str, list[str]]:
        """Data columns (non-key) for each registered cell annotation.

        Returns a fresh dict mapping annotation name to a fresh list of data
        column names. Consumers (e.g. free-function extractions of
        ``cell_summary`` / ``to_graph``) use this to enumerate annotation
        columns without reaching into the private storage tuple.
        """
        return {
            name: list(spec.data_cols) for name, spec in self._cell_annotations.items()
        }

    # ── internal column tracking ───────────────────────────────────────────

    def _current_columns(self) -> set[str]:
        """All column names present (or that will be present) in .df."""
        cols = set(self._syn_col_names)
        for spec in self._synapse_annotations.values():
            cols |= set(spec.data_cols)
        for spec in self._cell_annotations.values():
            cols |= {f"{c}_pre" for c in spec.data_cols}
            cols |= {f"{c}_post" for c in spec.data_cols}
        for spec in self._vertex_annotations.values():
            if spec.pre_vertex_col is not None:
                cols |= {f"{c}_pre" for c in spec.data_cols}
            if spec.post_vertex_col is not None:
                cols |= {f"{c}_post" for c in spec.data_cols}
        cols |= set(self._expressions)
        return cols

    # ── internal helpers ───────────────────────────────────────────────────

    # ── synapse annotations ────────────────────────────────────────────────

    def add_synapse_annotation(
        self, name: str, df, position_cols: list[str] | str | None = None
    ) -> None:
        """Register a synapse-level annotation, joined on id_col.

        Raises ValueError if any column already exists in the table.

        Parameters
        ----------
        name : str
            Identifier for this annotation; used to remove it later via
            remove_synapse_annotation.
        df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
            Annotation table. Must contain id_col plus data columns.
        position_cols : list[str] or str or None, optional
            Column name prefix(es) to auto-pack from split x/y/z format into a
            position struct. E.g. "ctr_pt_position" or ["ctr_pt_position"] will
            pack ctr_pt_position_x/y/z into a struct named ctr_pt_position.
        """
        if isinstance(position_cols, str):
            position_cols = [position_cols]
        lf = _to_lazy(df)
        for col in position_cols or []:
            lf = _auto_pack(lf, col)
        validate_unique_key(lf, self._id_col)
        data_cols = [c for c in lf.collect_schema().names() if c != self._id_col]
        collisions = set(data_cols) & self._current_columns()
        if collisions:
            raise ValueError(f"Columns already exist in table: {sorted(collisions)}")
        self._synapse_annotations[name] = SynapseAnnotationSpec(lf, data_cols)
        self._cache = None
        return self

    def remove_synapse_annotation(self, name: str) -> None:
        if name not in self._synapse_annotations:
            raise KeyError(f"No synapse annotation named {name!r}")
        del self._synapse_annotations[name]
        self._cache = None
        return self

    # ── cell annotations ───────────────────────────────────────────────────

    def add_cell_annotation(
        self,
        name: str,
        df,
        cell_id_col: str,
        join_on_alias: str | None = None,
        alias_col: str | None = None,
        alias_name: str | None = None,
        position_col: str | None = None,
        is_universe: bool = False,
    ) -> SynapseTable:
        """Register a cell-level annotation, joined symmetrically for pre and post.

        Each column in df (other than cell_id_col) produces two columns in
        .df: col_pre and col_post. Raises ValueError on any collision.

        Parameters
        ----------
        name : str
            Identifier for this annotation; used to remove it later via
            remove_cell_annotation.
        df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
            Annotation table. Must contain cell_id_col plus data columns.
        cell_id_col : str
            The column in df containing cell ids to join on.
        join_on_alias : str or None, optional
            If None (default), joins on root ID columns (pre_col/post_col).
            If a string, names the cell alias to join on — the alias must have
            been registered via set_cell_alias() or alias_col on a prior call.
        alias_col : str or None, optional
            If provided, registers this annotation as a cell alias source.
            Equivalent to calling set_cell_alias(name, alias_col, alias_name)
            immediately after registration.
        alias_name : str or None, optional
            Name under which to register the alias. Defaults to the annotation
            name. Only used when alias_col is also provided.
        position_col : str or None, optional
            Name of a data column in this annotation that carries the cell's
            position. May be either a struct with ``x`` / ``y`` / ``z`` fields
            or a column name whose ``_x`` / ``_y`` / ``_z`` triplet is present
            — in the latter case the triplet is auto-packed into a struct
            named ``position_col``. Declaring this role lets spatial filters
            (``filter_by_soma_distance``, ``add_spatial_features``) find the
            position without further configuration. For more exotic packing
            needs (multiple split columns in one annotation), pre-pack the
            DataFrame with ``trajan.spatial.pack_position`` before passing.
        is_universe : bool, optional
            If True, this annotation's cell-id set is treated as the
            authoritative cell universe — the set of cells that *exist* for
            the purposes of denominator-bearing statistics, including those
            with zero observed connections. Use ``annotation=`` on stats /
            resolvers to disambiguate when more than one is registered.
            See ``CellAnnotationSpec.is_universe``.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.
        """
        if join_on_alias is not None and join_on_alias not in self._cell_aliases:
            raise ValueError(
                f"No cell alias named {join_on_alias!r}. "
                f"Call set_cell_alias() first or use alias_col on a prior "
                f"add_cell_annotation call. Registered aliases: "
                f"{list(self._cell_aliases)}"
            )
        lf = _to_lazy(df)
        # position_col auto-packs: if the named column isn't a struct yet but
        # the split <name>_x/y/z triplet is present, pack it. Callers can then
        # pass position_col=<name> regardless of how the input stores positions.
        if position_col is not None:
            lf = _auto_pack(lf, position_col)
        validate_unique_key(lf, cell_id_col)
        data_cols = [c for c in lf.collect_schema().names() if c != cell_id_col]
        if position_col is not None and position_col not in data_cols:
            raise ValueError(
                f"position_col {position_col!r} not found in annotation data "
                f"columns. Available: {data_cols}"
            )
        new_cols = {f"{c}_pre" for c in data_cols} | {f"{c}_post" for c in data_cols}
        collisions = new_cols & self._current_columns()
        if collisions:
            raise ValueError(f"Columns already exist in table: {sorted(collisions)}")
        self._cell_annotations[name] = CellAnnotationSpec(
            lf,
            cell_id_col,
            join_on_alias,
            data_cols,
            position_col=position_col,
            is_universe=is_universe,
        )
        self._cache = None
        if alias_col is not None:
            self.set_cell_alias(name, alias_col, alias_name)
        return self

    def remove_cell_annotation(self, name: str) -> SynapseTable:
        if name not in self._cell_annotations:
            raise KeyError(f"No cell annotation named {name!r}")
        removed_aliases = [
            alias_name
            for alias_name, (ann_name, _) in self._cell_aliases.items()
            if ann_name == name
        ]
        if removed_aliases:
            broken = [
                n
                for n, spec in self._cell_annotations.items()
                if spec.join_on_alias in removed_aliases and n != name
            ]
            msg = (
                f"Removing annotation {name!r} which sourced cell "
                f"alias(es) {removed_aliases}. Those aliases have been cleared."
            )
            if broken:
                msg += (
                    f" The following annotations reference removed aliases and will "
                    f"fail until set_cell_alias() is called again: {broken}"
                )
            warnings.warn(msg, stacklevel=2)
            for alias_name in removed_aliases:
                del self._cell_aliases[alias_name]
        del self._cell_annotations[name]
        self._cache = None
        return self

    def set_cell_alias(
        self,
        annotation_name: str,
        col: str = "cell_id",
        alias_name: str | None = None,
    ) -> SynapseTable:
        """Declare a cell alias column produced by a cell annotation.

        Once registered, annotations indexed by this alias can be added with
        join_on_alias=<alias_name>.

        Parameters
        ----------
        annotation_name : str
            Name of an already-registered cell annotation whose data columns
            include col. This annotation must itself join on root ID
            (join_on_alias=None).
        col : str, optional
            Column within the annotation that holds the aliased cell IDs.
            Defaults to "cell_id".
        alias_name : str or None, optional
            Name used to reference this alias in join_on_alias. Defaults to
            the annotation name.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.

        Raises
        ------
        KeyError
            If annotation_name is not registered.
        ValueError
            If col is not in the annotation's data columns, or if the annotation
            itself uses join_on_alias (the alias source must be root-level).
        """
        if annotation_name not in self._cell_annotations:
            raise KeyError(f"No cell annotation named {annotation_name!r}")
        spec = self._cell_annotations[annotation_name]
        if spec.join_on_alias is not None:
            raise ValueError(
                f"Annotation {annotation_name!r} uses join_on_alias={spec.join_on_alias!r} "
                f"and cannot itself be a cell alias source. The alias source must "
                f"join on root ID (join_on_alias=None)."
            )
        if col not in spec.data_cols:
            raise ValueError(
                f"Column {col!r} not found in annotation {annotation_name!r}. "
                f"Available columns: {spec.data_cols}"
            )
        key = alias_name if alias_name is not None else annotation_name
        self._cell_aliases[key] = (annotation_name, col)
        return self

    def extend_cell_annotation(
        self,
        name: str,
        df,
        on: str,
        position_col: str | None = None,
        is_universe: bool | None = None,
    ) -> SynapseTable:
        """Join additional columns into an already-registered cell annotation.

        Uses a left join on `on`, which must be a column already present in the
        registered annotation (not the synapse table's cell id). Cells without a
        match in df receive nulls — row count is never expanded.

        Parameters
        ----------
        name : str
            Name of the existing cell annotation to extend.
        df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
            DataFrame or LazyFrame containing the new columns and the join key.
        on : str
            Column name to join on. Must exist in the already-registered annotation
            and must be unique in df.
        position_col : str or None, optional
            Declare a column being added by this extension as the annotation's
            position role. Overrides any previously declared ``position_col``
            on this annotation. May be either a struct with ``x`` / ``y`` / ``z``
            fields or a column whose ``_x`` / ``_y`` / ``_z`` triplet is present
            in ``df`` (auto-packed into a struct named ``position_col``). For
            exotic packing needs, pre-pack with ``trajan.spatial.pack_position``.
        is_universe : bool or None, optional
            If given, set the annotation's ``is_universe`` flag (True or
            False). ``None`` (default) leaves the existing flag unchanged.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.
        """
        if name not in self._cell_annotations:
            raise KeyError(f"No cell annotation named {name!r}")

        existing = self._cell_annotations[name]

        existing_schema = existing.lf.collect_schema().names()
        if on not in existing_schema:
            raise ValueError(
                f"Join key {on!r} not found in annotation {name!r}. "
                f"Available columns: {existing_schema}"
            )

        new_lf = _to_lazy(df)
        if position_col is not None:
            new_lf = _auto_pack(new_lf, position_col)

        validate_unique_key(new_lf, on)

        extra_cols = [c for c in new_lf.collect_schema().names() if c != on]
        new_pre_post = {f"{c}_pre" for c in extra_cols} | {
            f"{c}_post" for c in extra_cols
        }
        collisions = new_pre_post & self._current_columns()
        if collisions:
            raise ValueError(f"Columns already exist in table: {sorted(collisions)}")

        if position_col is not None and position_col not in extra_cols:
            raise ValueError(
                f"position_col {position_col!r} not found among extension data "
                f"columns. Available: {extra_cols}"
            )
        merged_position_col = (
            position_col if position_col is not None else existing.position_col
        )
        merged_is_universe = (
            is_universe if is_universe is not None else existing.is_universe
        )
        self._cell_annotations[name] = CellAnnotationSpec(
            lf=existing.lf.join(new_lf, on=on, how="left"),
            cell_id_col=existing.cell_id_col,
            join_on_alias=existing.join_on_alias,
            data_cols=existing.data_cols + extra_cols,
            position_col=merged_position_col,
            is_universe=merged_is_universe,
        )
        self._cache = None
        return self

    # ── vertex annotations ─────────────────────────────────────────────────

    def add_vertex_annotation(
        self,
        name: str,
        df,
        vertex_id_col: str,
        pre_vertex_col: str | None = None,
        post_vertex_col: str | None = None,
        position_cols: list[str] | str | None = None,
    ) -> SynapseTable:
        """Register a vertex-level annotation, joined for pre and/or post sides.

        Each column in df (other than vertex_id_col) produces a col_pre column
        if pre_vertex_col is given, and/or a col_post column if post_vertex_col
        is given. Raises ValueError on any collision or if neither side is given.

        Parameters
        ----------
        name : str
            Identifier for this annotation; used to remove it later via
            remove_vertex_annotation.
        df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
            Annotation table. Must contain vertex_id_col plus data columns.
        vertex_id_col : str
            The column in df containing vertex ids to join on.
        pre_vertex_col : str or None, optional
            Column in the synapse table holding pre-synaptic vertex ids.
            If provided, annotation columns appear as col_pre in .df.
        post_vertex_col : str or None, optional
            Column in the synapse table holding post-synaptic vertex ids.
            If provided, annotation columns appear as col_post in .df.
        position_cols : list[str] or str or None, optional
            Column name prefix(es) to auto-pack from split x/y/z format into a
            position struct. E.g. "pt_position" or ["pt_position"] will pack
            pt_position_x/y/z into a struct named pt_position.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.
        """
        if pre_vertex_col is None and post_vertex_col is None:
            raise ValueError(
                "At least one of pre_vertex_col or post_vertex_col must be provided"
            )
        if isinstance(position_cols, str):
            position_cols = [position_cols]
        lf = _to_lazy(df)
        for col in position_cols or []:
            lf = _auto_pack(lf, col)
        validate_unique_key(lf, vertex_id_col)
        data_cols = [c for c in lf.collect_schema().names() if c != vertex_id_col]
        new_cols = set()
        if pre_vertex_col is not None:
            new_cols |= {f"{c}_pre" for c in data_cols}
        if post_vertex_col is not None:
            new_cols |= {f"{c}_post" for c in data_cols}
        collisions = new_cols & self._current_columns()
        if collisions:
            raise ValueError(f"Columns already exist in table: {sorted(collisions)}")
        self._vertex_annotations[name] = VertexAnnotationSpec(
            lf=lf,
            vertex_id_col=vertex_id_col,
            pre_vertex_col=pre_vertex_col,
            post_vertex_col=post_vertex_col,
            data_cols=data_cols,
        )
        self._cache = None
        return self

    def remove_vertex_annotation(self, name: str) -> SynapseTable:
        if name not in self._vertex_annotations:
            raise KeyError(f"No vertex annotation named {name!r}")
        del self._vertex_annotations[name]
        self._cache = None
        return self

    # ── expressions ────────────────────────────────────────────────────────

    def _classify_expression(self, expr: pl.Expr) -> str | None:
        """Classify an expression as 'pre', 'post', 'both', or None.

        Thin wrapper over :func:`trajan._base.classify_by_cell_sides` that
        threads in this table's currently registered cell annotations and
        previously classified expressions.
        """
        return classify_by_cell_sides(
            expr, self._cell_annotations, self._expression_sides
        )

    def add_expression(self, name: str, expr: pl.Expr) -> SynapseTable:
        """Register a named computed column expression.

        Applied after all annotation joins and before filters, so any joined
        column is available. Expressions are applied in insertion order, so
        later expressions may reference earlier ones.

        Raises ValueError if name already exists as a column in the table.

        Parameters
        ----------
        name : str
            Output column name.
        expr : pl.Expr
            Polars expression. An alias of name is applied automatically.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.
        """
        if name in self._current_columns():
            raise ValueError(f"Column {name!r} already exists in the table")
        self._expressions[name] = expr.alias(name)
        self._expression_sides[name] = self._classify_expression(expr)
        self._cache = None
        return self

    def remove_expression(self, name: str) -> SynapseTable:
        if name not in self._expressions:
            raise KeyError(f"No expression named {name!r}")
        del self._expressions[name]
        del self._expression_sides[name]
        self._cache = None
        return self

    # ── weights ────────────────────────────────────────────────────────────

    def add_weight(self, col: str) -> SynapseTable:
        """Register a column to be auto-summed in all synapse aggregations.

        Registered weight columns are summed alongside n_syn in edgelist() and
        type_edgelist() without requiring an explicit agg argument. Weights are
        also available to matrix() and normalized() via their values parameter.

        Parameters
        ----------
        col : str
            Column name to register as a weight. Must exist in the table (either
            a base synapse column, a synapse annotation column, or an expression).

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.

        Raises
        ------
        ValueError
            If col does not exist in the table or is already registered as a weight.
        """
        if col not in self._current_columns():
            raise ValueError(
                f"Column {col!r} not found in table. "
                f"Add the column first via add_synapse_annotation or add_expression."
            )
        if col in self._weights:
            raise ValueError(f"Column {col!r} is already registered as a weight.")
        self._weights.append(col)
        return self

    def remove_weight(self, col: str) -> SynapseTable:
        """Remove a column from the registered weights list.

        Parameters
        ----------
        col : str
            Column name to remove.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.

        Raises
        ------
        KeyError
            If col is not currently registered as a weight.
        """
        if col not in self._weights:
            raise KeyError(f"No weight registered for column {col!r}")
        self._weights.remove(col)
        return self

    _WEIGHT_TRANSFORMS: dict = {
        "log1p": lambda c: pl.col(c).log1p(),
        "log": lambda c: pl.col(c).log(base=math.e),
        "sqrt": lambda c: pl.col(c).sqrt(),
    }

    def add_weight_transform(
        self,
        name: str,
        source_col: str,
        transform: str = "log1p",
        register_as_weight: bool = True,
    ) -> SynapseTable:
        """Register a transformed version of a column, optionally as a weight.

        A convenience wrapper around :meth:`add_expression` + :meth:`add_weight`
        that encodes common preprocessing conventions for synapse weight columns.
        ``log1p`` is the standard transform for synapse size/count data, which is
        approximately log-normally distributed; using ``log`` instead risks ``-inf``
        values for zero counts.

        Parameters
        ----------
        name : str
            Name for the new transformed column.
        source_col : str
            Existing column to transform. Must be present in the table at call time.
        transform : str, optional
            Transform to apply. One of:

            - ``"log1p"`` (default) — ``log(1 + x)``, safe for zero values
            - ``"log"`` — natural log; produces ``-inf`` for zero values
            - ``"sqrt"`` — square root

        register_as_weight : bool, optional
            If True (default), also register the new column as a weight so it is
            automatically summed in :meth:`edgelist`, :meth:`type_edgelist`, and
            :meth:`cell_summary`.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.

        Raises
        ------
        ValueError
            If ``transform`` is not a known transform name, or ``source_col`` is
            not found in the table.

        Examples
        --------
        >>> st.add_weight_transform("log_size", "size")
        # adds log1p(size) column and registers it as a weight
        """
        if transform not in self._WEIGHT_TRANSFORMS:
            raise ValueError(
                f"transform must be one of {list(self._WEIGHT_TRANSFORMS)}, "
                f"got {transform!r}"
            )
        if source_col not in self._current_columns():
            raise ValueError(f"Column {source_col!r} not found in table.")
        self.add_expression(name, self._WEIGHT_TRANSFORMS[transform](source_col))
        if register_as_weight:
            self.add_weight(name)
        return self

    # ── lazy plan construction ─────────────────────────────────────────────

    def build_lazy(self) -> pl.LazyFrame:
        """Construct (without collecting) the full annotated + filtered lazy plan.

        Applies all registered synapse, cell, and vertex annotation joins,
        computed expressions (in registration order), and accumulated filters.
        Does not hit the materialization cache — call ``.df`` for the
        cached collected result.

        This is the public entry point for consumers (free functions, free-
        standing statistics) that need the lazy plan without going through
        ``.df``. Renamed from the previous ``_build_lazy`` internal.
        """
        lf = self._syn_lf

        for spec in self._synapse_annotations.values():
            lf = lf.join(spec.lf, on=self._id_col, how="left")

        lf = join_cell_annotations_symmetric(
            lf,
            self._cell_annotations,
            pre_col=self._pre_col,
            post_col=self._post_col,
            aliases=self._cell_aliases,
        )

        for spec in self._vertex_annotations.values():
            if spec.pre_vertex_col is not None:
                pre_lf = spec.lf.rename({c: f"{c}_pre" for c in spec.data_cols})
                lf = lf.join(
                    pre_lf,
                    left_on=spec.pre_vertex_col,
                    right_on=spec.vertex_id_col,
                    how="left",
                )
            if spec.post_vertex_col is not None:
                post_lf = spec.lf.rename({c: f"{c}_post" for c in spec.data_cols})
                lf = lf.join(
                    post_lf,
                    left_on=spec.post_vertex_col,
                    right_on=spec.vertex_id_col,
                    how="left",
                )

        return apply_plan_tail(lf, self._expressions, self._filters)

    # ── df property (cached) ───────────────────────────────────────────────

    @property
    def df(self) -> pl.DataFrame:
        """Full merged synapse table with all registered annotations. Cached."""
        if self._cache is None:
            self._cache = self.build_lazy().collect()
        return self._cache

    # ── copy helper ────────────────────────────────────────────────────────

    def _copy(self) -> SynapseTable:
        new = SynapseTable.__new__(SynapseTable)
        new._syn_lf = self._syn_lf
        new._syn_col_names = self._syn_col_names
        new._pre_col = self._pre_col
        new._post_col = self._post_col
        new._id_col = self._id_col
        new._synapse_position_col = self._synapse_position_col
        new._synapse_annotations = self._synapse_annotations.copy()
        new._cell_annotations = self._cell_annotations.copy()
        new._vertex_annotations = self._vertex_annotations.copy()
        new._expressions = self._expressions.copy()
        new._expression_sides = self._expression_sides.copy()
        new._weights = self._weights.copy()
        new._cell_aliases = self._cell_aliases.copy()
        new._filters = self._filters.copy()
        new._filter_sides = self._filter_sides.copy()
        new.metadata = self.metadata.copy()
        new._cache = None
        new._n_syn_base = self._n_syn_base
        return new

    # ── selective view ─────────────────────────────────────────────────────

    def view(
        self,
        synapse_annotations: list[str] | None = None,
        cell_annotations: list[str] | None = None,
        vertex_annotations: list[str] | None = None,
        expressions: list[str] | None = None,
        keep_filters: bool = True,
    ) -> SynapseTable:
        """Return a new SynapseTable using only the specified annotations/expressions.

        Useful for building lightweight views without all registered annotations,
        or for reconstructing the table from scratch with granular control.

        Parameters
        ----------
        synapse_annotations : list[str] or None, optional
            Names to include. None keeps all registered; [] keeps none.
        cell_annotations : list[str] or None, optional
            Names to include. None keeps all registered; [] keeps none.
        vertex_annotations : list[str] or None, optional
            Names to include. None keeps all registered; [] keeps none.
        expressions : list[str] or None, optional
            Expression names to include. None keeps all; [] keeps none.
        keep_filters : bool, optional
            Whether to carry forward registered filters. Default True.

        Returns
        -------
        SynapseTable
            A new SynapseTable with only the requested annotations and expressions.

        Raises
        ------
        KeyError
            If any requested name is not registered.
        """

        def _select(d, names):
            if names is None:
                return d.copy()
            missing = set(names) - set(d)
            if missing:
                raise KeyError(f"Unknown name(s): {sorted(missing)}")
            return {k: v for k, v in d.items() if k in names}

        new = SynapseTable.__new__(SynapseTable)
        new._syn_lf = self._syn_lf
        new._syn_col_names = self._syn_col_names
        new._pre_col = self._pre_col
        new._post_col = self._post_col
        new._id_col = self._id_col
        new._synapse_position_col = self._synapse_position_col
        new._n_syn_base = self._n_syn_base
        new._cache = None
        new._synapse_annotations = _select(
            self._synapse_annotations, synapse_annotations
        )
        new._cell_annotations = _select(self._cell_annotations, cell_annotations)
        new._vertex_annotations = _select(self._vertex_annotations, vertex_annotations)
        new._expressions = _select(self._expressions, expressions)
        new._expression_sides = {
            k: v for k, v in self._expression_sides.items() if k in new._expressions
        }
        new._weights = self._weights.copy()
        new.metadata = self.metadata.copy()
        new._filters = self._filters.copy() if keep_filters else []
        new._filter_sides = self._filter_sides.copy() if keep_filters else []
        # carry only aliases whose source annotation is still present in the view
        new._cell_aliases = {
            alias_name: (ann_name, col)
            for alias_name, (ann_name, col) in self._cell_aliases.items()
            if ann_name in new._cell_annotations
        }
        return new

    # ── filtering ──────────────────────────────────────────────────────────

    def _annotation_null_expr(self, annotation_name: str, side: str) -> pl.Expr:
        """is_not_null() for the first data column of a cell/vertex annotation on pre or post."""
        if annotation_name in self._cell_annotations:
            data_cols = self._cell_annotations[annotation_name].data_cols
        elif annotation_name in self._vertex_annotations:
            data_cols = self._vertex_annotations[annotation_name].data_cols
        else:
            raise KeyError(f"No cell or vertex annotation named {annotation_name!r}")
        return pl.col(f"{data_cols[0]}_{side}").is_not_null()

    def filter_to_annotated(
        self, annotation_name: str, pre: bool = True, post: bool = True
    ) -> SynapseTable:
        """Return a new SynapseTable keeping only synapses where both pre and post
        cells have a non-null value for the given cell or vertex annotation.

        Equivalent to:
            st.filter(pl.col("col_pre").is_not_null() & pl.col("col_post").is_not_null())

        Parameters
        ----------
        annotation_name : str
            Name of a registered cell or vertex annotation to filter on.
        pre : bool, optional
            If True (default), require the pre-synaptic side to be non-null.
        post : bool, optional
            If True (default), require the post-synaptic side to be non-null.

        Returns
        -------
        SynapseTable
            A new SynapseTable with the annotation null filter applied.
        """
        expr = None
        if pre:
            expr = self._annotation_null_expr(annotation_name, "pre")
        if post:
            expr = (
                expr & self._annotation_null_expr(annotation_name, "post")
                if expr is not None
                else self._annotation_null_expr(annotation_name, "post")
            )
        return self.filter(expr)

    def filter(self, expr: pl.Expr) -> SynapseTable:
        """Return a new SynapseTable with expr applied to the lazy plan.

        The filter is pushed into the query plan after all annotation joins,
        so any column in .df is valid. Polars' optimizer will push
        predicates on base synapse columns before the joins automatically.

        Parameters
        ----------
        expr : pl.Expr
            A Polars boolean expression to filter rows by.

        Returns
        -------
        SynapseTable
            A new SynapseTable with the filter registered.

        Example
        -------
        st.filter(pl.col("cell_type_pre") == "L2/3 ET")
        st.filter(pl.col("pre_pt_root_id").is_in(cell_ids))
        """
        new = self._copy()
        new._filters = self._filters + [expr]
        new._filter_sides = self._filter_sides + [self._classify_expression(expr)]
        return new

    def filter_by_soma_distance(
        self,
        max_distance: float,
        *,
        annotation: str | None = None,
        distance_fn: Callable[[str, str], pl.Expr] = euclidean_distance,
    ) -> SynapseTable:
        """Return a new SynapseTable keeping only synapses where soma-soma
        distance is ≤ max_distance (in the same units as your position columns).

        Positions are looked up from a registered cell annotation whose
        ``position_col`` was declared at ``add_cell_annotation`` time. The
        position column must be packed as a struct with ``x`` / ``y`` / ``z``
        fields (see ``pack_position``).

        Parameters
        ----------
        max_distance : float
            Maximum soma-to-soma distance to retain, in the same units as
            the position columns.
        annotation : str or None, optional
            Name of the cell annotation whose ``position_col`` to use. If
            ``None`` (default), uses the unique position-bearing annotation;
            raises if zero or more than one are registered.
        distance_fn : Callable[[str, str], pl.Expr], optional
            Callable taking two column name strings and returning a pl.Expr for
            the distance. Defaults to euclidean_distance. Use radial_distance to
            ignore the z axis, or supply a custom function.

        Returns
        -------
        SynapseTable
            A new SynapseTable keeping only synapses within max_distance.
        """
        ann_name = self._resolve_position_annotation(annotation)
        pos_col = self._cell_annotations[ann_name].position_col
        return self.filter(
            distance_fn(f"{pos_col}_pre", f"{pos_col}_post") <= max_distance
        )

    def add_spatial_features(
        self,
        prefix: str = "",
        center: str = "pre",
        target: str = "post",
        depth_axis: str = "y",
        euclidean: bool = True,
        depth_diff: bool = True,
        spherical: bool = True,
        cylindrical: bool = True,
        *,
        annotation: str | None = None,
    ) -> SynapseTable:
        """Register a standard battery of spatial features for a two-point vector.

        Computes the vector **target_pos − center_pos** and decomposes it into
        euclidean distance, depth difference, spherical, and cylindrical coordinates.
        Each feature is registered via :meth:`add_expression` under the name
        ``{prefix}_{feature}`` (or just ``{feature}`` when prefix is empty).

        Parameters
        ----------
        prefix : str, optional
            Prefix prepended to all generated column names (joined with ``_``).
            Use distinct prefixes when calling this method multiple times.
        center : str, optional
            Origin of the vector. ``"pre"`` (default) or ``"post"``.
        target : str, optional
            Destination of the vector. ``"post"`` (default), ``"pre"``, or
            ``"syn"`` (synapse position).
        depth_axis : str, optional
            Axis representing cortical depth, with optional direction suffix.
            Plain ``"x"``, ``"y"``, or ``"z"`` means positive values go deeper.
            Append ``"_r"`` to reverse: ``"y_r"`` means positive y is towards the
            surface. Defaults to ``"y"``.
        euclidean : bool, optional
            Include ``euclidean`` — 3-D Euclidean distance.
        depth_diff : bool, optional
            Include ``depth_diff`` — signed depth component of the vector.
        spherical : bool, optional
            Include ``r`` (= euclidean), ``theta`` (polar angle from depth axis,
            [0, π]), and ``phi`` (azimuthal angle in lateral plane, [-π, π]).
        cylindrical : bool, optional
            Include ``rho`` (lateral distance), ``phi`` (shared with spherical),
            and ``dy`` (= depth_diff).
        annotation : str or None, optional
            Cell annotation whose ``position_col`` to use for soma positions.
            Auto-resolved when exactly one position-bearing annotation is
            registered; pass explicitly to disambiguate.

        Returns
        -------
        SynapseTable
            Self, for method chaining.

        Raises
        ------
        ValueError
            If ``center == target``, no position-bearing cell annotation is
            registered (or the named ``annotation`` lacks a ``position_col``),
            or ``target="syn"`` but ``synapse_position_col`` is not set.

        Examples
        --------
        Default pre→post soma features:

        >>> st.add_spatial_features(prefix="soma")
        # adds soma_euclidean, soma_depth_diff, soma_r, soma_theta, soma_phi, soma_rho, soma_dy

        Both centering perspectives (call twice):

        >>> st.add_spatial_features(prefix="from_pre")
        >>> st.add_spatial_features(prefix="from_post", center="post", target="pre")

        Pre soma → synapse:

        >>> st.add_spatial_features(prefix="pre_syn", center="pre", target="syn")
        """
        if center == target:
            raise ValueError(f"center and target must differ, both are {center!r}")
        if center not in ("pre", "post"):
            raise ValueError(f"center must be 'pre' or 'post', got {center!r}")
        if target not in ("pre", "post", "syn"):
            raise ValueError(f"target must be 'pre', 'post', or 'syn', got {target!r}")

        ann_name = self._resolve_position_annotation(annotation)
        soma_col = self._cell_annotations[ann_name].position_col
        soma_cols = {
            "pre": f"{soma_col}_pre",
            "post": f"{soma_col}_post",
        }
        from_col = soma_cols[center]

        if target == "syn":
            if self._synapse_position_col is None:
                raise ValueError(
                    "synapse_position_col not set on this SynapseTable; "
                    "required when target='syn'"
                )
            to_col = self._synapse_position_col
        else:
            to_col = soma_cols[target]

        for feat, expr in spatial_feature_exprs(
            from_col,
            to_col,
            depth_axis=depth_axis,
            euclidean=euclidean,
            depth_diff=depth_diff,
            spherical=spherical,
            cylindrical=cylindrical,
        ).items():
            self.add_expression(_spatial_col_name(prefix, feat), expr)

        return self

    def filter_by_ids(
        self,
        pre_ids=None,
        post_ids=None,
    ) -> SynapseTable:
        """Return a new SynapseTable keeping only synapses involving specified cell IDs.

        Filters are applied on the base pre/post root ID columns, so this works
        even before cell annotations are joined.

        Parameters
        ----------
        pre_ids : Iterable or None, optional
            Iterable of pre-synaptic cell IDs to keep. None means no filter on pre.
        post_ids : Iterable or None, optional
            Iterable of post-synaptic cell IDs to keep. None means no filter on post.

        Returns
        -------
        SynapseTable
            A new SynapseTable filtered to the specified cell IDs.

        Examples
        --------
        Keep only synapses from a specific set of pre cells:

        >>> st.filter_by_ids(pre_ids=[111, 222, 333])

        Keep synapses between two specific populations:

        >>> st.filter_by_ids(pre_ids=excitatory_ids, post_ids=inhibitory_ids)
        """
        return filter_by_id_sets(self, pre_ids, post_ids)

    def filter_by_min_synapses(
        self,
        n: int | float,
        weight_col: str | None = None,
    ) -> SynapseTable:
        """Return a new SynapseTable keeping only synapses from sufficiently strong pairs.

        Filters at the synapse level so that all downstream outputs — :meth:`edgelist`,
        :meth:`matrix`, :meth:`cell_summary` — consistently reflect the threshold.

        By default, pairs are qualified by synapse count. When ``weight_col`` is given,
        pairs are qualified by the sum of that column instead, which is useful for
        filtering on total synapse size or a pre-registered weight.

        Parameters
        ----------
        n : int or float
            Minimum threshold. Pairs with count (or weight sum) strictly below ``n``
            are removed along with all their synapses.
        weight_col : str or None, optional
            Column to sum per pair instead of counting synapses. Must be present in
            the table. Defaults to None (filter by synapse count).

        Returns
        -------
        SynapseTable
            A new SynapseTable with synapses from weak pairs removed.

        Raises
        ------
        ValueError
            If ``weight_col`` is given but not found in the table.

        Examples
        --------
        Keep only pairs with at least 3 synapses:

        >>> st.filter_by_min_synapses(3)

        Keep only pairs where total synapse size is at least 1000:

        >>> st.filter_by_min_synapses(1000, weight_col="size")
        """
        if weight_col is not None and weight_col not in self._current_columns():
            raise ValueError(f"Column {weight_col!r} not found in table.")

        lf = self.build_lazy()
        if weight_col is None:
            pair_agg = lf.group_by([self._pre_col, self._post_col]).agg(
                pl.len().alias("_agg")
            )
        else:
            pair_agg = lf.group_by([self._pre_col, self._post_col]).agg(
                pl.col(weight_col).sum().alias("_agg")
            )
        pair_keys = (
            pair_agg.filter(pl.col("_agg") >= n)
            .select([self._pre_col, self._post_col])
            .collect()
        )
        filtered_lf = self._syn_lf.join(
            pair_keys.lazy(), on=[self._pre_col, self._post_col], how="semi"
        )
        new = self._copy()
        new._syn_lf = filtered_lf
        new._syn_col_names = filtered_lf.collect_schema().names()
        new._n_syn_base = filtered_lf.select(pl.len()).collect().item()
        return new

    def filter_by_bbox(self, bbox) -> SynapseTable:
        """Filter synapses whose position falls within a bounding box.

        Requires synapse_position_col to be set, with the position column being
        a struct with x, y, z fields (see pack_position).

        Parameters
        ----------
        bbox : Sequence
            Sequence of two (x, y, z) corners: ((xmin, ymin, zmin), (xmax, ymax, zmax)).

        Returns
        -------
        SynapseTable
            A new SynapseTable keeping only synapses within the bounding box.
        """
        if self._synapse_position_col is None:
            raise ValueError("synapse_position_col not set on this SynapseTable")
        return self.filter(bbox_predicate(self._synapse_position_col, bbox))

    # ── edgelist ───────────────────────────────────────────────────────────

    def _build_agg_exprs(self, agg: dict[str, pl.Expr] | None) -> list[pl.Expr]:
        """Build the aggregation expr list shared by edgelist / type_edgelist.

        Prepends the auto ``n_syn`` count and the per-weight sums, then appends
        the caller's ``agg`` entries. Guards against name collisions: a user
        ``agg`` key (or a weight) that shadows ``n_syn`` or another weight would
        otherwise raise a cryptic polars DuplicateError or silently overwrite a
        column — here it raises a clear, actionable error instead.
        """
        if "n_syn" in self._weights:
            raise ValueError(
                "A weight column named 'n_syn' conflicts with the synapse count "
                "that edgelist()/type_edgelist() generate. Rename the weight."
            )
        if agg:
            reject_reserved_names(
                agg.keys(),
                {"n_syn", *self._weights},
                context="edgelist()/type_edgelist() agg",
            )
        agg_exprs = [pl.len().alias("n_syn")]
        agg_exprs.extend(pl.sum(w).alias(w) for w in self._weights)
        if agg:
            agg_exprs.extend(expr.alias(name) for name, expr in agg.items())
        return agg_exprs

    def edgelist(
        self,
        agg: dict[str, pl.Expr] | None = None,
    ) -> EdgeList:
        """Aggregate synapses into a cell-pair EdgeList.

        Always produces an ``n_syn`` weight column (synapse count per pair).
        Registered weights (``self.weights``) are auto-summed. Additional
        per-pair aggregations can be passed via ``agg``.

        Cell annotations are *propagated* — registered on the resulting
        EdgeList rather than inlined into the pair frame — so role
        declarations (``position_col``, ``is_universe``) and aliases survive
        the transition. Annotation columns still appear on ``el.df`` via the
        symmetric pre/post join performed at access time.

        Side-classified expressions (see ``expression_sides``) ride along as
        per-pair values via ``.first()`` aggregation.

        Parameters
        ----------
        agg : dict[str, pl.Expr] or None, optional
            {output_column_name: polars_expression} for additional per-pair
            aggregations. Example:
            ``{"mean_size": pl.mean("size"), "total_area": pl.sum("area")}``.

        Returns
        -------
        EdgeList
            Pair-level table with pre/post cell id columns, ``n_syn``, any
            registered weights, and ``agg`` outputs in the base pair frame;
            cell annotations (with their roles and aliases) registered for
            symmetric join via ``el.df``.
        """
        agg_exprs = self._build_agg_exprs(agg)

        pair_df = (
            self.build_lazy()
            .group_by([self._pre_col, self._post_col])
            .agg(agg_exprs)
            .collect()
        )

        weight_cols = ["n_syn"] + list(self._weights)
        el = EdgeList(
            pair_df,
            pre_col=self._pre_col,
            post_col=self._post_col,
            weight_cols=weight_cols,
            cell_aliases=dict(self._cell_aliases),
        )
        # Propagate cell annotations (including aliased ones — aliases are
        # 1:1 with root cell ids, so the alias join on the EdgeList yields
        # the same per-pair values that inlining via .first() would have).
        for ann_name, spec in self._cell_annotations.items():
            el.add_annotation(
                ann_name,
                spec.lf,
                cell_id_col=spec.cell_id_col,
                position_col=spec.position_col,
                is_universe=spec.is_universe,
                join_on_alias=spec.join_on_alias,
            )
        # Re-register cell-level expressions. Their bodies reference
        # annotation-derived columns (e.g. cell_type_pre) which the
        # EdgeList's annotation joins now provide; they evaluate live at
        # el.df access. Synapse-level expressions (side=None) drop here —
        # Phase 2 will give them a declared pair_agg= aggregation path.
        # Registration order matters when expressions depend on prior ones;
        # dict ordering preserves it.
        for name, side in self._expression_sides.items():
            if side in ("pre", "post", "both"):
                el.add_expression(name, self._expressions[name])
        return el

    def type_edgelist(
        self,
        pre_col: str,
        post_col: str | None = None,
        agg: dict[str, pl.Expr] | None = None,
    ) -> ConnectivityTable:
        """Aggregate synapses into a type-to-type ConnectivityTable.

        Groups synapses by label columns (typically cell-type annotation
        outputs such as ``cell_type_pre`` / ``cell_type_post``) rather than
        individual cell ids. Returns a ``ConnectivityTable`` — the axes are
        labels, not cells, so the result is Tier 2, not an EdgeList.

        Parameters
        ----------
        pre_col : str
            Column in the merged plan to use as the pre-side grouping key.
        post_col : str or None, optional
            Column for the post-side grouping key. Defaults to ``pre_col``
            with any trailing ``_pre`` replaced by ``_post``.
        agg : dict[str, pl.Expr] or None, optional
            Per-pair aggregations applied after the type grouping.

        Returns
        -------
        ConnectivityTable
        """
        if post_col is None:
            if pre_col.endswith("_pre"):
                post_col = pre_col[:-4] + "_post"
            else:
                post_col = pre_col

        agg_exprs = self._build_agg_exprs(agg)

        pair_df = (
            self.build_lazy().group_by([pre_col, post_col]).agg(agg_exprs).collect()
        )
        weight_cols = ["n_syn"] + list(self._weights)
        return ConnectivityTable(
            pair_df,
            pre_col=pre_col,
            post_col=post_col,
            weight_cols=weight_cols,
        )

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, folio: Union[str, Path, object], overwrite: bool = False) -> None:
        """Save the SynapseTable to a DataFolio.

        Materializes all lazy frames and writes them as Parquet tables.
        Filters are serialized as JSON expressions and stored in the config.

        Parameters
        ----------
        folio : str, Path, or DataFolio
            Path to a folio directory (opened or created) or an existing DataFolio
            instance.
        overwrite : bool, optional
            If True, overwrite existing items in the folio.
        """
        folio = _as_folio(folio)
        config = {
            "pre_col": self._pre_col,
            "post_col": self._post_col,
            "id_col": self._id_col,
            "synapse_position_col": self._synapse_position_col,
            "weights": self._weights,
            "filters": [f.meta.serialize(format="json") for f in self._filters],
            "filter_sides": list(self._filter_sides),
            "expressions": {
                name: base64.b64encode(expr.meta.serialize(format="binary")).decode(
                    "ascii"
                )
                for name, expr in self._expressions.items()
            },
            "synapse_annotations": {
                name: {"data_cols": spec.data_cols}
                for name, spec in self._synapse_annotations.items()
            },
            "cell_aliases": {
                alias_name: {"annotation_name": ann_name, "col": col}
                for alias_name, (ann_name, col) in self._cell_aliases.items()
            },
            "cell_annotations": {
                name: {
                    "cell_id_col": spec.cell_id_col,
                    "join_on_alias": spec.join_on_alias,
                    "data_cols": spec.data_cols,
                    "position_col": spec.position_col,
                    "is_universe": spec.is_universe,
                }
                for name, spec in self._cell_annotations.items()
            },
            "vertex_annotations": {
                name: {
                    "vertex_id_col": spec.vertex_id_col,
                    "pre_vertex_col": spec.pre_vertex_col,
                    "post_vertex_col": spec.post_vertex_col,
                    "data_cols": spec.data_cols,
                }
                for name, spec in self._vertex_annotations.items()
            },
        }
        folio.add_json("config", config, overwrite=overwrite)
        if self.metadata:
            folio.metadata.update(self.metadata)
        folio.add_table("synapses", self._syn_lf.collect(), overwrite=overwrite)
        for name, spec in self._synapse_annotations.items():
            folio.add_table(
                f"synapse_ann_{name}", spec.lf.collect(), overwrite=overwrite
            )
        for name, spec in self._cell_annotations.items():
            folio.add_table(f"cell_ann_{name}", spec.lf.collect(), overwrite=overwrite)
        for name, spec in self._vertex_annotations.items():
            folio.add_table(
                f"vertex_ann_{name}", spec.lf.collect(), overwrite=overwrite
            )

    @classmethod
    def load(cls, folio: Union[str, Path, object]) -> SynapseTable:
        """Load a SynapseTable from a DataFolio.

        Parameters
        ----------
        folio : str, Path, or DataFolio
            Path to a folio directory or an existing DataFolio instance previously
            written by .save().

        Returns
        -------
        SynapseTable
            A fully reconstructed SynapseTable with all annotations, expressions,
            and filters restored.
        """
        folio = _as_folio(folio)

        def _lf(name: str) -> pl.LazyFrame:
            return pl.scan_parquet(folio.get_data_path(name))

        config = folio.get_json("config")
        st = cls(
            _lf("synapses"),
            pre_col=config["pre_col"],
            post_col=config["post_col"],
            id_col=config["id_col"],
            synapse_position_col=config["synapse_position_col"],
        )
        for name in config["synapse_annotations"]:
            st.add_synapse_annotation(name, _lf(f"synapse_ann_{name}"))
        for name, meta in config["cell_annotations"].items():
            st.add_cell_annotation(
                name,
                _lf(f"cell_ann_{name}"),
                cell_id_col=meta["cell_id_col"],
                join_on_alias=meta.get("join_on_alias"),
                position_col=meta.get("position_col"),
                is_universe=meta.get("is_universe", False),
            )
        for alias_name, alias_meta in config.get("cell_aliases", {}).items():
            st.set_cell_alias(
                alias_meta["annotation_name"],
                alias_meta["col"],
                alias_name=alias_name,
            )
        for name, meta in config["vertex_annotations"].items():
            st.add_vertex_annotation(
                name,
                _lf(f"vertex_ann_{name}"),
                vertex_id_col=meta["vertex_id_col"],
                pre_vertex_col=meta["pre_vertex_col"],
                post_vertex_col=meta["post_vertex_col"],
            )
        for name, expr_val in config.get("expressions", {}).items():
            expr = None
            # New format: base64-encoded binary (round-trips all literals correctly)
            try:
                expr = pl.Expr.deserialize(base64.b64decode(expr_val), format="binary")
            except Exception:
                pass
            # Old format: JSON string (may fail for expressions containing NaN literals)
            if expr is None:
                try:
                    expr = pl.Expr.deserialize(expr_val.encode(), format="json")
                except Exception:
                    warnings.warn(
                        f"Could not deserialize expression {name!r} from folio "
                        f"(likely a Polars version incompatibility). "
                        f"Re-save the folio to fix this.",
                        stacklevel=2,
                    )
                    continue
            st.add_expression(name, expr)
        for col in config.get("weights", []):
            st.add_weight(col)
        # Use saved filter_sides when present (post-Universe+Scope refactor);
        # fall back to live reclassification for older folios. Both paths yield
        # the same classification given the order annotations are restored above.
        saved_sides = config.get("filter_sides")
        for i, f_json in enumerate(config["filters"]):
            st = st.filter(pl.Expr.deserialize(f_json.encode(), format="json"))
            if saved_sides is not None and i < len(saved_sides):
                st._filter_sides[-1] = saved_sides[i]
        _internal = {"created_at", "updated_at", "_datafolio"}
        st.metadata = {k: v for k, v in folio.metadata.items() if k not in _internal}
        return st
