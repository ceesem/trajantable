"""Shared machinery for trajan tables.

This module holds the pieces that are (or will be) common across the three
tiers — SynapseTable, EdgeList, ConnectivityTable — so that each tier can
compose them rather than reimplement them:

- Annotation-spec dataclasses: one per annotation kind. Replaces the ad-hoc
  tuple storage originally used on SynapseTable. The ``lf`` attribute is
  first on every spec so the existing ``_AnnotationProxy`` can continue to
  access the underlying LazyFrame uniformly.
- ``BlessedColumns``: a small dataclass holding the role → concrete-column
  mappings and the weight-column list. Centralizes the "semantic columns the
  library interprets" concept from the blessed-columns memory.
- Low-level input-coercion helpers (``_to_lazy``, ``_auto_pack``) that every
  tier needs when accepting user frames / positions.

This file intentionally stays small and free-standing. The real base class
(``_TrajanTable`` with filter / expression / cache machinery) is deferred
until the second and third tiers are implemented, so the shared surface can
be factored from observed duplication rather than speculated in advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

import polars as pl

from .spatial import pack_position

# ── scope vocabulary ──────────────────────────────────────────────────────────

# Shared across cells(), possible_pairs(), and any future per-cell or per-pair
# helper. The three values are minimal and orthogonal:
#
# - "universe"  : raw universe annotation, no filters applied
# - "filtered"  : universe ∩ cell-level filter projection
# - "observed"  : the cells / pairs actually present in the data
#
# cells() defaults to "universe" (the decorated-universe view); possible_pairs
# composes filters explicitly. Each consumer documents its own default.
#
# Defined once here and imported elsewhere so consumers stay in lockstep.
Scope = Literal["universe", "filtered", "observed"]

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ── annotation-spec dataclasses ───────────────────────────────────────────────


@dataclass
class SynapseAnnotationSpec:
    """Synapse-level annotation: keyed on synapse id.

    Joined onto the base synapse lazy plan, matching the annotation's
    ``syn_id_col`` against the table's ``id_col``. Contributes ``data_cols``
    verbatim to ``.df``.

    Parameters
    ----------
    lf : pl.LazyFrame
        The annotation table.
    syn_id_col : str
        Column in ``lf`` holding the synapse id to join on. Matched against the
        table's ``id_col``; the two need not share a name.
    data_cols : list[str]
        Columns in ``lf`` (excluding the join key) contributed to ``.df``.
    """

    lf: pl.LazyFrame
    syn_id_col: str
    data_cols: list[str]


@dataclass
class CellAnnotationSpec:
    """Cell-level annotation: keyed on cell id (or on an alias column).

    Joined symmetrically for pre and post cells; each ``data_col`` produces
    ``{col}_pre`` and ``{col}_post`` columns on ``.df``.

    Parameters
    ----------
    lf : pl.LazyFrame
        The annotation table.
    cell_id_col : str
        Column in ``lf`` holding the cell id to join on.
    join_on_alias : str or None
        If None, joins on the table's root-ID columns (``pre_col`` / ``post_col``).
        Otherwise names a registered cell alias.
    data_cols : list[str]
        Columns in ``lf`` (excluding the join key) that become
        ``_pre`` / ``_post`` columns on ``.df``.
    position_col : str or None
        Name of the data column (within ``data_cols``) that carries the cell's
        position as a struct with ``x`` / ``y`` / ``z`` fields. Declaring this
        role lets spatial filters (``filter_by_soma_distance``,
        ``filter_by_bbox``) locate the position without explicit per-call args.
        ``None`` means this annotation does not carry positions.
    is_universe : bool
        If True, this annotation's key set defines the authoritative cell
        universe — including cells with zero connections. Statistics that need
        a denominator (e.g. connection_probability) read membership from the
        universe annotation; null-model shufflers enumerate possible partners
        from it. See ``project_cell_annotation_as_universe.md``.

        A future ``Universe`` class will subsume this role; the flag is the
        migration anchor.
    """

    lf: pl.LazyFrame
    cell_id_col: str
    join_on_alias: str | None
    data_cols: list[str]
    position_col: str | None = None
    is_universe: bool = False


@dataclass
class VertexAnnotationSpec:
    """Vertex-level annotation: keyed on a supervoxel / vertex id, joined per side.

    Parameters
    ----------
    lf : pl.LazyFrame
        The annotation table.
    vertex_id_col : str
        Column in ``lf`` holding the vertex id to join on.
    pre_vertex_col : str or None
        Column on the base synapse table holding pre-side vertex ids.
        If given, annotation data columns appear as ``{col}_pre``.
    post_vertex_col : str or None
        Column on the base synapse table holding post-side vertex ids.
        If given, annotation data columns appear as ``{col}_post``.
    data_cols : list[str]
        Columns in ``lf`` (excluding the join key) that become
        ``_pre`` / ``_post`` columns on ``.df``.
    """

    lf: pl.LazyFrame
    vertex_id_col: str
    pre_vertex_col: str | None
    post_vertex_col: str | None
    data_cols: list[str]


# Discriminated union used by ``_AnnotationProxy`` and any generic helpers.
AnnotationSpec = Union[SynapseAnnotationSpec, CellAnnotationSpec, VertexAnnotationSpec]


# ── blessed columns ───────────────────────────────────────────────────────────


@dataclass
class BlessedColumns:
    """The semantically-meaningful columns the library owns for a given table.

    Trajan interprets only columns that are either (a) role-declared via one of
    the fields below, or (b) generated by trajan itself in a trajan-controlled
    namespace. Everything else in the table is user data with no library
    interpretation (see ``project_unified_blessed_columns.md``).

    Identifier roles (left empty when not applicable to this tier):

    - ``id_col`` — synapse id
    - ``pre_col`` / ``post_col`` — pre / post cell (or entity) id columns
    - ``pre_vertex_col`` / ``post_vertex_col`` — optional pre/post vertex id columns

    Value-type roles:

    - ``position_col`` — synapse position struct (``x`` / ``y`` / ``z``)
    - ``weights`` — columns carrying the weight contract (must-aggregate,
      default-sum, log-scale-compatible). ``n_syn`` appears here after
      aggregation to EdgeList / ConnectivityTable; user-registered weights
      (``size``, ``cleft_area``, etc.) live alongside it with no distinction.

    Notes
    -----
    On ``SynapseTable`` the ``pre_col`` / ``post_col`` fields hold cell-id
    columns; on EdgeList / ConnectivityTable they hold entity-id columns
    (cell for EdgeList, cell or label for ConnectivityTable). The role is
    the same — "the pre-side row index column" — but the semantic type
    differs by tier.
    """

    pre_col: str | None = None
    post_col: str | None = None
    id_col: str | None = None
    position_col: str | None = None
    pre_vertex_col: str | None = None
    post_vertex_col: str | None = None
    weights: list[str] = field(default_factory=list)


# ── input-coercion helpers ────────────────────────────────────────────────────


def _to_lazy(df) -> pl.LazyFrame:
    """Accept pl.DataFrame, pl.LazyFrame, or pd.DataFrame and return a LazyFrame."""
    if isinstance(df, pl.LazyFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    if _HAS_PANDAS and isinstance(df, pd.DataFrame):
        return pl.from_pandas(df).lazy()
    raise TypeError(
        f"Expected pl.DataFrame, pl.LazyFrame, or pd.DataFrame, got {type(df)}"
    )


def _auto_pack(lf: pl.LazyFrame, col: str | None) -> pl.LazyFrame:
    """Pack ``{col}_x/y/z`` into a struct ``col`` if the struct is absent but
    the triplet exists. No-op if ``col`` is ``None`` or already present.
    """
    if col is None:
        return lf
    names = lf.collect_schema().names()
    if col in names:
        return lf
    if all(f"{col}_{ax}" in names for ax in ("x", "y", "z")):
        return pack_position(lf, col)
    return lf


# ── annotation-store proxy ────────────────────────────────────────────────────


class _AnnotationProxy:
    """Lazy read-only view over a store of annotation specs.

    Used by SynapseTable's ``synapse_annotations`` / ``cell_annotations`` /
    ``vertex_annotations`` properties and ConnectivityTable's ``annotations``
    property. Indexing (``proxy[name]``) collects the named annotation's
    LazyFrame on demand; iteration and membership checks work without
    collecting any data.

    Lives in ``_base`` so every tier can reach for the same proxy without
    creating a circular dependency (previously imported from ``synapse_table``
    by ``connectivity_table``).
    """

    def __init__(self, store: dict) -> None:
        self._store = store

    def __getitem__(self, name: str) -> pl.DataFrame:
        if name not in self._store:
            raise KeyError(name)
        return self._store[name].lf.collect()

    def __iter__(self):
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: object) -> bool:
        return name in self._store

    def __repr__(self) -> str:
        return f"AnnotationProxy({list(self._store)})"


# ── name-collision helpers ────────────────────────────────────────────────────


def unique_name(base: str, taken) -> str:
    """Return ``base`` if free, else ``base`` with a numeric suffix not in ``taken``.

    Use for *internal* temporary column names (join keys, intermediate sums)
    that must not clash with user data but whose exact spelling is invisible to
    the caller. For *user-facing* output names that collide, prefer raising via
    :func:`reject_reserved_names` so intent isn't silently dropped.
    """
    taken = set(taken)
    if base not in taken:
        return base
    i = 1
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def reject_reserved_names(user_names, reserved, *, context: str) -> None:
    """Raise ``ValueError`` if any ``user_names`` collides with a ``reserved`` name.

    Guards user-supplied output names (custom aggregation names, weight columns)
    against the columns a transform auto-generates, so a silent ``DuplicateError``
    or shadowed column becomes a clear, actionable error at call time.
    """
    reserved = set(reserved)
    clash = sorted({n for n in user_names if n in reserved})
    if clash:
        raise ValueError(
            f"{context}: name(s) {clash} are reserved / auto-generated and would "
            f"collide. Rename them. Reserved here: {sorted(reserved)}."
        )


# ── join-key validation ───────────────────────────────────────────────────────


def validate_unique_key(
    lf: pl.LazyFrame, key_col: str, *, kind: str = "Annotation"
) -> None:
    """Raise ``ValueError`` if ``key_col`` has duplicate values in ``lf``.

    Annotation joins are left joins on a key column; if the key has duplicates,
    a single base row matches multiple annotation rows and the join silently
    expands the row count. Every tier (SynapseTable, ConnectivityTable,
    EdgeList, PairUniverse) must validate uniqueness up-front to keep this
    invariant.

    Parameters
    ----------
    lf : pl.LazyFrame
        The annotation lazy frame to validate.
    key_col : str
        Column whose values must be unique.
    kind : str
        Human-readable prefix for the error message (``"Annotation"`` for
        cell/synapse/vertex annotations, ``"Cross-product"`` etc. for other
        callers). Defaults to ``"Annotation"``.
    """
    n_total = lf.select(pl.len()).collect().item()
    n_unique = lf.select(pl.col(key_col).n_unique()).collect().item()
    if n_total != n_unique:
        raise ValueError(
            f"{kind} key {key_col!r} has {n_total - n_unique} duplicate "
            f"value(s); each id must appear at most once to avoid expanding "
            f"join cardinality."
        )


# ── annotation builder ───────────────────────────────────────────────────────


def build_cell_annotation_spec(
    df,
    *,
    cell_id_col: str,
    position_col: str | None,
    is_universe: bool,
    join_on_alias: str | None,
    current_columns: set[str],
) -> CellAnnotationSpec:
    """Validate inputs and return a ``CellAnnotationSpec``.

    Single source of truth for the pair-level ``add_annotation`` validation
    path shared by ``ConnectivityTable``, ``EdgeList``, and ``PairUniverse``.
    Each caller is responsible for upstream concerns it owns (alias-existence
    check, cache invalidation, storing the spec) — this function only builds
    and validates the spec itself.

    Performs, in order:

    1. Coerces ``df`` to a LazyFrame.
    2. Auto-packs ``{position_col}_x/y/z`` triplet into a struct if needed.
    3. Verifies ``cell_id_col`` is in the annotation schema.
    4. Validates id uniqueness via :func:`validate_unique_key`.
    5. Verifies ``position_col`` (if given) is in the data columns.
    6. Raises if any ``{col}_pre`` / ``{col}_post`` column would collide with
       an existing column on the receiving table.
    """
    lf = _to_lazy(df)
    if position_col is not None:
        lf = _auto_pack(lf, position_col)
    schema = lf.collect_schema().names()
    if cell_id_col not in schema:
        raise ValueError(
            f"cell_id_col {cell_id_col!r} not found in annotation. Available: {schema}"
        )
    validate_unique_key(lf, cell_id_col)
    data_cols = [c for c in schema if c != cell_id_col]
    if position_col is not None and position_col not in data_cols:
        raise ValueError(
            f"position_col {position_col!r} not found in annotation data "
            f"columns. Available: {data_cols}"
        )
    new_cols = {f"{c}_pre" for c in data_cols} | {f"{c}_post" for c in data_cols}
    collisions = new_cols & current_columns
    if collisions:
        raise ValueError(f"Columns already exist in table: {sorted(collisions)}")
    return CellAnnotationSpec(
        lf=lf,
        cell_id_col=cell_id_col,
        join_on_alias=join_on_alias,
        data_cols=data_cols,
        position_col=position_col,
        is_universe=is_universe,
    )


# ── pair-plan construction ────────────────────────────────────────────────────


def join_cell_annotations_symmetric(
    lf: pl.LazyFrame,
    specs: dict[str, CellAnnotationSpec],
    *,
    pre_col: str,
    post_col: str,
    aliases: dict[str, tuple[str, str]] | None = None,
) -> pl.LazyFrame:
    """Left-join each cell annotation onto ``lf`` once per side (pre and post).

    Each data column ``c`` of each spec becomes ``c_pre`` and ``c_post``. A spec
    with ``join_on_alias=None`` joins on ``pre_col`` / ``post_col``; an
    alias-keyed spec joins on ``{alias_col}_pre`` / ``{alias_col}_post`` where
    ``alias_col`` is looked up in ``aliases`` (so the alias-source annotation
    must be registered — and therefore iterated — before any spec that joins on
    its alias). Specs are joined in iteration order.

    Single source of truth for the symmetric cell-annotation join shared by
    ``SynapseTable``, ``ConnectivityTable``, ``EdgeList``, and ``PairUniverse``.
    ``ConnectivityTable`` has no aliases and passes ``aliases=None``; the alias
    branch is then never taken.
    """
    aliases = aliases or {}
    for spec in specs.values():
        if spec.join_on_alias is not None:
            alias_col = aliases[spec.join_on_alias][1]
            pre_key = f"{alias_col}_pre"
            post_key = f"{alias_col}_post"
        else:
            pre_key = pre_col
            post_key = post_col
        pre_lf = spec.lf.rename({c: f"{c}_pre" for c in spec.data_cols})
        lf = lf.join(pre_lf, left_on=pre_key, right_on=spec.cell_id_col, how="left")
        post_lf = spec.lf.rename({c: f"{c}_post" for c in spec.data_cols})
        lf = lf.join(post_lf, left_on=post_key, right_on=spec.cell_id_col, how="left")
    return lf


def apply_plan_tail(
    lf: pl.LazyFrame,
    expressions: dict[str, pl.Expr],
    filters: list[pl.Expr],
) -> pl.LazyFrame:
    """Apply named expressions (in order), then accumulated filters (in order).

    The common tail of every tier's ``build_lazy``: ``with_columns`` for each
    registered expression (registration order, so a later expression can
    reference an earlier one), then ``filter`` for each accumulated predicate.
    """
    for expr in expressions.values():
        lf = lf.with_columns(expr)
    for f in filters:
        lf = lf.filter(f)
    return lf


def build_pair_plan(
    base_lf: pl.LazyFrame,
    specs: dict[str, CellAnnotationSpec],
    *,
    pre_col: str,
    post_col: str,
    aliases: dict[str, tuple[str, str]] | None = None,
    expressions: dict[str, pl.Expr] | None = None,
    filters: list[pl.Expr] | None = None,
) -> pl.LazyFrame:
    """Assemble a pair-level lazy plan: symmetric annotation joins + tail.

    Joins every cell annotation onto ``base_lf`` (see
    :func:`join_cell_annotations_symmetric`), then applies expressions and
    filters (see :func:`apply_plan_tail`). This is the entirety of
    ``build_lazy`` for ``ConnectivityTable`` / ``EdgeList`` / ``PairUniverse``.
    ``SynapseTable`` composes the two sub-helpers directly instead, because it
    interleaves synapse- and vertex-level joins around the cell-annotation join.
    """
    lf = join_cell_annotations_symmetric(
        base_lf, specs, pre_col=pre_col, post_col=post_col, aliases=aliases
    )
    return apply_plan_tail(lf, expressions or {}, filters or [])


def aggregate_per_cell(
    lf: pl.LazyFrame,
    *,
    pre_col: str,
    post_col: str,
    count_expr: pl.Expr,
    out_count: str,
    in_count: str,
    pre_exprs: list[pl.Expr] | tuple = (),
    post_exprs: list[pl.Expr] | tuple = (),
    cell_id_col: str = "cell_id",
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Group ``lf`` by pre and post separately into per-cell aggregates.

    Each side is grouped on its id column and aggregated with ``count_expr``
    (aliased ``out_count`` on the pre side, ``in_count`` on the post side) plus
    any extra per-side expressions; the grouping key is renamed to
    ``cell_id_col``. Returns ``(pre_lf, post_lf)`` lazy frames, each keyed by
    ``cell_id_col``.

    Single source of truth for the per-side, per-cell grouping shared by
    :func:`trajan.stats.cell_summary` (rich aggregates over observed synapses)
    and :func:`trajan.cells` with ``participation=True`` (lightweight in/out
    synapse counts overlaid on the universe). The two differ only in how they
    join the pre and post frames afterward — full-outer for ``cell_summary``,
    left-onto-the-universe for ``cells`` — and in the extra expressions they
    pass, so the grouping itself lives here once.

    Parameters
    ----------
    lf : pl.LazyFrame
        Edge / synapse plan to aggregate. ``count_expr`` is interpreted
        against its rows (``pl.len()`` for synapse-level plans, ``pl.sum("n_syn")``
        for pair-level plans where one row is a pair).
    pre_col, post_col : str
        Pre / post id columns to group on.
    count_expr : pl.Expr
        Aggregation producing the per-cell count (e.g. ``pl.len()`` or
        ``pl.sum("n_syn")``). Aliased to ``out_count`` / ``in_count``.
    out_count, in_count : str
        Output names for the pre-side and post-side counts.
    pre_exprs, post_exprs : list[pl.Expr]
        Extra aggregations to compute alongside the count on each side.
    cell_id_col : str
        Name the grouping key is renamed to on both frames. Defaults to
        ``"cell_id"``.
    """
    pre_lf = (
        lf.group_by(pre_col)
        .agg(count_expr.alias(out_count), *pre_exprs)
        .rename({pre_col: cell_id_col})
    )
    post_lf = (
        lf.group_by(post_col)
        .agg(count_expr.alias(in_count), *post_exprs)
        .rename({post_col: cell_id_col})
    )
    return pre_lf, post_lf


def filter_by_id_sets(table, pre_ids, post_ids):
    """Restrict a synapse/pair table to the given pre/post id iterables.

    Shared implementation of ``filter_by_ids`` for ``SynapseTable``,
    ``EdgeList``, and ``PairUniverse``. Filters on the blessed
    ``pre_col`` / ``post_col`` directly (so it works before any annotation
    join) and goes through the table's own ``filter`` so the result keeps the
    caller's concrete type. With both id sets ``None`` it returns a copy,
    matching the no-op semantics of the per-class methods it replaces.
    """
    if pre_ids is None and post_ids is None:
        return table._copy()
    new = table
    if pre_ids is not None:
        new = new.filter(pl.col(table.pre_col).is_in(list(pre_ids)))
    if post_ids is not None:
        new = new.filter(pl.col(table.post_col).is_in(list(post_ids)))
    return new


# ── side-classification ───────────────────────────────────────────────────────


def classify_by_cell_sides(
    expr: pl.Expr,
    cell_annotations: dict[str, CellAnnotationSpec],
    expression_sides: dict[str, str | None] | None = None,
) -> str | None:
    """Classify ``expr`` as ``"pre"``, ``"post"``, ``"both"``, or ``None``.

    Inspects the root column names referenced by ``expr`` and checks whether
    they are all pre-side cell annotation outputs, all post-side, a mix of
    both, or contain any column that is not derived from a cell annotation
    (in which case the expression is not cell-level and returns ``None``).

    Used by both ``SynapseTable`` and ``ConnectivityTable`` / ``EdgeList`` to
    classify named expressions and accumulated filters. The classification
    enables side-decomposed projection of cell-level filters onto the cells
    frame and the pair-universe frame.

    Parameters
    ----------
    expr : pl.Expr
        Expression to classify.
    cell_annotations : dict[str, CellAnnotationSpec]
        Currently registered cell annotations. Their ``data_cols`` define
        the set of cell-side columns.
    expression_sides : dict[str, str | None] or None
        Previously classified expressions on this table. Expressions can
        reference other expressions; this lets the classifier propagate
        cell-side classification through derived columns. Pass ``None`` if
        the table does not maintain per-expression side classifications
        (e.g. plain ConnectivityTable).
    """
    cell_pre: set[str] = set()
    cell_post: set[str] = set()
    for spec in cell_annotations.values():
        cell_pre |= {f"{c}_pre" for c in spec.data_cols}
        cell_post |= {f"{c}_post" for c in spec.data_cols}

    root_names = expr.meta.root_names()
    if not root_names:
        return None

    has_pre = False
    has_post = False
    for col_name in root_names:
        if col_name in cell_pre:
            has_pre = True
        elif col_name in cell_post:
            has_post = True
        elif expression_sides is not None and col_name in expression_sides:
            side = expression_sides[col_name]
            if side == "pre":
                has_pre = True
            elif side == "post":
                has_post = True
            elif side == "both":
                has_pre = True
                has_post = True
            else:
                return None
        else:
            return None

    if has_pre and not has_post:
        return "pre"
    if has_post and not has_pre:
        return "post"
    if has_pre and has_post:
        return "both"
    return None


# ── annotation-role resolvers ─────────────────────────────────────────────────


def get_cell_annotation_store(table) -> dict[str, CellAnnotationSpec]:
    """Return ``table``'s cell-annotation dict.

    Every tier (SynapseTable, ConnectivityTable, EdgeList, PairUniverse)
    stores its cell annotations under ``_cell_annotations`` as
    ``dict[str, CellAnnotationSpec]``. SynapseTable additionally has
    ``_synapse_annotations`` and ``_vertex_annotations``; the pair-level
    tiers reject those kinds by construction. This helper exists as the
    intended access point for shared consumers (scope, stats, role
    resolvers) so they aren't poking at the private attribute name directly.
    """
    return table._cell_annotations


def resolve_universe_annotation(
    store: dict[str, CellAnnotationSpec],
    annotation: str | None,
    *,
    noun: str = "annotation",
    register_method: str = "add_annotation",
) -> str:
    """Return the name of the annotation that defines the universe.

    Shared implementation for every tier's ``_resolve_universe_annotation``.
    Behavior: if ``annotation`` is given, validate it exists in ``store`` and
    is marked ``is_universe=True``; otherwise find the unique annotation
    marked universe and error on zero matches or ambiguity.

    Parameters
    ----------
    store : dict[str, CellAnnotationSpec]
        Annotation store to search.
    annotation : str or None
        Explicit annotation name, or ``None`` to auto-resolve.
    noun : str
        Human-readable name for the annotation kind in error messages
        (e.g. ``"annotation"`` for CT/EL/PU, ``"cell annotation"`` for
        SynapseTable where multiple kinds coexist).
    register_method : str
        Name of the registration method, surfaced in error messages so the
        user knows what to call to fix the problem.
    """
    if annotation is not None:
        if annotation not in store:
            raise KeyError(f"No {noun} named {annotation!r}. Registered: {list(store)}")
        if not store[annotation].is_universe:
            raise ValueError(
                f"{noun.capitalize()} {annotation!r} is not marked "
                f"is_universe=True. Re-register with "
                f"{register_method}(..., is_universe=True)."
            )
        return annotation
    universes = [n for n, spec in store.items() if spec.is_universe]
    if not universes:
        raise ValueError(
            f"No {noun} is marked is_universe=True. Pass is_universe=True "
            f"to {register_method} on the annotation whose entity-id set "
            f"defines the authoritative universe."
        )
    if len(universes) > 1:
        raise ValueError(
            f"Multiple {noun}s are marked is_universe=True ({universes}); "
            f"pass annotation=<name> to disambiguate."
        )
    return universes[0]


def resolve_position_annotation(
    store: dict[str, CellAnnotationSpec],
    annotation: str | None,
    *,
    noun: str = "annotation",
    register_method: str = "add_annotation",
) -> str:
    """Return the name of the annotation whose ``position_col`` to use.

    Shared implementation for every tier's ``_resolve_position_annotation``.
    Same contract as :func:`resolve_universe_annotation` but matches on
    ``position_col is not None`` instead of ``is_universe``.
    """
    if annotation is not None:
        if annotation not in store:
            raise KeyError(f"No {noun} named {annotation!r}. Registered: {list(store)}")
        if store[annotation].position_col is None:
            raise ValueError(
                f"{noun.capitalize()} {annotation!r} has no position_col "
                f"declared. Re-register with "
                f"{register_method}(..., position_col=<col>)."
            )
        return annotation
    with_pos = [n for n, spec in store.items() if spec.position_col is not None]
    if not with_pos:
        raise ValueError(
            f"No {noun} with a position_col is registered. Pass "
            f"position_col=<col> to {register_method} when registering "
            f"the annotation that carries positions."
        )
    if len(with_pos) > 1:
        raise ValueError(
            f"Multiple {noun}s carry positions ({with_pos}); "
            f"pass annotation=<name> to disambiguate."
        )
    return with_pos[0]


# ── DataFolio helpers ─────────────────────────────────────────────────────────


def _as_folio(folio):
    """Return ``folio`` unchanged if it's already a DataFolio; otherwise open
    a DataFolio at the given path. Accepts ``str``, ``Path``, or DataFolio."""
    if isinstance(folio, (str, Path)):
        from datafolio import DataFolio

        return DataFolio(folio)
    return folio
