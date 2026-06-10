"""Scope-aware free functions: ``cells`` and the shared scope vocabulary.

This module is the consumer side of the ``is_universe`` annotation role. It
provides a single source of truth for per-cell views — ``cells(table, ...)``
— that statistics, viz helpers, and null-model shufflers can call to inherit
a consistent set of cells (with or without the table's cell-level filters
applied). Pair-level analogs (``possible_pairs``, ``PairUniverse``) live in
``trajan.pair_universe`` and use the same scope vocabulary.

The design is documented in ``DESIGN-universe.md`` (sections 1, 3, 5). The
key contract: any per-cell statistic or plot that needs to be "1:1 with the
analysis" should call ``cells()`` rather than reaching into annotations
directly, so the same filters that constrain ``.df`` also constrain the
cells frame.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Union

import polars as pl

from ._base import (
    CellAnnotationSpec,
    Scope,
    aggregate_per_cell,
    get_cell_annotation_store,
    unique_name,
)

if TYPE_CHECKING:
    from .edgelist import EdgeList
    from .synapse_table import SynapseTable


# ── internal helpers ──────────────────────────────────────────────────────────


def _build_cells_frame(
    table, universe_name: str, annotations="all"
) -> tuple[pl.LazyFrame, CellAnnotationSpec, list[str]]:
    """Build a per-cell lazy frame: universe annotation joined with siblings.

    Returns ``(lf, universe_spec, all_data_cols)``. ``all_data_cols`` is the
    union of every joined annotation's data columns and is what
    ``cells(scope="filtered")`` augments with ``_pre`` / ``_post`` aliases for
    side-projection.

    Sibling cell annotations are joined on their semantically-blessed
    ``cell_id_col``, not by string-matching column names. A non-aliased sibling
    always joins — its ``cell_id_col`` holds the same root-id value space as the
    universe's (both are keyed onto the table's ``pre_col`` / ``post_col``), so
    the universe's key is joined to the sibling's key on value even when the two
    columns are named differently. An alias-keyed sibling joins only when its
    alias column is already present on the cells frame; one whose alias column
    wasn't produced upstream is skipped silently, and downstream filters
    referencing its columns get caught by the column-availability check in
    ``cells()`` and warn there.

    ``annotations`` selects which sibling annotations to join: ``"all"``
    (default) joins every joinable sibling; a list / set of names joins only
    those; ``None`` joins none (universe columns only). The universe annotation
    itself is always included.

    This contract intentionally matches ``possible_pairs``, which re-registers
    every source cell annotation on the ``PairUniverse``. The two are 1:1 on
    what cells / pairs a given filter selects (when ``annotations="all"``).
    """
    store = get_cell_annotation_store(table)
    universe_spec = store[universe_name]
    lf = universe_spec.lf
    available = set(lf.collect_schema().names())
    all_data_cols = list(universe_spec.data_cols)

    aliases = getattr(table, "_cell_aliases", {}) or {}

    # Iterate in registration order so an alias-keyed sibling only joins after
    # the annotation that introduced its alias column.
    for name, spec in store.items():
        if name == universe_name:
            continue
        if annotations is None:
            continue
        if annotations != "all" and name not in annotations:
            continue
        if spec.join_on_alias is None:
            # Non-aliased sibling: by construction its cell_id_col holds the
            # same root-id values the universe is keyed on — both are joined
            # onto the source table's pre_col / post_col at the synapse / pair
            # level (see join_cell_annotations_symmetric). So we join the
            # universe's blessed key to the sibling's blessed key on value; the
            # two column *names* need not coincide. This honors the
            # semantically-blessed cell_id_col rather than string-matching
            # column names (the old behavior silently dropped siblings whose
            # cell_id_col happened to differ from the universe's, e.g. a
            # 'root_id'-keyed sibling against a 'pt_root_id'-keyed universe).
            left_on = universe_spec.cell_id_col
        else:
            # Alias-keyed sibling: alias source annotation registers a data
            # column whose values are the sibling's keys. We need that column
            # present on the cells frame to do the join.
            alias_meta = aliases.get(spec.join_on_alias)
            if alias_meta is None:
                continue
            _, alias_col = alias_meta
            if alias_col not in available:
                continue
            left_on = alias_col
        lf = lf.join(spec.lf, left_on=left_on, right_on=spec.cell_id_col, how="left")
        available |= set(spec.lf.collect_schema().names())
        all_data_cols.extend(spec.data_cols)

    return lf, universe_spec, all_data_cols


def _observed_cell_ids(table) -> pl.LazyFrame:
    """Return a single-column lazy frame of the cell ids present in ``.df``.

    Drawn from the union of pre / post id columns on the table's merged
    plan. The column is renamed to a stable internal name so the join in
    ``cells(scope="observed")`` doesn't depend on whether the pre/post
    columns share a name (they normally do, but the universe annotation's
    cell_id_col is the canonical join key).
    """
    pre = table.build_lazy().select(pl.col(table.pre_col).alias("__cell_id__"))
    post = table.build_lazy().select(pl.col(table.post_col).alias("__cell_id__"))
    return pl.concat([pre, post]).unique().drop_nulls()


def _participation_frame(table, key_name: str = "cell_id") -> pl.LazyFrame:
    """Per-cell participation in the table's observed (filtered) edges.

    Returns a lazy frame keyed on ``key_name`` with four columns:

    - ``in_pre`` / ``in_post`` (bool) — whether the cell appears as a
      presynaptic / postsynaptic cell in the observed edges.
    - ``n_syn_out`` / ``n_syn_in`` — synapse counts on each side.

    Computed from the table's observed edgelist, so all accumulated filters
    (synapse-, vertex-, and cell-level) are baked in — the participation
    columns are 1:1 with the analysis the caller holds, regardless of which
    ``scope`` row set they are overlaid onto.

    A ``SynapseTable`` is aggregated via ``.edgelist()`` first; an ``EdgeList``
    is used directly. Synapse counts come from summing the ``n_syn`` weight
    (always produced by ``.edgelist()``); if a hand-built ``EdgeList`` lacks
    ``n_syn``, the count falls back to the number of observed partner pairs.
    """
    from .edgelist import EdgeList

    el = table if isinstance(table, EdgeList) else table.edgelist()
    lf = el.build_lazy()
    count_expr = pl.sum("n_syn") if "n_syn" in el.weights else pl.len()
    pre_lf, post_lf = aggregate_per_cell(
        lf,
        pre_col=el.pre_col,
        post_col=el.post_col,
        count_expr=count_expr,
        out_count="n_syn_out",
        in_count="n_syn_in",
        cell_id_col=key_name,
    )
    return (
        pre_lf.join(post_lf, on=key_name, how="full", coalesce=True)
        .with_columns(
            pl.col("n_syn_out").fill_null(0),
            pl.col("n_syn_in").fill_null(0),
        )
        .with_columns(
            (pl.col("n_syn_out") > 0).alias("in_pre"),
            (pl.col("n_syn_in") > 0).alias("in_post"),
        )
    )


# ── public API ────────────────────────────────────────────────────────────────


def cells(
    table: Union["SynapseTable", "EdgeList"],
    *,
    annotations="all",
    scope: Scope = "universe",
    participation: bool = False,
    universe: str | None = None,
    strict: bool = True,
) -> pl.DataFrame:
    """Return the decorated cell universe at the requested scope.

    Single source of truth for cell-level views. Every viz helper and every
    per-cell statistic should call this rather than reaching into annotations
    directly — that's what keeps plots and stats trivially 1:1 with the
    analysis. The mental model: ``cells()`` is *the universe, decorated*. The
    universe annotation defines the rows; sibling annotations and optional
    participation columns decorate them.

    Parameters
    ----------
    table : SynapseTable or EdgeList
        Source table. Must have a cell annotation registered with
        ``is_universe=True`` (auto-resolved when exactly one is registered;
        pass ``universe=<name>`` to disambiguate).
    annotations : "all", list of str, or None, optional
        Which sibling cell annotations to join as decorations. ``"all"``
        (default) joins every joinable sibling; a list / set of names joins
        only those; ``None`` returns just the universe annotation's own
        columns. The universe annotation is always included.
    scope : {"universe", "filtered", "observed"}, optional
        Which set of cells to return. Default ``"universe"``.

        - ``"universe"`` — every cell in the universe, no filters applied.
        - ``"filtered"`` — universe intersected with cell-level filter
          projection (see §3.1 of ``DESIGN-universe.md``): the union of
          pre-eligible and post-eligible cells.
        - ``"observed"`` — cells appearing in ``.df`` (pre or post side),
          joined to the universe annotation.
    participation : bool, optional
        If True, add four columns describing how each cell participates in the
        table's observed (filtered) edges:

        - ``in_pre`` / ``in_post`` (bool) — appears as a presynaptic /
          postsynaptic cell.
        - ``n_syn_out`` / ``n_syn_in`` — synapse counts on each side.

        These are computed from the table's observed edgelist (all filters
        baked in) and overlaid onto whichever ``scope`` row set was selected;
        cells with no observed connections get ``False`` / ``0``. Replaces the
        old ``side=`` selector: instead of ``side="pre"``, write
        ``cells(table, participation=True).filter(pl.col("in_pre"))`` — and
        unlike the old flag, this works even with no cell-level filter present.
    universe : str or None, optional
        Name of the cell annotation marked ``is_universe=True``. Auto-resolved
        when exactly one is registered.
    strict : bool, optional
        Only meaningful when ``scope="observed"``. ``True`` (default) drops
        observed cells that aren't in the universe annotation. ``False``
        keeps them with null annotation columns (lenient mode, useful for
        data-quality diagnostics).

    Returns
    -------
    pl.DataFrame
        One row per cell. Columns are the universe annotation's
        ``cell_id_col`` plus its data columns, plus any selected sibling
        cell-annotation columns, plus the four participation columns when
        ``participation=True``. The column set is the same across all three
        scopes; only the row set differs.

    See Also
    --------
    trajan.cell_summary : Richer per-cell aggregates (summed weights, custom
        aggregations) over the *observed* cells only — not anchored to the
        universe, so zero-connection cells are absent. Use ``cells`` for the
        universe-anchored view; ``cell_summary`` for detailed per-cell numbers.

    Notes
    -----
    Cell-level filters classified as ``"both"`` (referencing both pre and
    post sides in a non-decomposable way, e.g. ``cell_type_pre !=
    cell_type_post``) are skipped with a warning under ``scope="filtered"`` —
    they apply to ``.df`` but cannot be cleanly projected to per-cell
    semantics. To get tight side-projection, split such filters into separate
    ``.filter()`` calls.

    Sibling cell annotations are joined into the cells frame on their
    semantically-blessed ``cell_id_col`` — the universe's key joined to the
    sibling's key on value, so a sibling keyed ``root_id`` joins onto a universe
    keyed ``pt_root_id`` without the names having to match. (Genuinely distinct
    id namespaces are handled via a registered alias instead.) This matches
    ``possible_pairs`` behavior: a filter on ``tag_pre`` from a sibling
    annotation narrows ``cells(scope="filtered")`` exactly as it narrows the
    pre side of the pair universe. Alias-keyed siblings whose alias column is
    unreachable from the universe frame are silently skipped; filters
    referencing their (or any other unavailable) columns warn and skip.
    Excluding a sibling via ``annotations=`` while a filter references its
    column likewise warns and skips that filter.
    """
    if scope not in ("universe", "filtered", "observed"):
        raise ValueError(
            f"scope must be 'universe', 'filtered', or 'observed', got {scope!r}"
        )

    universe_name = table._resolve_universe_annotation(universe)
    store = get_cell_annotation_store(table)
    universe_spec = store[universe_name]
    cell_id_col = universe_spec.cell_id_col

    if annotations not in ("all", None):
        annotations = list(annotations)
        unknown = [a for a in annotations if a not in store]
        if unknown:
            raise ValueError(
                f"Unknown annotation name(s) {unknown}; registered: {list(store)}"
            )

    # Join the selected sibling cell annotations onto the universe up front,
    # for *every* scope. This keeps the output column set identical across
    # scopes (a viz helper coloring by a sibling column works the same whether
    # scope is "universe", "filtered", or "observed"). Sibling joins are left
    # joins on a unique key, so they never change the universe row set — only
    # its width.
    base_lf, _, data_cols = _build_cells_frame(table, universe_name, annotations)

    if scope == "universe":
        # Every cell in the universe; no filters, no observed-restriction.
        result_lf = base_lf
    elif scope == "observed":
        observed = _observed_cell_ids(table)
        how = "inner" if strict else "left"
        # Join the cells frame onto the observed ids so the output schema
        # matches the cells frame. An inner join drops observed ids absent
        # from the universe; a left join from observed keeps them with null
        # annotation columns.
        result_lf = observed.join(
            base_lf,
            left_on="__cell_id__",
            right_on=cell_id_col,
            how=how,
        )
        if cell_id_col != "__cell_id__":
            result_lf = result_lf.rename({"__cell_id__": cell_id_col})
    else:
        # scope == "filtered" — side-decomposed projection, returning the union
        # of pre-eligible and post-eligible cells.
        result_lf = _filtered_cells_lf(table, base_lf, data_cols, cell_id_col)

    if participation:
        existing = set(result_lf.collect_schema().names())
        part_cols = ["n_syn_out", "n_syn_in", "in_pre", "in_post"]
        clash = [c for c in part_cols if c in existing]
        if clash:
            raise ValueError(
                f"cells(participation=True): output column(s) {clash} already exist "
                f"on the cells frame (from an annotation). Rename the colliding "
                f"annotation column(s), or drop participation=True."
            )
        # Join on a unique internal key so a universe column named like the
        # participation key can't shadow it.
        key = unique_name("__part_id__", existing | set(part_cols))
        part = _participation_frame(table, key_name=key)
        result_lf = result_lf.join(
            part, left_on=cell_id_col, right_on=key, how="left"
        ).with_columns(
            pl.col("n_syn_out").fill_null(0),
            pl.col("n_syn_in").fill_null(0),
            pl.col("in_pre").fill_null(False),
            pl.col("in_post").fill_null(False),
        )

    return result_lf.collect()


def _filtered_cells_lf(
    table, base_lf: pl.LazyFrame, data_cols: list[str], cell_id_col: str
) -> pl.LazyFrame:
    """Side-decomposed projection of cell-level filters onto the universe.

    Returns the lazy union of pre-eligible and post-eligible cells. Each
    single-sided filter constrains only its side; ``"both"`` filters and
    filters referencing columns absent from the cells frame are skipped with a
    warning (they still apply to ``.df``). See §3.1 of ``DESIGN-universe.md``.
    """
    pre_lf = base_lf.with_columns(*[pl.col(c).alias(f"{c}_pre") for c in data_cols])
    post_lf = base_lf.with_columns(*[pl.col(c).alias(f"{c}_post") for c in data_cols])

    base_schema = set(base_lf.collect_schema().names())
    available_pre = base_schema | {f"{c}_pre" for c in data_cols}
    available_post = base_schema | {f"{c}_post" for c in data_cols}

    # Apply cell-side expressions so a filter referencing an expression-derived
    # column resolves, keeping cells(scope="filtered") 1:1 with possible_pairs()
    # (which propagates the same expressions). Registration order preserves
    # dependencies; "both"/None expressions and those whose roots aren't on a
    # given side are skipped (a filter on a skipped column then warns below).
    expressions = getattr(table, "_expressions", {}) or {}
    expr_sides = getattr(table, "_expression_sides", {}) or {}
    pre_expr_cols: list[str] = []
    post_expr_cols: list[str] = []
    for name, expr in expressions.items():
        side = expr_sides.get(name)
        roots = set(expr.meta.root_names())
        if side == "pre" and roots <= available_pre:
            pre_lf = pre_lf.with_columns(expr)
            available_pre.add(name)
            pre_expr_cols.append(name)
        elif side == "post" and roots <= available_post:
            post_lf = post_lf.with_columns(expr)
            available_post.add(name)
            post_expr_cols.append(name)

    for f_expr, side_class in zip(table._filters, table._filter_sides):
        if side_class is None:
            continue  # synapse / vertex / pair-level: not a cell-level constraint
        if side_class == "both":
            warnings.warn(
                f"Cell-level filter classifies as 'both' and cannot be cleanly "
                f"projected onto cells(); skipping. The filter still applies to "
                f".df. Split into separate .filter() calls (one per side) for "
                f"tight projection. Filter: {f_expr!s}",
                stacklevel=3,
            )
            continue
        roots = set(f_expr.meta.root_names())
        if side_class == "pre":
            if roots - available_pre:
                # Filter references columns the cells frame doesn't expose
                # (a sibling annotation not selected, or a sibling whose join
                # key is unreachable). Skip with a warning.
                warnings.warn(
                    f"cells() cannot project filter referencing {sorted(roots - available_pre)} — "
                    f"those columns are not on the cells frame. Include the "
                    f"owning annotation (annotations=) or move them to the "
                    f"universe annotation. Filter: {f_expr!s}",
                    stacklevel=3,
                )
                continue
            pre_lf = pre_lf.filter(f_expr)
        elif side_class == "post":
            if roots - available_post:
                warnings.warn(
                    f"cells() cannot project filter referencing {sorted(roots - available_post)} — "
                    f"those columns are not on the cells frame. Include the "
                    f"owning annotation (annotations=) or move them to the "
                    f"universe annotation. Filter: {f_expr!s}",
                    stacklevel=3,
                )
                continue
            post_lf = post_lf.filter(f_expr)

    # Drop the side-alias and expression columns so the output schema matches
    # the joined cells frame (universe + selected siblings) and the two sides
    # are concat-compatible.
    pre_lf = pre_lf.drop([f"{c}_pre" for c in data_cols] + pre_expr_cols)
    post_lf = post_lf.drop([f"{c}_post" for c in data_cols] + post_expr_cols)

    return pl.concat([pre_lf, post_lf]).unique(subset=cell_id_col)
