"""Free-function statistics over trajan tables.

First landing: ``cell_summary`` extracted from the old ``SynapseTable.cell_summary``
method. Future entries (``connection_probability`` / ``connection_density``,
``reciprocity``, ``connectivity_similarity``, pair-correlation) will live here
and take EdgeList / ConnectivityTable as primary input — see
``project_architecture_principles.md`` and
``project_edgelist_abstraction.md``.

This module is the strategic home for the consolidation-mission work (see
``project_consolidation_mission.md``): one authoritative implementation of
statistics that are otherwise reimplemented across projects, with raw counts
surfaced and variants documented explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .synapse_table import SynapseTable


def cell_summary(
    st: SynapseTable,
    pre_agg: dict[str, pl.Expr] | None = None,
    post_agg: dict[str, pl.Expr] | None = None,
    include_annotations: bool = True,
) -> pl.DataFrame:
    """Aggregate synapse-level data into a per-cell summary DataFrame.

    Returns one row per unique cell ID, combining output (pre-side) and input
    (post-side) statistics. Cell-annotation values are included by default.

    Parameters
    ----------
    st : SynapseTable
        Source table. All registered weights are summed per direction and
        appear as ``{weight}_output`` / ``{weight}_input`` columns. Per-cell
        annotation values are included (suffix stripped) when
        ``include_annotations`` is True.
    pre_agg : dict[str, pl.Expr] or None, optional
        Additional aggregations when the cell is pre-synaptic.
        Example: ``{"mean_size_out": pl.mean("size")}``.
    post_agg : dict[str, pl.Expr] or None, optional
        Additional aggregations when the cell is post-synaptic.
    include_annotations : bool, optional
        If True (default), cell annotation columns are included in the output
        (with ``_pre`` / ``_post`` suffixes stripped). All registered cell
        annotations are included, regardless of whether they use
        ``join_on_alias``.

    Returns
    -------
    pl.DataFrame
        One row per cell with at minimum ``cell_id``, ``n_syn_output``,
        ``n_syn_input``; plus weight sums, agg outputs, and (optionally)
        cell-annotation columns.
    """
    lf = st.build_lazy()
    weights = st.weights
    pre_col, post_col = st.pre_col, st.post_col

    anno_cols: list[str] = [
        c for data_cols in st.cell_annotation_data_cols().values() for c in data_cols
    ]

    # Pre-side aggregation (cell as sender)
    pre_exprs: list[pl.Expr] = [pl.len().alias("n_syn_output")]
    pre_exprs.extend(pl.sum(w).alias(f"{w}_output") for w in weights)
    if pre_agg:
        pre_exprs.extend(expr.alias(name) for name, expr in pre_agg.items())
    if include_annotations:
        pre_exprs.extend(pl.col(f"{c}_pre").first().alias(c) for c in anno_cols)
    pre_df = lf.group_by(pre_col).agg(pre_exprs).rename({pre_col: "cell_id"}).collect()

    # Post-side aggregation (cell as receiver)
    post_exprs: list[pl.Expr] = [pl.len().alias("n_syn_input")]
    post_exprs.extend(pl.sum(w).alias(f"{w}_input") for w in weights)
    if post_agg:
        post_exprs.extend(expr.alias(name) for name, expr in post_agg.items())
    if include_annotations:
        post_exprs.extend(pl.col(f"{c}_post").first().alias(c) for c in anno_cols)
    post_df = (
        lf.group_by(post_col).agg(post_exprs).rename({post_col: "cell_id"}).collect()
    )

    # Full outer join; coalesce merges the two cell_id key columns.
    result = pre_df.join(post_df, on="cell_id", how="full", coalesce=True)

    # Annotation values are invariant per cell; coalesce the two copies the
    # outer join produces (suffixed _right on the post side).
    if include_annotations:
        for c in anno_cols:
            right = f"{c}_right"
            if right in result.columns:
                result = result.with_columns(pl.coalesce([c, right]).alias(c)).drop(
                    right
                )

    return result
