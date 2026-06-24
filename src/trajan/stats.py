"""Free-function statistics over trajan tables.

Strategic home for the consolidation-mission work (see
``project_consolidation_mission.md``): one authoritative implementation of
statistics that are otherwise reimplemented across projects, with raw counts
surfaced and variants documented explicitly.

Currently shipping
------------------
- ``cell_summary`` — per-cell aggregation over a SynapseTable.
- ``counts`` — the per-bin observed/possible primitive. Takes a PairUniverse
  *strictly* (the denominator frame) — handing it observed-only data would
  silently collapse every bin's ``p`` to ~1, so it refuses anything else.
- ``connection_probability`` / ``connection_density`` — ``counts`` + ``p = k/n``
  (+ optional CI estimator). Name-only twins: identical formula, the latter
  documents the *dense*-reconstruction interpretation (see
  ``project_connection_probability.md``). Both accept either a PairUniverse or
  a SynapseTable / EdgeList — a table is normalized to its possible-pairs
  denominator automatically, so the easy case stays a one-liner.
- ``wilson_ci`` / ``agresti_coull_ci`` — closed-form binomial-CI estimators
  that plug into ``connection_probability(..., estimator=...)``.
- ``cell_bootstrap_iter`` — cell-resampling iterator (the building block for
  custom CI / variance summaries over the bootstrap distribution).
- ``bootstrap_over_cells`` — percentile CI via cell-bootstrap; the recommended
  CI for dense connectomics where pair-level binomial CIs are overconfident.
- ``with_distance`` — register a per-pair distance expression resolved from
  the position-bearing cell annotation.

Tracked but not yet landed
--------------------------
- ``reciprocity``, ``connectivity_similarity``, pair-correlation.

The estimation entry points (``connection_probability``, ``connection_density``,
``bootstrap_over_cells``, ``cell_bootstrap_iter``) accept a PairUniverse or the
SynapseTable / EdgeList it derives from; ``counts`` is the strict low-level
primitive and takes a PairUniverse only. See ``project_architecture_principles.md``
and ``project_edgelist_abstraction.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator, Union

import polars as pl

from ._base import (
    aggregate_per_cell,
    get_cell_annotation_store,
    reject_reserved_names,
    unique_name,
)

if TYPE_CHECKING:
    from .edgelist import EdgeList
    from .pair_universe import PairUniverse
    from .synapse_table import SynapseTable

    # The estimation entry points accept either the denominator frame itself
    # or the table it derives from; `_as_pair_universe` normalizes the latter.
    StatInput = Union["PairUniverse", "SynapseTable", "EdgeList"]


# ── shared types ──────────────────────────────────────────────────────────────

# Each entry of a ``bin_by`` mapping accepts:
#   - list / tuple of numerics: edges passed to ``pl.Expr.cut`` — produces an
#     output column named ``{name}_bin`` carrying the bin label.
#   - ``None``: categorical pass-through — the raw column is used as a
#     group-by key directly. Useful for `cell_type_pre`, `compartment_post`,
#     etc.
# Joint binning is the cross product of multiple entries: ``bin_by={"d_rho":
# [...], "d_y": [...]}`` produces a 2-D grid of bins. Mixing continuous +
# categorical is fine: ``bin_by={"d_rho": [...], "cell_type_post": None}``.
BinSpec = Union[list, tuple, None]

# An ``Estimator`` consumes two polars expressions — k and n — and returns a
# dict of ``{output_col_name: expr}`` for new columns to attach to the
# ``counts`` DataFrame. Typically returns a point estimate plus CI bounds
# (e.g. ``{"p": ..., "p_lo": ..., "p_hi": ...}``). Kept as a pure
# ``(k, n) -> {col: expr}`` so estimators compose with whatever framework
# the user is computing in. See ``project_connection_probability.md`` for
# the rationale (estimation is a separable axis from binning).
Estimator = Callable[[pl.Expr, pl.Expr], dict[str, pl.Expr]]


# ── input normalization / guards ──────────────────────────────────────────────


def _as_pair_universe(
    table, *, universe: str | None = None, include_self: bool = False
) -> "PairUniverse":
    """Coerce ``table`` to a ``PairUniverse``, building one when needed.

    Returns a ``PairUniverse`` unchanged; passes a ``SynapseTable`` / ``EdgeList``
    through :func:`trajan.possible_pairs` to build the universe × universe
    denominator frame with observed counts overlaid.

    This is what lets the public estimators — ``connection_probability``,
    ``connection_density``, ``bootstrap_over_cells``, ``cell_bootstrap_iter`` —
    take a table directly instead of forcing the caller to spell
    ``possible_pairs(table)`` first. The low-level :func:`counts` primitive
    stays strict (PairUniverse only): handing it observed-only data (an
    ``EdgeList``) would treat every present row as both possible and observed,
    silently collapsing each bin's ``p`` to ~1.

    ``universe`` / ``include_self`` are forwarded to ``possible_pairs`` and so
    apply only when a table is passed; they are ignored when ``table`` is
    already a ``PairUniverse`` (which already fixed those choices at
    construction).
    """
    from .pair_universe import PairUniverse, possible_pairs

    if isinstance(table, PairUniverse):
        return table

    from .edgelist import EdgeList
    from .synapse_table import SynapseTable

    if isinstance(table, (SynapseTable, EdgeList)):
        return possible_pairs(table, universe=universe, include_self=include_self)
    raise TypeError(
        f"Expected a PairUniverse, SynapseTable, or EdgeList, got "
        f"{type(table).__name__}."
    )


def _require_pair_universe(pu) -> None:
    """Validate ``pu`` is a PairUniverse carrying an ``n_syn`` weight.

    Both :func:`counts` and the cell-bootstrap path depend on the PairUniverse
    contract: the full universe × universe frame (so ``n_possible`` is a real
    denominator that includes unobserved pairs) with ``n_syn`` overlaid as the
    observed-pair predicate. Passing observed-only data (an ``EdgeList``) would
    make ``n_possible`` count only observed pairs, collapsing ``p`` to ~1 with
    no error. This guard refuses anything but a PairUniverse, and a
    PairUniverse that has lost its ``n_syn`` weight.
    """
    from .pair_universe import PairUniverse

    if not isinstance(pu, PairUniverse):
        raise TypeError(
            f"counts() requires a PairUniverse — the universe × universe "
            f"denominator frame with observed counts overlaid — got "
            f"{type(pu).__name__}. Build one with possible_pairs(table), or call "
            f"connection_probability(table, ...) / connection_density(table, ...), "
            f"which construct it for you. (Passing an EdgeList here would use "
            f"observed-only pairs as the denominator, making every bin's p ~ 1.)"
        )
    if "n_syn" not in pu.weights:
        raise ValueError(
            "counts() requires the PairUniverse to carry an 'n_syn' weight (the "
            f"observed-pair predicate). Registered weights: {pu.weights}. "
            "PairUniverses from possible_pairs() always have n_syn; if you built "
            "one by hand, name the observed-count column 'n_syn'."
        )


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
        One row per cell with at minimum the identity column, ``n_syn_output``,
        ``n_syn_input``; plus weight sums, agg outputs, and (optionally)
        cell-annotation columns. The identity column is named ``cell_id`` unless
        that collides with an annotation / agg output column (e.g. an annotation
        literally named ``cell_id``), in which case it falls back to the universe
        annotation's ``cell_id_col``.

    Notes
    -----
    For a cell appearing on only one side, the *additive* columns of the absent
    side — ``n_syn_output`` / ``n_syn_input`` and the ``{weight}_output`` /
    ``{weight}_input`` sums — are ``0`` (a cell with no output synapses makes
    zero of them), matching :func:`trajan.cells` with ``participation=True``.
    Custom ``pre_agg`` / ``post_agg`` outputs are left ``null`` on the absent
    side, since an aggregation like a mean is undefined over zero rows.

    ``cell_summary`` is *observed-anchored*: only cells appearing as pre or post
    in the data are returned. For a universe-anchored view that includes
    zero-connection cells, use :func:`trajan.cells` with ``participation=True``.
    """
    lf = st.build_lazy()
    weights = st.weights
    pre_col, post_col = st.pre_col, st.post_col

    anno_cols: list[str] = [
        c for data_cols in st.cell_annotation_data_cols().values() for c in data_cols
    ]

    # Guard caller-supplied agg names against the columns we auto-generate, so a
    # collision is a clear error here rather than a cryptic polars DuplicateError
    # downstream. Reserved: the n_syn counts, the per-weight sums, and (when
    # included) the annotation columns.
    reserved = {"n_syn_output", "n_syn_input"}
    reserved |= {f"{w}_output" for w in weights} | {f"{w}_input" for w in weights}
    if include_annotations:
        reserved |= set(anno_cols)
    pre_names = set(pre_agg or {})
    post_names = set(post_agg or {})
    reject_reserved_names(
        pre_names | post_names, reserved, context="cell_summary pre_agg/post_agg"
    )
    overlap = pre_names & post_names
    if overlap:
        raise ValueError(
            f"cell_summary: name(s) {sorted(overlap)} appear in both pre_agg and "
            f"post_agg; a single output column can't hold both sides. Use distinct "
            f"names (e.g. an _out / _in suffix)."
        )

    # Pick the identity column name. Default "cell_id"; if that collides with an
    # output column (e.g. an annotation literally named cell_id, the exact
    # stable-id setup), fall back to the universe annotation's cell_id_col, then
    # to a unique suffix. The no-collision case keeps "cell_id" (non-breaking).
    output_names = reserved | pre_names | post_names
    key_name = "cell_id"
    if key_name in output_names:
        fallback = None
        try:
            uni = st._resolve_universe_annotation()
            fallback = get_cell_annotation_store(st)[uni].cell_id_col
        except (ValueError, KeyError):
            fallback = None
        if fallback is not None and fallback not in output_names:
            key_name = fallback
        else:
            key_name = unique_name("cell_id", output_names)

    # Extra per-side aggregations layered on top of the shared n_syn count:
    # summed weights, caller-supplied aggs, and (optionally) annotation values.
    pre_extra: list[pl.Expr] = [pl.sum(w).alias(f"{w}_output") for w in weights]
    post_extra: list[pl.Expr] = [pl.sum(w).alias(f"{w}_input") for w in weights]
    if pre_agg:
        pre_extra.extend(expr.alias(name) for name, expr in pre_agg.items())
    if post_agg:
        post_extra.extend(expr.alias(name) for name, expr in post_agg.items())
    if include_annotations:
        pre_extra.extend(pl.col(f"{c}_pre").first().alias(c) for c in anno_cols)
        post_extra.extend(pl.col(f"{c}_post").first().alias(c) for c in anno_cols)

    # Shared per-cell grouping: cell_summary counts synapses (one row = one
    # synapse), so the count is pl.len(). cells(participation=True) reuses the
    # same helper over the pair-level edgelist with pl.sum("n_syn").
    pre_lf, post_lf = aggregate_per_cell(
        lf,
        pre_col=pre_col,
        post_col=post_col,
        count_expr=pl.len(),
        out_count="n_syn_output",
        in_count="n_syn_input",
        pre_exprs=pre_extra,
        post_exprs=post_extra,
        cell_id_col=key_name,
    )
    pre_df = pre_lf.collect()
    post_df = post_lf.collect()

    # Full outer join; coalesce merges the two identity key columns.
    result = pre_df.join(post_df, on=key_name, how="full", coalesce=True)

    # Additive columns get 0 (not null) for cells absent from one side: a cell
    # that never appears as pre makes exactly 0 output synapses (and 0 summed
    # weight). This is the factually-correct value and matches
    # cells(participation=True). Custom pre_agg / post_agg outputs are left
    # null on the absent side — the user's aggregation (e.g. a mean) is
    # genuinely undefined over zero rows, where 0 would be wrong.
    additive = ["n_syn_output", "n_syn_input"]
    additive += [f"{w}_output" for w in weights] + [f"{w}_input" for w in weights]
    result = result.with_columns(
        pl.col(c).fill_null(0) for c in additive if c in result.columns
    )

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


# ── distance / feature helpers ───────────────────────────────────────────────


def with_distance(
    table,
    name: str,
    distance_fn: Callable[[str, str], pl.Expr],
    *,
    annotation: str | None = None,
):
    """Register a per-pair distance column derived from soma positions.

    Wraps ``add_expression`` with the position-column resolution boilerplate
    so callers don't need to remember the per-side naming convention. The
    typical idiom::

        from trajan import radial_distance, with_distance
        st = with_distance(st, "d_radial", radial_distance)

    is equivalent to::

        st.add_expression(
            "d_radial", radial_distance("soma_pre", "soma_post")
        )

    but composes cleanly with user-defined ``distance_fn`` callables — a
    cortical-curvature-corrected radial distance, an axon-dendrite
    proximity proxy, etc. The library doesn't bake in a "default" distance:
    pick the one that matches the geometry of your data, and bin on it.

    Works on ``SynapseTable``, ``EdgeList``, ``ConnectivityTable``, and
    ``PairUniverse``. For a ``SynapseTable``, the expression is classified
    as cell-side ("pre"/"post"/"both") and propagates through ``.edgelist()``
    and ``possible_pairs(...)`` automatically.

    Parameters
    ----------
    table
        Any trajan table with at least one position-bearing cell annotation.
    name : str
        Output column name for the distance expression.
    distance_fn : callable ``(col_a, col_b) -> pl.Expr``
        Takes two position column names (typically struct columns with
        ``x``/``y``/``z`` fields) and returns the distance expression. The
        shipped helpers :func:`trajan.euclidean_distance` and
        :func:`trajan.radial_distance` follow this signature; user-defined
        callables can do arbitrary geometry.
    annotation : str or None
        Position-bearing annotation to resolve positions from. Auto-resolved
        when exactly one such annotation is registered.

    Returns
    -------
    Same type as ``table``
        With the named distance expression registered. Use the new column
        as a bin / filter / group key in downstream stats.
    """
    ann_name = table._resolve_position_annotation(annotation)
    pos_col = get_cell_annotation_store(table)[ann_name].position_col
    return table.add_expression(name, distance_fn(f"{pos_col}_pre", f"{pos_col}_post"))


# ── binning primitive ────────────────────────────────────────────────────────


def _ordered_cut_labels(breaks: list) -> list[str]:
    """Polars' own ``cut`` labels for ``breaks``, in ascending numeric order.

    Derived by cutting one representative point inside each of the
    ``len(breaks) + 1`` bins (``(-inf, b0]`` … ``(b_last, inf]``) rather than
    formatting the labels by hand — so the strings match exactly what
    ``pl.Expr.cut`` produces for real data, and the subsequent ``Enum`` cast
    cannot mismatch. Used to make continuous bins sort numerically (see
    :func:`_resolve_bin_spec`).
    """
    pts = (
        [breaks[0] - 1]
        + [(breaks[i] + breaks[i + 1]) / 2 for i in range(len(breaks) - 1)]
        + [breaks[-1] + 1]
    )
    return pl.Series([float(p) for p in pts]).cut(breaks).cast(pl.Utf8).to_list()


def _resolve_bin_spec(name: str, spec: BinSpec) -> tuple[str, pl.Expr | None]:
    """Compile one ``(column, spec)`` entry into ``(output_col, optional_expr)``.

    Returns the column name to group by, plus an expression that materializes
    the bin column when needed (``None`` for categorical pass-through where
    no new column is required).

    Continuous bins are emitted as an ordered :class:`polars.Enum` (categories
    in ascending bin order) rather than the bare ``Categorical`` ``pl.cut``
    returns. A ``cut`` ``Categorical`` sorts *lexically* (so ``"(100, 200]"``
    lands before ``"(50, 100]"``) and only carries the bins present in the
    data; the ordered ``Enum`` makes ``sort`` / ``pivot`` / plotting axes honor
    numeric order automatically and includes every bin. Categorical
    pass-through (``spec is None``) is already user-controlled, so it is left
    untouched — the ordering issue only arises for numeric bins.
    """
    if spec is None:
        return name, None
    if isinstance(spec, (list, tuple)):
        out = f"{name}_bin"
        breaks = list(spec)
        labels = _ordered_cut_labels(breaks)
        expr = pl.col(name).cut(breaks).cast(pl.Enum(labels)).alias(out)
        return out, expr
    raise TypeError(
        f"bin_by spec for {name!r} must be a list of edges or None, "
        f"got {type(spec).__name__}"
    )


# ── shared binning helper ────────────────────────────────────────────────────


def _resolve_keys(
    bin_by: dict[str, BinSpec] | None,
    group_by: list[str] | str | None,
) -> list[str]:
    """Compute the ordered group-key list (group_by + bin output columns).

    The bin output column for each ``bin_by`` entry is determined by
    :func:`_resolve_bin_spec` (``{name}_bin`` for continuous, ``{name}``
    for categorical pass-through). Used by both :func:`_apply_bins` and by
    callers that already have a frame with bins materialized but need to
    re-derive the key list (e.g. ``bootstrap_over_cells`` for its final join).
    """
    if isinstance(group_by, str):
        group_by = [group_by]
    keys = list(group_by or [])
    for name, spec in (bin_by or {}).items():
        out_col, _ = _resolve_bin_spec(name, spec)
        keys.append(out_col)
    return keys


def _apply_bins(
    lf: pl.LazyFrame,
    bin_by: dict[str, BinSpec] | None,
    group_by: list[str] | str | None,
) -> tuple[pl.LazyFrame, list[str]]:
    """Materialize bin columns on ``lf`` and return the ordered group-key list.

    Single source of truth for binning. Both :func:`counts` and the bootstrap
    setup funnel through here — they cannot drift on bin-column naming or
    key ordering.

    Returns
    -------
    pl.LazyFrame
        ``lf`` with bin columns materialized via ``with_columns``.
    list[str]
        Ordered group / bin keys, matching what :func:`_resolve_keys` returns.
    """
    keys = _resolve_keys(bin_by, group_by)
    bin_exprs = [
        expr
        for _, expr in (
            _resolve_bin_spec(name, spec) for name, spec in (bin_by or {}).items()
        )
        if expr is not None
    ]
    if bin_exprs:
        lf = lf.with_columns(*bin_exprs)
    return lf, keys


# ── counts primitive ─────────────────────────────────────────────────────────


def counts(
    pu: "PairUniverse",
    *,
    bin_by: dict[str, BinSpec] | None = None,
    group_by: list[str] | str | None = None,
) -> pl.DataFrame:
    """Per-bin observed and possible counts from a ``PairUniverse``.

    This is the workhorse primitive: every density / probability statistic in
    this module composes from ``counts()`` + a transform. Raw counts are
    always returned so callers can plug in any estimator (Wilson,
    Clopper-Pearson, bootstrap-over-cells, ...) that this library doesn't
    yet ship — see ``project_connection_probability.md``.

    Parameters
    ----------
    pu : PairUniverse
        Pair frame with observed weights overlaid (typically from
        :func:`trajan.possible_pairs`). The ``n_syn`` weight defines the
        ``observed`` predicate; any other registered weight is summed.
    bin_by : dict[str, list | None] or None
        Per-column bin spec. ``list[float]`` of edges → continuous binning
        via ``pl.cut``; ``None`` → categorical pass-through. Joint binning
        is the cross product of multiple entries. Examples::

            bin_by={"d_rho": [0, 50, 100, 200]}                  # 1D scalar
            bin_by={"d_rho": [0,50,100], "d_y": [-100,0,100]}    # 2D joint
            bin_by={"d_rho": [0,50,100], "cell_type_post": None} # mixed
    group_by : list[str] or str or None
        Additional columns to group by alongside ``bin_by`` keys. Useful for
        categorical strata (``"cell_type_pre"``) where you don't want to
        rename the column to ``..._bin``.

    Returns
    -------
    pl.DataFrame
        One row per cross-product cell of group / bin keys, sorted by those
        keys. Columns: each group / bin key, then:

        - ``k_observed`` — number of pairs with ``n_syn > 0``
        - ``n_possible`` — total pairs in the bin (denominator)
        - ``sum_<weight>`` — sum of each registered weight on ``pu``

    Notes
    -----
    Pairs with ``n_syn == 0`` (unobserved possible pairs) contribute to
    ``n_possible`` but not ``k_observed``. That's the whole point: a
    connection-probability denominator needs both.

    Bins with ``n_possible == 0`` are simply absent from the output — the
    group-by never sees them. If you need empty bins represented, post-join
    against a reference frame of all expected bin labels.

    A pair whose binned feature is ``null`` (e.g. a soma-distance bin for a
    cell with no registered position) is kept under a ``null`` bin rather than
    dropped — it stays in ``n_possible``, so the denominator is never silently
    shrunk. This matches categorical ``None`` pass-through, which likewise
    keeps ``null`` as its own group. Filter such pairs out upstream if you want
    them excluded.
    """
    _require_pair_universe(pu)
    if not bin_by and not group_by:
        raise ValueError(
            "counts() requires at least one bin_by or group_by key — without "
            "grouping the result is one row, which you can get directly via "
            "pu.build_lazy().select(...)."
        )

    lf, keys = _apply_bins(pu.build_lazy(), bin_by, group_by)

    agg_exprs: list[pl.Expr] = [
        (pl.col("n_syn") > 0).sum().alias("k_observed"),
        pl.len().alias("n_possible"),
    ]
    for w in pu.weights:
        agg_exprs.append(pl.sum(w).alias(f"sum_{w}"))

    return lf.group_by(keys).agg(agg_exprs).sort(keys).collect()


# ── CI estimators ────────────────────────────────────────────────────────────


def _z_value(alpha: float) -> float:
    """Two-sided normal quantile ``z_{1-α/2}``. Lazy-imports scipy.

    Keeping scipy as an optional (not required) runtime dependency: a user
    who only ever calls :func:`counts` / :func:`connection_probability`
    without an estimator never touches this code path. Importing here gives
    a clear error message when a CI factory is called without scipy
    installed.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    try:
        from scipy.stats import norm
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "CI estimators require scipy for the normal-quantile lookup. "
            "Install with `pip install scipy` or `uv add scipy`."
        ) from e
    return float(norm.ppf(1 - alpha / 2))


def wilson_ci(alpha: float = 0.05) -> Estimator:
    """Wilson score interval factory.

    Returns an :data:`Estimator` callable suitable for
    ``connection_probability(estimator=wilson_ci(alpha=0.05))``. The
    estimator produces ``p_lo`` / ``p_hi`` columns; the point estimate
    ``p = k/n`` is set by :func:`connection_probability` itself, not the
    estimator (this is the library convention so the MLE stays consistent
    across estimator choices).

    Wilson is the standard recommended interval for binomial proportions:
    closed-form (no numerical inversion), naturally bounded in ``[0, 1]``
    at the extremes (``k = 0`` gives ``p_lo = 0``; ``k = n`` gives
    ``p_hi = 1``), and well-behaved for small ``k`` where the normal
    approximation collapses. See Brown, Cai & DasGupta (2001) for the
    definitive comparison; Wilson out-performs both Clopper-Pearson
    (conservative) and normal-approximation (over-narrow) on
    average-coverage metrics.

    Parameters
    ----------
    alpha : float
        Two-sided significance level. Default ``0.05`` (95% CI).

    Notes
    -----
    The Wilson interval is centered on a *shrunken* estimate
    ``p̃ = (k + z²/2) / (n + z²)``, not on the raw MLE. The interval is
    asymmetric around ``k/n`` and always contains it.
    """
    z = _z_value(alpha)
    z2 = z * z

    def _est(k: pl.Expr, n: pl.Expr) -> dict[str, pl.Expr]:
        p = k / n
        denom = 1.0 + z2 / n
        center = (p + z2 / (2 * n)) / denom
        margin = (z / denom) * (p * (1 - p) / n + z2 / (4 * n * n)).sqrt()
        return {"p_lo": center - margin, "p_hi": center + margin}

    return _est


def agresti_coull_ci(alpha: float = 0.05) -> Estimator:
    """Agresti–Coull interval factory.

    "Add ``z²/2`` successes and ``z²/2`` failures, then apply the
    normal-approximation interval to the adjusted proportion." Wider than
    Wilson but exceptionally simple to explain — the standard pedagogical
    interval. Recommended by Agresti & Coull (1998) for teaching contexts
    and as a robust default when you don't want to think too hard about
    coverage trade-offs.

    Parameters
    ----------
    alpha : float
        Two-sided significance level. Default ``0.05`` (95% CI).

    Notes
    -----
    The interval is **not** clipped to ``[0, 1]``: at extreme ``k`` (near
    ``0`` or ``n``) you can see ``p_lo < 0`` or ``p_hi > 1``. The
    mathematical interval is what's returned; clipping is one polars
    expression away if the caller wants it. This is deliberate — the
    library exposes the raw estimator output and leaves opinionated
    transforms to the caller.
    """
    z = _z_value(alpha)
    z2 = z * z

    def _est(k: pl.Expr, n: pl.Expr) -> dict[str, pl.Expr]:
        n_tilde = n + z2
        p_tilde = (k + z2 / 2) / n_tilde
        margin = z * (p_tilde * (1 - p_tilde) / n_tilde).sqrt()
        return {"p_lo": p_tilde - margin, "p_hi": p_tilde + margin}

    return _est


# ── connection_probability / connection_density ──────────────────────────────


def connection_probability(
    table: "StatInput",
    *,
    bin_by: dict[str, BinSpec] | None = None,
    group_by: list[str] | str | None = None,
    estimator: Estimator | None = None,
    universe: str | None = None,
    include_self: bool = False,
) -> pl.DataFrame:
    """Estimate P(connected) per bin from the universe of possible pairs.

    Built on :func:`counts`; adds a ``p`` column (= ``k_observed /
    n_possible``) and, when ``estimator`` is given, the columns the
    estimator produces (typically ``p_lo`` / ``p_hi``).

    Parameters
    ----------
    table : PairUniverse, SynapseTable, or EdgeList
        The denominator frame, or a table to derive it from. A
        ``SynapseTable`` / ``EdgeList`` is passed through
        :func:`trajan.possible_pairs` automatically, so the common case is a
        one-liner: ``connection_probability(st, bin_by=...)``. Pass a
        ``PairUniverse`` directly when you've already pruned the cross-product
        (spatial / id filters) and want to reuse it.
    bin_by, group_by
        See :func:`counts`.
    estimator : callable or None
        Pure ``(k_expr, n_expr) -> {col: expr}`` callable. Returned dict is
        added to the result via ``with_columns``. Pass one of the shipped
        factories — :func:`wilson_ci`, :func:`agresti_coull_ci` — or supply
        your own callable matching the protocol. For cell-resampling CIs
        (the recommended method for dense connectomics, where pair-level
        binomial CIs are overconfident) use :func:`bootstrap_over_cells`
        rather than passing an estimator here. See
        ``project_connection_probability.md``.
    universe : str or None
        Universe annotation name, forwarded to :func:`trajan.possible_pairs`
        when ``table`` is a SynapseTable / EdgeList. Ignored when a
        PairUniverse is passed (its universe was fixed at construction).
    include_self : bool
        Whether self-pairs (``pre == post``) count toward the denominator.
        Forwarded to :func:`trajan.possible_pairs`; ignored for a
        pre-built PairUniverse. Default ``False``.

    Returns
    -------
    pl.DataFrame
        :func:`counts` output extended with ``p`` (and any estimator columns).

    See Also
    --------
    connection_density : identical computation, documented for dense data.

    Notes
    -----
    **Terminology — see ``project_connection_probability.md``.** "Connection
    probability" usually denotes a *sampled* measurement (ephys, sparse
    reconstruction). For *dense* reconstructions — where every candidate pair
    is observed — "connection density" is the more accurate name. The formula
    is identical; the interpretation differs. :func:`connection_density` is
    the name-only twin for that case.
    """
    pu = _as_pair_universe(table, universe=universe, include_self=include_self)
    df = counts(pu, bin_by=bin_by, group_by=group_by)
    df = df.with_columns((pl.col("k_observed") / pl.col("n_possible")).alias("p"))
    if estimator is not None:
        new_cols = estimator(pl.col("k_observed"), pl.col("n_possible"))
        df = df.with_columns(*[expr.alias(name) for name, expr in new_cols.items()])
    return df


def connection_density(
    table: "StatInput",
    *,
    bin_by: dict[str, BinSpec] | None = None,
    group_by: list[str] | str | None = None,
    estimator: Estimator | None = None,
    universe: str | None = None,
    include_self: bool = False,
) -> pl.DataFrame:
    """Per-bin connection density for *dense* reconstructions.

    A name-only twin of :func:`connection_probability`: same formula
    (``p = k_observed / n_possible``), same columns, same arguments. The
    distinction is interpretive, not numerical — see
    ``project_connection_probability.md``. In a dense reconstruction every
    candidate pair is observed, so ``k / n`` is the *density* of realized
    connections among anatomically possible ones, not an estimate of an
    underlying sampling probability. Use whichever name matches how you'll
    describe the result; the point estimate stays in the ``p`` column either
    way so estimators and :func:`bootstrap_over_cells` compose unchanged.

    Parameters
    ----------
    table, bin_by, group_by, estimator, universe, include_self
        See :func:`connection_probability`.

    Returns
    -------
    pl.DataFrame
        Identical shape to :func:`connection_probability`.
    """
    return connection_probability(
        table,
        bin_by=bin_by,
        group_by=group_by,
        estimator=estimator,
        universe=universe,
        include_self=include_self,
    )


# ── cell-level bootstrap ─────────────────────────────────────────────────────


def _bootstrap_setup(
    pu: "PairUniverse",
    bin_by: dict[str, BinSpec] | None,
    group_by: list[str] | str | None,
    universe: str | None,
) -> tuple[pl.DataFrame, list[str], str, str, str, list, int, dict[str, str]]:
    """Materialize the binned pair frame and the universe cell list once.

    Returned tuple: ``(pair_df, keys, pre_col, post_col, cell_id_col,
    cell_ids, N, names)``. ``names`` maps the internal multiplicity / weight
    column roles (``m``, ``m_pre``, ``m_post``, ``w``) to collision-free
    column names for this call. ``pair_df`` is *eagerly collected* — bootstrap loops
    over it many times and re-collecting each iteration would dominate
    runtime.

    Requires the source ``PairUniverse`` to carry an ``n_syn`` weight; this
    is the "is the pair observed?" predicate. ``possible_pairs`` always
    provides it (the observed-counts overlay materializes ``n_syn`` from the
    edgelist's pair counts). If you constructed a PairUniverse by hand with
    a renamed weight, rename it back to ``n_syn`` first.
    """
    _require_pair_universe(pu)
    universe_name = pu._resolve_universe_annotation(universe)
    cell_id_col = pu._cell_annotations[universe_name].cell_id_col
    cell_ids = (
        pu._cell_annotations[universe_name]
        .lf.select(cell_id_col)
        .collect()[cell_id_col]
    ).to_list()
    N = len(cell_ids)

    lf, keys = _apply_bins(pu.build_lazy(), bin_by, group_by)
    pre_col = pu.pre_col
    post_col = pu.post_col
    # Only the columns we actually need downstream — keep the eagerly
    # collected frame slim so the per-resample joins are cheap.
    pair_df = lf.select([pre_col, post_col, "n_syn"] + keys).collect()

    # Derive internal column names that can't collide with a group_by/bin key
    # (or the id/count columns) — e.g. a user grouping on a column literally
    # named "m_pre".
    taken = set(pair_df.columns) | {cell_id_col}
    names = {
        "m": unique_name("m", {cell_id_col}),
        "m_pre": unique_name("m_pre", taken),
    }
    names["m_post"] = unique_name("m_post", taken | {names["m_pre"]})
    names["w"] = unique_name("w", taken | {names["m_pre"], names["m_post"]})
    return pair_df, keys, pre_col, post_col, cell_id_col, cell_ids, N, names


def _bootstrap_one(
    pair_df: pl.DataFrame,
    mult: pl.DataFrame,
    keys: list[str],
    pre_col: str,
    post_col: str,
    cell_id_col: str,
    names: dict[str, str],
) -> pl.DataFrame:
    """One bootstrap resample → ``(*keys, k_observed, n_possible, p)`` DataFrame.

    Column names match :func:`counts`. The dtype is ``Float64`` rather than
    integer because each pair contributes weight ``m_pre * m_post`` (cell
    multiplicities in the resample), not a single observation. The "weighted
    counts" semantics is the only thing that differs from :func:`counts`'s
    integer outputs.

    Cells absent from the resample (``m == 0``) are inner-joined out, which
    is semantically equivalent to keeping them with weight ``0`` and is
    much faster. By construction every retained pair has ``m_pre, m_post >
    0`` so ``n_possible = sum(_w) > 0`` per bin (no division-by-zero).
    """
    m, m_pre, m_post, w = names["m"], names["m_pre"], names["m_post"], names["w"]
    return (
        pair_df.lazy()
        .join(
            mult.lazy().rename({m: m_pre}),
            left_on=pre_col,
            right_on=cell_id_col,
            how="inner",
        )
        .join(
            mult.lazy().rename({m: m_post}),
            left_on=post_col,
            right_on=cell_id_col,
            how="inner",
        )
        .with_columns((pl.col(m_pre) * pl.col(m_post)).alias(w))
        .group_by(keys)
        .agg(
            (pl.col(w) * (pl.col("n_syn") > 0).cast(pl.UInt32))
            .sum()
            .cast(pl.Float64)
            .alias("k_observed"),
            pl.col(w).sum().cast(pl.Float64).alias("n_possible"),
        )
        .with_columns((pl.col("k_observed") / pl.col("n_possible")).alias("p"))
        .collect()
    )


def cell_bootstrap_iter(
    table: "StatInput",
    *,
    bin_by: dict[str, BinSpec] | None = None,
    group_by: list[str] | str | None = None,
    n_resamples: int = 1000,
    seed: int | None = None,
    universe: str | None = None,
) -> Iterator[pl.DataFrame]:
    """Yield ``n_resamples`` cell-bootstrap per-bin DataFrames.

    The building block underneath :func:`bootstrap_over_cells`. Use this
    directly when you want a CI method other than percentile (BCa,
    studentized, bias-corrected) or a non-CI summary (variance,
    higher-order moments, a custom test statistic) over the bootstrap
    distribution.

    Each yielded DataFrame has columns ``[*keys, k_observed, n_possible, p]``
    where ``keys`` is the cross product of ``group_by`` and the bin columns
    named by ``bin_by`` (same as :func:`counts`). The two count columns are
    ``Float64`` rather than integer because each pair contributes weight
    ``m_pre * m_post`` (cell-multiplicity product); names match
    :func:`counts` so estimators that take ``(k, n)`` expressions compose on
    either output. Bins with no pairs in a given resample are simply absent
    from that resample's frame — they contribute no rows.

    Parameters
    ----------
    table : PairUniverse, SynapseTable, or EdgeList
        The denominator frame, or a table to derive it from (normalized via
        :func:`trajan.possible_pairs`). See :func:`connection_probability`.
    bin_by, group_by, universe
        See :func:`counts` / :func:`bootstrap_over_cells`.
    n_resamples : int
        Number of resamples to draw. Default ``1000``.
    seed : int or None
        Numpy seed; pass an int for reproducibility.

    Yields
    ------
    pl.DataFrame
        ``[*keys, k_observed, n_possible, p]`` for one resample.

    Notes
    -----
    The resampling unit is a *cell*: each resample draws ``N`` cells with
    replacement from the universe annotation (where ``N`` is the universe
    size). For a pair ``(a, b)`` where ``a`` appears ``m_a`` times and
    ``b`` appears ``m_b`` times in a resample, its contribution to both
    ``k_observed`` and ``n_possible`` (the latter unconditional, the former
    gated on ``n_syn > 0``) is ``m_a * m_b``. Equivalent to physically
    duplicating cells in the universe and rebuilding ``possible_pairs``, but
    ~5 orders of magnitude cheaper because it never materializes the
    duplicated cross-product.
    """
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "cell_bootstrap_iter requires numpy. Install with "
            "`pip install numpy` or `uv add numpy`."
        ) from e

    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not bin_by and not group_by:
        raise ValueError(
            "cell_bootstrap_iter requires at least one bin_by or group_by key"
        )

    pu = _as_pair_universe(table, universe=universe)
    pair_df, keys, pre_col, post_col, cell_id_col, cell_ids, N, names = (
        _bootstrap_setup(pu, bin_by, group_by, universe)
    )
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        m = rng.multinomial(N, [1.0 / N] * N)
        # Drop zero-multiplicity cells from the join entirely. The inner
        # join then guarantees m_pre > 0 and m_post > 0 on every kept row,
        # so _w > 0 and per-bin n > 0. Without this filter, a bin where
        # every pair has at least one m=0 endpoint would produce sum(_w)=0
        # and a 0/0 NaN that poisons the percentile aggregation.
        mult = pl.DataFrame({cell_id_col: cell_ids, names["m"]: m}).filter(
            pl.col(names["m"]) > 0
        )
        yield _bootstrap_one(pair_df, mult, keys, pre_col, post_col, cell_id_col, names)


def bootstrap_over_cells(
    table: "StatInput",
    *,
    bin_by: dict[str, BinSpec] | None = None,
    group_by: list[str] | str | None = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
    universe: str | None = None,
) -> pl.DataFrame:
    """Cell-bootstrap percentile CI for per-bin connection density.

    The recommended CI method for *dense* connectomics data. Pair-level
    binomial CIs (Wilson, Clopper-Pearson) are overconfident here because
    pairs sharing a pre- or post-cell co-vary. Cell-level resampling
    preserves that correlation by re-drawing whole cells, each of which
    brings its full row/column of connections with it.

    Returns the same shape as ``connection_probability(estimator=wilson_ci())``
    — ``[*keys, k_observed, n_possible, sum_<weights>, p, p_lo, p_hi]`` —
    but the bounds are the empirical ``alpha/2`` and ``1 - alpha/2``
    quantiles of the bootstrap distribution rather than a closed-form
    formula. The point estimate ``p`` is the observed ``k/n`` (not the
    resample mean), matching the library convention that the MLE is
    estimator-independent.

    Parameters
    ----------
    table : PairUniverse, SynapseTable, or EdgeList
        The denominator frame, or a table to derive it from (normalized via
        :func:`trajan.possible_pairs`). See :func:`connection_probability`.
    bin_by, group_by, universe
        See :func:`counts`.
    n_resamples : int
        Number of bootstrap resamples. Default ``1000``; standard
        guidance is ``>=1000`` for percentile CIs, ``>=10000`` for stable
        tail probabilities. Higher ``n_resamples`` → tighter Monte-Carlo
        error on the CI bounds (not the CI width itself).
    alpha : float
        Two-sided significance level. Default ``0.05`` (95% CI).
    seed : int or None
        Numpy seed for reproducibility. ``None`` draws fresh randomness.

    Returns
    -------
    pl.DataFrame
        Joined point + CI per bin.

    See Also
    --------
    cell_bootstrap_iter : per-resample building block for custom summaries.
    connection_probability : pair-level binomial CI alternative.

    Notes
    -----
    Bins with no pairs in some resamples have their ``p_lo`` / ``p_hi``
    computed only over the resamples where they were sampled at all,
    which can bias the CI for very sparse bins. If your analysis has bins
    that disappear in many resamples, that is itself a signal — they are
    too thin for cell-bootstrap to characterize, and you likely want to
    widen them or report counts directly.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    pu = _as_pair_universe(table, universe=universe)
    keys = _resolve_keys(bin_by, group_by)
    point = connection_probability(pu, bin_by=bin_by, group_by=group_by)
    per_resample = list(
        cell_bootstrap_iter(
            pu,
            bin_by=bin_by,
            group_by=group_by,
            n_resamples=n_resamples,
            seed=seed,
            universe=universe,
        )
    )
    stacked = pl.concat(per_resample)
    # Use linear interpolation (numpy's default) so the percentile CI agrees
    # with hand-computed references and the broader scientific-Python
    # convention. Polars' default ``nearest`` would give answers off by a
    # ``1 / n_resamples`` step at percentile boundaries.
    ci = stacked.group_by(keys).agg(
        pl.col("p").quantile(alpha / 2, interpolation="linear").alias("p_lo"),
        pl.col("p").quantile(1 - alpha / 2, interpolation="linear").alias("p_hi"),
    )
    # join_nulls so a null group/bin key (e.g. an untyped cell, cell_type=None)
    # matches between point and ci; otherwise that bin keeps its computed p but
    # silently loses its bootstrap bounds.
    return point.join(ci, on=keys, how="left", nulls_equal=True)
