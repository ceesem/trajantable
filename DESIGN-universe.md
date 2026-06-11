# Design draft: universe + scope + cells + possible_pairs

**Status:** pre-implementation design draft. Pressure-test against the four worked analyses before committing code. Promote to `docs/guides/` (or strike) when implemented.

This draft closes the gap surfaced by four worked analyses: the framework has the foundation (blessed columns, role declarations at registration, annotation = properties + membership) but the **universe primitive has no consumer** and there is **no shared scope vocabulary**. Once those land, the stats and viz layers can bolt on without churning the foundation.

## 0. The pieces

Five additions, layered:

1. **Filter side-classification** — extend the existing `_expression_sides` mechanism to filters. Cell-level filters become referenceable from cell-level views.
2. **A shared scope vocabulary** — one enum, three values, defined once. Used by `cells()`, `possible_pairs()`, and any future stat or viz helper.
3. **`cells(table, *, scope, side, universe, strict)`** — free function returning the per-cell DataFrame at the requested scope. Single source of truth for cell-level views.
4. **`PairUniverse` class** — lazy n² pair frame, exposing `.filter`, `.group_by`, `.agg`, `.collect`, spatial filters, and annotation registry — but no `.df` property. Prevents accidental materialization of full cross-products.
5. **`possible_pairs(table, *, universe, include_self)`** — free function returning a `PairUniverse` with observed counts overlaid. The denominator primitive.

The universe annotation contract (the `is_universe=True` flag on `CellAnnotationSpec`) is already in place. These additions are its first real consumers.

## 1. Scope vocabulary

```python
from typing import Literal

Scope = Literal["universe", "filtered", "observed"]
```

Defined once, in `trajan._base`, imported everywhere. The values are minimal and orthogonal:

| Scope        | Cells frame                                                | Pair frame                                            |
| ---          | ---                                                        | ---                                                   |
| `universe`   | raw `is_universe` annotation, no filters applied           | universe × universe, no filters applied               |
| `filtered`   | universe ∩ cell-level filters                              | universe × universe ∩ cell-level filters (each side)  |
| `observed`   | cells appearing in `.df`, joined to the universe annotation| pairs appearing in `.df` (i.e. the regular EdgeList)  |

`filtered` is the default. It's what "1:1 with the analysis" almost always means — same universe, same cell-level filters as the table the user holds.

Notes:

- `cells(scope="observed")` is intersected with the universe annotation. Synapse-data rows referencing cell ids not in the universe are dropped from this view (consistent with "the universe is the authoritative cell set"); `strict=False` opts into a lenient mode that keeps orphan ids with null annotation columns.
- `possible_pairs(scope="observed")` returns the same rows as a plain `el.edgelist()` — provided for API symmetry, not because it's the right primitive for denominators.
- Synapse-level filters never apply to `cells()` (they require synapse context). They DO apply to `possible_pairs(scope="filtered")` because possible-pairs is computed by aggregating the source `SynapseTable.edgelist()` (synapse-level filters bake in there). Documented per-call.

### Asymmetric filters: pre and post may scope differently

The framework naturally supports analyses where the pre and post sides are scoped differently — e.g., "pre is restricted to proofread cells; post is the full universe." Single-sided filters (`pl.col("is_proofread_pre")`) classify as `side="pre"` and only constrain that side. The projection rule (below, §3.1) keeps the asymmetry without forcing the user to choose a side.

For `possible_pairs`, the asymmetry produces a non-square cross-product: |pre-eligible| × |post-eligible|. The 8K × 60K = 480M-row case is exactly the right shape for a "for each proofread pre cell, density against every universe post" analysis — kept lazy, aggregated downstream.

## 2. Filter side-classification (foundational)

Today `SynapseTable._filters: list[pl.Expr]`. Filters are accumulated and applied to the merged plan; the library doesn't know which are cell-level vs synapse-level.

The change:

```python
# Internal: parallel to _expression_sides
self._filter_sides: list[Literal["pre", "post", "both"] | None] = []

# In .filter(expr):
new._filters = self._filters + [expr]
new._filter_sides = self._filter_sides + [self._classify_expression(expr)]
```

`_classify_expression` already exists. Reuse it verbatim. Classification rules:

- All roots are cell-annotation outputs on one side (e.g. `cell_type_pre`, `soma_pre`) → `"pre"`.
- All roots are cell-annotation outputs on the other side → `"post"`.
- Roots from both sides → `"both"`.
- Any synapse- or vertex-level root → `None`.

**Caveat:** classification is captured at `filter()` time against the annotations registered then. Registering a cell annotation later doesn't reclassify existing filters. This matches the existing expression behavior — same caveat, document once.

**Persistence:** add to save/load alongside the filter expressions.

### 3.1 Side-decomposed projection onto the cells frame

Each cell-level filter contributes two projections:

- **Pre-projection:** the filter, with `_pre` references replaced by the bare column name and `_post` references either dropped (if the filter is a conjunction whose post conjunct can be cleanly removed) or left in place. A `side="pre"` filter's pre-projection is the meaningful constraint; its post-projection is `True`. A `side="post"` filter is symmetric.
- **Post-projection:** the symmetric form.
- **`side="both"` conjunctions** decompose: split AND-conjuncts, classify each, and route to its side's projection. Filters that combine sides non-decomposably (e.g., `cell_type_pre != cell_type_post`) project to `True` on both sides — they are pair-only constraints, skipped for cells() with a warning naming the filter, and still applied on `.df`.

A cell is **pre-eligible** if all pre-projections evaluate true for it; **post-eligible** if all post-projections do. `cells(scope="filtered")` returns the union of pre-eligible and post-eligible cells. `cells(scope="filtered", side="pre")` returns pre-eligible only.

Examples:

| Filter | Side | Pre-proj | Post-proj | cells(scope="filtered") |
| --- | --- | --- | --- | --- |
| `is_proofread_pre` | pre | `is_proofread` | `True` | proofread ∪ universe = universe |
| `cell_type_pre == "L23"` | pre | `cell_type == "L23"` | `True` | L23 ∪ universe = universe |
| `cell_type_pre == "L23"` AND `cell_type_post == "L4"` | both | `cell_type == "L23"` | `cell_type == "L4"` | L23 ∪ L4 |
| `cell_type_pre != cell_type_post` | both | `True` (skipped, warned) | `True` | universe |

The asymmetric case from the analyses (pre proofread, post universe) falls out: pre-eligible = 8K proofread, post-eligible = 60K universe, union = 60K. `possible_pairs` keeps the asymmetry: 8K × 60K lazy rows.

## 3. `cells(table, *, scope, universe)`

```python
def cells(
    table: SynapseTable | EdgeList,
    *,
    scope: Scope = "filtered",
    universe: str | None = None,
) -> pl.DataFrame:
    """Return the per-cell DataFrame derived from the registered universe annotation.

    Single source of truth for cell-level views. Every viz helper and every
    per-cell statistic should call this rather than reaching into annotations
    directly — that keeps plots and stats trivially 1:1.

    Parameters
    ----------
    table : SynapseTable or EdgeList
        Source table. Must have a cell annotation registered with
        ``is_universe=True`` for ``scope="universe"`` and ``scope="filtered"``;
        ``scope="observed"`` falls back to the observed cell ids when no
        universe is registered (but still left-joins to it when available).
    scope : "universe" | "filtered" | "observed"
        See the scope table above. Default ``"filtered"``.
    universe : str or None
        Name of the cell annotation marked ``is_universe=True``. Auto-resolved
        when exactly one is registered; pass explicitly to disambiguate.

    Returns
    -------
    pl.DataFrame
        One row per cell. Columns are the universe annotation's data columns
        (cell_id_col stays under its original name).
    """
```

Behavior:

- `scope="universe"`: return `table._cell_annotations[universe].lf.collect()`. No filters, no joins.
- `scope="filtered"`: same starting point, then `lf.filter(f)` for each `f` whose side-class is in {"pre", "post", "both"}. For two-sided filters (`cell_type_pre == "L23" AND cell_type_post == "L4"`), the filter is *unilaterally* applied to the cells frame by collapsing the suffix — this is a small substitution: rewrite `cell_type_pre`/`cell_type_post` → `cell_type` in the filter expression. Then dedupe. See open question #1 below.
- `scope="observed"`: `union(table.df[pre_col], table.df[post_col])` → distinct ids → left join the universe annotation. Cells observed in the data but absent from the universe are dropped (a strict-by-default choice; see open question #4).

## 4. `PairUniverse` and `possible_pairs(table, *, universe, include_self)`

```python
class PairUniverse:
    """A lazy universe × universe pair frame, used as the denominator primitive
    for density / probability / null-model statistics.

    Holds the same kind of state as ``EdgeList`` — blessed pre/post columns,
    a weight list, a registry of cell annotations, accumulated filters and
    expressions, the cell-alias registry — but does NOT expose ``.df``.
    Materialization is explicit via ``.collect()`` after the user has
    composed enough filters / aggregations to bring the row count down.

    Public surface mirrors EdgeList where it makes sense:

    - ``.filter(expr)`` — return a new PairUniverse with the filter accumulated
    - ``.filter_by_soma_distance(d, *, annotation=None)`` — spatial prune
    - ``.filter_by_bbox(bbox, *, annotation=None)`` — bbox prune
    - ``.filter_by_ids(pre_ids=None, post_ids=None)`` — id-set prune
    - ``.add_annotation(name, df, ...)`` — register additional annotations
    - ``.add_expression(name, expr)`` — register computed columns
    - ``.group_by(cols).agg(...)`` — direct polars aggregation pass-through
    - ``.collect()`` — materialize as a ``pl.DataFrame``
    - ``.to_edgelist()`` — explicit demotion to ``EdgeList`` (warns if row
      count is large; final escape hatch)

    No ``.df`` property. No automatic caching. Density stats consume this
    via group-by + sum without ever materializing the cross-product.
    """


def possible_pairs(
    table: SynapseTable | EdgeList,
    *,
    universe: str | None = None,
    include_self: bool = False,
) -> PairUniverse:
    """Enumerate the universe × universe pair frame with observed counts overlaid.

    Every possible (pre, post) pair drawn from the universe annotation. The
    ``n_syn`` weight is the observed synapse count (0 for unobserved pairs);
    any other registered weights on the source are likewise overlaid (0 where
    unobserved). All cell annotations on the source are propagated for
    symmetric pre/post joins, so spatial filters and cell-type filters
    compose on the result.

    This is the denominator primitive. Density / probability statistics
    aggregate over this; observed-only statistics use ``.edgelist()`` /
    the source table directly.

    Parameters
    ----------
    table : SynapseTable or EdgeList
        Source with a registered universe annotation. SynapseTable inputs
        are aggregated to pair-level via ``.edgelist()`` first (synapse-level
        filters bake in there).
    universe : str or None
        Name of the universe annotation; auto-resolved if unambiguous.
    include_self : bool
        If False (default), self-pairs (pre == post) are excluded. Most
        connectomics analyses exclude autapses.

    Returns
    -------
    PairUniverse
        Lazy n² (or n² − n) row plan with observed weights overlaid. For
        |universe| > 10k, materializing the full cross-product is impractical
        — the contract is "compose filters / aggregations on this; collect
        at the end."
    """
```

Behavior:

- Resolve the universe annotation via `_resolve_universe_annotation`.
- Build the cross-product as a `pl.LazyFrame`: `universe.lf.join(universe.lf, how="cross")` with renaming to produce `pre_col` and `post_col` matching the table's blessed columns.
- Drop self-pairs unless `include_self=True`.
- Left-join the source's observed `.edgelist()` on `(pre_col, post_col)`, filling missing weight columns with 0.
- Apply cell-level filters via the side-decomposed projection rule (§3.1): pre-projection on the pre side, post-projection on the post side.
- Return a `PairUniverse` carrying the source's cell annotations (for symmetric joining), aliases, weights, and the rewritten filters.

## 5. How `is_universe` is now consumed

Today: the flag is set on `CellAnnotationSpec`, persisted, and looked up by `_resolve_universe_annotation` — but nothing reads the resolved value. After this draft:

- `cells()` reads `table._cell_annotations[universe].lf` (and likewise on EdgeList via `table._annotations[universe].lf`).
- `possible_pairs()` reads the same.
- Future stats (`connection_density`, `cell_summary`, null shufflers) call `cells()` / `possible_pairs()` and inherit the resolution.

The migration anchor in `project_annotation_roles.md` ("the resolver becomes the place that returns a `Universe` object instead of an annotation name") still applies — when the `Universe` class lands, the resolver returns one and `cells()` / `possible_pairs()` upgrade by reading from it instead of the annotation spec directly. No call-site changes.

## 6. The four analyses, written against this API

### Analysis 1 — connection density per pre cell × (cell type, radial distance)

```python
st = (
    SynapseTable(syn)
    .add_cell_annotation("ct", cells_df, cell_id_col="root_id",
                         position_col="soma", is_universe=True)
)
st.add_spatial_features(prefix="d", center="pre", target="post")  # adds d_rho on .df
el = st.edgelist()                                                # observed pairs

# Denominator + numerator in one frame
pp = (
    possible_pairs(el)                          # universe × universe, n_syn=0 for unobserved
    .filter_by_soma_distance(500)               # prune cross-product before group_by
)

# Stats free function (future) — written against possible_pairs
density = connection_density(
    el,                                         # observed (numerator counts implicit)
    group_by=["pre_pt_root_id", "cell_type_post"],
    bin_by={"d_rho": [0, 50, 100, 200, 500]},
    universe="ct",
)
```

Internally `connection_density` calls `possible_pairs(el)` for the denominator and `el` for the numerator, both bound to the same universe annotation by definition. No drift.

### Analysis 2 — compartment filter → cell-type × cell-type, normalized by # target cells

This one needs the label-annotation kind (separate piece of work, sketched here for context — *not* part of this draft's scope):

```python
ct = (
    SynapseTable(syn)
    .add_cell_annotation("ct", cells_df, cell_id_col="root_id", is_universe=True)
    .add_vertex_annotation("compart", compart_df, vertex_id_col="l2_id",
                           pre_vertex_col="pre_l2_id", post_vertex_col="post_l2_id")
    .filter(pl.col("is_soma_post"))                          # synapse-level
    .type_edgelist(pre="cell_type_pre", post="cell_type_post")  # explicit args; no _pre sniffing
    .add_label_annotation("type_meta", types_df,             # NEW kind, future PR
                          label_col="cell_type",
                          denominator_col="n_cells")
)
ct.normalize(by="post", total="label")                       # uses denominator role
```

Universe/scope draft doesn't have to deliver `add_label_annotation` — it only needs to *not foreclose* it. The shape is identical to `add_cell_annotation`, just with an additional role flag. Compatible.

### Analysis 3 — cell-pair similarity vs spatial + annotation null

```python
el = SynapseTable(...).edgelist()

# Observed
sim_obs = cell_similarity(el, metric="cosine", universe="ct")  # future stats fn

# Null — uses possible_pairs to know who could swap with whom
from trajan.nulls import spatial_type_swap
gen = spatial_type_swap(
    st,
    radius=100,
    preserve=["cell_type"],
    universe="ct",     # explicit; the same annotation density uses
    n=1000,
)
sim_null = [cell_similarity(t.edgelist(), metric="cosine") for t in gen]
p = (sim_null >= sim_obs).mean(axis=0)
```

`cells()` and `possible_pairs()` here power the shuffler's "what's a legal swap partner?" lookup. The shuffler reads the universe annotation, joins positions and the preserve columns, builds a per-cell partner pool, and rebases the post column.

### Analysis 4 — plot somas, 1:1 with the analysis

```python
import trajan.viz as tv

tv.plot_somas(st)                              # internally: cells(st, scope="filtered")
tv.plot_somas(st, scope="observed",            # cells appearing in filtered data only
              color_by="cell_type")
tv.plot_somas(possible_pairs(el), agg="degree", color_by="cell_type")
```

The "1:1" guarantee comes from `tv.plot_somas` calling `cells()` internally. The user passes the same `st`/`el` they pass to stats; viz inherits scope semantics automatically.

## 7. Resolved design decisions

1. **Cell-level filter projection — RESOLVED.** Side-decomposed projection (see §3.1). Single-sided filters constrain only their side; two-sided AND-conjunctions decompose; pair-only filters (e.g. `cell_type_pre != cell_type_post`) project to `True` on both sides with a warning, and still apply on `.df`. `cells(scope="filtered")` returns the union of pre-eligible and post-eligible cells; `cells(side="pre" | "post")` returns one side only.

2. **Return type for `possible_pairs` — RESOLVED: separate class.** Introduce a `PairUniverse` (or similar — name TBD) type that holds a lazy plan and exposes `.filter`, `.group_by`, `.agg`, `.collect`, `.add_annotation`, and the spatial filters from `EdgeList`. No `.df` property — collection requires explicit `.collect()`. This prevents accidental materialization of n² rows. The type carries the source's cell annotations, aliases, weights, and rewritten cell-level filters so it composes with the same machinery as `EdgeList`.

3. **`possible_pairs` is a free function — RESOLVED.** Free function over the table. A major transformation (cross-product, lazy plan with observed overlay) sits more naturally outside the source class.

4. **Strict scope="observed" — RESOLVED: strict by default.** `cells(scope="observed")` intersects with the universe annotation by default; ids in the data but not in the universe are dropped. Opt into lenient mode with `strict=False` to keep orphans (with null annotation columns). A separate `orphans(table, universe=...)` diagnostic could also be added later for the data-quality use case.

5. **`include_self` default for `possible_pairs` — RESOLVED: False.** Excludes autapses by default. Labs studying autapses pass `include_self=True`.

6. **ConnectivityTable scope — RESOLVED: out of scope here.** `cells()` and `possible_pairs()` accept only `SynapseTable` and `EdgeList`. The analog for label-entity `ConnectivityTable` waits on the label-annotation kind (separate PR); when it lands, `labels(ct)` and `possible_label_pairs(ct)` follow the same template.

## 8. Implementation order

If the design above survives pressure-testing:

1. **Filter side-classification + side-decomposed projection helper** — small, mechanical, lands first. ~80 LOC + tests. The projection helper is needed by both `cells()` and `possible_pairs()`, so it lives in `_base.py` and is shared.
2. **Scope enum + `cells()`** — depends on (1). ~100 LOC + tests, covers analysis #4 and any per-cell statistic.
3. **`PairUniverse` class + `possible_pairs()`** — depends on (1). ~250 LOC + tests, covers the denominator path for analyses #1, #2, #3.
4. **Cross-tier consistency pass** — concurrent with (1–3) since it's mechanical and doesn't touch the new code paths.
5. **Stats free functions** — `connection_density`, `cell_similarity`, etc. — land on the new foundation, one per PR.
6. **Viz free functions** — `plot_somas`, `plot_pair_matrix`, etc. — wired against `cells()` and `PairUniverse` so 1:1 with stats is automatic.

Each step has independent test coverage and doesn't break the existing surface. The four analyses become writable end-to-end after step 5; the visualization side of analysis #4 needs step 6.
