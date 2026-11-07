# Connectivity representations

## Edgelist

`edgelist()` aggregates synapses into one row per cell pair with a synapse
count. Registered cell annotations — including their role declarations
(`position_col`, `is_universe`) and any cell aliases — are *propagated* onto
the returned `EdgeList` rather than inlined into the pair frame. Annotation
columns (`*_pre` / `*_post`) appear on `el.df` via a symmetric pre/post join
performed at access time:

```python
el = st.edgelist()
el.df  # pre_pt_root_id | post_pt_root_id | n_syn | cell_type_pre | cell_type_post | ...
```

`edgelist()`, `type_edgelist()`, and `aggregate_to_type()` build lazy plans.
Filter or project their results before calling `.collect()`, `.pairs`, or `.df`.
Categorical `count_by`/`fraction_by` summaries still scan upfront to discover
their output fields. Grouped row order is unspecified; sort explicitly when
order matters. `repr()` does not scan lazy pair tables to count them.

By default, `edgelist()` excludes synapses with a null pre **or** post cell ID
before aggregation. These rows remain in `st`; null annotations or measurements
on an otherwise complete cell pair are retained. To include unresolved endpoints
as groups explicitly, use `st.edgelist(drop_nulls=False)`.

`type_edgelist()` follows the same `drop_nulls=True` default, checking the
declared `pre_col` and `post_col` grouping keys, including type labels. Thus
missing target types do not become a null column in the resulting allocation
matrix. Use `st.type_edgelist(..., drop_nulls=False)` to retain those groups.
Only grouping keys are checked; null measurements and non-axis annotations
remain in the aggregation.

To drop an annotation's columns from `el.df`, remove the annotation:

```python
el.remove_annotation("types")
el.df  # pre_pt_root_id | post_pt_root_id | n_syn
```

Additional per-pair aggregations over any column in `.df`:

```python
el = st.edgelist(agg={
    "mean_size": pl.mean("size"),
    "total_area": pl.sum("area"),
})
# pre_pt_root_id | post_pt_root_id | n_syn | cell_type_pre | cell_type_post | mean_size | total_area
```

## EdgeList

`SynapseTable.edgelist()` returns an `EdgeList` — a pair-level table with
one row per `(pre, post)` cell pair. Work in pair-space continues through
the EdgeList (further filtering, aggregation to types, matrix export)
rather than dropping back to a raw DataFrame. Access the materialized pair
DataFrame via `el.df` when needed.

```python
el = st.edgelist()
el.df  # pl.DataFrame with [pre, post, n_syn, cell_type_pre, ...]
```

Pair-level filtering returns an `EdgeList`:

```python
strong = el.filter(pl.col("n_syn") >= 3)
```

Cell-id restriction:

```python
el.filter_by_ids(pre_ids=[111, 222], post_ids=[333, 444])
```

Spatial filters — declare the position column on the cell annotation that
carries it (`add_annotation(..., position_col="soma_pt_position")`); the
filter auto-resolves to it. Pass `annotation=<name>` to disambiguate if more
than one position-bearing annotation is registered:

```python
near = el.filter_by_radial_distance(100_000)  # lateral; same units as your position data
# or el.filter_by_euclidean_distance(100_000) for full 3-D (depth included)
```

## Connectivity matrix

Pivot an EdgeList into a dense pre × post matrix via `to_dense`:

```python
mat = st.edgelist().to_dense()                       # n_syn counts
mat = st.edgelist().to_dense(values="size")          # any weight column
mat = st.edgelist().to_dense(fill_value=-1)          # non-existing pairs
```

Retain row properties with `keep`:

```python
ct = st.type_edgelist(
    st.pre_col, "class_post", agg={"mean_size": pl.mean("size")}
)
features = ct.to_dense(
    values="mean_size", fill_value=float("nan"),
    keep=["class_pre", "soma_depth_um_pre"],
)
```

Each kept column must have one distinct value per pre entity; conflicting
values, including a mixture of null and non-null, raise an error. Kept
properties appear after the pre ID and before pivot columns, and retain their
dtypes and nulls. `fill_value` applies only to pivot measurements. Properties
must be available on the current table; `type_edgelist` now carries root-keyed
cell annotations on any axis that retains its original cell ID column, as
`EdgeList.aggregate_to_type` already does. Derived expressions and alias-keyed
annotations are not automatically transferred by `type_edgelist`.

If the post axis is a Polars Enum, pivot columns follow its declared category
order. Only observed categories appear; a null category, if present, comes
last. Other post dtypes retain the pivot's encounter order.

`ConnectivityTable` (returned by `type_edgelist` or by promoting an
EdgeList) is the general matrix tier — it supports the same `to_dense`,
plus shape-preserving operations like `normalize`, `binarize`, `log1p`.

## Normalized connectivity

### Population-adjusted allocation

`SynapseTable.cell_agg(column, agg=None)` groups the decorated cell population.
The default returns `n_cells`, counting rows including null-label groups.
Its default `scope="universe"` includes cells with no observed connections;
`scope="filtered"` and `scope="observed"` follow the same semantics as
`trajan.cells`. Filtered scope is the union of pre- and post-eligible cells,
not a post-only population. Supply `universe=` when multiple universe
annotations are registered. Custom Polars expressions, lists of expressions,
or dictionaries of named expressions can replace the default aggregation.

```python
allocation = (
    st.filter(pl.col("class_post") == "exc")
    .type_edgelist(st.pre_col, "subclass_post")
    .normalize(by="pre", values="n_syn", reweight_by=st.cell_agg("subclass"))
    .to_dense("fraction")
)

lengths = st.cell_agg(
    "subclass", agg={"total_length": pl.col("dendritic_length").sum()}
)
```

`reweight_by` divides each value by its opportunity denominator, then normalizes
the adjusted rates within each selected axis group. A dictionary matches post
categories for `by="pre"`, or pre categories for `by="post"`. Values must be
numeric, finite, and positive. Every observed category needs a key; extra keys
are allowed. Mapping coverage is checked with a narrow category scan when
`normalize` is called. The remaining normalization plan stays lazy.

The two-column DataFrame returned by `cell_agg` can be passed directly:
the first column supplies categories and the second supplies denominators.
Keys must be unique, and coverage and denominator validation follow the same
rules as a dictionary. For multi-column grouping or multiple custom measures,
select the appropriate two columns explicitly before passing the frame.

Alternatively, pass a numeric column name such as `reweight_by="n_cells_post"`
after attaching the population as a post-side annotation. Missing, nonfinite,
or nonpositive column denominators produce null fractions and are excluded
from the total. Groups without a positive finite adjusted total return null.
`reweight_by` cannot be combined with `total_col`.

Use `values="size"` for total-size allocation when size is registered as a
synapse weight. Cell abundance adjustment does not itself account for spatial
opportunity, dendritic exposure, or reconstruction completeness. Universe
counts ignore downstream synapse filters, so define the eligible target
population explicitly when registering the universe annotation. Type groups
without observed synapses contribute zero to the allocation totals, but are
not automatically added as rows or matrix columns.

### Enrichment relative to population abundance

Use `mode="enrichment"` when one should mean allocation exactly proportional
to the supplied opportunity weights:

```python
exc_population = st.cell_agg("subclass").filter(
    pl.col("subclass").is_in(EXC_SUBCLASS)
)

enrichment = (
    st.filter(pl.col("subclass_post").is_in(EXC_SUBCLASS))
    .type_edgelist(st.pre_col, "subclass_post")
    .normalize(
        by="pre", values="n_syn",
        reweight_by=exc_population, mode="enrichment",
    )
    .to_dense("enrichment")
)
```

For each neuron and target category, enrichment is
`(category_synapses / total_synapses) / (category_population / total_population)`.
One matches population abundance, two is twice the expected allocation, and
0.5 is half. Values do not sum to one. Use `values="size"` for total-size
enrichment when size is registered as an additive synapse weight.

Unlike fraction mode, **all supplied population categories contribute to the
expected population total**, including categories with no observed synapses.
Filter the supplied population to the intended target set explicitly: for an
excitatory subclass analysis, exclude inhibitory and other out-of-scope groups.
The observed total follows the current connectivity-table filters.

Enrichment requires a dictionary or two-column population DataFrame, since a
column attached only to observed connections cannot establish the full eligible
population. `total_col` cannot be combined with enrichment. A zero observed
total returns null enrichment. Explicit zero-weight connections yield zero
when their neuron's total is positive. Missing categories are not automatically
inserted into the output; zero-filled dense columns apply to absent pairs only
when that category is present elsewhere in the observed table.

`ConnectivityTable.normalize(by="pre" | "post")` produces a new
ConnectivityTable with a `fraction` column replacing the weight. Two
modes:

```python
# Internal mode — divide by the current table's axis sum.
# Semantics are dynamic: if the table has been filtered, fractions are
# relative to currently-visible totals.
frac = st.edgelist().normalize(by="post")

# External mode — divide by a user-supplied column.
# Trajan does not interpret what the column means, only divides by it.
# Useful when the denominator (e.g. the cell's true total input, from
# an annotation) should not change with filtering.
frac = st.edgelist().normalize(by="post", total_col="n_syn_input_post")
```

Collapse one axis to a cell-type annotation column via
`EdgeList.aggregate_to_type` before normalizing:

```python
# Fraction of each pre cell's output going to each post cell *type*
ct = st.edgelist().aggregate_to_type(post="cell_type_post")
frac = ct.normalize(by="pre")  # ConnectivityTable with fraction column
```

Custom edge columns need an explicit reduction when collapsing to types:

```python
el = st.filter(pl.col("is_proofread_pre")).edgelist(
    agg={"mean_size": pl.mean("size")}
)
ct = el.aggregate_to_type(
    post="class_post", agg={"mean_size": pl.mean("mean_size")}
)
```

This computes the mean of the **connection means**, giving each cell pair equal
weight. For a mean over individual synapses, preserve sufficient statistics:

```python
el = st.filter(pl.col("is_proofread_pre")).edgelist(
    agg={"size_sum": pl.sum("size"), "size_count": pl.col("size").count()}
)
ct = el.aggregate_to_type(
    post="class_post",
    agg={"mean_size": pl.sum("size_sum") / pl.sum("size_count")},
)
```

`size_count` excludes null sizes, matching Polars' mean. Custom `agg` outputs
remain data columns; only registered weights are automatically summed.

Pivot to a dense matrix at any step:

```python
frac.to_dense(values="fraction")
```

Dense export refuses matrices larger than 10 million entries before collecting
the full pair table or pivoting. Increase `max_cells` explicitly, or use
`max_cells=None` to disable the guard. The limit counts entries, not bytes.

For large matrices, install `trajan[sparse]` and use:

```python
matrix, pre_entities, post_entities = el.to_sparse()
```

The returned CSR array's rows and columns follow the returned Polars Series.
To include zero-connection cells, pass ordered `pre_ids=` and `post_ids=` from
your eligible cell population. Those axes must contain all observed identities.
No dense grid is constructed. Duplicate pairs raise unless an explicit
`aggregate=` reducer is supplied; null IDs or values require explicit filtering
or filling before sparse export.

## Per-cell summary

`trajan.cell_summary(st)` aggregates synapse-level data into one row per
cell, combining output and input statistics. It's a free function because
it's a derived statistic, not a chaining operation.

```python
import trajan

cs = trajan.cell_summary(st)
# cell_id | n_syn_output | n_syn_input | {weight}_output | {weight}_input | ...

# custom per-direction aggregations
cs = trajan.cell_summary(
    st,
    pre_agg={"mean_size_out": pl.mean("size")},
    post_agg={"mean_size_in": pl.mean("size")},
)

# skip annotation columns
cs = trajan.cell_summary(st, include_annotations=False)
```

## Selective view

`view()` returns a lightweight `SynapseTable` containing only the named
annotations. This is useful for building a focused output without the overhead
of unneeded joins:

```python
# Only cell_type annotation — faster edgelist build
el = st.view(cell_annotations=["cell_type"]).edgelist()

# Drop all annotations, keep filters
bare = st.view(
    synapse_annotations=[],
    cell_annotations=[],
    vertex_annotations=[],
)
```

`None` (the default for each parameter) keeps all registered annotations of
that type.

## Computed columns

`add_expression` registers a named Polars expression applied after joins and
before filters. This is useful for derived per-synapse values that you want
available in output or for filtering:

```python
from trajan import euclidean_distance

st = st.add_expression(
    "soma_dist",
    euclidean_distance("pt_position_pre", "pt_position_post"),
)
# Now available in .df and in edgelist agg
el = st.edgelist(agg={"mean_soma_dist": pl.mean("soma_dist")})
```

## Type edgelist

`type_edgelist(pre_col)` groups synapses by cell-type annotation columns instead
of individual cell IDs, producing one row per type pair with a synapse count.
The post column is inferred automatically by replacing `_pre` with `_post` in the
pre column name:

```python
el = st.type_edgelist("cell_type_pre")
# cell_type_pre | cell_type_post | n_syn
```

Pass `post_col` explicitly for asymmetric grouping (e.g. fine pre type vs. broad
post type):

```python
el = st.type_edgelist("cell_type_pre", post_col="broad_type_post")
# cell_type_pre | broad_type_post | n_syn
```

Additional per-pair aggregations work the same way as in `edgelist`:

```python
el = st.type_edgelist("cell_type_pre", agg={"mean_size": pl.mean("size")})
# cell_type_pre | cell_type_post | n_syn | mean_size
```

## Graph export

`trajan.to_graph(st, ...)` exports the synapse table as a directed graph. It is
a free function (not a method on `SynapseTable`) because it produces a derived
artifact rather than mutating or chaining state. Cell annotation columns
become node attributes with the `_pre` / `_post` suffix stripped:

```python
import trajan

# NetworkX (default) — requires: uv add networkx
G = trajan.to_graph(st)
G.nodes[111]   # {'cell_type': 'L2/3 ET'}
G.edges[111, 222]  # {'n_syn': 14}

# igraph — requires: uv add igraph
g = trajan.to_graph(st, backend="igraph")

# Scipy sparse matrix — requires: uv add scipy
mat, cell_ids = trajan.to_graph(st, backend="csgraph")
# returns (csr_array, array of cell IDs) rather than a graph object
```

Add edge attributes with `edge_agg` and per-cell node attributes with `cell_agg`:

```python
G = trajan.to_graph(
    st,
    edge_agg={"mean_size": pl.mean("size")},
    cell_agg={"n_output": pl.len()},  # computed from the pre side
)
```

## Pandas export

`trajan.to_dataframe(st)` materializes `.df` as a pandas DataFrame. Like
`to_graph`, it is a free function. Polars struct columns (positions) cannot be
represented directly in pandas, so they are automatically unpacked into flat
`_x` / `_y` / `_z` columns:

```python
df = trajan.to_dataframe(st)
# Returns a pandas DataFrame; e.g. ctr_pt_position → ctr_pt_position_x/y/z
```

The same unpacking is available as standalone utilities in the `trajan` namespace
for use on arbitrary Polars DataFrames:

```python
# Unpack a single position struct column
df = trajan.unpack_position(df, "soma_pt_position")

# Unpack all position struct columns at once
df = trajan.unpack_all_positions(st.df)
```
