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
near = el.filter_by_soma_distance(100_000)  # same units as your position data
```

## Connectivity matrix

Pivot an EdgeList into a dense pre × post matrix via `to_dense`:

```python
mat = st.edgelist().to_dense()                       # n_syn counts
mat = st.edgelist().to_dense(values="size")          # any weight column
mat = st.edgelist().to_dense(fill_value=-1)          # non-existing pairs
```

`ConnectivityTable` (returned by `type_edgelist` or by promoting an
EdgeList) is the general matrix tier — it supports the same `to_dense`,
plus shape-preserving operations like `normalize`, `binarize`, `log1p`.

## Normalized connectivity

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

Pivot to a dense matrix at any step:

```python
frac.to_dense(values="fraction")
```

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
