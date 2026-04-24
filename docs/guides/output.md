# Connectivity representations

## Edgelist

`edgelist()` aggregates synapses into one row per cell pair with a synapse count.
Cell annotation columns (the `*_pre` / `*_post` pairs from registered cell
annotations) are included automatically via `.first()`:

```python
el = st.edgelist()
# pre_pt_root_id | post_pt_root_id | n_syn | cell_type_pre | cell_type_post
```

Set `pre_anno=False` or `post_anno=False` to omit annotation columns on either
side — useful when only synapse counts are needed and you want to avoid the
overhead of carrying annotation data through the aggregation:

```python
el = st.edgelist(pre_anno=False, post_anno=False)
# pre_pt_root_id | post_pt_root_id | n_syn
```

Additional per-pair aggregations over any column in `.synapses`:

```python
el = st.edgelist(agg={
    "mean_size": pl.mean("size"),
    "total_area": pl.sum("area"),
})
# pre_pt_root_id | post_pt_root_id | n_syn | cell_type_pre | cell_type_post | mean_size | total_area
```

## Connectivity matrix

`matrix()` pivots the edgelist into a pre × post matrix:

```python
mat = st.matrix()          # synapse counts, all cells
mat = st.matrix(values="mean_size")   # any other edgelist column
```

Constrain or pad to a fixed cell set with `pre_ids` / `post_ids`:

```python
mat = st.matrix(
    pre_ids=excitatory_ids,
    post_ids=inhibitory_ids,
    fill_value=0,
)
```

Restrict to annotated cells only via `filter_annotated`:

```python
# Drop rows/columns for unannotated cells
mat = st.matrix(filter_annotated="cell_type")

# Per-side control
mat = st.matrix(filter_annotated={"pre": "cell_type", "post": "proofread"})
```

## Normalized connectivity

`normalized()` computes fractional output (by pre) or fractional input (by post):

```python
# Each pre cell's output weight distributed across post cells (tidy)
frac = st.normalized(by="pre")
# pre_pt_root_id | post_pt_root_id | fraction

# Each post cell's input weight distributed across pre cells
frac = st.normalized(by="post")
```

Collapse the "other" side by a cell type annotation before normalizing:

```python
# Fraction of each pre cell's output going to each post cell *type*
frac = st.normalized(by="pre", group_col="cell_type")
# pre_pt_root_id | cell_type_post | fraction
```

Pivot into a matrix directly:

```python
mat = st.normalized(by="pre", group_col="cell_type", pivot=True)
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
# Now available in .synapses and in edgelist agg
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

`trajan.to_dataframe(st)` materializes `.synapses` as a pandas DataFrame. Like
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
df = trajan.unpack_all_positions(st.synapses)
```
