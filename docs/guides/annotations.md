# Annotations

Annotations are additional DataFrames joined onto the synapse table.
There are three kinds, each joined differently.

## Synapse-level annotations

Joined on the synapse ID column (default `id`).
Use these for per-synapse attributes like confidence scores or cleft size.

```python
scores = pl.read_parquet("synapse_scores.parquet")  # id, confidence, cleft_size
st = st.add_synapse_annotation("scores", scores)

# Columns appear directly on .df
st.df[["id", "confidence", "cleft_size"]]
```

The source DataFrame must have one row per synapse ID.

## Cell-level annotations

Joined on a cell ID column, **symmetrically** for both pre- and post-synaptic cells.
Each data column in your annotation produces two columns in `.df`:
`{col}_pre` and `{col}_post`.

```python
cell_types = pl.read_parquet("cell_types.parquet")  # root_id, cell_type, layer
st = st.add_cell_annotation("cell_type", cell_types, cell_id_col="root_id")

# Produces cell_type_pre, cell_type_post, layer_pre, layer_post
st.df[["pre_pt_root_id", "cell_type_pre", "cell_type_post"]]
```

The source DataFrame must have one row per cell ID.

### Extending an existing annotation

Use `extend_cell_annotation` to join additional columns into an already-registered
annotation without replacing it:

```python
soma_positions = pl.read_parquet("soma_positions.parquet")  # root_id, pt_position
st = st.add_cell_annotation("soma", soma_positions, cell_id_col="root_id")

extra = pl.read_parquet("soma_extra.parquet")  # root_id, volume
st = st.extend_cell_annotation("soma", extra, on="root_id")
```

### Declaring annotation roles

Cell annotations can carry **role declarations** that let library operations
locate them without explicit per-call arguments:

- `position_col=<col>` — names a data column carrying the cell's position as
  a struct with `x` / `y` / `z` fields. Picked up by the distance filters
  (`filter_by_radial_distance` / `filter_by_euclidean_distance`) and
  `add_spatial_features`.
- `is_universe=True` — marks this annotation's cell-id set as the
  authoritative cell universe (the set of cells that *exist*, including those
  with zero observed connections). Picked up by denominator-bearing
  statistics and null-model shufflers.

```python
soma = pl.read_parquet("soma_positions.parquet")  # root_id, pt_position
st = st.add_cell_annotation(
    "soma",
    soma,
    cell_id_col="root_id",
    position_col="pt_position",   # spatial filters auto-locate this
    is_universe=True,             # this set defines "the cells we know about"
)
```

Resolution rules: with exactly one annotation declaring a given role, it's
auto-selected. With multiple, pass `annotation=<name>` to disambiguate. With
zero, the operation raises with a clear error.

!!! note
    A future `Universe` class will likely supersede the `is_universe` flag.
    The flag is the migration anchor — code that uses it today will get a
    clean upgrade path.

## Vertex-level annotations

Joined on a vertex (supervoxel) ID column for pre, post, or both sides.
Useful for supervoxel-level attributes like layer assignments.

```python
vertex_ann = pl.read_parquet("vertex_layers.parquet")  # sv_id, layer
st = st.add_vertex_annotation(
    "layer",
    vertex_ann,
    vertex_id_col="sv_id",
    pre_vertex_col="pre_pt_supervoxel_id",
    post_vertex_col="post_pt_supervoxel_id",
)
# Produces layer_pre, layer_post
```

Supply only `pre_vertex_col` or only `post_vertex_col` to annotate a single side.

## Removing annotations

All annotation types can be removed by name:

```python
st = st.remove_cell_annotation("cell_type")
st = st.remove_synapse_annotation("scores")
st = st.remove_vertex_annotation("layer")
```

!!! note
    Removing a cell annotation that is the source of a cell alias will
    also remove that alias and warn you. See [Cell ID aliasing](cell-aliases.md).

## Viewing a subset

`view()` returns a new `SynapseTable` containing only the named annotations:

```python
subset = st.view(cell_annotations=["cell_type"])
```

Pass `[]` to drop all annotations of a given type; `None` (default) keeps all.
