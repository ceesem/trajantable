# Filtering

All filter methods return a **new** `SynapseTable`; the original is unchanged.
Filters accumulate as Polars expressions in the lazy plan and are applied after
all annotation joins, so any column in `.df` is valid.

## Arbitrary Polars expressions

`filter(expr)` accepts any [Polars expression](https://docs.pola.rs/user-guide/expressions/):

```python
import polars as pl

# Keep synapses between specific cell types
st_ei = st.filter(
    (pl.col("cell_type_pre") == "excitatory")
    & (pl.col("cell_type_post") == "inhibitory")
)

# Keep synapses from a specific set of pre-synaptic cells
st_sub = st.filter(pl.col("pre_pt_root_id").is_in(my_cell_ids))

# Filter on synapse size
st_large = st.filter(pl.col("size") >= 100)
```

Filters can be chained — each call appends to the plan:

```python
st_filtered = (
    st.filter(pl.col("cell_type_pre") == "excitatory")
    .filter(pl.col("size") >= 50)
)
```

## Drop unannotated partners

`filter_to_annotated` keeps only synapses where **both** the pre and post cell
have a non-null value for a given annotation:

```python
# Drop synapses involving cells with no cell type annotation
st_typed = st.filter_to_annotated("cell_type")
```

This is useful after adding a sparse annotation that only covers proofread cells.

## Filter by cell IDs

`filter_by_ids` keeps only synapses where the pre and/or post root ID appears in
a given set. It operates directly on the root ID columns, before any annotation
joins are applied:

```python
# Keep only synapses from specific pre cells
st_sub = st.filter_by_ids(pre_ids=[111, 222, 333])

# Keep synapses between two populations
st_ei = st.filter_by_ids(pre_ids=excitatory_ids, post_ids=inhibitory_ids)
```

Omitting either argument leaves that side unconstrained. This is a more
convenient alternative to `filter(pl.col("pre_pt_root_id").is_in(...))` when
you only need to restrict by identity.

## Bounding box

Requires `synapse_position_col` to be set on the `SynapseTable`:

```python
# Keep synapses within a spatial bounding box (same units as your position data)
bbox = ((10_000, 20_000, 5_000), (50_000, 60_000, 30_000))  # (min_xyz, max_xyz)
st_local = st.filter_by_bbox(bbox)
```

## Soma-soma distance

Declare which cell annotation carries soma positions by passing
`position_col=` at registration. Spatial filters auto-resolve to the unique
position-bearing annotation; pass `annotation=<name>` to disambiguate when
more than one is registered.

```python
st = trajan.SynapseTable(pl.scan_parquet("synapses.parquet"))
soma = pl.read_parquet("soma_positions.parquet")  # root_id, pt_position (struct)
st = st.add_cell_annotation(
    "soma",
    soma,
    cell_id_col="root_id",
    position_col="pt_position",   # declare the position role on this annotation
)

# Keep only local connections (soma < 100 µm apart)
st_local = st.filter_by_soma_distance(100_000)  # units match your position data
```

By default uses Euclidean (3-D) distance. Pass `distance_fn=radial_distance` to
ignore the z axis:

```python
from trajan import radial_distance
st_local = st.filter_by_soma_distance(100_000, distance_fn=radial_distance)
```
