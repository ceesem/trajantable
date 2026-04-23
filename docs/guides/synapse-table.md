# SynapseTable

`SynapseTable` is the central class in trajan. It wraps a synapse list and owns
the full pipeline: annotation joins, computed columns, filters, and output
materialization. All other guides assume you already have one — this guide
explains how to build and inspect it.

## Construction

The only required argument is the synapse DataFrame. All other parameters are
optional and configure how the table interprets its columns.

**Minimal construction:**

```python
import polars as pl
import trajan

st = trajan.SynapseTable(pl.scan_parquet("synapses.parquet"))
```

**Typical CAVE connectome construction:**

```python
st = trajan.SynapseTable(
    pl.scan_parquet("synapses.parquet"),
    pre_col="pre_pt_root_id",       # default — pre-synaptic cell ID
    post_col="post_pt_root_id",     # default — post-synaptic cell ID
    id_col="id",                    # default — synapse ID, used for synapse annotations
    synapse_position_col="ctr_pt_position",  # required for spatial filtering
)
```

Because the CAVE defaults match trajan's defaults, the first example above is
equivalent for CAVE-sourced tables — the explicit form is shown here for
clarity.

**Pandas input:**

```python
import pandas as pd

df = pd.read_parquet("synapses.parquet")
st = trajan.SynapseTable(df)  # converted to Polars internally
```

## Expected columns

The synapse table must contain at minimum the pre-cell, post-cell, and synapse
ID columns named by `pre_col`, `post_col`, and `id_col`. All other columns
pass through unchanged and are available in `.synapses` and in output
aggregations.

```python
# Required columns only
# pre_pt_root_id | post_pt_root_id | id
```

Any additional columns — synapse size, cleft scores, supervoxel IDs, position
structs — pass through unchanged and require no special handling at construction
time. They appear in `.synapses` automatically and can be used in `filter()`,
`edgelist(agg=...)`, and `add_expression()`. A position column is needed for
spatial filtering; see [Spatial configuration](#spatial-configuration) below.

## The lazy plan

`SynapseTable` is built on Polars lazy frames. No data is loaded or processed
until you explicitly request it. Annotation joins, expressions, and filters all
accumulate as a query plan that is executed only when `.synapses` is accessed or
an output method like `edgelist()` is called.

For large datasets, pass a `pl.LazyFrame` at construction time so that even the
base table is never fully loaded into memory unless needed:

```python
# Recommended for large datasets: scan_parquet creates a lazy frame
st = trajan.SynapseTable(pl.scan_parquet("synapses.parquet"))

# Contrast: read_parquet loads the full table into memory immediately
st = trajan.SynapseTable(pl.read_parquet("synapses.parquet"))
```

When you chain annotations and filters, trajan composes them into a single
Polars lazy plan. Polars can push predicates into the scan, read only the
columns it needs, and parallelize joins — none of this is possible if the data
is already collected into a DataFrame.

## Accessing the merged table

The `.synapses` property builds and collects the full lazy plan and returns a
`pl.DataFrame` with all annotation columns joined in and all filters applied:

```python
df = st.synapses
# Returns a pl.DataFrame: pre_pt_root_id | post_pt_root_id | id | cell_type_pre | cell_type_post | ...
```

The result is cached. Accessing `.synapses` a second time returns the cached
DataFrame immediately without re-executing the plan. The cache is invalidated
automatically whenever you add or remove an annotation, expression, or filter —
the next access re-executes the full plan.

If you only need a subset of columns, avoid materializing the full merged table.
Use `view()` to drop unneeded annotations before accessing `.synapses`, or call
an output method directly:

```python
# Only the cell_type annotation — avoids joining soma positions, morphology, etc.
el = st.view(cell_annotations=["cell_type"]).edgelist()
```

## Spatial configuration

Set `synapse_position_col` when you need `filter_by_bbox` or
`filter_distance_to_point`. The column must hold synapse positions as a Polars
struct with `x`, `y`, `z` fields.

If your source data has separate `_x`, `_y`, `_z` columns instead of a struct,
trajan packs them automatically when the column name is given:

```python
# Source table has ctr_pt_position_x, ctr_pt_position_y, ctr_pt_position_z
st = trajan.SynapseTable(
    pl.scan_parquet("synapses.parquet"),
    synapse_position_col="ctr_pt_position",  # trajan packs x/y/z → struct automatically
)

# Source table already has a struct column named ctr_pt_position
st = trajan.SynapseTable(
    pl.scan_parquet("synapses.parquet"),
    synapse_position_col="ctr_pt_position",  # used directly, no packing needed
)
```

`soma_position_annotation` and `soma_position_col` are only needed for
`filter_by_soma_distance`. They name the registered cell annotation that holds
soma positions and the position column within it. See the
[Filtering guide](filtering.md) for a complete example.

## Updating configuration

If you forget to set `synapse_position_col` at construction time, use
`update_metadata()` rather than rebuilding the table from scratch. It updates
the position configuration without invalidating the annotation cache:

```python
st = trajan.SynapseTable(pl.scan_parquet("synapses.parquet"))
# ... add annotations, register aliases ...

# Later, realize you need spatial filtering:
st = st.update_metadata(synapse_position_col="ctr_pt_position")
```

`update_metadata` accepts `synapse_position_col`, `soma_position_annotation`,
and `soma_position_col` as keyword arguments. Any argument not passed is left
unchanged.

## Inspecting the table

The `repr` summarizes the table at a glance:

```python
st
# SynapseTable(n_syn=1_234_567, synapse_annotations=[], cell_annotations=['cell_type', 'soma'], vertex_annotations=[], expressions=['soma_dist'])
```

`n_syn` shows the base synapse count before filters when the cache is cold.
Once `.synapses` has been accessed it shows the filtered count. If filters are
registered but the cache is cold it shows `uncached`.

The annotation name list properties let you check what is registered without
materializing the table:

```python
st.synapse_annotation_names   # ['scores']
st.cell_annotation_names      # ['cell_type', 'soma']
st.vertex_annotation_names    # ['layer']
st.expression_names           # ['soma_dist']
st.cell_aliases               # {'proofread': ('proofread', 'cell_id')}
```

These are plain Python lists and dicts — safe to inspect at any time with no
query execution cost.
