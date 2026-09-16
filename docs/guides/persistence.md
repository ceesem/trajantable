# Persistence

`SynapseTable`, `EdgeList`, and `ConnectivityTable` can be saved to and loaded
from a [DataFolio](https://github.com/caseysm/datafolio), which stores each
table as a Parquet file and a JSON config alongside it.

trajan requires **DataFolio 2.0 or newer**. Folios written by DataFolio 1.x
still load without a conversion step — DataFolio migrates them to the unified
manifest on the next write.

## Saving

```python
import datafolio
import trajan

folio = datafolio.DataFolio("results/my_analysis")
st.save(folio)
```

Everything is preserved:

- The base synapse table
- All registered synapse, cell, and vertex annotations
- Cell aliases
- Computed column expressions
- Accumulated filters (serialized as Polars JSON expressions)

Use `overwrite=True` to replace an existing folio:

```python
st.save(folio, overwrite=True)
```

A `save()` is a single DataFolio commit: every table lands together or none
does, so an interrupted save never leaves a half-written folio behind. If you
are writing several tables into one folio, wrapping them in your own
`folio.batch()` block works too — `save()` stages into it rather than opening
a competing batch.

```python
with folio.batch():
    st.save(folio)
    el.save(folio)
```

Annotation names become DataFolio item names, so they may contain only
letters, digits, `.`, `_`, and `-`, and must start with a letter or digit. An
annotation named `"cell type"` is fine in memory but is rejected at `save()`.

## Loading

```python
st2 = trajan.SynapseTable.load(folio)
```

The loaded table is identical to the saved one: all annotations, aliases,
expressions, and filters are restored and the lazy plan is reconstructed
from the saved Parquet files — lazily, via `folio.scan_table()`, so opening a
folio does not read the Parquet data. The whole load reads one pinned,
consistent view of the folio even if another process commits to it midway.

`ConnectivityTable.load()` dispatches on the saved type, so it returns an
`EdgeList` when the folio holds one:

```python
ct = trajan.ConnectivityTable.load(folio)   # -> EdgeList if that is what was saved
el = trajan.EdgeList.load(folio)            # TypeError if the folio is not an EdgeList
```

## Round-trip example

```python
import polars as pl
import trajan
import datafolio

# Build a table
st = trajan.SynapseTable(pl.scan_parquet("synapses.parquet"))
cell_types = pl.read_parquet("cell_types.parquet")
st = st.add_cell_annotation("cell_type", cell_types, cell_id_col="root_id")
st = st.filter(pl.col("cell_type_pre") == "excitatory")

# Save
folio = datafolio.DataFolio("results/excitatory_output")
st.save(folio)

# Load in another session
st2 = trajan.SynapseTable.load(folio)
assert st.df.equals(st2.df)
```
