# Persistence

`SynapseTable` can be saved to and loaded from a
[DataFolio](https://github.com/caseysm/datafolio), which stores each table as a
Parquet file and a JSON config alongside it.

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

## Loading

```python
st2 = trajan.SynapseTable.load(folio)
```

The loaded table is identical to the saved one: all annotations, aliases,
expressions, and filters are restored and the lazy plan is reconstructed
from the saved Parquet files.

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
