# Persistence

## Delta Lake synapse storage

The base synapse records can live in a Delta Lake table. Install the optional
dependencies with `uv sync --extra delta` (or `pip install 'trajan[delta]'`).
Polars builds lazy queries over a pinned Delta snapshot:

```python
import trajan

st = trajan.SynapseTable.from_delta(
    "data/synapses",
    version=12,
    synapse_position_col="ctr_pt_position",
)
print(st.metadata["synapse_source"])  # format, absolute URI, resolved version
```

Omit `version` to resolve the latest commit once when opening. Later appends or
overwrites do not change an existing instance or its filtered views. Opening
another instance sees the newer commit. Opening reads the Delta log and file
metadata, but does not collect synapse rows or count them. `len(st)` performs a
count on demand; `repr(st)` shows `uncached` until a count or full collection
is available. Snapshot files must remain available: vacuuming old Delta files
can make historical versions unreadable.

Cloud URIs work through the same API:

```python
st = trajan.SynapseTable.from_delta(
    "s3://my-bucket/synapses",
    version=12,
    storage_options={"AWS_REGION": "us-west-2"},
)
```

Storage options are passed to Delta/Polars and are not copied into analysis
metadata. Keep credentials out of the URI itself.

To migrate an existing in-memory synapse frame, export its **base rows**, then
open the resulting table:

```python
st.write_synapses_delta("data/synapses")
st_delta = trajan.SynapseTable.from_delta("data/synapses")
```

This writer materializes the base frame. Filters, joined annotations, computed
expressions, and analysis metadata are excluded. Existing targets raise by
default. `mode="append"` adds rows without deduplicating IDs; `mode="overwrite"`
replaces rows while keeping Delta history. Schema changes require an explicit
upstream migration. `partition_by=["region"]` can help low-cardinality region
queries; avoid partitioning by cell or synapse IDs.

Delta uses signed integers. The exporter strictly converts top-level unsigned
integer columns to signed types; `UInt64` IDs above `2**63 - 1` raise before
writing. Supported IDs retain their integer values but load as signed columns.
Do not convert IDs to floating point. Unsupported Delta types are rejected by
the underlying writer.

For ingestion larger than RAM, use the
[delta-rs batch writer](https://delta-io.github.io/delta-rs/latest/usage/writing/)
upstream and then `from_delta`. Lazy reads use
[Polars scan_delta](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_delta.html).

## Saving an analysis with DataFolio

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

Every table tier exposes a free-form `metadata` dictionary. Synapse-to-edge,
type aggregation, candidate-pair construction, filtered copies, and matrix
transforms copy metadata deeply, so editing nested provenance on a derived
table does not change its source. Record annotation/taxonomy versions,
coordinate units/frame, and population definitions there. Delta loading
automatically records the synapse source URI and version. Metadata survives
save/load for SynapseTable, EdgeList, and ConnectivityTable; use values that
can be serialized to JSON.

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

For Delta-backed sources, `save()` copies the base rows into the folio's Parquet
snapshot and preserves `metadata["synapse_source"]` as provenance. `load()`
reads that copy; it does not reconnect to the original Delta table. Delta
commits and DataFolio commits are separate transactions. Reference-only folios
and a manifest pinning multiple independently versioned tables are future work.

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
