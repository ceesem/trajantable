# TrajanTable

TrajanTable is a Python library for working with connectome synapse tables.
It provides a lazy, composable pipeline for joining cell and synapse annotations,
filtering, and producing edgelists, connectivity matrices, and normalized outputs —
all built on [Polars](https://pola.rs) lazy frames.

## Install

```bash
pip install trajan
```

## Quickstart

```python
import polars as pl
import trajan

# Load synapses (any Polars-readable source works)
st = trajan.SynapseTable(
    pl.scan_parquet("synapses.parquet"),
    synapse_position_col="ctr_pt_position",
)

# Add a cell-type annotation keyed on root ID
cell_types = pl.read_parquet("cell_types.parquet")  # root_id, cell_type
st = st.add_cell_annotation("cell_type", cell_types, cell_id_col="root_id")

# Filter to excitatory→inhibitory pairs, get synapse counts
el = (
    st.filter(pl.col("cell_type_pre") == "excitatory")
    .filter(pl.col("cell_type_post") == "inhibitory")
    .edgelist()
)
```

## Key concepts

**SynapseTable** is the central class. It wraps a synapse list and owns the
annotation, filtering, and output pipeline. See the
[SynapseTable guide](guides/synapse-table.md).

**Annotations** are joined onto the synapse table lazily.
Cell annotations produce symmetric `_pre` / `_post` column pairs automatically.
See the [Annotations guide](guides/annotations.md).

**Cell ID aliases** let you chain annotations through a stable secondary ID
(e.g. a `cell_id` that survives root ID updates).
See the [Cell ID aliasing guide](guides/cell-aliases.md).

**Filters** accumulate as Polars expressions pushed into the lazy plan.
See the [Filtering guide](guides/filtering.md).

**Output methods** on `SynapseTable` (`edgelist`, `type_edgelist`, `matrix`,
`normalized`) materialize the lazy plan and reshape the result. Graph and
DataFrame export are free functions (`trajan.to_graph`, `trajan.to_dataframe`)
rather than methods, because they produce derived artifacts rather than
chaining state. Graph export supports NetworkX, igraph, and scipy sparse
backends. See the [Connectivity representations guide](guides/output.md).

**Persistence** via DataFolio round-trips the full table including all annotations,
filters, and expressions. See the [Persistence guide](guides/persistence.md).
