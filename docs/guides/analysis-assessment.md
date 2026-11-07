# Trajantable connectivity analysis assessment

Trajantable has a useful foundation for studying connectivity across synapses,
individual cells, cell types, and space. Keep its three table tiers and explicit
cell universe. Prioritize versioned data sources, valid spatial measurements,
and scalable denominator calculations before adding a large catalog of metrics.
Delta Lake support is the first storage improvement; it does not by itself
remove the expensive steps in the analysis pipeline.

## Current analysis tiers

| Level | Existing support | Main next improvement |
| --- | --- | --- |
| Individual synapses | `SynapseTable`, synapse and vertex annotations, positions, weights, expressions, spatial filters | Validate identifiers, coordinate contracts, source versions, and synapse quality before aggregation. |
| Individual cell pairs | `EdgeList`, synapse counts, summed weights, categorical/bin summaries, cell annotations and aliases | Add reciprocity and connectivity profiles using the lazy aggregation and sparse export foundations. |
| Cell types and other groups | `type_edgelist`, `aggregate_to_type`, `ConnectivityTable`, normalization and dense pivot | Carry taxonomy hierarchy, population counts, zero rows/columns, and denominator provenance through grouping. |
| Candidate pairs | `PairUniverse`, `possible_pairs`, scope-aware cell selection, binned counts, density/probability, cell bootstrap | Avoid evaluating the full Cartesian product for common type and spatial denominators. |

The distinction between observed edges and candidate pairs is a major strength.
Registered universe membership preserves cells with no observed connections;
grouped counts expose both numerator and denominator. Cell resampling is
available separately from binomial intervals. These are good foundations for
analysis across cell type and distance.

The current `ROADMAP.md` understates this capability: connection probability
and density already exist in `stats.py`. Reciprocity and connectivity similarity
remain proposed features. Update the roadmap around current code rather than
implementing the old probability proposal again.

## Delta Lake as the synapse source

`SynapseTable.from_delta()` opens a commit-pinned source through Polars.
`write_synapses_delta()` exports base records, and the optional `delta` extra
provides the storage dependencies. Constructor counts are deferred for lazy
frames, so opening a disk-backed source does not read its rows. The
[persistence guide](persistence.md) covers migration and snapshot semantics.

Delta adds a transaction log and historical snapshots. Polars still handles
joins, filters, projections, and aggregation. A table's version is resolved
once at opening, including when requesting the latest version. This gives
each analysis a stable synapse source while writers make newer commits.
[Polars scan_delta](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_delta.html)
provides the lazy scan;
[delta-rs](https://delta-io.github.io/delta-rs/latest/usage/writing/)
provides writes and batch ingestion without Spark.

The first implementation is deliberately limited: export collects base records
in memory; append does not enforce synapse-ID uniqueness; Delta stores no trajan
annotation or analysis configuration. DataFolio continues to save a complete
analysis by copying records into its own snapshot. It preserves the Delta
source URI/version as origin provenance. It is not yet a reference-only save.

Next, define a versioned analysis manifest containing dataset identity, CAVE
materialization or segmentation version, synapse Delta URI/version, annotation
versions, taxonomy release, coordinate frame, units, transformations, and
analysis configuration. Independent Delta tables do not share a transaction;
the manifest must pin a compatible set of commits. Pass credentials at runtime.
Carry this manifest through every tier transition and into result exports;
metadata now propagates to EdgeList, ConnectivityTable, and PairUniverse,
including save/load for the persistent table tiers.

Treat source ingestion separately from analysis. Use a stable synapse key,
validate duplicate/null keys and endpoints, and decide explicitly whether a
batch is append-only or updates existing synapses. A segmentation root ID needs
a dataset and segmentation version to be interpretable across snapshots.
Split/merge correspondence requires a dedicated mapping contract beyond a
one-to-one alias. Use strict integer conversion for Delta IDs and retain
high-cardinality identifiers as columns rather than partition keys.

## Spatial validity and observable populations

Declare units and coordinate frames on every position role. Include voxel
resolution, transform identity, depth convention, and validity mask. Require
compatible frames for subtraction; a column holding three numbers cannot
establish that positions are comparable.

Spatial helpers now reject incomplete or non-finite positions consistently.
The former partial-coordinate bug, where `(null, 0, 0)` and `(10, 0, 0)` could
produce zero distance, is fixed. Distances/depth/features return NaN for invalid
positions; bounding-box predicates exclude them, and transforms produce null
positions for invalid input or output. Report missing-position counts alongside
spatial statistics. Units and frames remain caller-declared provenance rather
than an enforced coordinate type system.

Keep three concepts separate in the API: synapse location, cell location, and
pair eligibility. A synapse bounding box restricts observed contacts. A soma
bounding box selects a cell population. A soma-distance cutoff selects
candidate pairs. None automatically establishes that an unconnected pair was
fully observable inside the reconstruction volume. Add explicit eligibility
for completeness, proofreading status, region, and the measurement window.
Counts should distinguish eligible zeros from unavailable observations.

Cell predicates involving both endpoints now constrain candidate pairs as well
as observations. `filter_pairs()` provides explicit eligibility based on cell
IDs/properties; source ID and soma-distance helpers use it automatically.
Eligibility survives synapse-to-edge and candidate-to-edge transitions. Synapse
measurements still select observed contacts only. Joint pair eligibility does
not redefine the cell universe or the bootstrap population.

## Cell types and connectivity profiles

Existing annotations can hold subclass, class, fine type, region, and quality
columns; retain that flexibility. Add taxonomy identity/version, explicit
parent relationships, and assignment status so analyses can roll up labels
without confusing classification releases. Keep unknown assignments visible.
Store confidence or alternative assignments separately from hard labels;
weighted analyses need their own numerator and denominator contract.

For every type pair, expose eligible pre/post cell counts, connected pair
counts, possible pair counts, total synapses, synapses per connected pair, and
per-cell input/output distributions. Preserve individual-cell results to
show heterogeneity hidden by type averages. Normalization should record whether
its denominator is the selected partner set or externally measured total
input/output; the existing `normalize(total_col=...)` distinction is useful.

Sparse export now provides explicit row/column identities and optional
zero-connection populations. Next add cosine/correlation/Jaccard similarity on bounded cohorts, nearest
neighbors, and stability across type resolution and spatial windows. Validate
type discrimination on held-out cells or another dataset; connectivity and
spatial covariates can otherwise create circular evidence for a taxonomy.

## Scale and priority order

Synapse-to-edge and type aggregation now build lazy plans.
`ConnectivityTable.to_dense()` guards against oversized allocation, while
`to_sparse()` returns a CSR array with explicit axis identities and optional
zero-connection populations. Metadata is deep-copied across tiers and persisted.
`possible_pairs()` constructs a Cartesian product, and the cell
bootstrap collects a compact representation of those pairs. Lazy execution
reduces unnecessary I/O, but does not guarantee bounded memory or subquadratic
work. A 60,000-cell directed universe has 3,599,940,000 non-self pairs; even
13 bytes per pair is about 46.8 GB before temporary buffers.

| Priority | Deliverable | Acceptance criterion |
| --- | --- | --- |
| Immediate | Delta snapshots and lazy opening, implemented in this change | No row collection at opening; old readers remain on the same commit after append/overwrite; identifiers retain exact values. |
| Implemented | Coordinate validity and pair eligibility | Partial/null/non-finite positions cannot produce valid zero distances; numerator and denominator share an explicit eligible-pair mask. |
| Implemented | Lazy pair aggregation and guarded sparse export | Filter/project/aggregate without eagerly collecting edges; sparse axes include eligible zero-connection cells; oversized dense requests fail before allocation. |
| Next | Scalable denominators | Compute type-only possible counts from population products with overlap/self-pair corrections; use blocks or a spatial candidate index for distance cutoffs; agree with Cartesian reference results on small fixtures. |
| Then | Versioned analysis manifest and taxonomy | All tiers/results retain compatible source, annotation, taxonomy, frame, and population definitions; reproduce a saved result after upstream changes. |
| Then | Reciprocity, profiles, and spatial inference | Validate on hand-computable networks; resample cells or spatial blocks as appropriate; null models preserve declared type, degree, and spatial constraints. |

Benchmark selective Delta reads, complete aggregation, type denominators,
distance bins, and bootstrap separately. Record row counts, file layout,
execution time, and peak memory. Compare unpartitioned data against a small
number of region or spatial partitions using real query patterns before
choosing a layout. Delta alone should not be credited with a performance gain
without those measurements.
