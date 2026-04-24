# trajantable — Feature Roadmap

Ideas that need more design work before implementation. Rough notes, not commitments.

---

## Connection Probability

`connection_probability(pre_type_col, post_type_col) -> pl.DataFrame`

The single most-reported statistic in cortical connectomics:
P(connection) = n_connected_pairs / n_possible_pairs.

The numerator (connected pairs) comes from the edgelist. The denominator requires
knowing total cell counts per type — including unconnected cells — which means going
back to the cell annotation tables, not just the synapse table. That makes it a method
rather than an expression, and requires thinking about how to expose total cell counts.

Output: `[pre_type, post_type, n_connected, n_possible, p_connection,
mean_syn_per_conn, total_syn]`.

**Open question:** Should the total-cell denominator come from the registered cell
annotations, or should the user pass it in?

---

## Reciprocal Connections

`reciprocal_pairs() -> pl.DataFrame`
`reciprocity(by_type=None) -> float | pl.DataFrame`

Overrepresentation of reciprocal connections (A→B and B→A both exist) is one of the
most-cited findings in cortical connectomics. Self-join on the edgelist with pre/post
swapped, filtered to A < B to avoid duplicates.

`reciprocity()` = fraction of connected pairs that are reciprocal; optionally
stratified by cell type pair.

---

## Connectivity Similarity

`connectivity_similarity(side, method, cell_ids, values) -> pl.DataFrame`

Cosine similarity of connectivity vectors — the foundation of connectivity-based cell
type classification. Build the connectivity matrix then compute pairwise similarity
(cosine / correlation / Jaccard) via scipy/numpy.

**Scaling constraint:** Full pairwise matrix at 140K cells ≈ 75 GB as float32.
`cell_ids` parameter is effectively required; warn/error if >10K cells without opt-in.
Typical use: similarity within a cell type of 500–2000 cells.

---

## Degree Statistics

`degree_stats(by=None) -> pl.DataFrame`

Thin wrapper over `cell_summary()` that computes mean, median, std, min, max of
`n_syn_input` and `n_syn_output`, optionally grouped by a cell type column.

Easy to compose manually but frequently needed; encoding it prevents common mistakes
(not handling cells that appear only on one side, using the wrong nulls strategy).

---

## Table Concatenation

`SynapseTable.concat(tables, labels=None) -> SynapseTable`

Multi-region or multi-condition analysis requires combining synapse tables while
preserving annotations. Requires identical annotation schemas (same names, same
columns) or a clear error.

Optionally adds a `source` label column from `labels`. Annotation tables are unioned
with deduplication on join keys.

**Open question:** What happens when the same cell ID appears in both tables with
different annotation values (e.g., after proofread root ID remapping)?

---

## Visualization

Separate from this roadmap — visualization features will be added to a dedicated
submodule.
