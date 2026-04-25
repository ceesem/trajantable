# trajan refactoring plan

Goal: migrate trajan from a single-class (`SynapseTable`) design to the three-tier architecture with unified blessed-columns + keyed-annotations machinery, as worked out in the design discussion captured in `memory/project_*.md`.

Core design references (living in `~/.claude/projects/-Users-caseysm-Work-Code-trajantable/memory/`):

- `project_unified_blessed_columns.md` — blessed identifier types (synapse/cell/vertex/label), value types (position, weight), behavioral contracts, type algebra under aggregation, weight-list pattern, column-semantic principle.
- `project_edgelist_abstraction.md` — EdgeList as pair-space abstraction; first-class entry point from published edgelists.
- `project_connection_table.md` — ConnectivityTable as matrix tier; normalization with internal-axis-sum vs external `total_col`.
- `project_cell_annotation_as_universe.md` — annotations carry properties AND membership.
- `project_architecture_principles.md` — keep classes narrow; free functions for stats/export/shufflers.
- `project_permutation_nulls.md` — `with_synapses` primitive for permutation-based null models.
- `project_consolidation_mission.md` — support variants, don't bake conventions, expose raw counts.
- `project_connection_probability.md` — denominator from annotation membership; binning is pluggable; CIs separable.

---

## Principles governing every phase

1. **Tests pass at every phase boundary.** No "will fix later" states committed to main.
2. **Break the API cleanly.** Three days old, one user — no method-forwarding shims, no deprecation warnings. Update docs and tests in the same commit as the change.
3. **`uv run` everything.** All commands are `uv run pytest …` / `uv run ruff …`.
4. **Semantics over names.** No column-name sniffing. Roles are declared via constructor params; trajan only owns columns it generated or that the user explicitly role-declared.
5. **Memories are the source of truth on intent.** If the plan and a memory disagree, the memory wins — flag the divergence and update the plan.

---

## Phase 0 — preparation (non-breaking)

**Purpose:** unblock every later extraction by promoting the minimum internal surface to public.

**Tasks:**

1. Add public properties on `SynapseTable`:
   - `pre_col`, `post_col`, `id_col` (just return the private fields).
2. Rename `_build_lazy` → `build_lazy` (method, not property). Keep a `_build_lazy` alias for one commit only if internal call sites are many; otherwise rewrite them.
3. Add a read-only accessor for cell-annotation data columns:
   - `cell_annotation_data_cols() -> dict[str, list[str]]` returning a copy. Needed by future free-function extractions (`cell_summary`, `to_graph`).
4. Add `run: uv run pytest -q` to CI preflight checklist if not already documented somewhere.

**Acceptance:**

- `uv run pytest` green.
- New properties covered by one new unit test each.
- No behavior change.

**Commit:** `prep: promote internal accessors to public surface`

---

## Phase 1 — extract `to_dataframe` and `to_graph` into `export.py`

**Purpose:** practice the extraction pattern on the simplest and largest self-contained methods before touching architectural concerns.

**Tasks:**

1. Create `src/trajan/export.py`.
2. Move `to_dataframe` (~30 LOC) as a free function `to_dataframe(st, *, unpack_positions=True)`.
3. Move `to_graph` (~180 LOC) as `to_graph(st, *, edge_agg=None, cell_agg=None, backend="networkx")`. Keep internal behavior identical.
4. Remove the methods from `SynapseTable`. Break the API — no shims.
5. Re-export from `trajan.__init__` for discoverability: `from .export import to_dataframe, to_graph`.
6. Update existing tests and docs (`docs/guides/output.md`) to use the free-function form.

**Acceptance:**

- `uv run pytest` green.
- All three graph backends (`networkx`, `igraph`, `csgraph`) still tested.
- `docs/guides/output.md` updated; Pandas export section updated.
- `src/trajan/synapse_table.py` shorter by ~210 lines.

**Commit:** `refactor: move to_dataframe and to_graph into trajan.export`

---

## Phase 2 — decide and implement the shared base machinery

**Purpose:** introduce the "table with blessed columns + keyed annotations" pattern as shared code, so SynapseTable, EdgeList, and ConnectivityTable all build on it.

**Design decisions to make before coding (flag for review):**

- **D2.1** — base class vs mixin vs shared helpers?
  - *Recommendation:* an internal abstract base `_TrajanTable` (underscore-prefixed, not part of public API) with concrete shared methods (filter/expression/weight registration, annotation registry, `build_lazy`, cache, view, save/load scaffolding). Subclasses override the join-plan builder specifics and declare which annotation kinds they accept.
- **D2.2** — how are annotations stored? The current `_cell_annotations: dict[name, (lf, cell_id_col, join_on_alias, data_cols)]` tuple is working but awkward. Options: keep tuples, switch to small dataclasses (`CellAnnotationSpec`, `LabelAnnotationSpec`, etc.), or a single `AnnotationSpec` with a `kind` field.
  - *Recommendation:* small dataclasses per kind — cheaper to evolve and carries clearer semantics.
- **D2.3** — where does "blessed column tracking" live? Suggest a single `BlessedColumns` dataclass on every tier instance: holds `{role_name: concrete_col_name}` for role-declared columns (pre, post, id, position, etc.) and the weight list.
- **D2.4** — membership enforcement. Should annotation registration require a full cell-set at register time, or at statistic-call time? Leaning register-time (a registered annotation always has membership); the statistic then only validates it exists.

**Tasks:**

1. Create `src/trajan/_base.py` (underscore — internal) with:
   - `@dataclass BlessedColumns` holding role → concrete-column mappings and the weight list.
   - `@dataclass`es for each annotation kind (`SynapseAnnotationSpec`, `CellAnnotationSpec`, `VertexAnnotationSpec`, `LabelAnnotationSpec`) carrying lazyframe + keys + data_cols.
   - `class _TrajanTable` with shared behavior: annotation add/remove, alias registry, filter accumulation, expression/weight registry, `build_lazy`, cache, `view`, save/load scaffolding. Join-plan methods dispatched to subclass hooks.
2. Refactor `SynapseTable` to subclass `_TrajanTable`. Public API unchanged except for what was already renamed in Phase 0.
3. Tests pass unchanged.

**Acceptance:**

- `uv run pytest` green with no test modifications.
- `src/trajan/synapse_table.py` is substantially shorter (target: <1200 LOC) with shared pieces living in `_base.py`.
- Design doc comment at top of `_base.py` referring to the memory files.

**Commit:** `refactor: extract TrajanTable base with blessed-columns + keyed-annotations`

---

## Phase 3 — implement `ConnectivityTable`

**Purpose:** introduce the Tier-2 abstraction and its normalization API. Stand-alone; doesn't yet replace `SynapseTable.matrix()`.

**Design decisions:**

- **D3.1** — dense vs sparse backend. Initial release: `pl.DataFrame` pivot for dense, `scipy.sparse.csr_array` for sparse. Transparent backend selection by shape/density with a `to_dense()` / `to_sparse()` explicit conversion. Start dense-only if sparse work looks like it blows Day-1 scope; add a TODO in code.
  - *Recommendation:* dense-only for the first landing; add sparse in a follow-up phase (Phase 3b).
- **D3.2** — does `ConnectivityTable` accept filters? Yes — same lazy-plan pattern as SynapseTable, accumulating filters that apply post-pivot. Keep symmetric with EdgeList.
- **D3.3** — label annotation shape. Define as: `LabelAnnotationSpec(lf, label_col, data_cols)` where `data_cols` must include at least one column nameable as the denominator (user-chosen; not sniffed). For future label-axis statistics.

**Tasks:**

1. Create `src/trajan/connectivity_table.py`.
2. Implement `ConnectivityTable(_TrajanTable)`:
   - Blessed columns: `pre_entity`, `post_entity`, weight list.
   - Construction signatures:
     - From a user frame: `ConnectivityTable(df, pre_col=..., post_col=..., weight_cols=[...])` — first-class entry point.
     - (Tier-transition factories added in later phases.)
   - Annotation acceptance: `cell` and `label` kinds. No synapse or vertex (reject at register time with a clear error).
   - Methods:
     - `normalize(by="pre" | "post", values=None, total_col=None) -> ConnectivityTable`
     - `binarize(threshold=0) -> ConnectivityTable`
     - `log1p(values=None) -> ConnectivityTable`
     - `to_dense() -> pl.DataFrame`
     - `to_sparse()` — deferred (Phase 3b).
     - `save` / `load` via DataFolio.
3. Tests in `tests/test_connectivity_table.py`:
   - Construction from a small hand-built frame.
   - `normalize(by="post")` with and without `total_col=`.
   - Cell annotation registration; label annotation registration; vertex/synapse rejection.
   - Save/load round-trip.

**Acceptance:**

- `uv run pytest tests/test_connectivity_table.py` green.
- `normalize` behaves correctly for filtered vs full tables (internal-axis vs `total_col` modes).
- `ConnectivityTable` constructible *without* an upstream `SynapseTable`.

**Commit:** `feat: introduce ConnectivityTable (Tier 2)`

---

## Phase 3b — sparse backend (may slip to a later day)

**Tasks:**

1. Add a sparse storage path (scipy.sparse CSR) to `ConnectivityTable` with auto-selection (dense under ~1% density and ~10k×10k; sparse otherwise; configurable).
2. `to_dense()` / `to_sparse()` conversions.
3. Tests on a synthetic 50k×50k sparse construction.

**Acceptance:**

- `uv run pytest` green.
- No behavioral change for dense-case tests from Phase 3.

**Commit:** `feat: add sparse backend to ConnectivityTable`

---

## Phase 4 — implement `EdgeList` as `ConnectivityTable` subclass

**Purpose:** Tier-1 abstraction. Inherits ConnectivityTable machinery; strengthens the contract to cells-on-both-axes and adds cell-specific operations.

**Design decisions:**

- **D4.1** — EdgeList refuses label annotations. An EdgeList means both axes are cells — registering a label annotation should be either rejected (simplest) or reinterpreted as a cell-annotation-with-label-column (ambiguous; avoid). Reject.
- **D4.2** — type promotion rules. An EdgeList method that collapses an axis to labels (e.g. `aggregate_to_type(pre="cell_type")`) returns a `ConnectivityTable`, not an `EdgeList`. Document clearly.

**Tasks:**

1. Create `src/trajan/edgelist.py`.
2. Implement `EdgeList(ConnectivityTable)`:
   - Constructor enforces cell-axis invariant (no label role).
   - Cell-specific methods:
     - `filter_by_ids(pre_ids=None, post_ids=None)`
     - `filter_by_bbox(bbox)` — requires a registered cell annotation with a position column (role-declared at cell-annotation registration? or via constructor?). Needs design ping — see D4.3.
     - `filter_by_soma_distance(max_distance, distance_fn=euclidean_distance)`
     - `add_spatial_features(...)` — the spatial-feature battery, now at pair level.
   - Tier-promotion methods:
     - `to_connectivity_table(pre="cell" | label_annotation_name, post="cell" | label_annotation_name) -> ConnectivityTable`
3. **D4.3** — where is the soma-position declared on an `EdgeList`? On `SynapseTable` it's `soma_position_annotation` + `soma_position_col`. Options for `EdgeList`: (a) same constructor params; (b) role-declared on the cell annotation at register time. Leaning (b) — simpler, declared at the right scope.
4. Tests in `tests/test_edgelist.py`.

**Acceptance:**

- `uv run pytest tests/test_edgelist.py` green.
- EdgeList passes every ConnectivityTable test (Liskov substitution sanity check).
- Spatial methods work; label annotations rejected.

**Commit:** `feat: introduce EdgeList <: ConnectivityTable (Tier 1)`

---

## Phase 5 — rewire `SynapseTable` aggregation outputs

**Purpose:** connect Tier 0 to Tiers 1–2. `SynapseTable.edgelist()` and `type_edgelist()` become factories for the new types; `matrix()` is removed in favor of `EdgeList.to_connectivity_table().to_dense()` or the ConnectivityTable API.

**Tasks:**

1. `SynapseTable.edgelist(...)` returns `EdgeList`. Cell annotations on the SynapseTable propagate onto the EdgeList (both properties *and* membership — every cell in a registered annotation becomes part of the EdgeList's universe, even cells with zero connections in the current view).
2. `SynapseTable.type_edgelist(pre_col, post_col=None, ...)` returns `ConnectivityTable`. Label annotations must be registered (or built on the fly from a cell annotation's group columns) — design ping.
3. **Remove `SynapseTable.matrix()`.** Replace call sites with `st.edgelist().to_connectivity_table().to_dense()` or equivalent.
4. Update `docs/guides/output.md` and relevant tests.

**Acceptance:**

- `uv run pytest` green with migrated test expectations.
- End-to-end example: load synapses → register cell annotation → filter → `edgelist()` → `EdgeList` carrying all registered cells, including zero-connection cells.

**Commit:** `refactor: SynapseTable aggregation outputs return EdgeList/ConnectivityTable`

---

## Phase 6 — extract stats into `stats.py`

**Purpose:** land `cell_summary` and `normalized` as free functions in the statistics module; this is where the roadmap statistics will live.

**Tasks:**

1. Create `src/trajan/stats.py`.
2. Move `cell_summary` as a free function `cell_summary(table, pre_agg=None, post_agg=None, include_annotations=True)` with type dispatch: accepts `SynapseTable` (full access to synapse-level data) or `EdgeList` (roll-up from pair-level only).
3. Remove `SynapseTable.normalized` — users normalize via `ConnectivityTable.normalize(...)` after converting from an `EdgeList`.
4. Re-export from `trajan.__init__`: `from .stats import cell_summary`.
5. Update `docs/guides/output.md` (the "normalized" and "cell summary" sections now point to the new paths).

**Acceptance:**

- `uv run pytest` green.
- `cell_summary(st)` and `cell_summary(el)` both work, with differing capabilities documented.

**Commit:** `refactor: move statistics into trajan.stats as free functions`

---

## Phase 7 — add the `with_synapses` permutation primitive

**Purpose:** unlock null-model work. Primitive only — specific shufflers can land later in `trajan.nulls`.

**Tasks:**

1. Add `SynapseTable.with_synapses(new_lf, *, preserve_synapse_ids=True) -> SynapseTable`:
   - Returns a new `SynapseTable` sharing annotations, aliases, filters, expressions, weights, metadata.
   - Validates `new_lf` contains `pre_col`, `post_col`, `id_col`.
   - When `preserve_synapse_ids=False`, document that synapse-level annotations become null-filled.
2. Tests: permute the `pre`/`post` columns on a small synapse LF, confirm annotations still join, confirm filters still apply.

**Acceptance:**

- `uv run pytest` green.
- Docstring includes the synapse-ID contract note (preserve vs regenerate).

**Commit:** `feat: add SynapseTable.with_synapses primitive for null-model workflows`

---

## Phase 8 — documentation rewrite

**Purpose:** teach the three-tier architecture up front. Users who don't understand which tier they're in will stay confused forever.

**Tasks:**

1. Rewrite `docs/index.md` around the tiered picture (blessed columns + keyed annotations + three classes).
2. New guide: `docs/guides/tiers.md` — teaches the SynapseTable → EdgeList → ConnectivityTable progression with worked examples.
3. Revise existing guides in place:
   - `synapse-table.md` — scoped to Tier 0.
   - `annotations.md` — extend to cover label annotations and the properties-vs-membership split.
   - `output.md` — rewritten around tier transitions (`SynapseTable.edgelist()` → `EdgeList`, `EdgeList.to_connectivity_table()`, etc.).
   - `filtering.md` — note which filters are available at which tier.
   - `cell-aliases.md` — unchanged content, but cross-link.
   - `persistence.md` — note that all three tiers persist via DataFolio.
4. `docs/reference/api.md` — regenerate / verify mkdocstrings autogen still works against the new structure.

**Acceptance:**

- `uv run poe doc-preview` renders cleanly.
- Every guide references the right tier.
- The "trajan doesn't interpret column meanings" principle is surfaced in the annotations guide.

**Commit:** `docs: rewrite around three-tier architecture`

---

## Day-1 scope recommendation

Realistically achievable tomorrow if you're heads-down:

- **Phase 0** (prep): ~45 min.
- **Phase 1** (export extraction): ~90 min.
- **Phase 2 design** (decide D2.1–D2.4, sketch `_base.py` skeleton): ~2 hr.
- **Phase 2 implementation** (refactor SynapseTable onto `_TrajanTable`): ~3 hr.

Stop at end of Phase 2 with green tests. Commit. That's a solid day — the hardest architectural decision is behind you and ConnectivityTable / EdgeList / stats extraction become mechanical from there.

If Phase 2 feels too ambitious in one day, split: Day 1 = Phases 0 + 1 + Phase 2 design doc only; Day 2 = Phase 2 implementation.

---

## Out of scope for this refactor

- Roadmap statistics (`connection_probability`, `reciprocity`, `connectivity_similarity`, `pair_correlation`) — these land *after* the refactor, on the new tier types.
- Null-model shufflers beyond the primitive — `trajan.nulls` module lands later.
- Visualization submodule — separate effort.
- Polars-JSON serialization hardening (version-matrix regression tests) — separate effort, unrelated to the refactor but worth a follow-up issue.

---

## Open questions to resolve tomorrow morning (before coding)

1. **D2.1** — shared-base shape: abstract base class, mixin, or helper functions? (Recommendation: abstract base.)
2. **D2.2** — annotation storage: tuples vs small dataclasses? (Recommendation: dataclasses.)
3. **D3.1** — sparse backend in Phase 3 or deferred to 3b? (Recommendation: defer.)
4. **D4.3** — where is soma-position declared on `EdgeList`? (Recommendation: on the cell annotation at registration.)
5. **D5** — how are label annotations built from a cell annotation's group columns in `type_edgelist()` — explicit user step, or inferred? (No recommendation yet; probably explicit.)

None of these block Phase 0 or Phase 1. They need resolution before Phase 2 implementation (D2.*) and Phase 3 (D3.*).
