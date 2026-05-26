# Cell ID aliasing

Some annotation tables are keyed on a **stable secondary ID** rather than the
root ID that changes when a cell is proofread. Cell aliases let you bridge these
two ID systems without pre-computing lookups outside trajan.

## The problem

Suppose you have:

- A `proofread` annotation keyed on `root_id`, which includes a stable `cell_id`
- A `morphology` annotation keyed on `cell_id`

You cannot pass `morphology` directly to `add_cell_annotation` with
`join_on_alias="cell_id"` — trajan doesn't yet know that `cell_id` in the
`proofread` table maps to cells. A **cell alias** declares this mapping.

## Registering an alias

### Shortcut: `alias_col` in `add_cell_annotation`

The most concise way is to declare the alias when you register the source annotation:

```python
proofread = pl.read_parquet("proofread.parquet")  # root_id, cell_id, proofread
st = st.add_cell_annotation(
    "proofread",
    proofread,
    cell_id_col="root_id",
    alias_col="cell_id",       # register "proofread" as a source of cell_id aliases
)
```

### Explicit: `set_cell_alias`

If you registered the annotation earlier, or want to name the alias differently:

```python
st = st.add_cell_annotation("proofread", proofread, cell_id_col="root_id")
st = st.set_cell_alias("proofread", col="cell_id")

# Custom alias name:
st = st.set_cell_alias("proofread", col="cell_id", alias_name="stable_id")
```

## Consuming an alias

Once registered, use `join_on_alias` in a subsequent `add_cell_annotation` call.
The annotation will be joined on the alias column (`cell_id_pre` / `cell_id_post`)
rather than the root ID columns:

```python
morphology = pl.read_parquet("morphology.parquet")  # cell_id, axon_length
st = st.add_cell_annotation(
    "morphology",
    morphology,
    cell_id_col="cell_id",
    join_on_alias="proofread",  # name matches alias_name (or annotation_name)
)
# .df now has axon_length_pre and axon_length_post
```

## Alias constraints

- The alias source annotation must itself join on root ID (`join_on_alias=None`).
  An annotation that already uses an alias cannot be an alias source.
- The aliased column must be in the annotation's data columns (not the join key).

## What happens on removal

Removing the source annotation also removes its aliases and warns you:

```python
st = st.remove_cell_annotation("proofread")
# UserWarning: Removing annotation 'proofread' which sourced cell alias(es) ['proofread'].
# Those aliases have been cleared. The following annotations reference removed aliases
# and will fail until set_cell_alias() is called again: ['morphology']
```

To recover, register a new alias source and call `set_cell_alias` again before
accessing `.df`.

## Inspecting registered aliases

```python
st.cell_aliases
# {'proofread': ('proofread', 'cell_id')}
```
