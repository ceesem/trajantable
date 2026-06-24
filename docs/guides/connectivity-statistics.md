# Connectivity statistics

> *"What fraction of the connections that **could** exist actually do — as a
> function of distance, cell type, or anything else — and how sure am I?"*

This guide covers the denominator-bearing machinery: the universe of possible
pairs, connection probability / density, confidence intervals, and the pair
sampler. The recurring theme is that every honest connectivity fraction needs a
**denominator** (the pairs that *could* connect), and every honest error bar
needs a **resampling unit that matches how the data was collected**.

If you just want the recipes, the [cheatsheet](cheatsheet.md) §5–6 has them. This
guide is the *why*.

---

## 1. The universe is the denominator

A connection fraction is `p = k_observed / n_possible`. The numerator is easy —
it's the observed edgelist. The denominator is the hard part: it's every
`(pre, post)` pair drawn from the **cell universe**, whether or not a synapse was
seen. `possible_pairs` builds exactly that frame, lazily:

```python
pu = trajan.possible_pairs(st)     # |U|² − |U| rows, observed n_syn overlaid (0 if unseen)
```

The zeros *are the point* — they're the unconnected possibilities that make the
fraction meaningful. That's why a `PairUniverse` keeps them and an `EdgeList`
does not, and why `counts()` accepts a `PairUniverse` only: hand it observed-only
data and every `p` collapses to ~1.

The universe is whatever annotation you marked `is_universe=True`. It is the
single source of truth — `cells()`, `possible_pairs()`, `connection_density()`,
and the bootstrap all resolve it automatically.

`PairUniverse` deliberately has no `.df`: a 60k-cell universe is 3.6 billion
pairs. **Prune (filter) before you collect**, and prefer
`group_by(...).agg(...)` over materializing the cross-product.

---

## 2. What restricts the denominator — and what only restricts the numerator

This is the subtlety that trips people up. There are three levers, and they do
different things.

### Restrict the *universe* → register a smaller universe annotation

The universe defines the cell set on **both** sides of the cross-product (pre
and post are drawn from it) **and** the population the cell bootstrap resamples.
To change it, change the annotation:

```python
sub = cells.filter(pl.col("region") == "V1")
st.add_cell_annotation("cells", sub, cell_id_col="root_id",
                       is_universe=True, position_col="soma")
pu = trajan.possible_pairs(st)     # now V1 × V1
```

### Restrict one *side* → filter on a cell-annotation column

A filter that references a cell-annotation column (`cell_type_post`,
`is_proofread_pre`, …) is *side-classified* and projects onto the matching side
of the cross-product, shrinking the denominator there:

```python
# denominator = pairs whose POST cell is an L5 PT cell
trajan.connection_probability(
    st.filter(pl.col("cell_type_post") == "L5 PT"),
    bin_by={"soma_rho": [0, 50, 100, 200]},
)
```

### Restrict to specific cells → filter the `PairUniverse`, not the source

!!! warning "Raw id filters on the *source* only move the numerator"
    A filter on the raw id column (`pre_pt_root_id`) is **not** a
    cell-annotation column, so it is classified as synapse/pair-level and bakes
    into the *observed counts only* — the denominator stays the full universe.
    `st.filter(pl.col("pre_pt_root_id") == X)` followed by `possible_pairs`
    gives a wrong (tiny) `p`.

    To restrict to specific cells, filter the **`PairUniverse`**, where filters
    apply directly to the cross-product:

    ```python
    pu = trajan.possible_pairs(st).filter_by_ids(pre_ids=my_cells)
    ```

    For per-cell curves, `group_by("pre_pt_root_id")` instead — that's an
    aggregation key, not a filter, so it's always safe.

!!! warning "Filtering the `PairUniverse` does not move the bootstrap population"
    The cell bootstrap reads its resampling population straight from the
    registered universe annotation, *ignoring* filters layered on the
    `PairUniverse`. If you want the resampled population to be a subpopulation,
    restrict the **universe annotation** (first lever above), not a downstream
    filter.

---

## 3. Connection probability / density

Same formula, two names for two interpretations:

```python
# sampled / sparse reconstruction
trajan.connection_probability(st, bin_by={"soma_rho": [0, 50, 100, 200]},
                              group_by=["cell_type_pre", "cell_type_post"])

# dense reconstruction
trajan.connection_density(st, bin_by={"soma_rho": [0, 50, 100, 200]})
```

`bin_by` maps a column to bin edges (`[0, 50, 100]` → continuous, output column
`"{col}_bin"`) or to `None` (categorical pass-through). The result has one row
per bin with `k_observed`, `n_possible`, `sum_<weight>`, and `p`.
`counts(pu, ...)` is the lower-level primitive if you want the raw numerator /
denominator without the ratio.

!!! note "Numeric bins sort in bin order"
    Continuous bin columns (`"{col}_bin"`) come back as an ordered `pl.Enum`
    (`"(0, 50]"` < `"(50, 100]"` < `"(100, 200]"` …), so `sort`, `pivot`, and
    plotting axes follow numeric bin order automatically — no `to_physical`
    trick needed. (Raw `pl.cut` returns an unordered `Categorical` that would
    sort lexically; the stats functions cast it for you.)

---

## 4. Confidence intervals: match the resampling unit to the experiment

The point estimate is always the observed `k/n`. The error bar depends entirely
on **what you imagine redrawing if you ran the experiment again** — and that is
set by how the data was collected, not by the data itself.

| Observational design | Resample unit | Tool |
|---|---|---|
| Dense EM reconstruction of *N* cells and all their connectivity | **cells** | `bootstrap_over_cells` |
| Paired recordings probing individual pairs, aggregated | **pairs** | `wilson_ci` (binary) / a pair draw (general) |
| Quick sparse estimate | independent pairs | `wilson_ci` / `agresti_coull_ci` |

### Cell draw — for dense reconstructions

When you reconstructed cells and read off *all* their connectivity, the thing
that varies between hypothetical re-runs is **which cells** you got. Pairs that
share a pre- or post-cell co-vary, so a pair-level binomial CI is overconfident.
`bootstrap_over_cells` redraws whole cells:

```python
trajan.bootstrap_over_cells(
    st,
    group_by=["cell_type_pre", "cell_type_post"],
    bin_by={"soma_rho": [0, 50, 100, 200]},
    n_resamples=1000, seed=0,
)   # adds p_lo / p_hi
```

Each resample draws `N` cells with replacement (a `multinomial` over the
universe), weights each pair by `m_pre · m_post`, and recomputes `p`; the CI is
the 2.5/97.5 percentiles across resamples. Because both endpoints are drawn from
the *same* universe draw, the pre and post populations are resampled jointly —
that is the whole point. Iterate with `cell_bootstrap_iter(...)` for custom
summaries.

!!! note "One pre cell? Don't cell-bootstrap"
    With a single pre cell, its multiplicity factors out of `p` and cancels, so
    the cell bootstrap degenerates to resampling partners (and wastes ~37% of
    draws on an empty pre side). Use `wilson_ci` — the partners are independent
    Bernoulli trials, which is exactly the binomial model.

### Pair draw — for paired-recording designs

When each experiment probes a specific pair and you aggregate across
experiments, the resampling unit is the **pair**. For a binary
connected/not outcome, the nonparametric pair bootstrap is just the bootstrap of
a proportion — i.e. **`wilson_ci` is the closed-form pair-draw CI**. You rarely
need to bootstrap it explicitly.

Where an explicit pair draw earns its keep:

- **non-binary per-pair outcomes** (mean synapse count, mean weight given
  connected) with no clean closed form;
- **matching an experimental sampling distribution** — e.g. "80 pairs between
  type A and B, drawn according to a target distance distribution" — which needs
  *weighted* pair sampling.

That is what [`sample_pairs`](#5-the-pair-sampler) provides, and what you build
custom pair-resampling on top of.

---

## 5. The pair sampler

`PairUniverse.sample_pairs` emits a random sample of pairs with their
connectivity. It materializes the filtered cross-product once and samples from
it, so **every filter is honored exactly** — prune first to keep that frame
small (the same ~10M-row warning as `collect()` applies).

```python
# uniform draw with replacement, reproducible
draw = (
    trajan.possible_pairs(st)
    .filter_by_radial_distance(200)
    .sample_pairs(500, seed=0)
)
# draw: ids, weights (n_syn, …), registered annotations, + a `connected` bool
```

### Weighted draws — matching an experimental distribution

Pass a column name or a polars expression as `weights`; pairs are drawn with
probability proportional to it. To mimic "80 recorded A→B pairs drawn according
to a target distance distribution", weight by a kernel of the pair distance:

```python
pu = trajan.possible_pairs(st).filter_by_ids(pre_ids=type_a, post_ids=type_b)

# any non-negative per-pair expression — here a Gaussian kernel centred at 50 µm
kernel = (-((pl.col("soma_rho") - 50) ** 2) / 200).exp()
synthetic_experiment = pu.sample_pairs(80, weights=kernel, seed=0)
fraction_connected = synthetic_experiment["connected"].mean()
```

Repeat the draw with different seeds to build a pair-level resampling
distribution — the "something fancier" that a plain binomial CI can't express
because it ignores the sampling design.

`replace=True` (default) is the bootstrap-friendly choice and lets `n` exceed
the eligible-pair count; `replace=False` draws distinct pairs. Weights must be
finite, non-negative, and sum to a positive value.

---

## See also

- [Cheatsheet](cheatsheet.md) — the same APIs as quick recipes
- [Filtering](filtering.md) — cell-level vs synapse-level filters and side
  classification
- [Adding space](cheatsheet.md#4-add-space-distances-and-spatial-features) —
  registering `soma_rho` and other spatial features used in `bin_by`
