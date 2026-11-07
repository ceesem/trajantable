"""Per-neuron target allocation across a class/subclass/cluster taxonomy."""

from __future__ import annotations

import polars as pl

from trajan import SynapseTable


def target_profiles(
    st: SynapseTable,
    target_cells: pl.DataFrame,
    *,
    cell_id_col: str = "root_id",
    size_col: str = "size",
    tag_col: str = "tag",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return wide neuron features and a long, denominator-bearing table.

    Parameters
    ----------
    st : SynapseTable
        Synapses, optionally already filtered by tag or distance. Existing
        filters are honored. Only non-null presynaptic IDs produce rows.
    target_cells : pl.DataFrame
        Eligible target population, including cells without observed synapses.
        Requires unique, non-null IDs and class/subclass/cluster columns.
        Classes are ``exc`` and ``inh``; other or missing classes are counted
        as unclassified synapses and excluded from allocation fractions.
    cell_id_col : str, optional
        ID column in target_cells.
    size_col : str, optional
        Synapse size column. Null sizes contribute to counts but not means.
    tag_col : str, optional
        Synapse compartment column. Count and total-size fractions for shaft,
        soma, spine, and other (missing/unexpected tags) are included per group.

    Returns
    -------
    summary : pl.DataFrame
        One row per observed presynaptic neuron, with diagnostic counts and
        flat columns such as ``subclass__exc__IT__count_fraction``. Cluster
        columns include their parent subclass to distinguish reused labels.
    long : pl.DataFrame
        One row per neuron and population-defined target group, including
        groups with zero synapses. Class fractions span exc/inh; subclass
        and cluster fractions are conditional on class. Null child labels
        form an explicit ``__unlabeled__`` group. A zero denominator gives
        null fractions. Population-adjusted fractions normalize count/cell
        or total-size/cell rates across groups in the same scope.

    Notes
    -----
    Population counts are supplied explicitly and do not shrink when synapses
    are filtered. They include the presynaptic cell when it belongs to the
    target population; autapses are retained. This adjusts for cell abundance,
    not dendritic exposure, reconstruction completeness, or spatial opportunity.
    Synapses are aggregated lazily before collection. The output itself has
    neuron-count times target-group-count rows and is materialized in memory.
    """
    required = [cell_id_col, "class", "subclass", "cluster"]
    cells = target_cells.select(required)
    ids = cells[cell_id_col]
    if ids.null_count() or ids.n_unique() != cells.height:
        raise ValueError("Target cell IDs must be unique and non-null.")
    if cell_id_col in {"class", "subclass", "cluster"}:
        raise ValueError("The cell ID column must be distinct from taxonomy columns.")
    if any(
        cells[c].cast(pl.String).eq("__unlabeled__").any()
        for c in ("subclass", "cluster")
    ):
        raise ValueError("__unlabeled__ is reserved for missing taxonomy labels.")
    cells = cells.with_columns(
        pl.col("class").cast(pl.String),
        pl.col("subclass", "cluster").cast(pl.String).fill_null("__unlabeled__"),
    ).rename({cell_id_col: "_target_id"})
    # Select only the needed columns from the fully filtered synapse plan.
    syn = (
        st.build_lazy()
        .select(
            pl.col(st.pre_col).alias("pre_id"),
            pl.col(st.post_col).alias("_target_id"),
            pl.col(size_col).alias("_size"),
            pl.col(tag_col).cast(pl.String).alias("_tag"),
        )
        .filter(pl.col("pre_id").is_not_null())
    )
    bad_size = (
        syn.select(
            (
                (pl.col("_size").is_not_null())
                & (~pl.col("_size").is_finite() | (pl.col("_size") < 0))
            ).any()
        )
        .collect()
        .item()
    )
    if bad_size:
        raise ValueError("Non-null synapse sizes must be finite and non-negative.")
    syn = syn.join(cells.lazy(), on="_target_id", how="left").with_columns(
        pl.col("class").is_in(["exc", "inh"]).fill_null(False).alias("_classified")
    )
    summary = (
        syn.group_by("pre_id")
        .agg(
            pl.len().alias("n_syn_total"),
            pl.col("_target_id").is_null().sum().alias("n_syn_missing_post"),
            (~pl.col("_classified")).sum().alias("n_syn_unclassified"),
            pl.col("_classified").sum().alias("n_syn_classified"),
            pl.col("_size").is_null().sum().alias("n_syn_missing_size"),
        )
        .collect()
        .sort("pre_id")
    )
    eligible = cells.filter(pl.col("class").is_in(["exc", "inh"]))
    tags = {
        tag: (pl.col("_tag") == tag).fill_null(False)
        for tag in ("shaft", "soma", "spine")
    }
    tags["other"] = ~pl.col("_tag").is_in(["shaft", "soma", "spine"]).fill_null(False)
    tag_aggregations = [
        expr
        for tag, predicate in tags.items()
        for expr in (
            predicate.sum().alias(f"_tag_{tag}_count"),
            pl.col("_size").filter(predicate).sum().alias(f"_tag_{tag}_size"),
        )
    ]
    fine = (
        syn.filter(pl.col("_classified"))
        .group_by("pre_id", "class", "subclass", "cluster")
        .agg(
            pl.len().alias("n_syn"),
            pl.col("_size").count().alias("n_size"),
            pl.col("_size").sum().alias("size_sum"),
            *tag_aggregations,
        )
        .collect()
    )
    outputs = []
    metrics = [
        "n_target_cells",
        "n_syn",
        "n_size",
        "size_sum",
        "mean_size",
        "count_fraction",
        "size_fraction",
        "synapses_per_target_cell",
        "size_per_target_cell",
        "adjusted_count_fraction",
        "adjusted_size_fraction",
    ]
    metrics += [
        f"tag_{tag}_{kind}_fraction" for tag in tags for kind in ("count", "size")
    ]
    additive = [
        "n_syn",
        "n_size",
        "size_sum",
        *[expr.meta.output_name() for expr in tag_aggregations],
    ]
    for level, keys in (
        ("class", ["class"]),
        ("subclass", ["class", "subclass"]),
        ("cluster", ["class", "subclass", "cluster"]),
    ):
        population = eligible.group_by(keys).agg(pl.len().alias("n_target_cells"))
        observed = fine.group_by("pre_id", *keys).agg(pl.col(additive).sum())
        frame = (
            summary.select("pre_id")
            .join(population, how="cross")
            .join(observed, on=["pre_id", *keys], how="left")
            .with_columns(pl.col(additive).fill_null(0))
        )
        scope = ["pre_id"] if level == "class" else ["pre_id", "class"]

        def ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
            return pl.when(denominator > 0).then(numerator / denominator)

        frame = frame.with_columns(
            *[
                ratio(pl.col(f"_tag_{tag}_{kind}"), pl.col(total)).alias(
                    f"tag_{tag}_{kind}_fraction"
                )
                for tag in tags
                for kind, total in (("count", "n_syn"), ("size", "size_sum"))
            ],
            ratio(pl.col("size_sum"), pl.col("n_size")).alias("mean_size"),
            ratio(pl.col("n_syn"), pl.col("n_syn").sum().over(scope)).alias(
                "count_fraction"
            ),
            ratio(pl.col("size_sum"), pl.col("size_sum").sum().over(scope)).alias(
                "size_fraction"
            ),
            (pl.col("n_syn") / pl.col("n_target_cells")).alias(
                "synapses_per_target_cell"
            ),
            (pl.col("size_sum") / pl.col("n_target_cells")).alias(
                "size_per_target_cell"
            ),
        ).with_columns(
            ratio(
                pl.col("synapses_per_target_cell"),
                pl.col("synapses_per_target_cell").sum().over(scope),
            ).alias("adjusted_count_fraction"),
            ratio(
                pl.col("size_per_target_cell"),
                pl.col("size_per_target_cell").sum().over(scope),
            ).alias("adjusted_size_fraction"),
            pl.lit(level).alias("level"),
            pl.col(level).alias("target_type"),
            pl.concat_str(
                pl.lit(level), *[pl.col(k) for k in keys], separator="__"
            ).alias("_feature"),
        )
        # Pivot each measure separately so feature names remain explicit.
        for metric in metrics:
            pivoted = frame.select(
                "pre_id",
                pl.concat_str("_feature", pl.lit(metric), separator="__").alias(
                    "feature"
                ),
                metric,
            ).pivot(on="feature", index="pre_id", values=metric, sort_columns=True)
            summary = summary.join(pivoted, on="pre_id", how="left")
        if "subclass" not in keys:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("subclass"))
        outputs.append(
            frame.select(
                "pre_id", "level", "class", "subclass", "target_type", *metrics
            )
        )
    long = pl.concat(outputs).sort(
        "pre_id", "level", "class", "subclass", "target_type"
    )
    return summary, long
