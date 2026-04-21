from __future__ import annotations

from typing import Callable

import polars as pl

from .spatial import euclidean_distance, pack_position

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def _auto_pack(lf: pl.LazyFrame, col: str | None) -> pl.LazyFrame:
    """Pack {col}_x/y/z into a struct named col, if col is absent but the triplet exists."""
    if col is None:
        return lf
    names = lf.collect_schema().names()
    if col in names:
        return lf
    if all(f"{col}_{ax}" in names for ax in ("x", "y", "z")):
        return pack_position(lf, col)
    return lf


def _to_lazy(df) -> pl.LazyFrame:
    if isinstance(df, pl.LazyFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    if _HAS_PANDAS and isinstance(df, pd.DataFrame):
        return pl.from_pandas(df).lazy()
    raise TypeError(
        f"Expected pl.DataFrame, pl.LazyFrame, or pd.DataFrame, got {type(df)}"
    )


class SynapseTable:
    """Synapse list with automatic pre/post cell annotation merging.

    Owns the synapse list → edgelist → connectivity matrix → normalized
    connectivity pipeline. Cell annotations are merged symmetrically:
    adding an annotation named "cell_type" with a cell_type column produces
    cell_type_pre and cell_type_post on .synapses automatically.

    Built on Polars lazy frames; the merged table is cached and invalidated
    only when annotations are added or removed.

    Parameters
    ----------
    syn_df:
        Synapse table, one row per synapse.
    pre_col:
        Column name for the pre-synaptic cell id.
    post_col:
        Column name for the post-synaptic cell id.
    id_col:
        Column name for the synapse id (used to join synapse annotations).
    synapse_position_col:
        Column in syn_df holding synapse positions as a struct with x, y, z fields.
        Required for filter_by_bbox.
    soma_position_annotation:
        Name of the registered cell annotation that holds soma positions.
        Required for filter_by_soma_distance.
    soma_position_col:
        Column within the soma position annotation that holds positions as a struct
        with x, y, z fields. Required for filter_by_soma_distance.
    """

    def __init__(
        self,
        syn_df,
        pre_col: str = "pre_pt_root_id",
        post_col: str = "post_pt_root_id",
        id_col: str = "id",
        synapse_position_col: str | None = None,
        soma_position_annotation: str | None = None,
        soma_position_col: str | None = None,
    ):
        self._syn_lf = _auto_pack(_to_lazy(syn_df), synapse_position_col)
        self._pre_col = pre_col
        self._post_col = post_col
        self._id_col = id_col
        self._synapse_position_col = synapse_position_col
        self._soma_position_annotation = soma_position_annotation
        self._soma_position_col = soma_position_col

        # Cache base schema names once — _syn_lf never changes
        self._syn_col_names: list[str] = self._syn_lf.collect_schema().names()

        # name → (LazyFrame, data_cols)  — data_cols excludes the join key
        self._synapse_annotations: dict[str, tuple[pl.LazyFrame, list[str]]] = {}
        # name → (LazyFrame, cell_id_col, data_cols)
        self._cell_annotations: dict[str, tuple[pl.LazyFrame, str, list[str]]] = {}
        # name → (LazyFrame, vertex_id_col, pre_vertex_col, post_vertex_col, data_cols)
        self._vertex_annotations: dict[
            str, tuple[pl.LazyFrame, str, str | None, str | None, list[str]]
        ] = {}

        self._filters: list[pl.Expr] = []
        self._cache: pl.DataFrame | None = None
        self._n_syn_base: int = self._syn_lf.select(pl.len()).collect().item()

    # ── repr ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self._cache is not None:
            n = len(self._cache)
        elif not self._filters:
            n = self._n_syn_base
        else:
            n = "uncached"
        return (
            f"SynapseTable("
            f"n_syn={n}, "
            f"synapse_annotations={list(self._synapse_annotations)}, "
            f"cell_annotations={list(self._cell_annotations)}, "
            f"vertex_annotations={list(self._vertex_annotations)})"
        )

    # ── annotation name lists ──────────────────────────────────────────────

    @property
    def synapse_annotation_names(self) -> list[str]:
        return list(self._synapse_annotations)

    @property
    def cell_annotation_names(self) -> list[str]:
        return list(self._cell_annotations)

    @property
    def vertex_annotation_names(self) -> list[str]:
        return list(self._vertex_annotations)

    # ── internal column tracking ───────────────────────────────────────────

    def _current_columns(self) -> set[str]:
        """All column names present (or that will be present) in .synapses."""
        cols = set(self._syn_col_names)
        for _, data_cols in self._synapse_annotations.values():
            cols |= set(data_cols)
        for _, _, data_cols in self._cell_annotations.values():
            cols |= {f"{c}_pre" for c in data_cols}
            cols |= {f"{c}_post" for c in data_cols}
        for _, _, pre_v_col, post_v_col, data_cols in self._vertex_annotations.values():
            if pre_v_col is not None:
                cols |= {f"{c}_pre" for c in data_cols}
            if post_v_col is not None:
                cols |= {f"{c}_post" for c in data_cols}
        return cols

    # ── internal helpers ───────────────────────────────────────────────────

    def _validate_join_key_unique(self, lf: pl.LazyFrame, key_col: str) -> None:
        """Raise ValueError if key_col has duplicate values in lf."""
        n_total = lf.select(pl.len()).collect().item()
        n_unique = lf.select(pl.col(key_col).n_unique()).collect().item()
        if n_total != n_unique:
            raise ValueError(
                f"Annotation join key {key_col!r} has {n_total - n_unique} duplicate "
                f"value(s); each id must appear at most once to avoid expanding synapse rows."
            )

    # ── synapse annotations ────────────────────────────────────────────────

    def add_synapse_annotation(
        self, name: str, df, position_cols: list[str] | str | None = None
    ) -> None:
        """Register a synapse-level annotation, joined on id_col.

        Raises ValueError if any column already exists in the table.

        Parameters
        ----------
        position_cols:
            Column name prefix(es) to auto-pack from split x/y/z format into a
            position struct. E.g. "ctr_pt_position" or ["ctr_pt_position"] will
            pack ctr_pt_position_x/y/z into a struct named ctr_pt_position.
        """
        if isinstance(position_cols, str):
            position_cols = [position_cols]
        lf = _to_lazy(df)
        for col in position_cols or []:
            lf = _auto_pack(lf, col)
        self._validate_join_key_unique(lf, self._id_col)
        data_cols = [c for c in lf.collect_schema().names() if c != self._id_col]
        collisions = set(data_cols) & self._current_columns()
        if collisions:
            raise ValueError(f"Columns already exist in table: {sorted(collisions)}")
        self._synapse_annotations[name] = (lf, data_cols)
        self._cache = None
        return self

    def remove_synapse_annotation(self, name: str) -> None:
        if name not in self._synapse_annotations:
            raise KeyError(f"No synapse annotation named {name!r}")
        del self._synapse_annotations[name]
        self._cache = None
        return self

    # ── cell annotations ───────────────────────────────────────────────────

    def add_cell_annotation(
        self,
        name: str,
        df,
        cell_id_col: str,
        position_cols: list[str] | str | None = None,
    ) -> None:
        """Register a cell-level annotation, joined symmetrically for pre and post.

        Each column in df (other than cell_id_col) produces two columns in
        .synapses: col_pre and col_post. Raises ValueError on any collision.

        Parameters
        ----------
        cell_id_col:
            The column in df containing cell ids to join on.
        position_cols:
            Column name prefix(es) to auto-pack from split x/y/z format into a
            position struct. E.g. "soma_pt_position" or ["soma_pt_position"] will
            pack soma_pt_position_x/y/z into a struct named soma_pt_position.
        """
        if isinstance(position_cols, str):
            position_cols = [position_cols]
        lf = _to_lazy(df)
        for col in position_cols or []:
            lf = _auto_pack(lf, col)
        self._validate_join_key_unique(lf, cell_id_col)
        data_cols = [c for c in lf.collect_schema().names() if c != cell_id_col]
        new_cols = {f"{c}_pre" for c in data_cols} | {f"{c}_post" for c in data_cols}
        collisions = new_cols & self._current_columns()
        if collisions:
            raise ValueError(f"Columns already exist in table: {sorted(collisions)}")
        self._cell_annotations[name] = (lf, cell_id_col, data_cols)
        self._cache = None
        return self

    def remove_cell_annotation(self, name: str) -> None:
        if name not in self._cell_annotations:
            raise KeyError(f"No cell annotation named {name!r}")
        del self._cell_annotations[name]
        self._cache = None
        return self

    def extend_cell_annotation(
        self,
        name: str,
        df,
        on: str,
        position_cols: list[str] | str | None = None,
    ) -> SynapseTable:
        """Join additional columns into an already-registered cell annotation.

        Uses a left join on `on`, which must be a column already present in the
        registered annotation (not the synapse table's cell id). Cells without a
        match in df receive nulls — row count is never expanded.

        Parameters
        ----------
        name:
            Name of the existing cell annotation to extend.
        df:
            DataFrame or LazyFrame containing the new columns and the join key.
        on:
            Column name to join on. Must exist in the already-registered annotation
            and must be unique in df.
        position_cols:
            Column name prefix(es) to auto-pack from split x/y/z into a position
            struct before registering.
        """
        if name not in self._cell_annotations:
            raise KeyError(f"No cell annotation named {name!r}")
        if isinstance(position_cols, str):
            position_cols = [position_cols]

        existing_lf, cell_id_col, existing_data_cols = self._cell_annotations[name]

        existing_schema = existing_lf.collect_schema().names()
        if on not in existing_schema:
            raise ValueError(
                f"Join key {on!r} not found in annotation {name!r}. "
                f"Available columns: {existing_schema}"
            )

        new_lf = _to_lazy(df)
        for col in position_cols or []:
            new_lf = _auto_pack(new_lf, col)

        self._validate_join_key_unique(new_lf, on)

        extra_cols = [c for c in new_lf.collect_schema().names() if c != on]
        new_pre_post = {f"{c}_pre" for c in extra_cols} | {
            f"{c}_post" for c in extra_cols
        }
        collisions = new_pre_post & self._current_columns()
        if collisions:
            raise ValueError(f"Columns already exist in table: {sorted(collisions)}")

        merged_lf = existing_lf.join(new_lf, on=on, how="left")
        self._cell_annotations[name] = (
            merged_lf,
            cell_id_col,
            existing_data_cols + extra_cols,
        )
        self._cache = None
        return self

    # ── vertex annotations ─────────────────────────────────────────────────

    def add_vertex_annotation(
        self,
        name: str,
        df,
        vertex_id_col: str,
        pre_vertex_col: str | None = None,
        post_vertex_col: str | None = None,
        position_cols: list[str] | str | None = None,
    ) -> SynapseTable:
        """Register a vertex-level annotation, joined for pre and/or post sides.

        Each column in df (other than vertex_id_col) produces a col_pre column
        if pre_vertex_col is given, and/or a col_post column if post_vertex_col
        is given. Raises ValueError on any collision or if neither side is given.

        Parameters
        ----------
        vertex_id_col:
            The column in df containing vertex ids to join on.
        pre_vertex_col:
            Column in the synapse table holding pre-synaptic vertex ids.
            If provided, annotation columns appear as col_pre in .synapses.
        post_vertex_col:
            Column in the synapse table holding post-synaptic vertex ids.
            If provided, annotation columns appear as col_post in .synapses.
        position_cols:
            Column name prefix(es) to auto-pack from split x/y/z format into a
            position struct. E.g. "pt_position" or ["pt_position"] will pack
            pt_position_x/y/z into a struct named pt_position.
        """
        if pre_vertex_col is None and post_vertex_col is None:
            raise ValueError(
                "At least one of pre_vertex_col or post_vertex_col must be provided"
            )
        if isinstance(position_cols, str):
            position_cols = [position_cols]
        lf = _to_lazy(df)
        for col in position_cols or []:
            lf = _auto_pack(lf, col)
        self._validate_join_key_unique(lf, vertex_id_col)
        data_cols = [c for c in lf.collect_schema().names() if c != vertex_id_col]
        new_cols = set()
        if pre_vertex_col is not None:
            new_cols |= {f"{c}_pre" for c in data_cols}
        if post_vertex_col is not None:
            new_cols |= {f"{c}_post" for c in data_cols}
        collisions = new_cols & self._current_columns()
        if collisions:
            raise ValueError(f"Columns already exist in table: {sorted(collisions)}")
        self._vertex_annotations[name] = (
            lf,
            vertex_id_col,
            pre_vertex_col,
            post_vertex_col,
            data_cols,
        )
        self._cache = None
        return self

    def remove_vertex_annotation(self, name: str) -> SynapseTable:
        if name not in self._vertex_annotations:
            raise KeyError(f"No vertex annotation named {name!r}")
        del self._vertex_annotations[name]
        self._cache = None
        return self

    # ── lazy plan construction ─────────────────────────────────────────────

    def _build_lazy(self) -> pl.LazyFrame:
        lf = self._syn_lf

        for ann_lf, _ in self._synapse_annotations.values():
            lf = lf.join(ann_lf, on=self._id_col, how="left")

        for ann_lf, cell_id_col, data_cols in self._cell_annotations.values():
            pre_lf = ann_lf.rename({c: f"{c}_pre" for c in data_cols})
            lf = lf.join(
                pre_lf, left_on=self._pre_col, right_on=cell_id_col, how="left"
            )
            post_lf = ann_lf.rename({c: f"{c}_post" for c in data_cols})
            lf = lf.join(
                post_lf, left_on=self._post_col, right_on=cell_id_col, how="left"
            )

        for (
            ann_lf,
            vertex_id_col,
            pre_v_col,
            post_v_col,
            data_cols,
        ) in self._vertex_annotations.values():
            if pre_v_col is not None:
                pre_lf = ann_lf.rename({c: f"{c}_pre" for c in data_cols})
                lf = lf.join(
                    pre_lf, left_on=pre_v_col, right_on=vertex_id_col, how="left"
                )
            if post_v_col is not None:
                post_lf = ann_lf.rename({c: f"{c}_post" for c in data_cols})
                lf = lf.join(
                    post_lf, left_on=post_v_col, right_on=vertex_id_col, how="left"
                )

        for f in self._filters:
            lf = lf.filter(f)

        return lf

    # ── synapses property (cached) ─────────────────────────────────────────

    @property
    def synapses(self) -> pl.DataFrame:
        """Full merged synapse table with all registered annotations. Cached."""
        if self._cache is None:
            self._cache = self._build_lazy().collect()
        return self._cache

    # ── copy helper ────────────────────────────────────────────────────────

    def _copy(self) -> SynapseTable:
        new = SynapseTable.__new__(SynapseTable)
        new._syn_lf = self._syn_lf
        new._syn_col_names = self._syn_col_names
        new._pre_col = self._pre_col
        new._post_col = self._post_col
        new._id_col = self._id_col
        new._synapse_position_col = self._synapse_position_col
        new._soma_position_annotation = self._soma_position_annotation
        new._soma_position_col = self._soma_position_col
        new._synapse_annotations = self._synapse_annotations.copy()
        new._cell_annotations = self._cell_annotations.copy()
        new._vertex_annotations = self._vertex_annotations.copy()
        new._filters = self._filters.copy()
        new._cache = None
        new._n_syn_base = self._n_syn_base
        return new

    # ── filtering ──────────────────────────────────────────────────────────

    def _annotation_null_expr(self, annotation_name: str, side: str) -> pl.Expr:
        """is_not_null() for the first data column of a cell/vertex annotation on pre or post."""
        if annotation_name in self._cell_annotations:
            data_cols = self._cell_annotations[annotation_name][2]
        elif annotation_name in self._vertex_annotations:
            data_cols = self._vertex_annotations[annotation_name][4]
        else:
            raise KeyError(f"No cell or vertex annotation named {annotation_name!r}")
        return pl.col(f"{data_cols[0]}_{side}").is_not_null()

    def filter_to_annotated(self, annotation_name: str) -> SynapseTable:
        """Return a new SynapseTable keeping only synapses where both pre and post
        cells have a non-null value for the given cell or vertex annotation.

        Equivalent to:
            st.filter(pl.col("col_pre").is_not_null() & pl.col("col_post").is_not_null())
        """
        expr = self._annotation_null_expr(
            annotation_name, "pre"
        ) & self._annotation_null_expr(annotation_name, "post")
        return self.filter(expr)

    def filter(self, expr: pl.Expr) -> SynapseTable:
        """Return a new SynapseTable with expr applied to the lazy plan.

        The filter is pushed into the query plan after all annotation joins,
        so any column in .synapses is valid. Polars' optimizer will push
        predicates on base synapse columns before the joins automatically.

        Example
        -------
        st.filter(pl.col("cell_type_pre") == "L2/3 ET")
        st.filter(pl.col("pre_pt_root_id").is_in(cell_ids))
        """
        new = self._copy()
        new._filters = self._filters + [expr]
        return new

    def filter_by_soma_distance(
        self,
        max_distance: float,
        distance_fn: Callable[[str, str], pl.Expr] = euclidean_distance,
    ) -> SynapseTable:
        """Return a new SynapseTable keeping only synapses where soma-soma
        distance is ≤ max_distance (in the same units as your position columns).

        Requires soma_position_annotation and soma_position_col to be set, with
        the position column being a struct with x, y, z fields (see pack_position).

        Parameters
        ----------
        distance_fn:
            Callable taking two column name strings and returning a pl.Expr for
            the distance. Defaults to euclidean_distance. Use radial_distance to
            ignore the z axis, or supply a custom function.
        """
        if self._soma_position_annotation is None:
            raise ValueError("soma_position_annotation not set on this SynapseTable")
        if self._soma_position_annotation not in self._cell_annotations:
            raise ValueError(
                f"Soma position annotation {self._soma_position_annotation!r} "
                f"not registered. Call add_cell_annotation first."
            )
        if self._soma_position_col is None:
            raise ValueError("soma_position_col not set on this SynapseTable")
        pre_col = f"{self._soma_position_col}_pre"
        post_col = f"{self._soma_position_col}_post"
        return self.filter(distance_fn(pre_col, post_col) <= max_distance)

    def filter_by_bbox(self, bbox) -> SynapseTable:
        """Filter synapses whose position falls within a bounding box.

        Requires synapse_position_col to be set, with the position column being
        a struct with x, y, z fields (see pack_position).

        Parameters
        ----------
        bbox:
            Sequence of two (x, y, z) corners: ((xmin, ymin, zmin), (xmax, ymax, zmax)).
        """
        if self._synapse_position_col is None:
            raise ValueError("synapse_position_col not set on this SynapseTable")
        (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox
        col = pl.col(self._synapse_position_col)
        return self.filter(
            (col.struct.field("x") >= xmin)
            & (col.struct.field("x") <= xmax)
            & (col.struct.field("y") >= ymin)
            & (col.struct.field("y") <= ymax)
            & (col.struct.field("z") >= zmin)
            & (col.struct.field("z") <= zmax)
        )

    # ── edgelist ───────────────────────────────────────────────────────────

    def edgelist(self, agg: dict[str, pl.Expr] | None = None) -> pl.DataFrame:
        """Aggregate synapses into a cell-pair edgelist.

        Always includes n_syn (synapse count per pair). Additional aggregations
        over any column in .synapses can be passed via agg.

        Parameters
        ----------
        agg:
            {output_column_name: polars_expression} for additional aggregations.
            Example: {"mean_size": pl.mean("size"), "total_area": pl.sum("area")}
        """
        agg_exprs = [pl.len().alias("n_syn")]
        if agg:
            agg_exprs.extend(expr.alias(name) for name, expr in agg.items())
        return (
            self._build_lazy()
            .group_by([self._pre_col, self._post_col])
            .agg(agg_exprs)
            .collect()
        )

    # ── matrix ─────────────────────────────────────────────────────────────

    def matrix(
        self,
        values: str = "n_syn",
        fill_value: float = 0,
        pre_ids=None,
        post_ids=None,
        filter_annotated: str | dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """Pivot the edgelist into a pre × post connectivity matrix.

        Parameters
        ----------
        values:
            Column to use as matrix entries. "n_syn" (synapse count) is always
            available; any other column is summed per cell pair.
        fill_value:
            Value for missing pairs.
        pre_ids, post_ids:
            Constrain or pad rows/columns to a fixed cell set. Missing cells
            are filled with fill_value.
        filter_annotated:
            Restrict to synapses where cells have non-null annotation values.
            str: filter both pre and post on the named annotation.
            dict: per-side control, e.g. {"pre": "cell_id"} or
            {"pre": "cell_id", "post": "other_annotation"}.
        """
        st = self
        if isinstance(filter_annotated, str):
            st = st.filter_to_annotated(filter_annotated)
        elif isinstance(filter_annotated, dict):
            for side, ann in filter_annotated.items():
                st = st.filter(st._annotation_null_expr(ann, side))
        if values == "n_syn":
            el = st.edgelist()
        else:
            el = st.edgelist(agg={values: pl.sum(values)})

        result = el.pivot(
            on=self._post_col,
            index=self._pre_col,
            values=values,
            aggregate_function="first",
        ).fill_null(fill_value)

        if pre_ids is not None:
            pre_df = pl.DataFrame({self._pre_col: list(pre_ids)})
            result = pre_df.join(result, on=self._pre_col, how="left").fill_null(
                fill_value
            )

        if post_ids is not None:
            existing = set(result.columns) - {self._pre_col}
            post_str = [str(p) for p in post_ids]
            for pid in post_str:
                if pid not in existing:
                    result = result.with_columns(pl.lit(fill_value).alias(pid))
            result = result.select([self._pre_col] + post_str)

        return result

    # ── normalized ─────────────────────────────────────────────────────────

    def normalized(
        self,
        by: str = "pre",
        values: str = "n_syn",
        group_col: str | None = None,
        pivot: bool = False,
    ) -> pl.DataFrame:
        """Compute fractional connectivity normalized by pre or post cell totals.

        Parameters
        ----------
        by:
            "pre" — normalize by each pre cell's total output synaptic weight.
            "post" — normalize by each post cell's total input synaptic weight.
        values:
            Column to normalize. "n_syn" counts synapses per pair; any other
            column is summed per pair before normalizing.
        group_col:
            Cell annotation name (without _pre/_post suffix) to collapse the
            "other" side before normalizing. Resolved to _pre or _post
            automatically from `by`.

            Example: group_col="broad_type" with by="pre" computes what
            fraction of each pre cell's output goes to each post cell type.
            group_col=None preserves individual cell identity on both sides.
        pivot:
            False (default): return tidy DataFrame with a "fraction" column.
            True: pivot into a matrix (self_col rows × other/group columns).
        """
        if by == "pre":
            self_col = self._pre_col
            other_col = self._post_col
            group_suffix = "_post"
        elif by == "post":
            self_col = self._post_col
            other_col = self._pre_col
            group_suffix = "_pre"
        else:
            raise ValueError(f"by must be 'pre' or 'post', got {by!r}")

        if group_col is not None:
            resolved = f"{group_col}{group_suffix}"
            group_by_cols = [self_col, resolved]
            other_label = resolved
        else:
            group_by_cols = [self_col, other_col]
            other_label = other_col

        if values == "n_syn":
            val_expr = pl.len().alias("n_syn")
        else:
            val_expr = pl.sum(values).alias(values)

        agg_lf = self._build_lazy().group_by(group_by_cols).agg(val_expr)
        totals_lf = agg_lf.group_by(self_col).agg(pl.sum(values).alias("_total"))

        result = (
            agg_lf.join(totals_lf, on=self_col)
            .with_columns((pl.col(values) / pl.col("_total")).alias("fraction"))
            .drop(["_total", values])
            .collect()
        )

        if pivot:
            result = result.pivot(
                on=other_label,
                index=self_col,
                values="fraction",
                aggregate_function="first",
            ).fill_null(0.0)

        return result

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, folio, overwrite: bool = False) -> None:
        """Save the SynapseTable to a DataFolio.

        Materializes all lazy frames and writes them as Parquet tables.
        Filters are serialized as JSON expressions and stored in the config.

        Parameters
        ----------
        folio:
            A DataFolio instance to save into.
        overwrite:
            If True, overwrite existing items in the folio.
        """
        config = {
            "pre_col": self._pre_col,
            "post_col": self._post_col,
            "id_col": self._id_col,
            "synapse_position_col": self._synapse_position_col,
            "soma_position_annotation": self._soma_position_annotation,
            "soma_position_col": self._soma_position_col,
            "filters": [f.meta.serialize(format="json") for f in self._filters],
            "synapse_annotations": {
                name: {"data_cols": data_cols}
                for name, (_, data_cols) in self._synapse_annotations.items()
            },
            "cell_annotations": {
                name: {"cell_id_col": cell_id_col, "data_cols": data_cols}
                for name, (_, cell_id_col, data_cols) in self._cell_annotations.items()
            },
            "vertex_annotations": {
                name: {
                    "vertex_id_col": vertex_id_col,
                    "pre_vertex_col": pre_v_col,
                    "post_vertex_col": post_v_col,
                    "data_cols": data_cols,
                }
                for name, (
                    _,
                    vertex_id_col,
                    pre_v_col,
                    post_v_col,
                    data_cols,
                ) in self._vertex_annotations.items()
            },
        }
        folio.add_json("config", config, overwrite=overwrite)
        folio.add_table(
            "synapses", self._syn_lf.collect().to_pandas(), overwrite=overwrite
        )
        for name, (lf, _) in self._synapse_annotations.items():
            folio.add_table(
                f"synapse_ann_{name}", lf.collect().to_pandas(), overwrite=overwrite
            )
        for name, (lf, _, _) in self._cell_annotations.items():
            folio.add_table(
                f"cell_ann_{name}", lf.collect().to_pandas(), overwrite=overwrite
            )
        for name, (lf, _, _, _, _) in self._vertex_annotations.items():
            folio.add_table(
                f"vertex_ann_{name}", lf.collect().to_pandas(), overwrite=overwrite
            )

    @classmethod
    def load(cls, folio) -> SynapseTable:
        """Load a SynapseTable from a DataFolio.

        Parameters
        ----------
        folio:
            A DataFolio instance previously written by .save().
        """

        def _lf(name: str) -> pl.LazyFrame:
            return pl.scan_parquet(folio.get_data_path(name))

        config = folio.get_json("config")
        st = cls(
            _lf("synapses"),
            pre_col=config["pre_col"],
            post_col=config["post_col"],
            id_col=config["id_col"],
            synapse_position_col=config["synapse_position_col"],
            soma_position_annotation=config["soma_position_annotation"],
            soma_position_col=config["soma_position_col"],
        )
        for name in config["synapse_annotations"]:
            st.add_synapse_annotation(name, _lf(f"synapse_ann_{name}"))
        for name, meta in config["cell_annotations"].items():
            st.add_cell_annotation(
                name, _lf(f"cell_ann_{name}"), cell_id_col=meta["cell_id_col"]
            )
        for name, meta in config["vertex_annotations"].items():
            st.add_vertex_annotation(
                name,
                _lf(f"vertex_ann_{name}"),
                vertex_id_col=meta["vertex_id_col"],
                pre_vertex_col=meta["pre_vertex_col"],
                post_vertex_col=meta["post_vertex_col"],
            )
        for f_json in config["filters"]:
            st = st.filter(pl.Expr.deserialize(f_json.encode(), format="json"))
        return st
