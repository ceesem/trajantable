from __future__ import annotations

import warnings
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
    syn_df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
        Synapse table, one row per synapse.
    pre_col : str, optional
        Column name for the pre-synaptic cell id.
    post_col : str, optional
        Column name for the post-synaptic cell id.
    id_col : str, optional
        Column name for the synapse id (used to join synapse annotations).
    synapse_position_col : str or None, optional
        Column in syn_df holding synapse positions as a struct with x, y, z fields.
        Required for filter_by_bbox.
    soma_position_annotation : str or None, optional
        Name of the registered cell annotation that holds soma positions.
        Required for filter_by_soma_distance.
    soma_position_col : str or None, optional
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
        # name → (LazyFrame, cell_id_col, join_on_alias, data_cols)
        # join_on_alias=None joins on _pre_col/_post_col; a string names the alias to join on
        self._cell_annotations: dict[
            str, tuple[pl.LazyFrame, str, str | None, list[str]]
        ] = {}
        # name → (LazyFrame, vertex_id_col, pre_vertex_col, post_vertex_col, data_cols)
        self._vertex_annotations: dict[
            str, tuple[pl.LazyFrame, str, str | None, str | None, list[str]]
        ] = {}

        self._filters: list[pl.Expr] = []
        # name → aliased pl.Expr, applied in insertion order after joins and before filters
        self._expressions: dict[str, pl.Expr] = {}
        # alias_name → (annotation_name, col_in_annotation)
        # populated via set_cell_alias() or alias_col in add_cell_annotation()
        self._cell_aliases: dict[str, tuple[str, str]] = {}
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
            f"vertex_annotations={list(self._vertex_annotations)}, "
            f"expressions={list(self._expressions)})"
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

    @property
    def expression_names(self) -> list[str]:
        return list(self._expressions)

    @property
    def cell_aliases(self) -> dict[str, tuple[str, str]]:
        """Registered cell aliases: {alias_name: (annotation_name, col)}."""
        return dict(self._cell_aliases)

    # ── internal column tracking ───────────────────────────────────────────

    def _current_columns(self) -> set[str]:
        """All column names present (or that will be present) in .synapses."""
        cols = set(self._syn_col_names)
        for _, data_cols in self._synapse_annotations.values():
            cols |= set(data_cols)
        for _, _, _, data_cols in self._cell_annotations.values():
            cols |= {f"{c}_pre" for c in data_cols}
            cols |= {f"{c}_post" for c in data_cols}
        for _, _, pre_v_col, post_v_col, data_cols in self._vertex_annotations.values():
            if pre_v_col is not None:
                cols |= {f"{c}_pre" for c in data_cols}
            if post_v_col is not None:
                cols |= {f"{c}_post" for c in data_cols}
        cols |= set(self._expressions)
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
        name : str
            Identifier for this annotation; used to remove it later via
            remove_synapse_annotation.
        df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
            Annotation table. Must contain id_col plus data columns.
        position_cols : list[str] or str or None, optional
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
        join_on_alias: str | None = None,
        alias_col: str | None = None,
        alias_name: str | None = None,
        position_cols: list[str] | str | None = None,
    ) -> SynapseTable:
        """Register a cell-level annotation, joined symmetrically for pre and post.

        Each column in df (other than cell_id_col) produces two columns in
        .synapses: col_pre and col_post. Raises ValueError on any collision.

        Parameters
        ----------
        name : str
            Identifier for this annotation; used to remove it later via
            remove_cell_annotation.
        df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
            Annotation table. Must contain cell_id_col plus data columns.
        cell_id_col : str
            The column in df containing cell ids to join on.
        join_on_alias : str or None, optional
            If None (default), joins on root ID columns (pre_col/post_col).
            If a string, names the cell alias to join on — the alias must have
            been registered via set_cell_alias() or alias_col on a prior call.
        alias_col : str or None, optional
            If provided, registers this annotation as a cell alias source.
            Equivalent to calling set_cell_alias(name, alias_col, alias_name)
            immediately after registration.
        alias_name : str or None, optional
            Name under which to register the alias. Defaults to the annotation
            name. Only used when alias_col is also provided.
        position_cols : list[str] or str or None, optional
            Column name prefix(es) to auto-pack from split x/y/z format into a
            position struct. E.g. "soma_pt_position" or ["soma_pt_position"] will
            pack soma_pt_position_x/y/z into a struct named soma_pt_position.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.
        """
        if join_on_alias is not None and join_on_alias not in self._cell_aliases:
            raise ValueError(
                f"No cell alias named {join_on_alias!r}. "
                f"Call set_cell_alias() first or use alias_col on a prior "
                f"add_cell_annotation call. Registered aliases: "
                f"{list(self._cell_aliases)}"
            )
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
        self._cell_annotations[name] = (lf, cell_id_col, join_on_alias, data_cols)
        self._cache = None
        if alias_col is not None:
            self.set_cell_alias(name, alias_col, alias_name)
        return self

    def remove_cell_annotation(self, name: str) -> SynapseTable:
        if name not in self._cell_annotations:
            raise KeyError(f"No cell annotation named {name!r}")
        removed_aliases = [
            alias_name
            for alias_name, (ann_name, _) in self._cell_aliases.items()
            if ann_name == name
        ]
        if removed_aliases:
            broken = [
                n
                for n, (_, _, join_on_alias, _) in self._cell_annotations.items()
                if join_on_alias in removed_aliases and n != name
            ]
            msg = (
                f"Removing annotation {name!r} which sourced cell "
                f"alias(es) {removed_aliases}. Those aliases have been cleared."
            )
            if broken:
                msg += (
                    f" The following annotations reference removed aliases and will "
                    f"fail until set_cell_alias() is called again: {broken}"
                )
            warnings.warn(msg, stacklevel=2)
            for alias_name in removed_aliases:
                del self._cell_aliases[alias_name]
        del self._cell_annotations[name]
        self._cache = None
        return self

    def set_cell_alias(
        self,
        annotation_name: str,
        col: str = "cell_id",
        alias_name: str | None = None,
    ) -> SynapseTable:
        """Declare a cell alias column produced by a cell annotation.

        Once registered, annotations indexed by this alias can be added with
        join_on_alias=<alias_name>.

        Parameters
        ----------
        annotation_name : str
            Name of an already-registered cell annotation whose data columns
            include col. This annotation must itself join on root ID
            (join_on_alias=None).
        col : str, optional
            Column within the annotation that holds the aliased cell IDs.
            Defaults to "cell_id".
        alias_name : str or None, optional
            Name used to reference this alias in join_on_alias. Defaults to
            the annotation name.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.

        Raises
        ------
        KeyError
            If annotation_name is not registered.
        ValueError
            If col is not in the annotation's data columns, or if the annotation
            itself uses join_on_alias (the alias source must be root-level).
        """
        if annotation_name not in self._cell_annotations:
            raise KeyError(f"No cell annotation named {annotation_name!r}")
        _, _, join_on_alias, data_cols = self._cell_annotations[annotation_name]
        if join_on_alias is not None:
            raise ValueError(
                f"Annotation {annotation_name!r} uses join_on_alias={join_on_alias!r} "
                f"and cannot itself be a cell alias source. The alias source must "
                f"join on root ID (join_on_alias=None)."
            )
        if col not in data_cols:
            raise ValueError(
                f"Column {col!r} not found in annotation {annotation_name!r}. "
                f"Available columns: {data_cols}"
            )
        key = alias_name if alias_name is not None else annotation_name
        self._cell_aliases[key] = (annotation_name, col)
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
        name : str
            Name of the existing cell annotation to extend.
        df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
            DataFrame or LazyFrame containing the new columns and the join key.
        on : str
            Column name to join on. Must exist in the already-registered annotation
            and must be unique in df.
        position_cols : list[str] or str or None, optional
            Column name prefix(es) to auto-pack from split x/y/z into a position
            struct before registering.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.
        """
        if name not in self._cell_annotations:
            raise KeyError(f"No cell annotation named {name!r}")
        if isinstance(position_cols, str):
            position_cols = [position_cols]

        existing_lf, cell_id_col, join_on_alias, existing_data_cols = (
            self._cell_annotations[name]
        )

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
            join_on_alias,
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
        name : str
            Identifier for this annotation; used to remove it later via
            remove_vertex_annotation.
        df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
            Annotation table. Must contain vertex_id_col plus data columns.
        vertex_id_col : str
            The column in df containing vertex ids to join on.
        pre_vertex_col : str or None, optional
            Column in the synapse table holding pre-synaptic vertex ids.
            If provided, annotation columns appear as col_pre in .synapses.
        post_vertex_col : str or None, optional
            Column in the synapse table holding post-synaptic vertex ids.
            If provided, annotation columns appear as col_post in .synapses.
        position_cols : list[str] or str or None, optional
            Column name prefix(es) to auto-pack from split x/y/z format into a
            position struct. E.g. "pt_position" or ["pt_position"] will pack
            pt_position_x/y/z into a struct named pt_position.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.
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

    # ── expressions ────────────────────────────────────────────────────────

    def add_expression(self, name: str, expr: pl.Expr) -> SynapseTable:
        """Register a named computed column expression.

        Applied after all annotation joins and before filters, so any joined
        column is available. Expressions are applied in insertion order, so
        later expressions may reference earlier ones.

        Raises ValueError if name already exists as a column in the table.

        Parameters
        ----------
        name : str
            Output column name.
        expr : pl.Expr
            Polars expression. An alias of name is applied automatically.

        Returns
        -------
        SynapseTable
            Returns self to allow method chaining.
        """
        if name in self._current_columns():
            raise ValueError(f"Column {name!r} already exists in the table")
        self._expressions[name] = expr.alias(name)
        self._cache = None
        return self

    def remove_expression(self, name: str) -> SynapseTable:
        if name not in self._expressions:
            raise KeyError(f"No expression named {name!r}")
        del self._expressions[name]
        self._cache = None
        return self

    # ── lazy plan construction ─────────────────────────────────────────────

    def _build_lazy(self) -> pl.LazyFrame:
        lf = self._syn_lf

        for ann_lf, _ in self._synapse_annotations.values():
            lf = lf.join(ann_lf, on=self._id_col, how="left")

        for (
            ann_lf,
            cell_id_col,
            join_on_alias,
            data_cols,
        ) in self._cell_annotations.values():
            if join_on_alias is not None:
                alias_col = self._cell_aliases[join_on_alias][1]
                pre_key = f"{alias_col}_pre"
                post_key = f"{alias_col}_post"
            else:
                pre_key = self._pre_col
                post_key = self._post_col
            pre_lf = ann_lf.rename({c: f"{c}_pre" for c in data_cols})
            lf = lf.join(pre_lf, left_on=pre_key, right_on=cell_id_col, how="left")
            post_lf = ann_lf.rename({c: f"{c}_post" for c in data_cols})
            lf = lf.join(post_lf, left_on=post_key, right_on=cell_id_col, how="left")

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

        for expr in self._expressions.values():
            lf = lf.with_columns(expr)

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
        new._expressions = self._expressions.copy()
        new._cell_aliases = self._cell_aliases.copy()
        new._filters = self._filters.copy()
        new._cache = None
        new._n_syn_base = self._n_syn_base
        return new

    # ── selective view ─────────────────────────────────────────────────────

    def view(
        self,
        synapse_annotations: list[str] | None = None,
        cell_annotations: list[str] | None = None,
        vertex_annotations: list[str] | None = None,
        expressions: list[str] | None = None,
        keep_filters: bool = True,
    ) -> SynapseTable:
        """Return a new SynapseTable using only the specified annotations/expressions.

        Useful for building lightweight views without all registered annotations,
        or for reconstructing the table from scratch with granular control.

        Parameters
        ----------
        synapse_annotations : list[str] or None, optional
            Names to include. None keeps all registered; [] keeps none.
        cell_annotations : list[str] or None, optional
            Names to include. None keeps all registered; [] keeps none.
        vertex_annotations : list[str] or None, optional
            Names to include. None keeps all registered; [] keeps none.
        expressions : list[str] or None, optional
            Expression names to include. None keeps all; [] keeps none.
        keep_filters : bool, optional
            Whether to carry forward registered filters. Default True.

        Returns
        -------
        SynapseTable
            A new SynapseTable with only the requested annotations and expressions.

        Raises
        ------
        KeyError
            If any requested name is not registered.
        """

        def _select(d, names):
            if names is None:
                return d.copy()
            missing = set(names) - set(d)
            if missing:
                raise KeyError(f"Unknown name(s): {sorted(missing)}")
            return {k: v for k, v in d.items() if k in names}

        new = SynapseTable.__new__(SynapseTable)
        new._syn_lf = self._syn_lf
        new._syn_col_names = self._syn_col_names
        new._pre_col = self._pre_col
        new._post_col = self._post_col
        new._id_col = self._id_col
        new._synapse_position_col = self._synapse_position_col
        new._soma_position_annotation = self._soma_position_annotation
        new._soma_position_col = self._soma_position_col
        new._n_syn_base = self._n_syn_base
        new._cache = None
        new._synapse_annotations = _select(
            self._synapse_annotations, synapse_annotations
        )
        new._cell_annotations = _select(self._cell_annotations, cell_annotations)
        new._vertex_annotations = _select(self._vertex_annotations, vertex_annotations)
        new._expressions = _select(self._expressions, expressions)
        new._filters = self._filters.copy() if keep_filters else []
        # carry only aliases whose source annotation is still present in the view
        new._cell_aliases = {
            alias_name: (ann_name, col)
            for alias_name, (ann_name, col) in self._cell_aliases.items()
            if ann_name in new._cell_annotations
        }
        return new

    # ── filtering ──────────────────────────────────────────────────────────

    def _annotation_null_expr(self, annotation_name: str, side: str) -> pl.Expr:
        """is_not_null() for the first data column of a cell/vertex annotation on pre or post."""
        if annotation_name in self._cell_annotations:
            data_cols = self._cell_annotations[annotation_name][3]
        elif annotation_name in self._vertex_annotations:
            data_cols = self._vertex_annotations[annotation_name][4]
        else:
            raise KeyError(f"No cell or vertex annotation named {annotation_name!r}")
        return pl.col(f"{data_cols[0]}_{side}").is_not_null()

    def filter_to_annotated(
        self, annotation_name: str, pre: bool = True, post: bool = True
    ) -> SynapseTable:
        """Return a new SynapseTable keeping only synapses where both pre and post
        cells have a non-null value for the given cell or vertex annotation.

        Equivalent to:
            st.filter(pl.col("col_pre").is_not_null() & pl.col("col_post").is_not_null())

        Parameters
        ----------
        annotation_name : str
            Name of a registered cell or vertex annotation to filter on.
        pre : bool, optional
            If True (default), require the pre-synaptic side to be non-null.
        post : bool, optional
            If True (default), require the post-synaptic side to be non-null.

        Returns
        -------
        SynapseTable
            A new SynapseTable with the annotation null filter applied.
        """
        expr = None
        if pre:
            expr = self._annotation_null_expr(annotation_name, "pre")
        if post:
            expr = (
                expr & self._annotation_null_expr(annotation_name, "post")
                if expr is not None
                else self._annotation_null_expr(annotation_name, "post")
            )
        return self.filter(expr)

    def filter(self, expr: pl.Expr) -> SynapseTable:
        """Return a new SynapseTable with expr applied to the lazy plan.

        The filter is pushed into the query plan after all annotation joins,
        so any column in .synapses is valid. Polars' optimizer will push
        predicates on base synapse columns before the joins automatically.

        Parameters
        ----------
        expr : pl.Expr
            A Polars boolean expression to filter rows by.

        Returns
        -------
        SynapseTable
            A new SynapseTable with the filter registered.

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
        max_distance : float
            Maximum soma-to-soma distance to retain, in the same units as
            the position columns.
        distance_fn : Callable[[str, str], pl.Expr], optional
            Callable taking two column name strings and returning a pl.Expr for
            the distance. Defaults to euclidean_distance. Use radial_distance to
            ignore the z axis, or supply a custom function.

        Returns
        -------
        SynapseTable
            A new SynapseTable keeping only synapses within max_distance.
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

    def filter_by_ids(
        self,
        pre_ids=None,
        post_ids=None,
    ) -> SynapseTable:
        """Return a new SynapseTable keeping only synapses involving specified cell IDs.

        Filters are applied on the base pre/post root ID columns, so this works
        even before cell annotations are joined.

        Parameters
        ----------
        pre_ids : Iterable or None, optional
            Iterable of pre-synaptic cell IDs to keep. None means no filter on pre.
        post_ids : Iterable or None, optional
            Iterable of post-synaptic cell IDs to keep. None means no filter on post.

        Returns
        -------
        SynapseTable
            A new SynapseTable filtered to the specified cell IDs.

        Examples
        --------
        Keep only synapses from a specific set of pre cells:

        >>> st.filter_by_ids(pre_ids=[111, 222, 333])

        Keep synapses between two specific populations:

        >>> st.filter_by_ids(pre_ids=excitatory_ids, post_ids=inhibitory_ids)
        """
        if pre_ids is None and post_ids is None:
            return self._copy()
        new = self
        if pre_ids is not None:
            new = new.filter(pl.col(self._pre_col).is_in(list(pre_ids)))
        if post_ids is not None:
            new = new.filter(pl.col(self._post_col).is_in(list(post_ids)))
        return new

    def filter_by_bbox(self, bbox) -> SynapseTable:
        """Filter synapses whose position falls within a bounding box.

        Requires synapse_position_col to be set, with the position column being
        a struct with x, y, z fields (see pack_position).

        Parameters
        ----------
        bbox : Sequence
            Sequence of two (x, y, z) corners: ((xmin, ymin, zmin), (xmax, ymax, zmax)).

        Returns
        -------
        SynapseTable
            A new SynapseTable keeping only synapses within the bounding box.
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

    def edgelist(
        self,
        agg: dict[str, pl.Expr] | None = None,
        pre_anno: bool = True,
        post_anno: bool = True,
    ) -> pl.DataFrame:
        """Aggregate synapses into a cell-pair edgelist.

        Always includes n_syn (synapse count per pair). Additional aggregations
        over any column in .synapses can be passed via agg.

        Parameters
        ----------
        agg : dict[str, pl.Expr] or None, optional
            {output_column_name: polars_expression} for additional aggregations.
            Example: {"mean_size": pl.mean("size"), "total_area": pl.sum("area")}
        pre_anno : bool, optional
            If True (default), include all cell annotation columns for the
            pre-synaptic side (``*_pre``) using ``.first()``.
        post_anno : bool, optional
            If True (default), include all cell annotation columns for the
            post-synaptic side (``*_post``) using ``.first()``.

        Returns
        -------
        pl.DataFrame
            Edgelist with columns [pre_col, post_col, "n_syn"] plus any agg columns
            and, if requested, cell annotation columns.
        """
        agg_exprs = [pl.len().alias("n_syn")]
        if agg:
            agg_exprs.extend(expr.alias(name) for name, expr in agg.items())

        if pre_anno or post_anno:
            for _, _, _, data_cols in self._cell_annotations.values():
                if pre_anno:
                    agg_exprs.extend(pl.col(f"{c}_pre").first() for c in data_cols)
                if post_anno:
                    agg_exprs.extend(pl.col(f"{c}_post").first() for c in data_cols)

        return (
            self._build_lazy()
            .group_by([self._pre_col, self._post_col])
            .agg(agg_exprs)
            .collect()
        )

    def type_edgelist(
        self,
        pre_col: str,
        post_col: str | None = None,
        agg: dict[str, pl.Expr] | None = None,
    ) -> pl.DataFrame:
        """Aggregate synapses into a type-to-type edgelist.

        Groups synapses by cell-type annotation columns rather than individual
        cell IDs, producing synapse counts (and optional aggregations) between
        cell-type categories. This is a fast path to the type-level connectivity
        table that would otherwise require joining the result of edgelist() back
        to annotation data.

        Parameters
        ----------
        pre_col : str
            Column in .synapses to use as the pre-synaptic grouping key.
            Typically a cell annotation column such as ``"cell_type_pre"``.
        post_col : str or None, optional
            Column in .synapses to use as the post-synaptic grouping key.
            Defaults to the corresponding ``*_post`` column: if pre_col ends
            with ``_pre``, post_col becomes the same name with ``_post``; otherwise
            post_col defaults to pre_col.
        agg : dict[str, pl.Expr] or None, optional
            {output_column_name: polars_expression} for additional per-pair
            aggregations, applied after the type grouping.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns [pre_col, post_col, "n_syn"] plus any agg columns.

        Examples
        --------
        Synapse counts between cell types:

        >>> el = st.type_edgelist("cell_type_pre")
        >>> el.columns
        ['cell_type_pre', 'cell_type_post', 'n_syn']

        Using an asymmetric grouping:

        >>> el = st.type_edgelist("cell_type_pre", post_col="broad_type_post")

        With extra aggregations:

        >>> el = st.type_edgelist(
        ...     "cell_type_pre",
        ...     agg={"mean_size": pl.mean("size")},
        ... )
        """
        if post_col is None:
            if pre_col.endswith("_pre"):
                post_col = pre_col[:-4] + "_post"
            else:
                post_col = pre_col

        agg_exprs = [pl.len().alias("n_syn")]
        if agg:
            agg_exprs.extend(expr.alias(name) for name, expr in agg.items())

        return self._build_lazy().group_by([pre_col, post_col]).agg(agg_exprs).collect()

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
        values : str, optional
            Column to use as matrix entries. "n_syn" (synapse count) is always
            available; any other column is summed per cell pair.
        fill_value : float, optional
            Value for missing pairs.
        pre_ids : Iterable or None, optional
            Constrain or pad rows to a fixed cell set. Missing cells
            are filled with fill_value.
        post_ids : Iterable or None, optional
            Constrain or pad columns to a fixed cell set. Missing cells
            are filled with fill_value.
        filter_annotated : str or dict[str, str] or None, optional
            Restrict to synapses where cells have non-null annotation values.
            str: filter both pre and post on the named annotation.
            dict: per-side control, e.g. {"pre": "cell_id"} or
            {"pre": "cell_id", "post": "other_annotation"}.

        Returns
        -------
        pl.DataFrame
            Pivot table with pre cell IDs as rows and post cell IDs as columns.
        """
        st = self
        if isinstance(filter_annotated, str):
            st = st.filter_to_annotated(filter_annotated)
        elif isinstance(filter_annotated, dict):
            for side, ann in filter_annotated.items():
                st = st.filter(st._annotation_null_expr(ann, side))
        if values == "n_syn":
            el = st.edgelist(pre_anno=False, post_anno=False)
        else:
            el = st.edgelist(
                agg={values: pl.sum(values)}, pre_anno=False, post_anno=False
            )

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
        by : str, optional
            "pre" — normalize by each pre cell's total output synaptic weight.
            "post" — normalize by each post cell's total input synaptic weight.
        values : str, optional
            Column to normalize. "n_syn" counts synapses per pair; any other
            column is summed per pair before normalizing.
        group_col : str or None, optional
            Cell annotation name (without _pre/_post suffix) to collapse the
            "other" side before normalizing. Resolved to _pre or _post
            automatically from `by`.

            Example: group_col="broad_type" with by="pre" computes what
            fraction of each pre cell's output goes to each post cell type.
            group_col=None preserves individual cell identity on both sides.
        pivot : bool, optional
            False (default): return tidy DataFrame with a "fraction" column.
            True: pivot into a matrix (self_col rows × other/group columns).

        Returns
        -------
        pl.DataFrame
            Tidy DataFrame with a "fraction" column, or a pivot matrix if
            pivot=True.
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

    # ── graph export ───────────────────────────────────────────────────────

    def to_graph(
        self,
        edge_agg: dict[str, pl.Expr] | None = None,
        cell_agg: dict[str, pl.Expr] | None = None,
        backend: str = "networkx",
    ):
        """Convert the synapse table to a directed graph.

        Nodes are cell IDs. Cell annotation columns (``*_pre`` / ``*_post``) are
        stored as node attributes, with the ``_pre`` / ``_post`` suffix stripped.
        Where pre and post values differ for the same cell, the first encountered
        value is kept. Edge attributes are ``n_syn`` plus any ``edge_agg`` columns.

        Parameters
        ----------
        edge_agg : dict[str, pl.Expr] or None, optional
            Additional per-cell-pair aggregations forwarded to :meth:`edgelist`.
            Results become edge attributes.
        cell_agg : dict[str, pl.Expr] or None, optional
            Additional per-cell aggregations computed by grouping the full synapse
            lazy plan by cell ID. Results become node attributes. Aggregations are
            computed once over all synapses where the cell appears on either side,
            with the pre-side result taking precedence on conflicts.
        backend : str, optional
            Graph library to use. One of ``"networkx"`` (default), ``"igraph"``,
            or ``"csgraph"``. The matching library must be installed.

            ``"csgraph"`` returns a ``(scipy.sparse.csr_array, cell_ids)`` tuple
            rather than a graph object; node annotations are not representable in
            a sparse matrix and are omitted.

        Returns
        -------
        networkx.DiGraph or igraph.Graph or tuple[scipy.sparse.csr_array, list]

        Examples
        --------
        >>> G = st.to_graph()
        >>> G.nodes[111]
        {'cell_type': 'L2/3 ET'}
        >>> G.edges[111, 222]
        {'n_syn': 14}

        With per-cell aggregation as node attributes:

        >>> G = st.to_graph(cell_agg={"n_output": pl.len()})

        igraph backend:

        >>> g = st.to_graph(backend="igraph")
        >>> g.vs["name"]
        [111, 222, 333]

        Scipy sparse matrix:

        >>> mat, cell_ids = st.to_graph(backend="csgraph")
        """
        if backend not in ("networkx", "igraph", "csgraph"):
            raise ValueError(
                f"backend must be 'networkx', 'igraph', or 'csgraph', got {backend!r}"
            )

        el = self.edgelist(agg=edge_agg)

        # ── shared setup ──────────────────────────────────────────────────────
        # Cell annotation attribute columns (present in el as *_pre / *_post)
        anno_cols: list[str] = []
        for _, _, _, data_cols in self._cell_annotations.values():
            anno_cols.extend(data_cols)

        anno_suffixed = {f"{c}_pre" for c in anno_cols} | {
            f"{c}_post" for c in anno_cols
        }
        edge_cols = [
            c
            for c in el.columns
            if c not in {self._pre_col, self._post_col} and c not in anno_suffixed
        ]

        # Ordered unique cell IDs (pre union post, preserving first-seen order)
        seen: dict = {}
        for row in el.iter_rows(named=True):
            seen.setdefault(row[self._pre_col], None)
            seen.setdefault(row[self._post_col], None)
        cell_ids = list(seen)
        idx_map = {cid: i for i, cid in enumerate(cell_ids)}

        # Node attribute dict: cell_id → {attr: value}
        node_attrs: dict = {cid: {} for cid in cell_ids}
        for row in el.iter_rows(named=True):
            for cell_id, side in [
                (row[self._pre_col], "pre"),
                (row[self._post_col], "post"),
            ]:
                attrs = node_attrs[cell_id]
                if not attrs:  # first encounter — populate from annotation cols
                    attrs.update(
                        {
                            c: row[f"{c}_{side}"]
                            for c in anno_cols
                            if f"{c}_{side}" in el.columns
                        }
                    )

        # Merge cell_agg results into node_attrs
        if cell_agg:
            agg_exprs = [expr.alias(name) for name, expr in cell_agg.items()]
            lf = self._build_lazy()
            for id_col in (self._post_col, self._pre_col):  # pre wins on conflict
                per_cell = lf.group_by(id_col).agg(agg_exprs).collect()
                for row in per_cell.iter_rows(named=True):
                    cid = row[id_col]
                    if cid in node_attrs:
                        node_attrs[cid].update(
                            {k: v for k, v in row.items() if k != id_col}
                        )

        # ── backend dispatch ──────────────────────────────────────────────────
        if backend == "networkx":
            try:
                import networkx as nx
            except ImportError as e:
                raise ImportError(
                    "NetworkX is required for backend='networkx'. "
                    "Install it with: uv add networkx"
                ) from e

            G = nx.DiGraph()
            for cid in cell_ids:
                G.add_node(cid, **node_attrs[cid])
            for row in el.iter_rows(named=True):
                G.add_edge(
                    row[self._pre_col],
                    row[self._post_col],
                    **{c: row[c] for c in edge_cols},
                )
            return G

        elif backend == "igraph":
            try:
                import igraph
            except ImportError as e:
                raise ImportError(
                    "igraph is required for backend='igraph'. "
                    "Install it with: uv add igraph"
                ) from e

            g = igraph.Graph(n=len(cell_ids), directed=True)
            g.vs["name"] = cell_ids
            # Set node attribute arrays
            attr_keys = {k for attrs in node_attrs.values() for k in attrs}
            for key in attr_keys:
                g.vs[key] = [node_attrs[cid].get(key) for cid in cell_ids]
            # Add edges
            g.add_edges(
                [
                    (idx_map[row[self._pre_col]], idx_map[row[self._post_col]])
                    for row in el.iter_rows(named=True)
                ]
            )
            for col in edge_cols:
                g.es[col] = el[col].to_list()
            return g

        else:  # csgraph
            try:
                import scipy.sparse as sp
            except ImportError as e:
                raise ImportError(
                    "SciPy is required for backend='csgraph'. "
                    "Install it with: uv add scipy"
                ) from e

            n = len(cell_ids)
            data = el["n_syn"].to_numpy()
            row_idx = [idx_map[v] for v in el[self._pre_col].to_list()]
            col_idx = [idx_map[v] for v in el[self._post_col].to_list()]
            matrix = sp.csr_array((data, (row_idx, col_idx)), shape=(n, n))
            return matrix, cell_ids

    # ── dataframe export ───────────────────────────────────────────────────

    def to_dataframe(self, unpack_positions: bool = True):
        """Return .synapses as a pandas DataFrame.

        Requires pandas (``pip install pandas`` or ``uv add pandas``).

        Parameters
        ----------
        unpack_positions : bool, optional
            If True (default), unpack any struct columns with x, y, z fields
            into flat ``{col}_x``, ``{col}_y``, ``{col}_z`` columns. This is
            necessary because pandas cannot natively represent Polars structs.

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> df = st.to_dataframe()
        >>> df.columns
        Index(['pre_pt_root_id', 'post_pt_root_id', 'soma_pt_position_x', ...])
        """
        if not _HAS_PANDAS:
            raise ImportError(
                "pandas is required for to_dataframe(). Install it with: uv add pandas"
            )
        df = self.synapses
        if unpack_positions:
            from .spatial import unpack_all_positions

            df = unpack_all_positions(df)
        return df.to_pandas()

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, folio, overwrite: bool = False) -> None:
        """Save the SynapseTable to a DataFolio.

        Materializes all lazy frames and writes them as Parquet tables.
        Filters are serialized as JSON expressions and stored in the config.

        Parameters
        ----------
        folio : DataFolio
            A DataFolio instance to save into.
        overwrite : bool, optional
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
            "expressions": {
                name: expr.meta.serialize(format="json")
                for name, expr in self._expressions.items()
            },
            "synapse_annotations": {
                name: {"data_cols": data_cols}
                for name, (_, data_cols) in self._synapse_annotations.items()
            },
            "cell_aliases": {
                alias_name: {"annotation_name": ann_name, "col": col}
                for alias_name, (ann_name, col) in self._cell_aliases.items()
            },
            "cell_annotations": {
                name: {
                    "cell_id_col": cell_id_col,
                    "join_on_alias": join_on_alias,
                    "data_cols": data_cols,
                }
                for name, (
                    _,
                    cell_id_col,
                    join_on_alias,
                    data_cols,
                ) in self._cell_annotations.items()
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
        folio.add_table("synapses", self._syn_lf.collect(), overwrite=overwrite)
        for name, (lf, _) in self._synapse_annotations.items():
            folio.add_table(f"synapse_ann_{name}", lf.collect(), overwrite=overwrite)
        for name, (lf, _, _, _) in self._cell_annotations.items():
            folio.add_table(f"cell_ann_{name}", lf.collect(), overwrite=overwrite)
        for name, (lf, _, _, _, _) in self._vertex_annotations.items():
            folio.add_table(f"vertex_ann_{name}", lf.collect(), overwrite=overwrite)

    @classmethod
    def load(cls, folio) -> SynapseTable:
        """Load a SynapseTable from a DataFolio.

        Parameters
        ----------
        folio : DataFolio
            A DataFolio instance previously written by .save().

        Returns
        -------
        SynapseTable
            A fully reconstructed SynapseTable with all annotations, expressions,
            and filters restored.
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
                name,
                _lf(f"cell_ann_{name}"),
                cell_id_col=meta["cell_id_col"],
                join_on_alias=meta.get("join_on_alias"),
            )
        for alias_name, alias_meta in config.get("cell_aliases", {}).items():
            st.set_cell_alias(
                alias_meta["annotation_name"],
                alias_meta["col"],
                alias_name=alias_name,
            )
        for name, meta in config["vertex_annotations"].items():
            st.add_vertex_annotation(
                name,
                _lf(f"vertex_ann_{name}"),
                vertex_id_col=meta["vertex_id_col"],
                pre_vertex_col=meta["pre_vertex_col"],
                post_vertex_col=meta["post_vertex_col"],
            )
        for name, expr_json in config.get("expressions", {}).items():
            st.add_expression(
                name, pl.Expr.deserialize(expr_json.encode(), format="json")
            )
        for f_json in config["filters"]:
            st = st.filter(pl.Expr.deserialize(f_json.encode(), format="json"))
        return st
