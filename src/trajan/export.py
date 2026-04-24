"""Export helpers: SynapseTable → pandas / NetworkX / igraph / scipy sparse.

These are free functions, not methods, consistent with the architectural
principle that derivation / export belongs outside the narrow
``SynapseTable`` class. They consume only the public accessor surface
(``pre_col``, ``post_col``, ``build_lazy``, ``cell_annotation_data_cols``,
``synapses``, ``edgelist``), so they serve as a template for the other
free-function extractions (statistics, visualization) still to come.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .synapse_table import SynapseTable


try:
    import pandas as pd  # noqa: F401

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def to_dataframe(st: SynapseTable, *, unpack_positions: bool = True):
    """Return ``st.synapses`` as a pandas DataFrame.

    Requires pandas (``uv add pandas``).

    Parameters
    ----------
    st : SynapseTable
        The synapse table to export. ``st.synapses`` is materialized.
    unpack_positions : bool, optional
        If True (default), unpack any struct columns with ``x`` / ``y`` / ``z``
        fields into flat ``{col}_x`` / ``{col}_y`` / ``{col}_z`` columns.
        Pandas cannot natively represent Polars struct types.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> from trajan import to_dataframe
    >>> df = to_dataframe(st)
    """
    if not _HAS_PANDAS:
        raise ImportError(
            "pandas is required for to_dataframe(). Install it with: uv add pandas"
        )
    df = st.synapses
    if unpack_positions:
        from .spatial import unpack_all_positions

        df = unpack_all_positions(df)
    return df.to_pandas()


def to_graph(
    st: SynapseTable,
    *,
    edge_agg: dict[str, pl.Expr] | None = None,
    cell_agg: dict[str, pl.Expr] | None = None,
    backend: str = "networkx",
):
    """Convert a SynapseTable to a directed graph.

    Nodes are cell IDs. Cell-annotation columns (``*_pre`` / ``*_post``) become
    node attributes with the side suffix stripped; when pre and post values
    differ for the same cell, the first encountered value is kept. Edge
    attributes are ``n_syn`` plus any ``edge_agg`` columns.

    Parameters
    ----------
    st : SynapseTable
        Source table. Its edgelist (optionally with ``edge_agg``) provides
        edges; its lazy plan provides ``cell_agg`` aggregations.
    edge_agg : dict[str, pl.Expr] or None, optional
        Per-cell-pair aggregations forwarded to ``st.edgelist(agg=...)``.
        Become edge attributes.
    cell_agg : dict[str, pl.Expr] or None, optional
        Per-cell aggregations computed by grouping the full annotated plan
        by cell ID. Become node attributes. Computed twice — once on the
        post side, once on the pre side — with pre-side values taking
        precedence when a cell appears on both sides.
    backend : str, optional
        Graph library. One of ``"networkx"`` (default), ``"igraph"``, or
        ``"csgraph"``. The matching library must be installed.

        ``"csgraph"`` returns a ``(scipy.sparse.csr_array, cell_ids)`` tuple
        rather than a graph object; node annotations are not representable
        in a sparse matrix and are omitted.

    Returns
    -------
    networkx.DiGraph or igraph.Graph or tuple[scipy.sparse.csr_array, list]

    Examples
    --------
    >>> from trajan import to_graph
    >>> G = to_graph(st)
    >>> G.nodes[111]
    {'cell_type': 'L2/3 ET'}
    >>> G.edges[111, 222]
    {'n_syn': 14}

    With per-cell aggregation as node attributes:

    >>> G = to_graph(st, cell_agg={"n_output": pl.len()})

    igraph backend:

    >>> g = to_graph(st, backend="igraph")

    Scipy sparse matrix:

    >>> mat, cell_ids = to_graph(st, backend="csgraph")
    """
    if backend not in ("networkx", "igraph", "csgraph"):
        raise ValueError(
            f"backend must be 'networkx', 'igraph', or 'csgraph', got {backend!r}"
        )

    pre_col, post_col = st.pre_col, st.post_col
    el = st.edgelist(agg=edge_agg)

    # Cell annotation attribute columns (present in el as *_pre / *_post)
    anno_cols: list[str] = []
    for data_cols in st.cell_annotation_data_cols().values():
        anno_cols.extend(data_cols)

    anno_suffixed = {f"{c}_pre" for c in anno_cols} | {f"{c}_post" for c in anno_cols}
    edge_cols = [
        c for c in el.columns if c not in {pre_col, post_col} and c not in anno_suffixed
    ]

    # Ordered unique cell IDs (pre union post, preserving first-seen order)
    seen: dict = {}
    for row in el.iter_rows(named=True):
        seen.setdefault(row[pre_col], None)
        seen.setdefault(row[post_col], None)
    cell_ids = list(seen)
    idx_map = {cid: i for i, cid in enumerate(cell_ids)}

    # Node attribute dict: cell_id → {attr: value}
    node_attrs: dict = {cid: {} for cid in cell_ids}
    for row in el.iter_rows(named=True):
        for cell_id, side in [(row[pre_col], "pre"), (row[post_col], "post")]:
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
        lf = st.build_lazy()
        for id_col in (post_col, pre_col):  # pre wins on conflict
            per_cell = lf.group_by(id_col).agg(agg_exprs).collect()
            for row in per_cell.iter_rows(named=True):
                cid = row[id_col]
                if cid in node_attrs:
                    node_attrs[cid].update(
                        {k: v for k, v in row.items() if k != id_col}
                    )

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
                row[pre_col],
                row[post_col],
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
        attr_keys = {k for attrs in node_attrs.values() for k in attrs}
        for key in attr_keys:
            g.vs[key] = [node_attrs[cid].get(key) for cid in cell_ids]
        g.add_edges(
            [
                (idx_map[row[pre_col]], idx_map[row[post_col]])
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
                "SciPy is required for backend='csgraph'. Install it with: uv add scipy"
            ) from e

        n = len(cell_ids)
        data = el["n_syn"].to_numpy()
        row_idx = [idx_map[v] for v in el[pre_col].to_list()]
        col_idx = [idx_map[v] for v in el[post_col].to_list()]
        matrix = sp.csr_array((data, (row_idx, col_idx)), shape=(n, n))
        return matrix, cell_ids
