import sys
from unittest.mock import patch

import polars as pl
import pytest

from trajan import cell_summary, to_graph
from trajan.synapse_table import SynapseTable


@pytest.fixture
def base_synapses():
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "pre_pt_root_id": [10, 10, 20, 20, 30],
            "post_pt_root_id": [20, 30, 10, 30, 10],
        }
    )


@pytest.fixture
def st(base_synapses):
    return SynapseTable(base_synapses)


# ── repr / n_syn ───────────────────────────────────────────────────────────────


def test_repr_shows_count_without_cache(st):
    """n_syn should be available in repr without triggering a collect."""
    r = repr(st)
    assert "n_syn=5" in r
    assert st._cache is None  # no side-effect collect


def test_repr_shows_count_after_annotation(st):
    """Adding an annotation invalidates cache but n_syn should still be visible."""
    cell_ann = pl.DataFrame({"root_id": [10, 20, 30], "cell_type": ["A", "B", "C"]})
    st.add_cell_annotation("types", cell_ann, cell_id_col="root_id")
    assert st._cache is None
    assert "n_syn=5" in repr(st)


def test_repr_uncached_when_filtered(st):
    """After a filter with no collected cache, repr should say uncached."""
    filtered = st.filter(pl.col("pre_pt_root_id") == 10)
    assert "uncached" in repr(filtered)


def test_repr_shows_count_after_collect(st):
    """After accessing .synapses the repr should reflect the actual count."""
    _ = st.synapses
    assert "n_syn=5" in repr(st)


# ── duplicate join-key validation ──────────────────────────────────────────────


def test_synapse_annotation_duplicate_id_raises(st):
    """add_synapse_annotation must reject annotation with duplicate id values."""
    dup_ann = pl.DataFrame({"id": [1, 1, 2], "score": [0.1, 0.9, 0.5]})
    with pytest.raises(ValueError, match="duplicate"):
        st.add_synapse_annotation("scores", dup_ann)


def test_cell_annotation_duplicate_id_raises(st):
    """add_cell_annotation must reject annotation with duplicate cell_id values."""
    dup_ann = pl.DataFrame({"root_id": [10, 10, 20], "cell_type": ["A", "A2", "B"]})
    with pytest.raises(ValueError, match="duplicate"):
        st.add_cell_annotation("types", dup_ann, cell_id_col="root_id")


def test_vertex_annotation_duplicate_id_raises(st):
    """add_vertex_annotation must reject annotation with duplicate vertex_id values."""
    dup_ann = pl.DataFrame({"vid": [100, 100], "label": ["x", "y"]})
    # add a vertex column to the base table first
    synapses = pl.DataFrame(
        {
            "id": [1, 2],
            "pre_pt_root_id": [10, 20],
            "post_pt_root_id": [20, 10],
            "pre_vid": [100, 200],
        }
    )
    st2 = SynapseTable(synapses)
    with pytest.raises(ValueError, match="duplicate"):
        st2.add_vertex_annotation(
            "labels", dup_ann, vertex_id_col="vid", pre_vertex_col="pre_vid"
        )


def test_synapse_annotation_does_not_expand_rows(st):
    """A valid synapse annotation must not change the synapse count."""
    ann = pl.DataFrame({"id": [1, 2, 3, 4, 5], "score": [0.1, 0.2, 0.3, 0.4, 0.5]})
    st.add_synapse_annotation("scores", ann)
    assert len(st.synapses) == 5


def test_cell_annotation_does_not_expand_rows(st):
    """A valid cell annotation must not change the synapse count."""
    cell_ann = pl.DataFrame({"root_id": [10, 20, 30], "cell_type": ["A", "B", "C"]})
    st.add_cell_annotation("types", cell_ann, cell_id_col="root_id")
    assert len(st.synapses) == 5


def test_cell_annotation_partial_coverage_does_not_expand_rows(st):
    """Cell annotation that doesn't cover all ids leaves nulls, not extra rows."""
    partial_ann = pl.DataFrame({"root_id": [10, 20], "cell_type": ["A", "B"]})
    st.add_cell_annotation("types", partial_ann, cell_id_col="root_id")
    result = st.synapses
    assert len(result) == 5
    # cell 30 has no annotation — should produce nulls on post side for row where post=30
    assert result["cell_type_post"].null_count() > 0


# ── extend_cell_annotation ─────────────────────────────────────────────────────


def test_extend_cell_annotation_adds_columns(st):
    """extend_cell_annotation joins new columns into an existing annotation."""
    cells = pl.DataFrame({"root_id": [10, 20, 30], "nucleus_id": [100, 200, 300]})
    types = pl.DataFrame({"nucleus_id": [100, 200], "cell_type": ["A", "B"]})
    st.add_cell_annotation("cells", cells, cell_id_col="root_id")
    st.extend_cell_annotation("cells", types, on="nucleus_id")
    result = st.synapses
    assert len(result) == 5
    assert "cell_type_pre" in result.columns
    assert "cell_type_post" in result.columns


def test_extend_cell_annotation_partial_coverage_nulls(st):
    """Cells without a match in extend df get nulls, not dropped rows."""
    cells = pl.DataFrame({"root_id": [10, 20, 30], "nucleus_id": [100, 200, 300]})
    types = pl.DataFrame({"nucleus_id": [100, 200], "cell_type": ["A", "B"]})
    st.add_cell_annotation("cells", cells, cell_id_col="root_id")
    st.extend_cell_annotation("cells", types, on="nucleus_id")
    result = st.synapses
    # root_id=30 has no cell_type — expect nulls on whichever side it appears
    assert (
        result["cell_type_pre"].null_count() > 0
        or result["cell_type_post"].null_count() > 0
    )


def test_extend_cell_annotation_duplicate_key_raises(st):
    """extend_cell_annotation rejects df with duplicate `on` values."""
    cells = pl.DataFrame({"root_id": [10, 20, 30], "nucleus_id": [100, 200, 300]})
    types = pl.DataFrame({"nucleus_id": [100, 100], "cell_type": ["A", "B"]})
    st.add_cell_annotation("cells", cells, cell_id_col="root_id")
    with pytest.raises(ValueError, match="duplicate"):
        st.extend_cell_annotation("cells", types, on="nucleus_id")


def test_extend_cell_annotation_unknown_on_raises(st):
    """extend_cell_annotation raises if `on` is not in the existing annotation."""
    cells = pl.DataFrame({"root_id": [10, 20, 30], "nucleus_id": [100, 200, 300]})
    st.add_cell_annotation("cells", cells, cell_id_col="root_id")
    types = pl.DataFrame({"other_id": [100], "cell_type": ["A"]})
    with pytest.raises(ValueError, match="not found in annotation"):
        st.extend_cell_annotation("cells", types, on="other_id")


def test_extend_cell_annotation_unknown_name_raises(st):
    """extend_cell_annotation raises KeyError for an unregistered annotation name."""
    types = pl.DataFrame({"nucleus_id": [100], "cell_type": ["A"]})
    with pytest.raises(KeyError):
        st.extend_cell_annotation("nonexistent", types, on="nucleus_id")


def test_extend_cell_annotation_column_collision_raises(st):
    """extend_cell_annotation raises if new columns collide with existing table columns."""
    cells = pl.DataFrame(
        {
            "root_id": [10, 20, 30],
            "nucleus_id": [100, 200, 300],
            "cell_type": ["A", "B", "C"],
        }
    )
    st.add_cell_annotation("cells", cells, cell_id_col="root_id")
    extra = pl.DataFrame({"nucleus_id": [100, 200, 300], "cell_type": ["X", "Y", "Z"]})
    with pytest.raises(ValueError, match="already exist"):
        st.extend_cell_annotation("cells", extra, on="nucleus_id")


# ── edgelist annotation columns ──────────────────────────────────────────────


@pytest.fixture
def st_with_cell_ann(st):
    """SynapseTable with a cell annotation registered."""
    cell_ann = pl.DataFrame({"root_id": [10, 20, 30], "cell_type": ["A", "B", "C"]})
    st.add_cell_annotation("types", cell_ann, cell_id_col="root_id")
    return st


def test_edgelist_includes_anno_columns_by_default(st_with_cell_ann):
    """edgelist() includes cell annotation columns for both sides by default."""
    el = st_with_cell_ann.edgelist().pairs
    assert "cell_type_pre" in el.columns
    assert "cell_type_post" in el.columns


def test_edgelist_no_anno_columns(st_with_cell_ann):
    """edgelist(pre_anno=False, post_anno=False) excludes annotation columns."""
    el = st_with_cell_ann.edgelist(pre_anno=False, post_anno=False).pairs
    assert "cell_type_pre" not in el.columns
    assert "cell_type_post" not in el.columns
    assert "n_syn" in el.columns


def test_edgelist_pre_anno_only(st_with_cell_ann):
    """edgelist(post_anno=False) includes only pre-side annotation columns."""
    el = st_with_cell_ann.edgelist(pre_anno=True, post_anno=False).pairs
    assert "cell_type_pre" in el.columns
    assert "cell_type_post" not in el.columns


def test_edgelist_post_anno_only(st_with_cell_ann):
    """edgelist(pre_anno=False) includes only post-side annotation columns."""
    el = st_with_cell_ann.edgelist(pre_anno=False, post_anno=True).pairs
    assert "cell_type_pre" not in el.columns
    assert "cell_type_post" in el.columns


def test_edgelist_anno_values_correct(st_with_cell_ann):
    """Annotation values in edgelist match the cell annotation data."""
    el = st_with_cell_ann.edgelist().pairs.sort(
        [st_with_cell_ann._pre_col, st_with_cell_ann._post_col]
    )
    # pre=10 -> "A", pre=20 -> "B", pre=30 -> "C"
    for row in el.iter_rows(named=True):
        pre_id = row["pre_pt_root_id"]
        expected = {10: "A", 20: "B", 30: "C"}
        assert row["cell_type_pre"] == expected[pre_id]
        post_id = row["post_pt_root_id"]
        assert row["cell_type_post"] == expected[post_id]


def test_edgelist_no_cell_annotations(st):
    """edgelist() with no cell annotations and pre_anno/post_anno=True works fine."""
    el = st.edgelist().pairs
    assert "n_syn" in el.columns
    assert len(el.columns) == 3  # pre_col, post_col, n_syn


# ── to_graph ──────────────────────────────────────────────────────────────────


@pytest.fixture
def st_graph(base_synapses):
    """SynapseTable with a cell annotation, ready for graph export."""
    st = SynapseTable(base_synapses)
    cell_ann = pl.DataFrame({"root_id": [10, 20, 30], "cell_type": ["A", "B", "C"]})
    st.add_cell_annotation("types", cell_ann, cell_id_col="root_id")
    return st


def test_to_graph_invalid_backend(st_graph):
    """Unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="backend"):
        to_graph(st_graph, backend="gephi")


def test_to_graph_missing_networkx(st_graph):
    """Missing networkx raises ImportError with install hint."""
    with patch.dict(sys.modules, {"networkx": None}):
        with pytest.raises(ImportError, match="networkx"):
            to_graph(st_graph, backend="networkx")


def test_to_graph_missing_igraph(st_graph):
    """Missing igraph raises ImportError with install hint."""
    with patch.dict(sys.modules, {"igraph": None}):
        with pytest.raises(ImportError, match="igraph"):
            to_graph(st_graph, backend="igraph")


def test_to_graph_missing_scipy(st_graph):
    """Missing scipy raises ImportError with install hint."""
    with patch.dict(sys.modules, {"scipy.sparse": None, "scipy": None}):
        with pytest.raises(ImportError, match="[Ss]ci[Pp]y"):
            to_graph(st_graph, backend="csgraph")


def test_to_graph_networkx_structure(st_graph):
    """networkx graph has correct node/edge counts and attributes."""
    nx = pytest.importorskip("networkx")
    G = to_graph(st_graph, backend="networkx")
    assert isinstance(G, nx.DiGraph)
    assert set(G.nodes) == {10, 20, 30}
    assert G.number_of_edges() == 5  # 5 unique pre/post pairs in base_synapses
    # cell annotation as node attr (suffix stripped)
    assert G.nodes[10]["cell_type"] == "A"
    assert G.nodes[20]["cell_type"] == "B"
    # n_syn as edge attr
    assert G.edges[10, 20]["n_syn"] == 1


def test_to_graph_igraph_structure(st_graph):
    """igraph graph has correct vertex/edge counts and attributes."""
    igraph = pytest.importorskip("igraph")
    g = to_graph(st_graph, backend="igraph")
    assert isinstance(g, igraph.Graph)
    assert g.is_directed()
    assert g.vcount() == 3
    assert g.ecount() == 5
    assert set(g.vs["name"]) == {10, 20, 30}
    assert set(g.vs["cell_type"]) == {"A", "B", "C"}
    assert all(w >= 1 for w in g.es["n_syn"])


def test_to_graph_csgraph_structure(st_graph):
    """csgraph returns a (sparse_matrix, cell_ids) tuple with correct shape."""
    pytest.importorskip("scipy")
    mat, cell_ids = to_graph(st_graph, backend="csgraph")
    assert len(cell_ids) == 3
    assert mat.shape == (3, 3)
    # total synapse count across all edges should equal 5 (one per synapse row)
    assert mat.sum() == 5


def test_to_graph_cell_agg_node_attrs(st_graph):
    """cell_agg results appear as node attributes."""
    nx = pytest.importorskip("networkx")
    G = to_graph(st_graph, cell_agg={"total_syn": pl.len()}, backend="networkx")
    for node in G.nodes:
        assert "total_syn" in G.nodes[node]
        assert G.nodes[node]["total_syn"] >= 1


def test_to_graph_edge_agg(st_graph):
    """edge_agg columns appear as edge attributes."""
    nx = pytest.importorskip("networkx")
    # add a size column to work with
    syn_with_size = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "pre_pt_root_id": [10, 10, 20, 20, 30],
            "post_pt_root_id": [20, 30, 10, 30, 10],
            "size": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    st2 = SynapseTable(syn_with_size)
    G = to_graph(st2, edge_agg={"mean_size": pl.mean("size")}, backend="networkx")
    for _, _, data in G.edges(data=True):
        assert "mean_size" in data


# ── expression classification / edgelist propagation ─────────────────────────


@pytest.fixture
def st_with_depth(base_synapses):
    """SynapseTable with a soma-depth cell annotation and derived expressions."""
    st = SynapseTable(base_synapses)
    soma_ann = pl.DataFrame(
        {
            "root_id": [10, 20, 30],
            "depth": [100.0, 200.0, 300.0],
        }
    )
    st.add_cell_annotation("soma", soma_ann, cell_id_col="root_id")
    return st


def test_expression_classified_pre(st_with_depth):
    """Expression referencing only a *_pre cell annotation column → side 'pre'."""
    st_with_depth.add_expression("depth_pre_scaled", pl.col("depth_pre") * 2)
    assert st_with_depth.expression_sides["depth_pre_scaled"] == "pre"


def test_expression_classified_post(st_with_depth):
    """Expression referencing only a *_post cell annotation column → side 'post'."""
    st_with_depth.add_expression("depth_post_scaled", pl.col("depth_post") * 2)
    assert st_with_depth.expression_sides["depth_post_scaled"] == "post"


def test_expression_classified_both(st_with_depth):
    """Expression referencing both *_pre and *_post cell annotation columns → side 'both'."""
    st_with_depth.add_expression(
        "depth_diff", pl.col("depth_post") - pl.col("depth_pre")
    )
    assert st_with_depth.expression_sides["depth_diff"] == "both"


def test_expression_classified_none_synapse_col(st_with_depth):
    """Expression referencing a synapse column → side None."""
    st_with_depth.add_expression("id_doubled", pl.col("id") * 2)
    assert st_with_depth.expression_sides["id_doubled"] is None


def test_expression_classified_none_mixed_cell_and_synapse(st_with_depth):
    """Expression referencing both a cell annotation col and a synapse col → None."""
    st_with_depth.add_expression(
        "mixed", pl.col("depth_pre") + pl.col("pre_pt_root_id")
    )
    assert st_with_depth.expression_sides["mixed"] is None


def test_expression_classified_transitive_pre(st_with_depth):
    """Expression referencing a pre-side expression → side 'pre' (transitive)."""
    st_with_depth.add_expression("depth_pre_scaled", pl.col("depth_pre") * 2)
    st_with_depth.add_expression("depth_pre_final", pl.col("depth_pre_scaled") + 1)
    assert st_with_depth.expression_sides["depth_pre_final"] == "pre"


def test_expression_classified_transitive_none(st_with_depth):
    """Expression referencing a non-cell-level expression → None (transitive)."""
    st_with_depth.add_expression("id_doubled", pl.col("id") * 2)
    st_with_depth.add_expression("id_tripled", pl.col("id_doubled") + pl.col("id"))
    assert st_with_depth.expression_sides["id_tripled"] is None


def test_expression_sides_property(st_with_depth):
    """expression_sides returns a copy of the classification dict."""
    st_with_depth.add_expression("depth_pre_scaled", pl.col("depth_pre") * 2)
    sides = st_with_depth.expression_sides
    assert isinstance(sides, dict)
    assert "depth_pre_scaled" in sides
    # modifying returned dict doesn't affect internal state
    sides["depth_pre_scaled"] = "post"
    assert st_with_depth.expression_sides["depth_pre_scaled"] == "pre"


def test_edgelist_includes_pre_expression(st_with_depth):
    """edgelist() includes pre-side expression column when pre_anno=True."""
    st_with_depth.add_expression("depth_pre_scaled", pl.col("depth_pre") * 2)
    el = st_with_depth.edgelist().pairs
    assert "depth_pre_scaled" in el.columns


def test_edgelist_includes_post_expression(st_with_depth):
    """edgelist() includes post-side expression column when post_anno=True."""
    st_with_depth.add_expression("depth_post_scaled", pl.col("depth_post") * 2)
    el = st_with_depth.edgelist().pairs
    assert "depth_post_scaled" in el.columns


def test_edgelist_includes_both_expression(st_with_depth):
    """edgelist() includes 'both'-side expression column when pre_anno or post_anno=True."""
    st_with_depth.add_expression(
        "depth_diff", pl.col("depth_post") - pl.col("depth_pre")
    )
    el = st_with_depth.edgelist().pairs
    assert "depth_diff" in el.columns


def test_edgelist_excludes_pre_expression_when_no_pre_anno(st_with_depth):
    """edgelist(pre_anno=False) excludes pre-side expression columns."""
    st_with_depth.add_expression("depth_pre_scaled", pl.col("depth_pre") * 2)
    el = st_with_depth.edgelist(pre_anno=False).pairs
    assert "depth_pre_scaled" not in el.columns


def test_edgelist_excludes_post_expression_when_no_post_anno(st_with_depth):
    """edgelist(post_anno=False) excludes post-side expression columns."""
    st_with_depth.add_expression("depth_post_scaled", pl.col("depth_post") * 2)
    el = st_with_depth.edgelist(post_anno=False).pairs
    assert "depth_post_scaled" not in el.columns


def test_edgelist_excludes_synapse_level_expression(st_with_depth):
    """edgelist() never auto-includes expressions that reference synapse columns."""
    st_with_depth.add_expression("id_doubled", pl.col("id") * 2)
    el = st_with_depth.edgelist().pairs
    assert "id_doubled" not in el.columns


def test_edgelist_expression_value_correct(st_with_depth):
    """Expression values in edgelist match the expected per-cell values."""
    st_with_depth.add_expression("depth_pre_scaled", pl.col("depth_pre") * 2)
    el = st_with_depth.edgelist().pairs.sort(
        [st_with_depth._pre_col, st_with_depth._post_col]
    )
    # depth_pre for cell 10 = 100.0 → scaled = 200.0
    # depth_pre for cell 20 = 200.0 → scaled = 400.0
    # depth_pre for cell 30 = 300.0 → scaled = 600.0
    expected = {10: 200.0, 20: 400.0, 30: 600.0}
    for row in el.iter_rows(named=True):
        assert row["depth_pre_scaled"] == expected[row["pre_pt_root_id"]]


# ── weights ───────────────────────────────────────────────────────────────────


@pytest.fixture
def st_with_size(base_synapses):
    """SynapseTable with a per-synapse size column registered as a weight."""
    syn = base_synapses.with_columns(pl.Series("size", [1.0, 2.0, 3.0, 4.0, 5.0]))
    return SynapseTable(syn)


def test_add_weight_unknown_col_raises(st_with_size):
    """add_weight raises ValueError for a column not in the table."""
    with pytest.raises(ValueError, match="not found"):
        st_with_size.add_weight("nonexistent")


def test_add_weight_duplicate_raises(st_with_size):
    """add_weight raises ValueError if the column is already registered."""
    st_with_size.add_weight("size")
    with pytest.raises(ValueError, match="already registered"):
        st_with_size.add_weight("size")


def test_remove_weight_unknown_raises(st_with_size):
    """remove_weight raises KeyError for a column not registered as a weight."""
    with pytest.raises(KeyError):
        st_with_size.remove_weight("size")


def test_weights_property(st_with_size):
    """weights property returns a copy of the registered weight list."""
    st_with_size.add_weight("size")
    w = st_with_size.weights
    assert w == ["size"]
    w.append("other")
    assert st_with_size.weights == ["size"]  # copy, not a reference


def test_edgelist_includes_weight_sum(st_with_size):
    """Registered weight appears summed in edgelist alongside n_syn."""
    st_with_size.add_weight("size")
    el = st_with_size.edgelist(pre_anno=False, post_anno=False).pairs
    assert "n_syn" in el.columns
    assert "size" in el.columns


def test_edgelist_weight_values_correct(st_with_size):
    """Weight column in edgelist is summed per cell pair, not .first()."""
    st_with_size.add_weight("size")
    el = st_with_size.edgelist(pre_anno=False, post_anno=False).pairs.sort(
        ["pre_pt_root_id", "post_pt_root_id"]
    )
    # pre=10, post=20: synapse 1 → size=1.0
    # pre=10, post=30: synapse 2 → size=2.0
    # pre=20, post=10: synapse 3 → size=3.0
    # pre=20, post=30: synapse 4 → size=4.0
    # pre=30, post=10: synapse 5 → size=5.0
    expected = {
        (10, 20): 1.0,
        (10, 30): 2.0,
        (20, 10): 3.0,
        (20, 30): 4.0,
        (30, 10): 5.0,
    }
    for row in el.iter_rows(named=True):
        key = (row["pre_pt_root_id"], row["post_pt_root_id"])
        assert row["size"] == pytest.approx(expected[key])


def test_type_edgelist_includes_weight_sum(st_with_size):
    """Registered weight appears summed in type_edgelist."""
    cell_ann = pl.DataFrame({"root_id": [10, 20, 30], "ct": ["A", "B", "A"]})
    st_with_size.add_cell_annotation("types", cell_ann, cell_id_col="root_id")
    st_with_size.add_weight("size")
    el = st_with_size.type_edgelist("ct_pre").pairs
    assert "n_syn" in el.columns
    assert "size" in el.columns


def test_edgelist_to_dense_weight_column(st_with_size):
    """EdgeList.to_dense(values=weight) produces a dense matrix with totals."""
    st_with_size.add_weight("size")
    mat = st_with_size.edgelist(pre_anno=False, post_anno=False).to_dense(values="size")
    assert mat.shape[0] > 0
    total = sum(mat.select(pl.exclude("pre_pt_root_id")).sum().row(0))
    assert total == pytest.approx(15.0)  # 1+2+3+4+5


def test_remove_weight(st_with_size):
    """After remove_weight the column no longer appears in edgelist automatically."""
    st_with_size.add_weight("size")
    st_with_size.remove_weight("size")
    el = st_with_size.edgelist(pre_anno=False, post_anno=False).pairs
    assert "size" not in el.columns


# ── annotation proxies ────────────────────────────────────────────────────────


def test_cell_annotations_proxy_returns_dataframe(st_with_cell_ann):
    """st.cell_annotations["name"] returns a collected pl.DataFrame."""
    df = st_with_cell_ann.cell_annotations["types"]
    assert isinstance(df, pl.DataFrame)
    assert "cell_type" in df.columns


def test_cell_annotations_proxy_unknown_raises(st_with_cell_ann):
    """st.cell_annotations["unknown"] raises KeyError."""
    with pytest.raises(KeyError):
        _ = st_with_cell_ann.cell_annotations["nonexistent"]


def test_cell_annotations_proxy_contains(st_with_cell_ann):
    """'name' in st.cell_annotations works without collecting."""
    assert "types" in st_with_cell_ann.cell_annotations
    assert "other" not in st_with_cell_ann.cell_annotations


def test_cell_annotations_proxy_iter(st_with_cell_ann):
    """Iterating st.cell_annotations yields annotation names."""
    assert list(st_with_cell_ann.cell_annotations) == ["types"]


def test_cell_annotations_proxy_len(st_with_cell_ann):
    """len(st.cell_annotations) equals number of registered cell annotations."""
    assert len(st_with_cell_ann.cell_annotations) == 1


def test_synapse_annotations_proxy(st):
    """st.synapse_annotations["name"] returns the annotation DataFrame."""
    ann = pl.DataFrame({"id": [1, 2, 3, 4, 5], "score": [0.1, 0.2, 0.3, 0.4, 0.5]})
    st.add_synapse_annotation("scores", ann)
    df = st.synapse_annotations["scores"]
    assert isinstance(df, pl.DataFrame)
    assert "score" in df.columns


def test_vertex_annotations_proxy(base_synapses):
    """st.vertex_annotations["name"] returns the annotation DataFrame."""
    syn = base_synapses.with_columns(pl.Series("pre_vid", [100, 100, 200, 200, 300]))
    st2 = SynapseTable(syn)
    ann = pl.DataFrame({"vid": [100, 200, 300], "label": ["x", "y", "z"]})
    st2.add_vertex_annotation(
        "labels", ann, vertex_id_col="vid", pre_vertex_col="pre_vid"
    )
    df = st2.vertex_annotations["labels"]
    assert isinstance(df, pl.DataFrame)
    assert "label" in df.columns


# ── cell_summary ──────────────────────────────────────────────────────────────


@pytest.fixture
def st_summary(base_synapses):
    """SynapseTable with cell annotation and a size weight for cell_summary tests."""
    st = SynapseTable(
        base_synapses.with_columns(pl.Series("size", [1.0, 2.0, 3.0, 4.0, 5.0]))
    )
    cell_ann = pl.DataFrame({"root_id": [10, 20, 30], "cell_type": ["A", "B", "C"]})
    st.add_cell_annotation("types", cell_ann, cell_id_col="root_id")
    st.add_weight("size")
    return st


def test_cell_summary_row_count(st_summary):
    """cell_summary returns one row per unique cell."""
    cs = cell_summary(st_summary)
    assert len(cs) == 3
    assert set(cs["cell_id"].to_list()) == {10, 20, 30}


def test_cell_summary_n_syn_totals(st_summary):
    """n_syn_output + n_syn_input across all cells equals 2 * total synapses."""
    cs = cell_summary(st_summary)
    total_out = cs["n_syn_output"].fill_null(0).sum()
    total_in = cs["n_syn_input"].fill_null(0).sum()
    assert total_out == 5  # 5 synapses, each contributes 1 to output
    assert total_in == 5  # 5 synapses, each contributes 1 to input


def test_cell_summary_weight_sums(st_summary):
    """Weight sums in cell_summary are correct per cell per direction."""
    cs = cell_summary(st_summary).sort("cell_id")
    # pre=10: synapses 1,2 → size=1+2=3 output
    # pre=20: synapses 3,4 → size=3+4=7 output
    # pre=30: synapse 5 → size=5 output
    row = {r["cell_id"]: r for r in cs.iter_rows(named=True)}
    assert row[10]["size_output"] == pytest.approx(3.0)
    assert row[20]["size_output"] == pytest.approx(7.0)
    assert row[30]["size_output"] == pytest.approx(5.0)
    # post=20: synapse 1 → size=1 input
    # post=30: synapses 2,4 → size=2+4=6 input
    # post=10: synapses 3,5 → size=3+5=8 input
    assert row[20]["size_input"] == pytest.approx(1.0)
    assert row[30]["size_input"] == pytest.approx(6.0)
    assert row[10]["size_input"] == pytest.approx(8.0)


def test_cell_summary_annotation_columns(st_summary):
    """Cell annotation columns appear in cell_summary with suffix stripped."""
    cs = cell_summary(st_summary)
    assert "cell_type" in cs.columns
    row = {r["cell_id"]: r["cell_type"] for r in cs.iter_rows(named=True)}
    assert row[10] == "A"
    assert row[20] == "B"
    assert row[30] == "C"


def test_cell_summary_no_annotations(st_summary):
    """include_annotations=False omits cell annotation columns."""
    cs = cell_summary(st_summary, include_annotations=False)
    assert "cell_type" not in cs.columns
    assert "n_syn_output" in cs.columns


def test_cell_summary_custom_agg(st_summary):
    """Custom pre_agg and post_agg columns appear in output."""
    cs = cell_summary(
        st_summary,
        pre_agg={"mean_size_out": pl.mean("size")},
        post_agg={"mean_size_in": pl.mean("size")},
    )
    assert "mean_size_out" in cs.columns
    assert "mean_size_in" in cs.columns


def test_cell_summary_null_for_absent_side(base_synapses):
    """Cells that only appear on one side have nulls for the other side's count."""
    # Cell 40 only appears as post, cell 30 only as pre in this subset
    syn = pl.DataFrame(
        {
            "id": [1, 2],
            "pre_pt_root_id": [10, 10],
            "post_pt_root_id": [20, 40],
        }
    )
    st2 = SynapseTable(syn)
    cs = cell_summary(st2, include_annotations=False)
    row = {r["cell_id"]: r for r in cs.iter_rows(named=True)}
    # Cell 40 never appears as pre
    assert row[40]["n_syn_output"] is None
    # Cell 20 never appears as pre
    assert row[20]["n_syn_output"] is None


# ── metadata ──────────────────────────────────────────────────────────────────


def test_metadata_default_empty(st):
    """metadata is an empty dict by default."""
    assert st.metadata == {}


def test_metadata_mutable(st):
    """metadata can be mutated directly."""
    st.metadata["description"] = "test table"
    assert st.metadata["description"] == "test table"


def test_metadata_copy_is_independent(st):
    """Copying the SynapseTable does not share the metadata dict."""
    st.metadata["key"] = "original"
    filtered = st.filter(pl.col("pre_pt_root_id") == 10)
    filtered.metadata["key"] = "modified"
    assert st.metadata["key"] == "original"


def test_metadata_view_is_independent(st):
    """view() carries metadata but mutations are independent."""
    st.metadata["key"] = "original"
    v = st.view()
    v.metadata["key"] = "modified"
    assert st.metadata["key"] == "original"


def test_metadata_save_load_roundtrip(base_synapses):
    """metadata round-trips through save()/load() via folio.metadata."""
    import tempfile

    import datafolio

    with tempfile.TemporaryDirectory() as d:
        folio = datafolio.DataFolio(d + "/folio")
        st = SynapseTable(base_synapses)
        st.metadata["description"] = "V1DD dataset"
        st.metadata["version"] = "2026-04"
        st.save(folio)
        st2 = SynapseTable.load(folio)
        assert st2.metadata["description"] == "V1DD dataset"
        assert st2.metadata["version"] == "2026-04"


def test_save_load_with_path(base_synapses):
    """save() and load() accept str/Path in addition to DataFolio instances."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        folio_path = Path(d) / "folio"
        st = SynapseTable(base_synapses)
        st.metadata["note"] = "path test"
        st.save(folio_path)
        st2 = SynapseTable.load(str(folio_path))
        assert st2.synapses.shape == st.synapses.shape
        assert st2.metadata["note"] == "path test"


# ── add_spatial_features ───────────────────────────────────────────────────────


@pytest.fixture
def st_spatial(base_synapses):
    """SynapseTable with soma positions and synapse positions configured."""
    # Cells 10, 20, 30 with soma positions
    soma_df = pl.DataFrame(
        {
            "root_id": [10, 20, 30],
            "soma_x": [0.0, 3.0, 0.0],
            "soma_y": [0.0, 0.0, 4.0],
            "soma_z": [0.0, 0.0, 0.0],
        }
    )
    # Synapse positions (one per synapse id 1..5)
    syn_pos_df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "ctr_pt_position_x": [1.5, 1.5, 1.5, 1.5, 0.0],
            "ctr_pt_position_y": [0.0, 0.0, 0.0, 2.0, 2.0],
            "ctr_pt_position_z": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    return (
        SynapseTable(
            base_synapses,
            synapse_position_col="ctr_pt_position",
        )
        .add_synapse_annotation("ctr_pt", syn_pos_df, position_cols="ctr_pt_position")
        .add_cell_annotation(
            "soma",
            soma_df,
            cell_id_col="root_id",
            position_cols="soma",
            position_col="soma",
        )
    )


def test_add_spatial_features_default_columns(st_spatial):
    st_spatial.add_spatial_features(prefix="soma")
    cols = st_spatial.synapses.columns
    for feat in (
        "soma_euclidean",
        "soma_depth_diff",
        "soma_r",
        "soma_theta",
        "soma_phi",
        "soma_rho",
        "soma_dy",
    ):
        assert feat in cols, f"missing {feat}"


def test_add_spatial_features_phi_not_duplicated(st_spatial):
    st_spatial.add_spatial_features(prefix="s")
    assert st_spatial.synapses.columns.count("s_phi") == 1


def test_add_spatial_features_euclidean_value(st_spatial):
    import math

    st_spatial.add_spatial_features(prefix="soma")
    syn = st_spatial.synapses
    # Synapse 1: pre=cell10 (0,0,0), post=cell20 (3,0,0) → euclidean=3
    row = syn.filter(pl.col("id") == 1).row(0, named=True)
    assert math.isclose(row["soma_euclidean"], 3.0)


def test_add_spatial_features_post_pre_negates_depth(st_spatial):
    import math

    st_spatial.add_spatial_features(prefix="fwd")
    st_spatial.add_spatial_features(prefix="rev", center="post", target="pre")
    syn = st_spatial.synapses
    # Synapse 5: pre=cell30 (0,4,0), post=cell10 (0,0,0) → fwd depth_diff = -4
    row = syn.filter(pl.col("id") == 5).row(0, named=True)
    assert math.isclose(row["fwd_depth_diff"], -4.0)
    assert math.isclose(row["rev_depth_diff"], 4.0)
    assert math.isclose(row["fwd_euclidean"], row["rev_euclidean"])


def test_add_spatial_features_target_syn(st_spatial):
    import math

    st_spatial.add_spatial_features(prefix="pre_syn", center="pre", target="syn")
    cols = st_spatial.synapses.columns
    for feat in ("pre_syn_euclidean", "pre_syn_depth_diff"):
        assert feat in cols


def test_add_spatial_features_soma_soma_in_edgelist(st_spatial):
    """Soma→soma features (classified 'both') appear in edgelist."""
    st_spatial.add_spatial_features(prefix="soma")
    el = st_spatial.edgelist().pairs
    assert "soma_euclidean" in el.columns


def test_add_spatial_features_syn_not_in_edgelist(st_spatial):
    """Soma→synapse features (classified None) do not appear in edgelist."""
    st_spatial.add_spatial_features(prefix="pre_syn", center="pre", target="syn")
    el = st_spatial.edgelist().pairs
    assert "pre_syn_euclidean" not in el.columns


def test_add_spatial_features_center_equals_target_raises(st_spatial):
    with pytest.raises(ValueError, match="center and target must differ"):
        st_spatial.add_spatial_features(center="pre", target="pre")


def test_add_spatial_features_no_soma_config_raises(base_synapses):
    st = SynapseTable(base_synapses)
    with pytest.raises(ValueError, match="position_col"):
        st.add_spatial_features()


def test_add_spatial_features_target_syn_no_synapse_col_raises(base_synapses):
    soma_df = pl.DataFrame(
        {
            "root_id": [10, 20, 30],
            "soma_x": [0.0] * 3,
            "soma_y": [0.0] * 3,
            "soma_z": [0.0] * 3,
        }
    )
    st = SynapseTable(base_synapses).add_cell_annotation(
        "soma",
        soma_df,
        cell_id_col="root_id",
        position_cols="soma",
        position_col="soma",
    )
    with pytest.raises(ValueError, match="synapse_position_col"):
        st.add_spatial_features(target="syn")


def test_resolve_universe_annotation_single(base_synapses):
    cells = pl.DataFrame({"root_id": [10, 20, 30, 40], "ct": ["a", "b", "c", "d"]})
    st = SynapseTable(base_synapses).add_cell_annotation(
        "cells", cells, cell_id_col="root_id", is_universe=True
    )
    assert st._resolve_universe_annotation() == "cells"


def test_resolve_universe_annotation_zero_raises(base_synapses):
    st = SynapseTable(base_synapses)
    with pytest.raises(ValueError, match="No cell annotation is marked is_universe"):
        st._resolve_universe_annotation()


def test_resolve_universe_annotation_ambiguous_raises(base_synapses):
    a = pl.DataFrame({"root_id": [10, 20], "ta": ["x"] * 2})
    b = pl.DataFrame({"root_id": [30, 40], "tb": ["y"] * 2})
    st = (
        SynapseTable(base_synapses)
        .add_cell_annotation("a", a, cell_id_col="root_id", is_universe=True)
        .add_cell_annotation("b", b, cell_id_col="root_id", is_universe=True)
    )
    with pytest.raises(
        ValueError, match="Multiple cell annotations are marked is_universe"
    ):
        st._resolve_universe_annotation()
    assert st._resolve_universe_annotation("a") == "a"


def test_is_universe_persists_through_save_load(base_synapses, tmp_path):
    cells = pl.DataFrame({"root_id": [10, 20, 30, 40], "ct": ["a", "b", "c", "d"]})
    st = SynapseTable(base_synapses).add_cell_annotation(
        "cells", cells, cell_id_col="root_id", is_universe=True
    )
    folio_path = tmp_path / "folio"
    st.save(str(folio_path))
    loaded = SynapseTable.load(str(folio_path))
    assert loaded._cell_annotations["cells"].is_universe is True


def test_extend_cell_annotation_preserves_is_universe(base_synapses):
    cells = pl.DataFrame({"root_id": [10, 20, 30, 40], "ct": ["a", "b", "c", "d"]})
    more = pl.DataFrame({"root_id": [10, 20, 30, 40], "extra": [1, 2, 3, 4]})
    st = SynapseTable(base_synapses).add_cell_annotation(
        "cells", cells, cell_id_col="root_id", is_universe=True
    )
    st.extend_cell_annotation("cells", more, on="root_id")
    assert st._cell_annotations["cells"].is_universe is True


def test_resolve_position_annotation_ambiguous_raises(base_synapses):
    soma_a = pl.DataFrame(
        {"root_id": [10, 20, 30], "pos_a": [{"x": 0.0, "y": 0.0, "z": 0.0}] * 3}
    )
    soma_b = pl.DataFrame(
        {"root_id": [10, 20, 30], "pos_b": [{"x": 0.0, "y": 0.0, "z": 0.0}] * 3}
    )
    st = (
        SynapseTable(base_synapses)
        .add_cell_annotation("a", soma_a, cell_id_col="root_id", position_col="pos_a")
        .add_cell_annotation("b", soma_b, cell_id_col="root_id", position_col="pos_b")
    )
    with pytest.raises(ValueError, match="Multiple cell annotations"):
        st.filter_by_soma_distance(10.0)
    # Disambiguating works
    out = st.filter_by_soma_distance(10.0, annotation="a")
    assert isinstance(out, SynapseTable)


# ── filter_by_min_synapses ─────────────────────────────────────────────────────


# Pairs: (10,20)=3 syn size=6, (20,30)=1 syn size=5, (30,10)=2 syn size=2
@pytest.fixture
def st_multi_syn():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "pre_pt_root_id": [10, 10, 10, 20, 30, 30],
            "post_pt_root_id": [20, 20, 20, 30, 10, 10],
            "size": [1.0, 2.0, 3.0, 5.0, 1.0, 1.0],
        }
    )
    return SynapseTable(df)


def test_filter_by_min_synapses_removes_weak_pairs(st_multi_syn):
    st2 = st_multi_syn.filter_by_min_synapses(2)
    # (20,30) had only 1 synapse — should be gone; 5 remain
    assert len(st2.synapses) == 5
    pairs = set(
        map(tuple, st2.synapses.select(["pre_pt_root_id", "post_pt_root_id"]).rows())
    )
    assert (20, 30) not in pairs


def test_filter_by_min_synapses_keeps_exact_threshold(st_multi_syn):
    st2 = st_multi_syn.filter_by_min_synapses(2)
    pairs = set(
        map(tuple, st2.synapses.select(["pre_pt_root_id", "post_pt_root_id"]).rows())
    )
    assert (30, 10) in pairs  # exactly 2 synapses — should be kept


def test_filter_by_min_synapses_all_removed(st_multi_syn):
    st2 = st_multi_syn.filter_by_min_synapses(10)
    assert len(st2.synapses) == 0


def test_filter_by_min_synapses_by_weight(st_multi_syn):
    # sum(size): (10,20)=6, (20,30)=5, (30,10)=2 — threshold 3 removes (30,10)
    st2 = st_multi_syn.filter_by_min_synapses(3, weight_col="size")
    assert len(st2.synapses) == 4
    pairs = set(
        map(tuple, st2.synapses.select(["pre_pt_root_id", "post_pt_root_id"]).rows())
    )
    assert (30, 10) not in pairs
    assert (20, 30) in pairs  # size sum=5 >= 3


def test_filter_by_min_synapses_edgelist_reflects_threshold(st_multi_syn):
    st2 = st_multi_syn.filter_by_min_synapses(2)
    el = st2.edgelist().pairs
    assert (20, 30) not in set(
        map(tuple, el.select(["pre_pt_root_id", "post_pt_root_id"]).rows())
    )


def test_filter_by_min_synapses_unknown_weight_raises(st_multi_syn):
    with pytest.raises(ValueError, match="not found"):
        st_multi_syn.filter_by_min_synapses(2, weight_col="nonexistent")


def test_filter_by_min_synapses_preserves_annotations(st_multi_syn):
    """Annotations and expressions registered before the filter still work after."""
    st_multi_syn.add_expression("size_sq", pl.col("size").pow(2))
    st2 = st_multi_syn.filter_by_min_synapses(2)
    assert "size_sq" in st2.synapses.columns


# ── add_weight_transform ───────────────────────────────────────────────────────


@pytest.fixture
def st_const_size(base_synapses):
    df = base_synapses.with_columns(pl.lit(2.0).alias("size"))
    return SynapseTable(df)


def test_add_weight_transform_column_appears(st_const_size):
    st_const_size.add_weight_transform("log_size", "size")
    assert "log_size" in st_const_size.synapses.columns


def test_add_weight_transform_log1p_values(st_const_size):
    import math

    st_const_size.add_weight_transform("log_size", "size")
    vals = st_const_size.synapses["log_size"].to_list()
    assert all(math.isclose(v, math.log1p(2.0)) for v in vals)


def test_add_weight_transform_registered_as_weight_by_default(st_const_size):
    st_const_size.add_weight_transform("log_size", "size")
    assert "log_size" in st_const_size.weights


def test_add_weight_transform_sums_in_edgelist(st_const_size):
    st_const_size.add_weight_transform("log_size", "size")
    el = st_const_size.edgelist().pairs
    assert "log_size" in el.columns


def test_add_weight_transform_no_weight_registration(st_const_size):
    st_const_size.add_weight_transform("log_size", "size", register_as_weight=False)
    assert "log_size" not in st_const_size.weights
    el = st_const_size.edgelist().pairs
    assert "log_size" not in el.columns


def test_add_weight_transform_sqrt(st_const_size):
    import math

    st_const_size.add_weight_transform("sqrt_size", "size", transform="sqrt")
    vals = st_const_size.synapses["sqrt_size"].to_list()
    assert all(math.isclose(v, math.sqrt(2.0)) for v in vals)


def test_add_weight_transform_unknown_transform_raises(st_const_size):
    with pytest.raises(ValueError, match="transform must be one of"):
        st_const_size.add_weight_transform("x", "size", transform="cube")


def test_add_weight_transform_unknown_source_raises(st_const_size):
    with pytest.raises(ValueError, match="not found"):
        st_const_size.add_weight_transform("x", "nonexistent")


# ── info ──────────────────────────────────────────────────────────────────────


def test_info_core_columns(st):
    """info() should show core column names."""
    text = st.info()
    assert "pre_pt_root_id" in text
    assert "post_pt_root_id" in text
    assert "id_col" in text
    assert "5 synapses" in text


def test_info_synapse_annotation(st):
    ann = pl.DataFrame({"id": [1, 2, 3, 4, 5], "score": [0.1, 0.2, 0.3, 0.4, 0.5]})
    st.add_synapse_annotation("scores", ann)
    text = st.info()
    assert "Synapse annotations (1)" in text
    assert "'scores'" in text
    assert "score" in text


def test_info_cell_annotation(st):
    cell_ann = pl.DataFrame({"root_id": [10, 20, 30], "cell_type": ["A", "B", "C"]})
    st.add_cell_annotation("types", cell_ann, cell_id_col="root_id")
    text = st.info()
    assert "Cell annotations (1)" in text
    assert "cell_type  ->  cell_type_pre, cell_type_post" in text


def test_info_cell_alias(st):
    lookup = pl.DataFrame({"root_id": [10, 20, 30], "cell_id": [100, 200, 300]})
    st.add_cell_annotation("lookup", lookup, cell_id_col="root_id", alias_col="cell_id")
    text = st.info()
    assert "Cell aliases (1)" in text
    assert "cell_id from 'lookup'" in text


def test_info_vertex_annotation():
    synapses = pl.DataFrame(
        {
            "id": [1, 2],
            "pre_pt_root_id": [10, 20],
            "post_pt_root_id": [20, 10],
            "pre_vid": [100, 200],
        }
    )
    st = SynapseTable(synapses)
    ann = pl.DataFrame({"vid": [100, 200], "label": ["x", "y"]})
    st.add_vertex_annotation(
        "labels", ann, vertex_id_col="vid", pre_vertex_col="pre_vid"
    )
    text = st.info()
    assert "Vertex annotations (1)" in text
    assert "pre via 'pre_vid'" in text
    assert "label  ->  label_pre" in text


def test_info_expression(st):
    cell_ann = pl.DataFrame({"root_id": [10, 20, 30], "depth": [1.0, 2.0, 3.0]})
    st.add_cell_annotation("coords", cell_ann, cell_id_col="root_id")
    st.add_expression("depth_diff", pl.col("depth_pre") - pl.col("depth_post"))
    text = st.info()
    assert "Expressions (1)" in text
    assert "depth_diff" in text
    assert "(both)" in text
    assert "depth_pre" in text


def test_info_weights(st):
    ann = pl.DataFrame({"id": [1, 2, 3, 4, 5], "size": [1.0, 2.0, 3.0, 4.0, 5.0]})
    st.add_synapse_annotation("sizes", ann)
    st.add_weight("size")
    text = st.info()
    assert "Weights (1)" in text
    assert "size" in text


def test_info_position_tag(st):
    cell_ann = pl.DataFrame(
        {
            "root_id": [10, 20, 30],
            "soma": [
                {"x": 1.0, "y": 2.0, "z": 3.0},
                {"x": 4.0, "y": 5.0, "z": 6.0},
                {"x": 7.0, "y": 8.0, "z": 9.0},
            ],
        }
    )
    st.add_cell_annotation("coords", cell_ann, cell_id_col="root_id")
    text = st.info()
    assert "[position]" in text


def test_info_filter_count(st):
    filtered = st.filter(pl.col("pre_pt_root_id") == 10)
    text = filtered.info()
    assert "1 filter(s)" in text


# ── role-declared public accessors (Phase 0) ──────────────────────────────────


def test_role_accessors_default(st):
    assert st.pre_col == "pre_pt_root_id"
    assert st.post_col == "post_pt_root_id"
    assert st.id_col == "id"
    assert st.synapse_position_col is None


def test_position_col_declared_on_cell_annotation(base_synapses):
    soma_df = pl.DataFrame(
        {
            "root_id": [10, 20, 30],
            "pt_position_x": [0.0, 3.0, 0.0],
            "pt_position_y": [0.0, 0.0, 4.0],
            "pt_position_z": [0.0, 0.0, 0.0],
        }
    )
    st = SynapseTable(base_synapses).add_cell_annotation(
        "soma",
        soma_df,
        cell_id_col="root_id",
        position_cols="pt_position",
        position_col="pt_position",
    )
    spec = st._cell_annotations["soma"]
    assert spec.position_col == "pt_position"
    # round-trip via the resolution helper
    assert st._resolve_position_annotation() == "soma"


def test_build_lazy_is_public(st):
    lf = st.build_lazy()
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().equals(st.synapses)


def test_cell_annotation_data_cols(st):
    assert st.cell_annotation_data_cols() == {}
    ann = pl.DataFrame({"root_id": [10, 20, 30], "cell_type": ["a", "b", "c"]})
    st.add_cell_annotation("types", ann, cell_id_col="root_id")
    assert st.cell_annotation_data_cols() == {"types": ["cell_type"]}
    # mutating the returned dict must not affect internal state
    d = st.cell_annotation_data_cols()
    d["types"].append("bogus")
    d["other"] = ["nope"]
    assert st.cell_annotation_data_cols() == {"types": ["cell_type"]}
