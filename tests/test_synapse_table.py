import polars as pl
import pytest

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
