"""Tests for trajan.scope: cells() and the scope vocabulary.

Pressure-tests the contract laid out in DESIGN-universe.md §3:

- scope='universe' returns the raw universe annotation
- scope='filtered' applies cell-level filter projection (single-sided,
  asymmetric, and 'both' cases) returning the pre/post-eligible union
- scope='observed' returns cells appearing in .df, joined to the universe
- annotations= selects which sibling decorations to join
- participation=True overlays in_pre/in_post/n_syn_out/n_syn_in
- Filters referencing unavailable columns are skipped with a warning
- Non-cell-level filters (synapse / pair weight) do not narrow cells()
"""

import polars as pl
import pytest

from trajan import EdgeList, SynapseTable, cells

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def universe_cells():
    """A small universe: 5 cells, with cell_type and is_proofread."""
    return pl.DataFrame(
        {
            "root_id": [10, 20, 30, 40, 50],
            "cell_type": ["L23", "L23", "L4", "L5", "L5"],
            "is_proofread": [True, True, False, True, False],
        }
    )


@pytest.fixture
def synapses():
    """Synapses only mention 3 of the 5 universe cells — 40 and 50 are zero-connection."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "pre_pt_root_id": [10, 10, 20, 20, 30, 30],
            "post_pt_root_id": [20, 30, 10, 30, 10, 20],
            "size": [100, 50, 200, 150, 80, 60],
        }
    )


@pytest.fixture
def st(synapses, universe_cells):
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", universe_cells, cell_id_col="root_id", is_universe=True
    )
    return st


# ── scope='universe' ──────────────────────────────────────────────────────────


def test_universe_scope_returns_raw_annotation(st):
    """scope='universe' returns the registered universe annotation as-is."""
    df = cells(st, scope="universe")
    assert len(df) == 5
    assert set(df["root_id"].to_list()) == {10, 20, 30, 40, 50}
    assert "cell_type" in df.columns
    assert "is_proofread" in df.columns


def test_universe_scope_ignores_filters(st):
    """scope='universe' does not apply any filter — universe is universe."""
    filtered = st.filter(pl.col("cell_type_pre") == "L23")
    df = cells(filtered, scope="universe")
    assert len(df) == 5


# ── scope='filtered' single-sided ─────────────────────────────────────────────


def test_filtered_no_filters_returns_universe(st):
    """With no filters, scope='filtered' equals scope='universe'."""
    df = cells(st, scope="filtered").sort("root_id")
    expected = cells(st, scope="universe").sort("root_id")
    assert df.equals(expected)


def test_filtered_pre_only_filter_returns_universe(st):
    """A pre-only filter constrains pre side; post side is universe; union = universe."""
    filtered = st.filter(pl.col("cell_type_pre") == "L23")
    df = cells(filtered, scope="filtered")
    assert len(df) == 5  # universe ∪ {10, 20} = universe


# ── scope='filtered' two-sided decomposable via separate .filter() calls ─────


def test_filtered_two_separate_filters_decompose_cleanly(st):
    """Two single-sided .filter() calls decompose: pre L23 ∪ post L4."""
    filtered = st.filter(pl.col("cell_type_pre") == "L23").filter(
        pl.col("cell_type_post") == "L4"
    )
    df = cells(filtered, scope="filtered").sort("root_id")
    # pre-eligible: {10, 20} (L23); post-eligible: {30} (L4); union {10, 20, 30}
    assert set(df["root_id"].to_list()) == {10, 20, 30}


# ── scope='filtered' two-sided in one expression: 'both' → skip + warn ───────


def test_filtered_both_filter_skipped_with_warning(st):
    """A single AND-conjunction referencing both sides classifies as 'both';
    cells() warns and skips it (the filter still applies to .df)."""
    filtered = st.filter(
        (pl.col("cell_type_pre") == "L23") & (pl.col("cell_type_post") == "L4")
    )
    with pytest.warns(UserWarning, match="both"):
        df = cells(filtered, scope="filtered")
    # Skipped → universe-wide result.
    assert len(df) == 5


def test_filtered_pair_only_filter_skipped_with_warning(st):
    """A pair-only filter like cell_type_pre != cell_type_post classifies as
    'both' and skips with a warning."""
    filtered = st.filter(pl.col("cell_type_pre") != pl.col("cell_type_post"))
    with pytest.warns(UserWarning, match="both"):
        df = cells(filtered, scope="filtered")
    assert len(df) == 5


# ── scope='filtered' ignores non-cell-level filters ──────────────────────────


def test_filtered_ignores_synapse_level_filter(st):
    """Synapse-level filters (side=None) do not affect cells() — they require
    synapse context that cells() does not have."""
    filtered = st.filter(pl.col("size") > 100)
    df = cells(filtered, scope="filtered")
    assert len(df) == 5  # universe unchanged


# ── scope='observed' ─────────────────────────────────────────────────────────


def test_observed_returns_cells_in_df(st):
    """scope='observed' returns cells appearing as pre or post in .df."""
    df = cells(st, scope="observed").sort("root_id")
    # Synapses mention 10, 20, 30 only.
    assert set(df["root_id"].to_list()) == {10, 20, 30}
    assert "cell_type" in df.columns  # annotation columns are joined


def test_observed_strict_drops_orphans(synapses, universe_cells):
    """Default strict=True drops observed cells absent from the universe.

    Universe excludes cell 30; observed includes 30 → strict drops it.
    """
    partial_universe = universe_cells.filter(pl.col("root_id") != 30)
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", partial_universe, cell_id_col="root_id", is_universe=True
    )
    df = cells(st, scope="observed").sort("root_id")
    assert set(df["root_id"].to_list()) == {10, 20}


def test_observed_lenient_keeps_orphans_with_nulls(synapses, universe_cells):
    """strict=False keeps orphans with null annotation columns."""
    partial_universe = universe_cells.filter(pl.col("root_id") != 30)
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", partial_universe, cell_id_col="root_id", is_universe=True
    )
    df = cells(st, scope="observed", strict=False).sort("root_id")
    assert set(df["root_id"].to_list()) == {10, 20, 30}
    # Cell 30's cell_type should be null.
    row_30 = df.filter(pl.col("root_id") == 30)
    assert row_30["cell_type"].item() is None


# ── EdgeList path ────────────────────────────────────────────────────────────


def test_cells_on_edgelist(st):
    """cells() works on EdgeList; annotations propagate via .edgelist()."""
    el = st.edgelist()
    df = cells(el, scope="universe")
    assert set(df["root_id"].to_list()) == {10, 20, 30, 40, 50}


def test_cells_on_edgelist_participation(st):
    """participation=True works on an EdgeList; counts come from n_syn weight."""
    el = st.edgelist()
    df = cells(el, participation=True).sort("root_id")
    assert set(df.filter(pl.col("in_pre"))["root_id"].to_list()) == {10, 20, 30}
    # Cells 40, 50 are zero-connection in the universe.
    iso = df.filter(pl.col("root_id").is_in([40, 50]))
    assert iso["n_syn_out"].to_list() == [0, 0]
    assert iso["n_syn_in"].to_list() == [0, 0]


# ── validation / error cases ─────────────────────────────────────────────────


def test_cells_no_universe_annotation_raises(synapses):
    st = SynapseTable(synapses)
    with pytest.raises(ValueError, match="No cell annotation is marked is_universe"):
        cells(st)


def test_cells_explicit_universe_name(st):
    """Passing universe= disambiguates explicitly."""
    df = cells(st, universe="cells", scope="universe")
    assert len(df) == 5


def test_cells_invalid_scope_raises(st):
    with pytest.raises(ValueError, match="scope must be"):
        cells(st, scope="bogus")


def test_filter_sides_classified_at_filter_time(st):
    """filter() classifies each added filter at registration time and exposes
    the classification via .filter_sides."""
    pre_filter = st.filter(pl.col("cell_type_pre") == "L23")
    assert pre_filter.filter_sides == ["pre"]

    post_filter = pre_filter.filter(pl.col("cell_type_post") == "L4")
    assert post_filter.filter_sides == ["pre", "post"]

    both_filter = post_filter.filter(
        pl.col("cell_type_pre") != pl.col("cell_type_post")
    )
    assert both_filter.filter_sides == ["pre", "post", "both"]

    syn_filter = both_filter.filter(pl.col("size") > 50)
    assert syn_filter.filter_sides == ["pre", "post", "both", None]


def test_filter_sides_persist_through_save_load(st, tmp_path):
    """Saved filter_sides round-trip; classification stays stable across reload."""
    filtered = st.filter(pl.col("cell_type_pre") == "L23").filter(pl.col("size") > 50)
    folio = tmp_path / "folio"
    filtered.save(str(folio))
    loaded = SynapseTable.load(str(folio))
    assert loaded.filter_sides == filtered.filter_sides == ["pre", None]


def test_cells_joins_sibling_cell_annotations(synapses, universe_cells):
    """Sibling cell annotations are joined into the cells frame on compatible
    keys, so filters that reference their columns project correctly. This
    matches possible_pairs() behavior — both must agree on which cells are
    selected by a given filter."""
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", universe_cells, cell_id_col="root_id", is_universe=True
    )
    sibling = pl.DataFrame({"root_id": [10, 20, 30, 40, 50], "tag": list("ABCDE")})
    st.add_cell_annotation("tags", sibling, cell_id_col="root_id")
    # Two single-sided filters on sibling columns decompose cleanly:
    # pre-eligible {10} (tag A) ∪ post-eligible {20} (tag B).
    filtered = st.filter(pl.col("tag_pre") == "A").filter(pl.col("tag_post") == "B")
    df = cells(filtered, scope="filtered")
    assert set(df["root_id"].to_list()) == {10, 20}


def test_cells_joins_sibling_with_differently_named_cell_id_col(
    synapses, universe_cells
):
    """A sibling annotation keyed on a *differently-named* cell_id_col still
    joins into the cells frame. cells() joins on the semantically-blessed
    cell_id_col (universe key ↔ sibling key on value), not by string-matching
    column names — so a 'pt_root_id'-keyed sibling joins onto a 'root_id'-keyed
    universe. Regression for the old behavior that silently dropped such
    siblings (they appear correctly in .df via the symmetric pre/post join, so
    the drop was a surprising cells()-only inconsistency)."""
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", universe_cells, cell_id_col="root_id", is_universe=True
    )
    # Same root-id values as the universe, but under a different column name.
    sibling = pl.DataFrame({"pt_root_id": [10, 20, 30, 40, 50], "tag": list("ABCDE")})
    st.add_cell_annotation("tags", sibling, cell_id_col="pt_root_id")

    df = cells(st).sort("root_id")
    assert "tag" in df.columns
    assert df["tag"].to_list() == list("ABCDE")
    # The sibling's own key column is consumed by the join, not leaked.
    assert "pt_root_id" not in df.columns

    # And a filter on the differently-keyed sibling still projects.
    filtered = st.filter(pl.col("tag_pre") == "A").filter(pl.col("tag_post") == "B")
    proj = cells(filtered, scope="filtered")
    assert set(proj["root_id"].to_list()) == {10, 20}


def test_cells_schema_is_scope_invariant(synapses, universe_cells):
    """All three scopes return the same column set (including sibling cell
    annotations). Only the row set differs. This is the contract that lets a
    viz helper switch scope without its color/feature columns disappearing."""
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", universe_cells, cell_id_col="root_id", is_universe=True
    )
    sibling = pl.DataFrame({"root_id": [10, 20, 30, 40, 50], "tag": list("ABCDE")})
    st.add_cell_annotation("tags", sibling, cell_id_col="root_id")

    u = cells(st, scope="universe")
    f = cells(st, scope="filtered")
    o = cells(st, scope="observed")
    # Sibling column present in every scope.
    assert "tag" in u.columns
    assert set(u.columns) == set(f.columns) == set(o.columns)
    # The universe-scope sibling join is a left join on a unique key, so it
    # doesn't change the universe row count.
    assert len(u) == 5


def test_cells_projects_expression_derived_filters(synapses, universe_cells):
    """A filter on a registered cell-side EXPRESSION column projects onto the
    cells frame (the expression is re-applied per side), matching what
    possible_pairs does — they must agree on which cells a filter selects."""
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", universe_cells, cell_id_col="root_id", is_universe=True
    )
    # ct_upper is a cell-side ("pre") expression. Two single-sided filters on it
    # decompose: pre L23 ∪ post L4.
    st.add_expression("ct_pre_up", pl.col("cell_type_pre").str.to_uppercase())
    st.add_expression("ct_post_up", pl.col("cell_type_post").str.to_uppercase())
    filtered = st.filter(pl.col("ct_pre_up") == "L23").filter(
        pl.col("ct_post_up") == "L4"
    )
    df = cells(filtered, scope="filtered")
    # pre-eligible {10, 20} (L23) ∪ post-eligible {30} (L4); expression columns
    # are not leaked into the output schema.
    assert set(df["root_id"].to_list()) == {10, 20, 30}
    assert "ct_pre_up" not in df.columns
    assert "ct_post_up" not in df.columns


def test_cells_warns_when_filter_references_unavailable_column(
    synapses, universe_cells
):
    """A filter referencing a column unreachable from the cells frame — e.g. a
    sibling annotation excluded via annotations= — is skipped with a warning."""
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", universe_cells, cell_id_col="root_id", is_universe=True
    )
    sibling = pl.DataFrame({"root_id": [10, 20, 30, 40, 50], "tag": list("ABCDE")})
    st.add_cell_annotation("tags", sibling, cell_id_col="root_id")
    filtered = st.filter(pl.col("tag_pre") == "A")
    # Exclude the 'tags' sibling, so tag_pre is not on the cells frame.
    with pytest.warns(UserWarning, match="not on the cells frame"):
        df = cells(filtered, scope="filtered", annotations=None)
    # Filter is skipped → universe-wide result.
    assert len(df) == 5


# ── default scope ────────────────────────────────────────────────────────────


def test_default_scope_is_universe(st):
    """cells() with no scope returns the full universe (the decorated-universe
    default), ignoring filters."""
    filtered = st.filter(pl.col("cell_type_pre") == "L23")
    df = cells(filtered)
    assert len(df) == 5
    assert df.equals(cells(filtered, scope="universe"))


# ── annotations selection ────────────────────────────────────────────────────


@pytest.fixture
def st_two_siblings(synapses, universe_cells):
    st = SynapseTable(synapses)
    st.add_cell_annotation(
        "cells", universe_cells, cell_id_col="root_id", is_universe=True
    )
    st.add_cell_annotation(
        "tags",
        pl.DataFrame({"root_id": [10, 20, 30, 40, 50], "tag": list("ABCDE")}),
        cell_id_col="root_id",
    )
    st.add_cell_annotation(
        "groups",
        pl.DataFrame({"root_id": [10, 20, 30, 40, 50], "grp": [1, 1, 2, 2, 3]}),
        cell_id_col="root_id",
    )
    return st


def test_annotations_all_joins_every_sibling(st_two_siblings):
    df = cells(st_two_siblings)  # annotations="all" by default
    assert "cell_type" in df.columns  # universe's own column
    assert "tag" in df.columns
    assert "grp" in df.columns


def test_annotations_none_universe_columns_only(st_two_siblings):
    df = cells(st_two_siblings, annotations=None)
    assert "cell_type" in df.columns  # universe's own column stays
    assert "tag" not in df.columns
    assert "grp" not in df.columns


def test_annotations_list_selects_specific_siblings(st_two_siblings):
    df = cells(st_two_siblings, annotations=["tags"])
    assert "tag" in df.columns
    assert "grp" not in df.columns


def test_annotations_unknown_name_raises(st):
    with pytest.raises(ValueError, match="Unknown annotation"):
        cells(st, annotations=["nope"])


# ── participation columns ────────────────────────────────────────────────────


def test_participation_adds_columns_over_universe(st):
    """participation=True overlays in_pre/in_post/n_syn_out/n_syn_in onto the
    universe; zero-connection cells get False/0."""
    df = cells(st, participation=True).sort("root_id")
    assert {"in_pre", "in_post", "n_syn_out", "n_syn_in"} <= set(df.columns)
    assert len(df) == 5  # still the full universe
    iso = df.filter(pl.col("root_id").is_in([40, 50]))
    assert iso["in_pre"].to_list() == [False, False]
    assert iso["in_post"].to_list() == [False, False]
    assert iso["n_syn_out"].to_list() == [0, 0]
    assert iso["n_syn_in"].to_list() == [0, 0]


def test_participation_recovers_presynaptic_cells(st):
    """The replacement for side='pre': filter on in_pre. Works with no filter
    present, unlike the old side= selector."""
    df = cells(st, participation=True)
    assert set(df.filter(pl.col("in_pre"))["root_id"].to_list()) == {10, 20, 30}
    assert set(df.filter(pl.col("in_post"))["root_id"].to_list()) == {10, 20, 30}


def test_participation_synapse_counts(st):
    """n_syn_out / n_syn_in are synapse counts per side."""
    df = cells(st, participation=True)
    row10 = df.filter(pl.col("root_id") == 10)
    # Cell 10: pre in synapses 1,2 → n_syn_out=2; post in synapses 3,5 → n_syn_in=2.
    assert row10["n_syn_out"].item() == 2
    assert row10["n_syn_in"].item() == 2


def test_participation_reflects_filters(st):
    """Participation comes from the filtered observed edges (1:1 with analysis)."""
    filtered = st.filter(pl.col("cell_type_pre") == "L23")  # pre ∈ {10, 20}
    df = cells(filtered, participation=True)
    assert set(df.filter(pl.col("in_pre"))["root_id"].to_list()) == {10, 20}


def test_participation_under_observed_scope(st):
    """participation overlays onto whichever scope row set was selected."""
    df = cells(st, scope="observed", participation=True).sort("root_id")
    assert set(df["root_id"].to_list()) == {10, 20, 30}
    assert {"in_pre", "in_post", "n_syn_out", "n_syn_in"} <= set(df.columns)
