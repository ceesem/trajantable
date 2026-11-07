"""Tests for the full-collect memory guard (trajan._guard)."""

import polars as pl
import pytest

import trajan
from trajan import (
    CollectTooLargeError,
    SynapseTable,
    allow_large_collect,
    estimate_bytes,
    possible_pairs,
    set_max_collect_bytes,
)


@pytest.fixture
def st():
    syn = pl.DataFrame(
        {
            "id": list(range(100)),
            "pre_pt_root_id": [i % 10 for i in range(100)],
            "post_pt_root_id": [(i * 3) % 10 for i in range(100)],
            "size": [1.0] * 100,
        }
    )
    cells = pl.DataFrame({"root_id": list(range(10)), "ct": ["a"] * 10})
    st = SynapseTable(syn)
    st.add_cell_annotation("cells", cells, cell_id_col="root_id", is_universe=True)
    return st


@pytest.fixture
def tiny_limit():
    prev = trajan.get_max_collect_bytes()
    set_max_collect_bytes(100)
    yield
    set_max_collect_bytes(prev)


def test_estimate_bytes_fixed_width():
    schema = pl.Schema({"a": pl.Int64, "b": pl.Float32, "c": pl.Boolean})
    assert estimate_bytes(schema, 10) == 10 * (8 + 4 + 1)


def test_estimate_bytes_struct_sums_fields():
    schema = pl.Schema({"p": pl.Struct({"x": pl.Float64, "y": pl.Float64})})
    assert estimate_bytes(schema, 2) == 32


def test_df_raises_over_limit_and_leaves_no_cache(st, tiny_limit):
    with pytest.raises(CollectTooLargeError, match="SynapseTable.df"):
        st.df
    assert st._cache is None


def test_guard_error_is_memory_error(st, tiny_limit):
    with pytest.raises(MemoryError):
        st.df


def test_raw_synapses_guarded(st, tiny_limit):
    with pytest.raises(CollectTooLargeError, match="SynapseTable.synapses"):
        st.synapses
    # the lazy view is always fine
    assert isinstance(st.synapses_lazy, pl.LazyFrame)


def test_collect_force_bypasses(st, tiny_limit):
    df = st.collect(force=True)
    assert len(df) == 100
    assert st.df is df  # cached; subsequent .df does not re-check


def test_allow_large_collect_context(st, tiny_limit):
    with allow_large_collect():
        assert len(st.df) == 100


def test_narrow_collect_checked_on_projection(st):
    prev = trajan.get_max_collect_bytes()
    # 100 rows x one Int64 column = 800 B; whole table is much wider
    set_max_collect_bytes(1000)
    try:
        assert st.collect("id").shape == (100, 1)
        with pytest.raises(CollectTooLargeError):
            st.df
    finally:
        set_max_collect_bytes(prev)


def test_edgelist_and_pairs_guarded(st, tiny_limit):
    with allow_large_collect():
        el = st.edgelist()
    with pytest.raises(CollectTooLargeError):
        el.df
    with pytest.raises(CollectTooLargeError, match="pairs"):
        el.pairs


def test_pair_universe_guarded(st, tiny_limit):
    pu = possible_pairs(st)
    with pytest.raises(CollectTooLargeError, match="PairUniverse.collect"):
        pu.collect()
    with pytest.raises(CollectTooLargeError, match="to_pair_frame"):
        pu.to_pair_frame()
    assert len(pu.collect(force=True)) == 90


def test_preview_and_len_unguarded(st, tiny_limit):
    assert len(st) == 100
    assert len(st.preview(5)) == 5


def test_disabled_guard(st):
    prev = trajan.get_max_collect_bytes()
    set_max_collect_bytes(None)
    try:
        assert len(st.df) == 100
    finally:
        set_max_collect_bytes(prev)


def test_set_limit_rejects_nonpositive():
    with pytest.raises(ValueError):
        set_max_collect_bytes(0)
