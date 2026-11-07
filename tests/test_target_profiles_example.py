"""Exercise the notebook recipe with unequal and unconnected populations."""

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

from trajan import SynapseTable

spec = importlib.util.spec_from_file_location(
    "target_profiles_example",
    Path(__file__).parents[1] / "examples" / "target_profiles.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
target_profiles = module.target_profiles


@pytest.fixture
def inputs():
    cells = pl.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5],
            "class": ["exc", "exc", "exc", "inh", "inh"],
            "subclass": ["A", "A", "B", "C", "D"],
            "cluster": ["one", "two", "one", "three", None],
        }
    )
    syn = pl.DataFrame(
        {
            "id": list(range(8)),
            "pre": [10, 10, 10, 10, 10, 10, 10, 11],
            "post": [1, 1, 2, 3, 4, None, 999, None],
            "size": [2.0, 4.0, None, 6.0, 10.0, 100.0, 20.0, None],
            "tag": [
                "shaft",
                "spine",
                "soma",
                "spine",
                "soma",
                None,
                "unexpected",
                None,
            ],
            "dist_to_root": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        }
    )
    return SynapseTable(syn, pre_col="pre", post_col="post"), cells


def group(long, level, target, pre=10):
    return long.filter(
        (pl.col("pre_id") == pre)
        & (pl.col("level") == level)
        & (pl.col("target_type") == target)
    ).row(0, named=True)


def test_allocations_population_adjustment_and_tags(inputs):
    wide, long = target_profiles(*inputs)
    assert wide["pre_id"].to_list() == [10, 11]
    exc = group(long, "class", "exc")
    assert exc["n_syn"] == 4
    assert exc["count_fraction"] == pytest.approx(4 / 5)
    assert exc["mean_size"] == 4  # Per-synapse mean, not mean of connection means.
    assert exc["tag_spine_count_fraction"] == 0.5
    assert exc["tag_spine_size_fraction"] == pytest.approx(10 / 12)
    a = group(long, "subclass", "A")
    assert a["n_target_cells"] == 2
    assert a["count_fraction"] == 0.75
    assert a["size_fraction"] == 0.5
    assert a["adjusted_count_fraction"] == pytest.approx(0.6)
    assert a["adjusted_size_fraction"] == pytest.approx(1 / 3)
    assert group(long, "subclass", "D")["n_syn"] == 0
    assert group(long, "subclass", "D")["mean_size"] is None
    assert group(long, "class", "exc", pre=11)["count_fraction"] is None
    assert "cluster__exc__A__one__n_syn" in wide.columns
    assert "cluster__exc__B__one__n_syn" in wide.columns
    assert wide.filter(pl.col("pre_id") == 10)["n_syn_unclassified"].item() == 2


def test_synapse_filters_keep_target_population_fixed(inputs):
    st, cells = inputs
    _, long = target_profiles(st.filter(pl.col("tag") == "spine"), cells)
    a = group(long, "subclass", "A")
    assert a["n_target_cells"] == 2
    assert a["n_syn"] == 1
    assert a["adjusted_count_fraction"] == pytest.approx(1 / 3)
    _, long = target_profiles(st.filter(pl.col("dist_to_root") < 25), cells)
    assert group(long, "subclass", "B")["n_syn"] == 0
    assert group(long, "subclass", "B")["n_target_cells"] == 1


def test_empty_invalid_and_missing_taxonomy(inputs):
    st, cells = inputs
    with pytest.raises(ValueError, match="unique"):
        target_profiles(st, pl.concat([cells, cells.head(1)]))
    with pytest.raises(ValueError, match="finite"):
        target_profiles(
            SynapseTable(
                pl.DataFrame(
                    {
                        "id": [1],
                        "pre": [10],
                        "post": [1],
                        "size": [-1.0],
                        "tag": ["spine"],
                    }
                ),
                pre_col="pre",
                post_col="post",
            ),
            cells,
        )
    wide, long = target_profiles(st.filter(pl.lit(False)), cells)
    assert wide.height == long.height == 0
    _, long = target_profiles(
        st,
        cells.with_columns(
            pl.when(pl.col("root_id") == 1)
            .then(None)
            .otherwise(pl.col("subclass"))
            .alias("subclass")
        ),
    )
    assert group(long, "subclass", "__unlabeled__")["n_syn"] == 2


def test_missing_tags_zero_sizes_and_unclassified_population(inputs):
    st, cells = inputs
    syn = pl.DataFrame(
        {
            "id": [1, 2],
            "pre": [10, 10],
            "post": [1, 1],
            "size": [0.0, 0.0],
            "tag": [None, "unexpected"],
        }
    )
    source = SynapseTable(syn, pre_col="pre", post_col="post")
    _, long = target_profiles(source, cells)
    exc = group(long, "class", "exc")
    assert exc["mean_size"] == 0
    assert exc["size_fraction"] is None
    assert exc["tag_other_count_fraction"] == 1
    assert exc["tag_other_size_fraction"] is None
    wide, long = target_profiles(
        st, cells.with_columns(pl.lit(None, dtype=pl.String).alias("class"))
    )
    assert wide.height == 2
    assert long.height == 0
    assert wide["n_syn_classified"].sum() == 0


def test_trajan_notebook_workflow(inputs):
    st, cells = inputs
    notebook = json.loads(
        (Path(__file__).parents[1] / "examples" / "target_allocation.ipynb").read_text()
    )
    namespace = {"st": st, "cell_types": cells}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(
                compile("".join(cell["source"]), "target_allocation.ipynb", "exec"),
                namespace,
            )
    summary = namespace["summary"].filter(pl.col("pre") == 10)
    assert summary["class__exc__n_syn"].item() == 4
    assert summary["class__exc__mean_size"].item() == 4
    assert summary["exc_subclass__A__adjusted_count_fraction"].item() == pytest.approx(
        0.6
    )
    assert summary["exc_subclass__A__adjusted_size_fraction"].item() == pytest.approx(
        1 / 3
    )
    assert summary["inh_subclass__D__n_syn"].item() == 0
    assert summary["class__exc__tag_spine_fraction"].item() == 0.5
    assert namespace["summary"].height == 2
