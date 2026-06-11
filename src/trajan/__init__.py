"""Reshapable and visualizable connectivity tables"""

__version__ = "0.0.1"

from ._base import Scope
from .connectivity_table import ConnectivityTable
from .edgelist import EdgeList
from .export import to_dataframe, to_graph
from .pair_universe import PairUniverse, possible_pairs
from .scope import cells
from .spatial import (
    euclidean_distance,
    pack_all_positions,
    pack_position,
    radial_distance,
    unpack_all_positions,
    unpack_position,
)
from .stats import (
    agresti_coull_ci,
    bootstrap_over_cells,
    cell_bootstrap_iter,
    cell_summary,
    connection_density,
    connection_probability,
    counts,
    wilson_ci,
    with_distance,
)
from .synapse_table import SynapseTable

__all__ = [
    "ConnectivityTable",
    "EdgeList",
    "PairUniverse",
    "Scope",
    "SynapseTable",
    "agresti_coull_ci",
    "bootstrap_over_cells",
    "cell_bootstrap_iter",
    "cell_summary",
    "cells",
    "connection_density",
    "connection_probability",
    "counts",
    "euclidean_distance",
    "pack_all_positions",
    "pack_position",
    "possible_pairs",
    "radial_distance",
    "to_dataframe",
    "to_graph",
    "unpack_all_positions",
    "unpack_position",
    "wilson_ci",
    "with_distance",
]
