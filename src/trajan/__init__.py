"""Reshapable and visualizable connectivity tables"""

__version__ = "0.0.1"

from ._base import Scope
from .completeness import (
    edge_distance_expr,
    interior_share,
    universe_share_by_edge_distance,
)
from .connectivity_table import ConnectivityTable
from .edge_correction import LateralBox, edge_weight, lateral_box, ring_inside_fraction
from .edgelist import EdgeList
from .export import to_dataframe, to_graph
from .pair_universe import PairUniverse, possible_pairs
from .scope import cells
from .spatial import (
    depth_component,
    euclidean_distance,
    lateral_mahalanobis_distance,
    lateral_spread,
    pack_all_positions,
    pack_position,
    position_is_valid,
    radial_distance,
    transform_point,
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
    pivot_grid,
    wilson_ci,
    with_distance,
)
from .synapse_table import SynapseTable

__all__ = [
    "ConnectivityTable",
    "EdgeList",
    "LateralBox",
    "edge_distance_expr",
    "edge_weight",
    "interior_share",
    "universe_share_by_edge_distance",
    "lateral_box",
    "ring_inside_fraction",
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
    "depth_component",
    "euclidean_distance",
    "lateral_mahalanobis_distance",
    "lateral_spread",
    "pack_all_positions",
    "pack_position",
    "pivot_grid",
    "possible_pairs",
    "position_is_valid",
    "radial_distance",
    "to_dataframe",
    "to_graph",
    "transform_point",
    "unpack_all_positions",
    "unpack_position",
    "wilson_ci",
    "with_distance",
]
