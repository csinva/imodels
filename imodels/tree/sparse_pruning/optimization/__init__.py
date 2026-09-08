"""Numerical point and regularization-path solvers for sparse pruning."""

from ._result import RegularizationPath
from .apa_point import hiCAP_classification, hiCAP_regression, proj_l1_ball
from .apa import apa_apg_classification_path, apa_apg_regression_path
from .diagnostics import (
    group_linf_lambda_max,
    group_linf_quadratic_kkt_diagnostic,
    group_linf_regression_kkt_diagnostic,
    group_linf_regression_lambda_max,
)
from .hicap import hicap_regression_path
from .diagonal_homotopy import tree_group_linf_exact_coefficient_path
from .proximal import (
    laminar_group_linf_regression,
    laminar_group_linf_regression_path,
)
from .tree_prox import LaminarGroupLinfProx, prox_laminar_group_linf
from .topology import (
    TreeTopologyPath,
    laminar_group_linf_exact_topology_path,
    tree_group_linf_exact_topology_path,
)

__all__ = [
    "hiCAP_classification",
    "hiCAP_regression",
    "proj_l1_ball",
    "RegularizationPath",
    "apa_apg_classification_path",
    "apa_apg_regression_path",
    "group_linf_lambda_max",
    "group_linf_quadratic_kkt_diagnostic",
    "group_linf_regression_kkt_diagnostic",
    "group_linf_regression_lambda_max",
    "hicap_regression_path",
    "tree_group_linf_exact_coefficient_path",
    "LaminarGroupLinfProx",
    "laminar_group_linf_regression",
    "laminar_group_linf_regression_path",
    "prox_laminar_group_linf",
    "TreeTopologyPath",
    "laminar_group_linf_exact_topology_path",
    "tree_group_linf_exact_topology_path",
]
