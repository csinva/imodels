"""Solver selection and nonmutating solves for one fitted pruning tree."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sklearn.base import is_regressor
from sklearn.tree import DecisionTreeRegressor

from .fitted_tree import (
    _MEAN_REGRESSION_CRITERIA,
    _fitted_tree_statistics,
    fitted_tree_linf_exact_coefficient_path,
    fitted_tree_linf_exact_topology_path,
)
from .optimization import (
    LaminarGroupLinfProx,
    RegularizationPath,
    TreeTopologyPath,
    tree_group_linf_exact_topology_path,
)
from .optimization.apa_point import hiCAP_classification, hiCAP_regression


_SOLVERS = ("auto", "topology", "proximal", "coefficient_path", "hicap", "apa_apg2")


def validate_solver(name: str) -> str:
    """Validate the public solver name without silently changing its meaning."""
    if not isinstance(name, str) or name not in _SOLVERS:
        raise ValueError(f"solver must be one of {_SOLVERS}; got {name!r}")
    return name


def _nonnegative_scalar(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
        or np.iscomplexobj(value)
    ):
        raise ValueError(f"{name} must be a nonnegative finite scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite scalar")
    return value


def _native_tree_eligible(estimator: Any, *, require_fitted: bool = True) -> bool:
    """Check tree prerequisites without fitting data just to select a solver.

    CV can inspect an unfitted template with ``require_fitted=False``. A real
    solve still requires one fitted output and validates the stored statistics.
    """
    constraints = getattr(estimator, "monotonic_cst", None)
    return bool(
        isinstance(estimator, DecisionTreeRegressor)
        and (not require_fitted or (
            hasattr(estimator, "tree_")
            and getattr(estimator, "n_outputs_", None) == 1
        ))
        and str(estimator.criterion) in _MEAN_REGRESSION_CRITERIA
        and (constraints is None or not np.any(np.asarray(constraints) != 0))
    )


def resolve_solver(
    name: str,
    *,
    estimator: Any,
    ord: int | str,
    matched_training: bool,
    support_tol: float | None,
    custom_solver: bool = False,
) -> str:
    """Select a compatible algorithm, never a fallback after solver failure.

    An overridden legacy point-solver hook keeps the ``apa_apg2`` dispatch
    name. Explicitly selecting another algorithm conflicts with that hook.
    Native paths require the original fitting rows/weights and mathematical
    zero support; positive support thresholds use coefficient point solves.
    """
    name = validate_solver(name)
    if (
        isinstance(ord, (bool, np.bool_)) or not np.isscalar(ord)
        or np.iscomplexobj(ord) or ord not in (2, "inf", np.inf)
    ):
        raise ValueError("ord must be 2, 'inf', or np.inf")
    if not isinstance(matched_training, (bool, np.bool_)):
        raise ValueError("matched_training must be a boolean")
    if not isinstance(custom_solver, (bool, np.bool_)):
        raise ValueError("custom_solver must be a boolean")
    support_tol = 0.0 if support_tol is None else _nonnegative_scalar(
        support_tol, "support_tol"
    )
    if custom_solver:
        if name not in ("auto", "apa_apg2"):
            raise ValueError("an explicit solver conflicts with the custom point-solver hook")
        return "apa_apg2"

    linf_regression = is_regressor(estimator) and ord in ("inf", np.inf)
    native = linf_regression and matched_training and _native_tree_eligible(estimator)
    if name == "auto":
        if native and support_tol == 0:
            return "topology"
        return "proximal" if linf_regression else "apa_apg2"
    if name in ("topology", "coefficient_path"):
        if not native:
            raise ValueError(
                f"solver={name!r} requires a fitted single-output, mean-based "
                "regression tree, ord=inf, its fitting rows/weights, and no "
                "active monotonic constraints"
            )
        if support_tol > 0:
            raise ValueError(f"solver={name!r} requires support_tol=None or 0")
    elif name in ("proximal", "hicap") and not linf_regression:
        raise ValueError(f"solver={name!r} requires regression with ord=inf")
    return name


@dataclass(frozen=True)
class FittedTreeSolution:
    """A solve on the original tree, before the estimator prunes or compacts it.

    Coefficients, when present, use unnormalized local stumps in ``node_ids``
    order. Topology-only solves deliberately return ``coefficients=None``.
    """

    topology_path: TreeTopologyPath
    coefficient_path: RegularizationPath | None
    coefficients: np.ndarray | None
    intercept: float
    node_ids: np.ndarray
    retained_node_ids: np.ndarray
    info: dict[str, Any]


def _snapshot(values: Any, dtype: Any) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _topology_from_statistics(scores, parents, node_ids, source) -> TreeTopologyPath:
    path = tree_group_linf_exact_topology_path(scores, parents, node_ids=node_ids)
    return replace(path, metadata={
        **path.metadata,
        "problem": "fitted_regression_tree",
        "source": "fitted_tree_sufficient_statistics",
        "tree_node_ids": tuple(int(node) for node in node_ids),
        "fitted_tree_fingerprint": source["fitted_tree_fingerprint"],
        "training_design_materialized": False,
    })


def _diagonal_coefficients(scores, diagonal, alpha, tolerance, operator):
    if scores.size == 0:
        return np.empty(0), {"certified": True, "certificate_residual": 0.0}
    sqrt_diagonal = np.sqrt(diagonal)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = scores / sqrt_diagonal
    theta, certificate = operator(value, alpha, return_info=True)
    residuals = np.asarray([
        abs(certificate["raw_relative_duality_gap"]),
        certificate["max_relative_dual_l1_violation"],
        certificate["relative_moreau_residual"],
    ])
    if not np.all(np.isfinite(residuals)) or np.max(residuals) > max(
        tolerance, 1024 * np.finfo(float).eps
    ):
        raise RuntimeError("the fitted-tree proximal solution could not be certified")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        coefficients = theta / sqrt_diagonal
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("fitted-tree coefficients are not representable; rescale the problem")
    return coefficients, {
        "certified": True,
        "certificate_residual": float(np.max(residuals)),
    }


def iter_fitted_tree_solutions(
    estimator: DecisionTreeRegressor,
    alphas,
    solver: str,
    *,
    tol: float,
    max_iter: int,
):
    """Yield point results, preparing one tree's statistics/path only once.

    Supports ``topology``, ``proximal``, and ``coefficient_path``; the caller
    routes general design-based algorithms separately. ``max_iter`` caps
    coefficient-path events, not the noniterative topology/proximal sweeps.
    For compatibility, alpha zero retains every original split even when its
    mathematical coefficient is zero. Numerical failures are raised, not
    silently replaced by another algorithm.
    """
    solver = validate_solver(solver)
    if solver not in ("topology", "proximal", "coefficient_path"):
        raise ValueError("solve_fitted_tree requires topology, proximal, or coefficient_path")
    alphas = [_nonnegative_scalar(alpha, "alpha") for alpha in alphas]
    tol = _nonnegative_scalar(tol, "tol")
    if tol == 0:
        raise ValueError("tol must be positive")
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0
    ):
        raise ValueError("max_iter must be a positive integer")

    coefficient_path = None
    info = {"solver": solver, "status": "complete", "converged": True,
            "n_iter": 0, "relative_step_norm": 0.0}
    if solver == "topology":
        topology = fitted_tree_linf_exact_topology_path(estimator)
        node_ids = topology.node_ids
    elif solver == "coefficient_path":
        coefficient_path = fitted_tree_linf_exact_coefficient_path(
            estimator, tolerance=tol, max_events=max_iter
        )
        if not coefficient_path.exact or coefficient_path.status != "complete":
            raise RuntimeError(
                "a complete certified coefficient path is required; got "
                f"{coefficient_path.status!r}: "
                f"{coefficient_path.metadata.get('failure_message', '')}"
            )
        source = coefficient_path.metadata
        node_ids = np.asarray(source["tree_node_ids"], dtype=np.intp)
        topology = _topology_from_statistics(
            source["linear_scores"], source["parent_indices"], node_ids, source
        )
        info.update(certified=True, n_iter=int(source.get("n_oracle_calls", 0)))
    else:
        scores, diagonal, parents, node_ids, source = _fitted_tree_statistics(estimator)
        topology = _topology_from_statistics(scores, parents, node_ids, source)
    intercept = float(estimator.tree_.value[0, 0, 0])
    node_ids = _snapshot(node_ids, np.intp)
    operator = None
    for alpha in alphas:
        coefficients = None
        point_info = dict(info)
        if solver == "coefficient_path":
            coefficients = (
                np.zeros(node_ids.size) if alpha >= coefficient_path.lambdas[0]
                else coefficient_path.at(alpha)[0]
            )
        elif solver == "proximal":
            if alpha == 0:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    coefficients = scores / diagonal
            elif not np.any(topology.activation_lambdas > alpha):
                coefficients = np.zeros(node_ids.size)
            else:
                if operator is None:
                    # Only interior queries need explicit descendant groups.
                    groups = [[node] for node in range(scores.size)]
                    for child in range(scores.size - 1, -1, -1):
                        if parents[child] >= 0:
                            groups[int(parents[child])].extend(groups[child])
                    operator = LaminarGroupLinfProx(
                        groups, scores.size, coordinate_weights=1 / np.sqrt(diagonal)
                    )
                coefficients, certificate = _diagonal_coefficients(
                    scores, diagonal, alpha, tol, operator
                )
                point_info.update(certificate, n_iter=1)
            if not np.all(np.isfinite(coefficients)):
                raise ValueError("fitted-tree coefficients are not representable; rescale the problem")
            point_info.setdefault("certified", True)
            point_info.setdefault("certificate_residual", 0.0)

        retained = node_ids if alpha == 0 else topology.tree_nodes_at(alpha)
        yield FittedTreeSolution(
            topology_path=topology, coefficient_path=coefficient_path,
            coefficients=None if coefficients is None else _snapshot(coefficients, float),
            intercept=intercept, node_ids=node_ids,
            retained_node_ids=_snapshot(retained, np.intp), info=point_info,
        )


def solve_fitted_tree(estimator, alpha, solver, *, tol, max_iter):
    """Solve a single fitted-tree penalty; use the iterator for repeated queries."""
    return next(iter_fitted_tree_solutions(
        estimator, [alpha], solver, tol=tol, max_iter=max_iter
    ))


def _callable_accepts_keyword(function, keyword):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get(keyword)
    return (
        parameter is not None and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
    ) or any(
        value.kind == inspect.Parameter.VAR_KEYWORD
        for value in signature.parameters.values()
    )


def solve_design(X, y, groups, alpha, solver, *, point_solver,
                 sample_weight, beta_init, gamma1, a, max_iter, tol, ord, verbose):
    """Dispatch a general design solve without mutating any estimator.

    Groups index stump columns (no intercept). Returns intercept-first
    coefficients, point diagnostics, and an optional complete coefficient path.
    The historical custom hook still receives an explicit intercept column.
    """
    path = None
    if solver == "proximal":
        from .optimization.proximal import laminar_group_linf_regression

        coefficients, info = laminar_group_linf_regression(
            X, y, groups, alpha, sample_weight=sample_weight, fit_intercept=True,
            max_iter=max_iter, tol=tol, return_info=True,
        )
        info = dict(info, solver=solver)
        if not info.get("certified", False):
            raise RuntimeError(
                "The proximal pruning solution did not pass its certificate; "
                "increase max_iter or relax tol."
            )
        result = (np.r_[info["intercept"], coefficients], info)
    elif solver == "hicap":
        from .optimization.hicap import hicap_regression_path

        path = hicap_regression_path(
            X, y, groups, sample_weight=sample_weight, fit_intercept=True,
            tolerance=tol, max_events=max_iter,
        )
        return coefficient_path_solution(path, alpha)
    elif solver == "apa_apg2":
        kwargs = dict(
            X=np.column_stack((np.ones(len(y)), X)), y=y,
            groups=[group + 1 for group in groups], lam=alpha,
            beta_init=beta_init, gamma1=gamma1, a=a, max_iter=max_iter,
            tol=tol, ord=ord, verbose=verbose,
        )
        if point_solver in (hiCAP_regression, hiCAP_classification):
            result = point_solver(return_info=True, sample_weight=sample_weight, **kwargs)
        else:
            if sample_weight is not None:
                if not _callable_accepts_keyword(point_solver, "sample_weight"):
                    raise TypeError(
                        "A custom hiCAP solver used with sample_weight must "
                        "accept a sample_weight keyword argument"
                    )
                kwargs["sample_weight"] = sample_weight
            result = point_solver(**kwargs)
    else:
        raise ValueError(f"solver={solver!r} does not support general design inputs")

    beta, info = result if (
        isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict)
    ) else (result, None)
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (X.shape[1] + 1,):
        raise ValueError(
            f"hiCAP solver returned coefficients with shape {beta.shape}; "
            f"expected ({X.shape[1] + 1},)"
        )
    if not np.all(np.isfinite(beta)):
        raise FloatingPointError("hiCAP solver returned non-finite coefficients")
    return beta, info, path


def coefficient_path_solution(path, alpha):
    """Read a certified reference path, including its constant upper tail."""
    if not path.exact or path.status != "complete":
        raise RuntimeError(f"hiCAP coefficient path is incomplete: {path.status}")
    coefficients, intercept = (
        (np.zeros(path.coefficients.shape[1]), float(path.intercepts[0]))
        if alpha >= path.lambdas[0] else path.at(alpha)
    )
    return np.r_[intercept, coefficients], {
        "solver": "hicap", "converged": True, "n_iter": path.n_points,
        "relative_step_norm": 0.0, "status": "complete",
    }, path
