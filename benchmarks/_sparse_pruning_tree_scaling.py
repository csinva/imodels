"""Shared utilities for real-tree sparse-pruning path benchmarks.

This module deliberately contains no command-line or plotting code.  It
provides the deterministic data/problem construction, solver adapters, result
schema, comparisons, aggregation, and serialization used by those layers.

The benchmark design is the local-stump basis produced from an actual fitted
``DecisionTreeRegressor``.  It is centered once, so both the exact hiCAP path
and APA-APG2 solve the same intercept-free normalized least-squares problem::

    0.5 * ||y - X beta||_2**2 / n
        + lambda * sum(group in groups) ||beta[group]||_inf
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import URLError

import numpy as np
import scipy
import sklearn
from scipy.optimize import linprog
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF

from imodels.importance.local_stumps import make_stumps, tree_feature_transform
from imodels.tree.sparse_pruning.optimization.apa import apa_apg_regression_path
from imodels.tree.sparse_pruning.optimization.hicap import hicap_regression_path


SCHEMA_VERSION = 2
METHOD_EXACT = "hicap_exact"
METHOD_APA_WARM = "apa_apg2_warm"
SUPPORTED_METHODS = (METHOD_EXACT, METHOD_APA_WARM)


def _readonly_float_array(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=float, order="C", copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _dataset_digest(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]
) -> str:
    """Return a stable digest independent of native endian and array layout."""

    digest = hashlib.sha256()
    for array in (X, y):
        canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    digest.update(
        json.dumps(list(feature_names), ensure_ascii=True, separators=(",", ":"))
        .encode("utf-8")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class RegressionDataset:
    """A numeric regression dataset with provenance for reproducible runs."""

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    name: str
    requested_name: str
    source: str = "sklearn"
    digest: str = ""
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        X = _readonly_float_array(self.X, ndim=2, name="X")
        y = _readonly_float_array(self.y, ndim=1, name="y")
        if X.shape[0] != y.size or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("dataset X and y must have compatible nonempty shapes")
        names = tuple(str(value) for value in self.feature_names)
        if len(names) != X.shape[1]:
            raise ValueError("feature_names must have one entry per X column")
        digest = self.digest or _dataset_digest(X, y, names)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "digest", digest)


def _diabetes_dataset(requested_name: str, fallback_reason: str | None = None):
    bunch = load_diabetes()
    return RegressionDataset(
        X=bunch.data,
        y=bunch.target,
        feature_names=tuple(str(value) for value in bunch.feature_names),
        name="diabetes",
        requested_name=requested_name,
        fallback_reason=fallback_reason,
    )


def load_regression_dataset(
    name: str = "diabetes",
    *,
    data_home: str | os.PathLike[str] | None = None,
    allow_download: bool = False,
    fallback_to_diabetes: bool = True,
) -> RegressionDataset:
    """Load a stable sklearn regression dataset without implicit networking.

    ``diabetes`` is bundled with sklearn.  ``california`` uses an existing
    sklearn cache unless ``allow_download=True``.  If California is unavailable
    and ``fallback_to_diabetes`` is true, the returned provenance explicitly
    records that fallback so results can never silently claim another dataset.
    """

    requested = str(name).strip().lower().replace("-", "_")
    if requested in {"diabetes", "diabetes_regression", "diabetes_regr"}:
        return _diabetes_dataset(requested)
    if requested not in {"california", "california_housing"}:
        raise ValueError("name must be 'diabetes' or 'california_housing'")

    try:
        bunch = fetch_california_housing(
            data_home=None if data_home is None else os.fspath(data_home),
            download_if_missing=bool(allow_download),
        )
    except (OSError, URLError) as exc:
        if not fallback_to_diabetes:
            raise RuntimeError(
                "California housing is not cached; pass allow_download=True "
                "or enable the diabetes fallback"
            ) from exc
        reason = f"{type(exc).__name__}: {exc}"
        return _diabetes_dataset(requested, fallback_reason=reason)

    return RegressionDataset(
        X=bunch.data,
        y=bunch.target,
        feature_names=tuple(str(value) for value in bunch.feature_names),
        name="california_housing",
        requested_name=requested,
    )


def subset_regression_dataset(
    dataset: RegressionDataset,
    n_samples: int,
    *,
    seed: int,
) -> RegressionDataset:
    """Select a deterministic without-replacement subset.

    Reusing the same seed for several sizes gives nested subsets because each
    size takes a prefix of the same random permutation.
    """

    if isinstance(n_samples, (bool, np.bool_)) or not isinstance(
        n_samples, (int, np.integer)
    ):
        raise ValueError("n_samples must be a positive integer")
    n_samples = int(n_samples)
    if n_samples < 2 or n_samples > dataset.y.size:
        raise ValueError(
            f"n_samples must lie in [2, {dataset.y.size}]; got {n_samples}"
        )
    order = np.random.default_rng(seed).permutation(dataset.y.size)[:n_samples]
    # Preserve the dataset's original row ordering after choosing the subset;
    # tree results then do not depend on the arbitrary permutation order.
    indices = np.sort(order)
    X = dataset.X[indices]
    y = dataset.y[indices]
    return RegressionDataset(
        X=X,
        y=y,
        feature_names=dataset.feature_names,
        name=dataset.name,
        requested_name=dataset.requested_name,
        source=dataset.source,
        fallback_reason=dataset.fallback_reason,
    )


@dataclass(frozen=True)
class TreePathProblem:
    """Centered local-stump design and hierarchy for one fitted real tree."""

    X: np.ndarray
    y: np.ndarray
    groups: tuple[np.ndarray, ...]
    parent_indices: np.ndarray
    node_ids: np.ndarray
    x_mean: np.ndarray
    y_mean: float
    tree_linear_scores: np.ndarray
    tree_gram_diagonal: np.ndarray
    dataset_name: str
    dataset_digest: str
    seed: int
    requested_internal_nodes: int
    actual_internal_nodes: int
    raw_feature_count: int
    tree_node_count: int
    tree_depth: int
    design_rank: int
    gram_condition: float
    gram_max_off_diagonal: float
    gram_max_off_diagonal_correlation: float
    group_memberships: int
    epigraph_rows: int
    tree_fit_seconds: float
    transform_seconds: float

    def __post_init__(self) -> None:
        X = _readonly_float_array(self.X, ndim=2, name="X")
        y = _readonly_float_array(self.y, ndim=1, name="y")
        if X.shape != (y.size, self.actual_internal_nodes):
            raise ValueError("tree-path X shape does not match y/nodes")
        if len(self.groups) != self.actual_internal_nodes:
            raise ValueError("there must be one descendant group per internal node")
        groups: list[np.ndarray] = []
        for number, raw_group in enumerate(self.groups):
            group = np.asarray(raw_group, dtype=np.intp)
            if group.ndim != 1 or group.size == 0:
                raise ValueError(f"group {number} must be a nonempty vector")
            if np.any(group < 0) or np.any(group >= self.actual_internal_nodes):
                raise ValueError(f"group {number} contains an invalid node index")
            group = np.unique(group)
            group.setflags(write=False)
            groups.append(group)
        if not np.array_equal(groups[0], np.arange(self.actual_internal_nodes)):
            raise ValueError("the root descendant group must contain every node")
        parents = np.asarray(self.parent_indices, dtype=np.intp)
        node_ids = np.asarray(self.node_ids, dtype=np.intp)
        x_mean = np.asarray(self.x_mean, dtype=float)
        tree_linear_scores = np.asarray(self.tree_linear_scores, dtype=float)
        tree_gram_diagonal = np.asarray(self.tree_gram_diagonal, dtype=float)
        expected = (self.actual_internal_nodes,)
        if (
            parents.shape != expected
            or node_ids.shape != expected
            or tree_linear_scores.shape != expected
            or tree_gram_diagonal.shape != expected
        ):
            raise ValueError("parent_indices and node_ids must match node count")
        if parents.size and parents[0] != -1:
            raise ValueError("the root parent index must be -1")
        if x_mean.shape != expected:
            raise ValueError("x_mean must match node count")
        if (
            np.any(~np.isfinite(tree_linear_scores))
            or np.any(~np.isfinite(tree_gram_diagonal))
            or np.any(tree_gram_diagonal <= 0.0)
        ):
            raise ValueError("tree sufficient statistics must be finite and positive")
        parents = np.ascontiguousarray(parents)
        node_ids = np.ascontiguousarray(node_ids)
        x_mean = np.ascontiguousarray(x_mean)
        tree_linear_scores = np.ascontiguousarray(tree_linear_scores)
        tree_gram_diagonal = np.ascontiguousarray(tree_gram_diagonal)
        parents.setflags(write=False)
        node_ids.setflags(write=False)
        x_mean.setflags(write=False)
        tree_linear_scores.setflags(write=False)
        tree_gram_diagonal.setflags(write=False)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "groups", tuple(groups))
        object.__setattr__(self, "parent_indices", parents)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "x_mean", x_mean)
        object.__setattr__(self, "tree_linear_scores", tree_linear_scores)
        object.__setattr__(self, "tree_gram_diagonal", tree_gram_diagonal)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def _internal_nodes_and_parents(tree: Any) -> tuple[np.ndarray, np.ndarray]:
    node_ids: list[int] = []
    parent_indices: list[int] = []

    def traverse(node_id: int, parent_position: int) -> None:
        if node_id == TREE_LEAF or tree.feature[node_id] < 0:
            return
        position = len(node_ids)
        node_ids.append(int(node_id))
        parent_indices.append(parent_position)
        traverse(int(tree.children_left[node_id]), position)
        traverse(int(tree.children_right[node_id]), position)

    traverse(0, -1)
    return np.asarray(node_ids, dtype=np.intp), np.asarray(
        parent_indices, dtype=np.intp
    )


def descendant_groups(parent_indices: Sequence[int]) -> tuple[np.ndarray, ...]:
    """Construct one zero-based internal-descendant group per tree node."""

    parents = np.asarray(parent_indices, dtype=np.intp)
    if parents.ndim != 1 or parents.size == 0 or parents[0] != -1:
        raise ValueError("parent_indices must be a nonempty vector rooted at -1")
    children: list[list[int]] = [[] for _ in range(parents.size)]
    for child, parent in enumerate(parents[1:], start=1):
        if parent < 0 or parent >= child:
            raise ValueError("parents must precede their children in traversal order")
        children[int(parent)].append(child)

    groups: list[np.ndarray | None] = [None] * parents.size

    def collect(node: int) -> list[int]:
        values = [node]
        for child in children[node]:
            values.extend(collect(child))
        groups[node] = np.asarray(values, dtype=np.intp)
        return values

    collect(0)
    return tuple(group for group in groups if group is not None)


def build_tree_path_problem(
    X: np.ndarray,
    y: np.ndarray,
    requested_internal_nodes: int,
    *,
    seed: int = 0,
    dataset_name: str = "custom",
    dataset_digest: str | None = None,
    min_samples_leaf: int = 1,
) -> TreePathProblem:
    """Fit a regression tree and construct its centered local-stump problem."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("X and y must be a compatible matrix/vector pair")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must contain at least two rows and one column")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values")
    for value, name in (
        (requested_internal_nodes, "requested_internal_nodes"),
        (min_samples_leaf, "min_samples_leaf"),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
    requested_internal_nodes = int(requested_internal_nodes)
    min_samples_leaf = int(min_samples_leaf)

    estimator = DecisionTreeRegressor(
        max_leaf_nodes=requested_internal_nodes + 1,
        min_samples_leaf=min_samples_leaf,
        random_state=int(seed),
    )
    start = time.perf_counter()
    estimator.fit(X, y)
    tree_fit_seconds = time.perf_counter() - start
    tree = estimator.tree_
    node_ids, parents = _internal_nodes_and_parents(tree)
    if node_ids.size == 0:
        raise ValueError("the fitted tree has no internal nodes")

    start = time.perf_counter()
    stumps = make_stumps(tree)
    X_tree = tree_feature_transform(stumps, X)
    transform_seconds = time.perf_counter() - start
    if X_tree.shape[1] != node_ids.size:
        raise RuntimeError(
            "local-stump traversal order did not match internal-node traversal"
        )

    x_mean = np.mean(X_tree, axis=0)
    y_mean = float(np.mean(y))
    centered_X = np.asarray(X_tree - x_mean, dtype=float)
    centered_y = np.asarray(y - y_mean, dtype=float)
    groups = descendant_groups(parents)
    p = int(node_ids.size)
    H = centered_X.T @ centered_X / centered_X.shape[0]
    design_rank = int(np.linalg.matrix_rank(centered_X))
    gram_condition = float(np.linalg.cond(H))
    gram_diagonal = np.diag(H).copy()
    off_diagonal = H.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    gram_max_off_diagonal = float(
        np.max(np.abs(off_diagonal), initial=0.0)
    )
    correlation_scale = np.sqrt(
        np.maximum(gram_diagonal, 0.0)[:, None]
        * np.maximum(gram_diagonal, 0.0)[None, :]
    )
    off_diagonal_correlation = np.zeros_like(H)
    positive_scale = correlation_scale > 0.0
    off_diagonal_correlation[positive_scale] = (
        np.abs(off_diagonal[positive_scale])
        / correlation_scale[positive_scale]
    )
    off_diagonal_correlation[
        ~positive_scale & (off_diagonal != 0.0)
    ] = np.inf
    gram_max_off_diagonal_correlation = float(
        np.max(off_diagonal_correlation, initial=0.0)
    )
    tree_weight = np.asarray(tree.weighted_n_node_samples, dtype=float)
    tree_value = np.asarray(tree.value, dtype=float)
    root_weight = float(tree_weight[0])
    tree_linear_scores = np.empty(p, dtype=float)
    tree_gram_diagonal = np.empty(p, dtype=float)
    for position, node_id in enumerate(node_ids):
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        left_weight = float(tree_weight[left])
        right_weight = float(tree_weight[right])
        tree_linear_scores[position] = (
            np.sqrt(left_weight * right_weight)
            * (float(tree_value[right, 0, 0]) - float(tree_value[left, 0, 0]))
            / root_weight
        )
        tree_gram_diagonal[position] = tree_weight[node_id] / root_weight
    memberships = int(sum(group.size for group in groups))
    if dataset_digest is None:
        names = tuple(f"x{index}" for index in range(X.shape[1]))
        dataset_digest = _dataset_digest(X, y, names)
    return TreePathProblem(
        X=centered_X,
        y=centered_y,
        groups=groups,
        parent_indices=parents,
        node_ids=node_ids,
        x_mean=x_mean,
        y_mean=y_mean,
        tree_linear_scores=tree_linear_scores,
        tree_gram_diagonal=tree_gram_diagonal,
        dataset_name=str(dataset_name),
        dataset_digest=str(dataset_digest),
        seed=int(seed),
        requested_internal_nodes=requested_internal_nodes,
        actual_internal_nodes=p,
        raw_feature_count=int(X.shape[1]),
        tree_node_count=int(tree.node_count),
        tree_depth=int(tree.max_depth),
        design_rank=design_rank,
        gram_condition=gram_condition,
        gram_max_off_diagonal=gram_max_off_diagonal,
        gram_max_off_diagonal_correlation=(
            gram_max_off_diagonal_correlation
        ),
        group_memberships=memberships,
        epigraph_rows=2 * memberships,
        tree_fit_seconds=float(tree_fit_seconds),
        transform_seconds=float(transform_seconds),
    )


def zero_solution_lambda_max(problem: TreePathProblem) -> float:
    """Compute the exact zero-solution threshold by a subgradient LP."""

    p = problem.n_features
    m = len(problem.groups)
    row_group: list[int] = []
    row_feature: list[int] = []
    row_sign: list[float] = []
    for group_number, group in enumerate(problem.groups):
        for feature in group:
            for sign in (1.0, -1.0):
                row_group.append(group_number)
                row_feature.append(int(feature))
                row_sign.append(sign)
    row_group_array = np.asarray(row_group, dtype=np.intp)
    row_feature_array = np.asarray(row_feature, dtype=np.intp)
    row_sign_array = np.asarray(row_sign, dtype=float)
    n_rows = row_group_array.size
    equality = np.zeros((p + m, n_rows + 1), dtype=float)
    equality[row_feature_array, np.arange(n_rows)] = row_sign_array
    equality[p + row_group_array, np.arange(n_rows)] = 1.0
    equality[p:, -1] = -1.0
    score = problem.X.T @ problem.y / problem.n_samples
    rhs = np.r_[score, np.zeros(m)]
    objective = np.zeros(n_rows + 1)
    objective[-1] = 1.0
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=rhs,
        bounds=[(0.0, None)] * (n_rows + 1),
        method="highs-ds",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"could not compute lambda_max: {result.message}")
    residual = float(np.max(np.abs(equality @ result.x - rhs)))
    scale = max(1.0, float(np.max(np.abs(score))))
    if residual > 1e-7 * scale:
        raise RuntimeError("lambda_max LP failed its residual check")
    return max(0.0, float(result.x[-1]))


@dataclass(frozen=True)
class LambdaGrid:
    lambdas: np.ndarray
    lambda_max: float
    lambda_upper: float
    minimum_ratio: float
    includes_zero: bool
    setup_seconds: float = 0.0

    def __post_init__(self) -> None:
        lambdas = _readonly_float_array(self.lambdas, ndim=1, name="lambdas")
        if lambdas.size == 0 or np.any(lambdas < 0) or np.any(np.diff(lambdas) > 0):
            raise ValueError("lambdas must be nonempty, nonnegative, and descending")
        object.__setattr__(self, "lambdas", lambdas)


def make_lambda_grid(
    problem: TreePathProblem,
    n_points: int = 40,
    minimum_ratio: float = 1e-3,
    *,
    lambda_max: float | None = None,
    upper_factor: float = 1.05,
    include_zero: bool = False,
) -> LambdaGrid:
    """Create a common descending APA grid around the true zero threshold."""

    if isinstance(n_points, (bool, np.bool_)) or not isinstance(
        n_points, (int, np.integer)
    ) or int(n_points) < 2:
        raise ValueError("n_points must be an integer of at least two")
    if not np.isfinite(minimum_ratio) or not 0 < minimum_ratio < 1:
        raise ValueError("minimum_ratio must lie strictly between zero and one")
    if not np.isfinite(upper_factor) or upper_factor < 1:
        raise ValueError("upper_factor must be finite and at least one")
    setup_started = time.perf_counter()
    if lambda_max is None:
        lambda_max = zero_solution_lambda_max(problem)
    lambda_max = float(lambda_max)
    if not np.isfinite(lambda_max) or lambda_max <= 0:
        raise ValueError("lambda_max must be a positive finite value")
    lambda_upper = float(upper_factor * lambda_max)
    positive_points = int(n_points) - int(bool(include_zero))
    if positive_points < 1:
        raise ValueError("include_zero leaves no positive grid point")
    lambdas = np.geomspace(
        lambda_upper,
        lambda_upper * float(minimum_ratio),
        num=positive_points,
    )
    if include_zero:
        lambdas = np.r_[lambdas, 0.0]
    return LambdaGrid(
        lambdas=lambdas,
        lambda_max=lambda_max,
        lambda_upper=lambda_upper,
        minimum_ratio=float(minimum_ratio),
        includes_zero=bool(include_zero),
        setup_seconds=float(time.perf_counter() - setup_started),
    )


@dataclass(frozen=True)
class ScalingConfig:
    """Numerical controls shared by subprocess solver workers."""

    max_iter: int = 2_000
    tolerance: float = 1e-7
    support_tolerance: float = 1e-6
    max_events: int = 10_000

    def __post_init__(self) -> None:
        if self.max_iter < 1 or self.max_events < 1:
            raise ValueError("max_iter and max_events must be positive")
        if self.tolerance <= 0 or self.support_tolerance <= 0:
            raise ValueError("tolerances must be positive")


@dataclass(frozen=True)
class SolverPayload:
    problem: TreePathProblem
    method: str
    lambda_grid: LambdaGrid
    config: ScalingConfig = field(default_factory=ScalingConfig)

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {SUPPORTED_METHODS}")


@dataclass(frozen=True)
class SolverResult:
    """Pickle-friendly worker result; path arrays are excluded from CSV rows."""

    method: str
    outcome: str
    status: str
    exact: bool
    wall_seconds: float | None
    lambdas: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray | None = None
    native_points: int = 0
    total_iterations: int | None = None
    all_points_converged: bool | None = None
    n_regions: int | None = None
    probe_attempts: int | None = None
    max_kkt_residual: float | None = None
    max_active_kkt_condition: float | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        lambdas = np.asarray(self.lambdas, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        if lambdas.ndim != 1 or coefficients.ndim != 2:
            raise ValueError("solver path arrays must be 1D/2D")
        if coefficients.shape[0] != lambdas.size:
            raise ValueError("coefficient rows must match lambda count")
        intercepts = self.intercepts
        if intercepts is not None:
            intercepts = np.asarray(intercepts, dtype=float)
            if intercepts.shape != lambdas.shape:
                raise ValueError("intercepts must match lambda count")
        object.__setattr__(self, "lambdas", lambdas)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercepts", intercepts)


def terminal_solver_result(
    problem: TreePathProblem,
    method: str,
    outcome: str,
    *,
    wall_seconds: float | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> SolverResult:
    """Construct an empty timeout/crash/skip result in the shared schema."""

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")
    if outcome not in {"timeout", "failed", "crashed", "skipped"}:
        raise ValueError("outcome must describe a terminal non-complete run")
    return SolverResult(
        method=method,
        outcome=outcome,
        status=outcome,
        exact=False,
        wall_seconds=wall_seconds,
        lambdas=np.empty(0),
        coefficients=np.empty((0, problem.n_features)),
        error_type=error_type,
        error_message=error_message,
    )


def run_solver_payload(payload: SolverPayload) -> SolverResult:
    """Run one method and convert all Python failures to structured output."""

    problem = payload.problem
    config = payload.config
    start = time.perf_counter()
    try:
        if payload.method == METHOD_EXACT:
            path = hicap_regression_path(
                problem.X,
                problem.y,
                problem.groups,
                fit_intercept=False,
                tolerance=config.tolerance,
                max_events=config.max_events,
                oracle_max_iter=config.max_iter,
            )
        elif payload.method == METHOD_APA_WARM:
            path = apa_apg_regression_path(
                problem.X,
                problem.y,
                problem.groups,
                payload.lambda_grid.lambdas,
                max_iter=config.max_iter,
                tol=config.tolerance,
                ord="inf",
            )
        else:  # guarded by SolverPayload, retained for untrusted deserialization
            raise ValueError(f"unsupported method {payload.method!r}")
        wall_seconds = time.perf_counter() - start
        iterations = [
            int(item["n_iter"])
            for item in path.diagnostics
            if isinstance(item, Mapping) and item.get("n_iter") is not None
        ]
        convergence = [
            bool(item["converged"])
            for item in path.diagnostics
            if isinstance(item, Mapping) and item.get("converged") is not None
        ]
        metadata = dict(path.metadata)
        failure_message = metadata.get("failure_message")
        return SolverResult(
            method=payload.method,
            outcome="complete",
            status=str(path.status),
            exact=bool(path.exact),
            wall_seconds=float(wall_seconds),
            lambdas=np.asarray(path.lambdas),
            coefficients=np.asarray(path.coefficients),
            intercepts=(
                None if path.intercepts is None else np.asarray(path.intercepts)
            ),
            native_points=int(path.n_points),
            total_iterations=sum(iterations) if iterations else None,
            all_points_converged=all(convergence) if convergence else None,
            n_regions=_optional_int(metadata.get("n_regions")),
            probe_attempts=_optional_int(metadata.get("probe_attempts")),
            max_kkt_residual=_optional_float(metadata.get("max_kkt_residual")),
            max_active_kkt_condition=_optional_float(
                metadata.get("max_active_kkt_condition")
            ),
            error_type=(
                "PathContinuationFailure" if failure_message is not None else None
            ),
            error_message=(
                None if failure_message is None else str(failure_message)
            ),
        )
    except Exception as exc:  # worker boundary must always return a record
        return SolverResult(
            method=payload.method,
            outcome="failed",
            status="failed",
            exact=False,
            wall_seconds=float(time.perf_counter() - start),
            lambdas=np.empty(0),
            coefficients=np.empty((0, problem.n_features)),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def evaluate_solver_result(
    result: SolverResult, lambdas: Sequence[float]
) -> np.ndarray:
    """Interpolate path coefficients with the shared right-continuous rule."""

    query = np.asarray(lambdas, dtype=float)
    if result.outcome != "complete" or result.lambdas.size == 0:
        raise ValueError("only completed nonempty paths can be evaluated")
    order = np.argsort(result.lambdas, kind="stable")
    xp = result.lambdas[order]
    coefficients = result.coefficients[order]
    unique_xp, unique_indices = np.unique(xp, return_index=True)
    coefficients = coefficients[unique_indices]
    return np.column_stack(
        [
            np.interp(query, unique_xp, coefficients[:, column])
            for column in range(coefficients.shape[1])
        ]
    )


def group_penalty(beta: np.ndarray, groups: Sequence[np.ndarray]) -> float:
    return float(sum(np.max(np.abs(beta[group])) for group in groups))


def objective_values(
    problem: TreePathProblem,
    lambdas: Sequence[float],
    coefficients: np.ndarray,
) -> np.ndarray:
    values = []
    for lam, beta in zip(np.asarray(lambdas), np.asarray(coefficients)):
        residual = problem.y - problem.X @ beta
        values.append(
            0.5 * float(residual @ residual) / problem.n_samples
            + float(lam) * group_penalty(beta, problem.groups)
        )
    return np.asarray(values)


def topology_signature(
    beta: np.ndarray, parent_indices: np.ndarray, tolerance: float
) -> tuple[int, ...]:
    """Return the ancestor closure of coefficients exceeding ``tolerance``."""

    topology: set[int] = set()
    for active in np.flatnonzero(np.abs(beta) > tolerance):
        node = int(active)
        while node >= 0:
            topology.add(node)
            node = int(parent_indices[node])
    return tuple(sorted(topology))


def compare_paths(
    problem: TreePathProblem,
    reference: SolverResult,
    candidate: SolverResult,
    lambdas: Sequence[float],
    *,
    support_tolerance: float = 1e-6,
) -> dict[str, float | int]:
    """Compare two completed paths on a common grid."""

    query = np.asarray(lambdas, dtype=float)
    reference_beta = evaluate_solver_result(reference, query)
    candidate_beta = evaluate_solver_result(candidate, query)
    differences = np.linalg.norm(candidate_beta - reference_beta, axis=1)
    denominators = np.maximum(np.linalg.norm(reference_beta, axis=1), 1.0)
    relative = differences / denominators
    reference_objective = objective_values(problem, query, reference_beta)
    candidate_objective = objective_values(problem, query, candidate_beta)
    objective_denominator = np.maximum(np.abs(reference_objective), 1.0)
    objective_excess = (
        candidate_objective - reference_objective
    ) / objective_denominator
    reference_supports = np.abs(reference_beta) > support_tolerance
    candidate_supports = np.abs(candidate_beta) > support_tolerance
    support_agreement = np.all(reference_supports == candidate_supports, axis=1)
    reference_topologies = [
        topology_signature(beta, problem.parent_indices, support_tolerance)
        for beta in reference_beta
    ]
    candidate_topologies = [
        topology_signature(beta, problem.parent_indices, support_tolerance)
        for beta in candidate_beta
    ]
    topology_agreement = np.asarray(
        [
            left == right
            for left, right in zip(reference_topologies, candidate_topologies)
        ]
    )
    return {
        "coefficient_relative_error_median": float(np.median(relative)),
        "coefficient_relative_error_max": float(np.max(relative)),
        "objective_relative_excess_median": float(np.median(objective_excess)),
        "objective_relative_excess_max": float(np.max(objective_excess)),
        "support_agreement_fraction": float(np.mean(support_agreement)),
        "topology_agreement_fraction": float(np.mean(topology_agreement)),
        "reference_unique_supports": len(
            {tuple(np.flatnonzero(row).tolist()) for row in reference_supports}
        ),
        "candidate_unique_supports": len(
            {tuple(np.flatnonzero(row).tolist()) for row in candidate_supports}
        ),
        "reference_unique_topologies": len(set(reference_topologies)),
        "candidate_unique_topologies": len(set(candidate_topologies)),
    }


@dataclass
class ScalingRecord:
    """Flat, CSV-safe schema for one method/repetition/scaling case."""

    schema_version: int = SCHEMA_VERSION
    scaling_axis: str = ""
    method: str = ""
    repeat: int = 0
    seed: int = 0
    dataset_name: str = ""
    dataset_digest: str = ""
    requested_samples: int = 0
    n_samples: int = 0
    requested_internal_nodes: int = 0
    n_internal_nodes: int = 0
    raw_feature_count: int = 0
    tree_node_count: int = 0
    tree_depth: int = 0
    n_groups: int = 0
    group_memberships: int = 0
    epigraph_rows: int = 0
    design_rank: int = 0
    gram_condition: float | None = None
    lambda_max: float | None = None
    lambda_upper: float | None = None
    lambda_minimum_ratio: float | None = None
    requested_path_points: int = 0
    metric_points: int = 0
    max_iter: int = 0
    max_events: int = 0
    min_samples_leaf: int = 0
    threads: int = 0
    tolerance: float | None = None
    support_tolerance: float | None = None
    timeout_seconds: float | None = None
    outcome: str = "pending"
    censored: bool = False
    solver_status: str = ""
    exact: bool = False
    parent_wall_seconds: float | None = None
    solver_wall_seconds: float | None = None
    tree_fit_seconds: float | None = None
    transform_seconds: float | None = None
    lambda_setup_seconds: float | None = None
    native_points: int | None = None
    total_iterations: int | None = None
    all_points_converged: bool | None = None
    n_regions: int | None = None
    probe_attempts: int | None = None
    max_kkt_residual: float | None = None
    max_active_kkt_condition: float | None = None
    coefficient_relative_error_median: float | None = None
    coefficient_relative_error_max: float | None = None
    objective_relative_excess_median: float | None = None
    objective_relative_excess_max: float | None = None
    support_agreement_fraction: float | None = None
    topology_agreement_fraction: float | None = None
    sample_coefficient_relative_error_median: float | None = None
    sample_coefficient_relative_error_max: float | None = None
    sample_objective_relative_excess_median: float | None = None
    sample_objective_relative_excess_max: float | None = None
    sample_support_agreement_fraction: float | None = None
    sample_topology_agreement_fraction: float | None = None
    reference_unique_supports: int | None = None
    candidate_unique_supports: int | None = None
    reference_unique_topologies: int | None = None
    candidate_unique_topologies: int | None = None
    error_type: str | None = None
    error_message: str | None = None


def make_scaling_record(
    problem: TreePathProblem,
    grid: LambdaGrid,
    config: ScalingConfig,
    result: SolverResult,
    *,
    scaling_axis: str,
    requested_samples: int,
    repeat: int,
    timeout_seconds: float | None = None,
    parent_wall_seconds: float | None = None,
    censored: bool = False,
    comparison: Mapping[str, Any] | None = None,
    metric_points: int = 0,
    min_samples_leaf: int = 1,
    threads: int = 1,
) -> ScalingRecord:
    """Combine immutable problem/result metadata into one flat row."""

    comparison = {} if comparison is None else comparison
    return ScalingRecord(
        scaling_axis=str(scaling_axis),
        method=result.method,
        repeat=int(repeat),
        seed=problem.seed,
        dataset_name=problem.dataset_name,
        dataset_digest=problem.dataset_digest,
        requested_samples=int(requested_samples),
        n_samples=problem.n_samples,
        requested_internal_nodes=problem.requested_internal_nodes,
        n_internal_nodes=problem.actual_internal_nodes,
        raw_feature_count=problem.raw_feature_count,
        tree_node_count=problem.tree_node_count,
        tree_depth=problem.tree_depth,
        n_groups=len(problem.groups),
        group_memberships=problem.group_memberships,
        epigraph_rows=problem.epigraph_rows,
        design_rank=problem.design_rank,
        gram_condition=problem.gram_condition,
        lambda_max=grid.lambda_max,
        lambda_upper=grid.lambda_upper,
        lambda_minimum_ratio=grid.minimum_ratio,
        requested_path_points=int(grid.lambdas.size),
        metric_points=int(metric_points),
        max_iter=config.max_iter,
        max_events=config.max_events,
        min_samples_leaf=int(min_samples_leaf),
        threads=int(threads),
        tolerance=config.tolerance,
        support_tolerance=config.support_tolerance,
        timeout_seconds=timeout_seconds,
        outcome=result.outcome,
        censored=bool(censored),
        solver_status=result.status,
        exact=result.exact,
        parent_wall_seconds=parent_wall_seconds,
        solver_wall_seconds=result.wall_seconds,
        tree_fit_seconds=problem.tree_fit_seconds,
        transform_seconds=problem.transform_seconds,
        lambda_setup_seconds=grid.setup_seconds,
        native_points=result.native_points,
        total_iterations=result.total_iterations,
        all_points_converged=result.all_points_converged,
        n_regions=result.n_regions,
        probe_attempts=result.probe_attempts,
        max_kkt_residual=result.max_kkt_residual,
        max_active_kkt_condition=result.max_active_kkt_condition,
        coefficient_relative_error_median=_optional_float(
            comparison.get("coefficient_relative_error_median")
        ),
        coefficient_relative_error_max=_optional_float(
            comparison.get("coefficient_relative_error_max")
        ),
        objective_relative_excess_median=_optional_float(
            comparison.get("objective_relative_excess_median")
        ),
        objective_relative_excess_max=_optional_float(
            comparison.get("objective_relative_excess_max")
        ),
        support_agreement_fraction=_optional_float(
            comparison.get("support_agreement_fraction")
        ),
        topology_agreement_fraction=_optional_float(
            comparison.get("topology_agreement_fraction")
        ),
        sample_coefficient_relative_error_median=_optional_float(
            comparison.get("sample_coefficient_relative_error_median")
        ),
        sample_coefficient_relative_error_max=_optional_float(
            comparison.get("sample_coefficient_relative_error_max")
        ),
        sample_objective_relative_excess_median=_optional_float(
            comparison.get("sample_objective_relative_excess_median")
        ),
        sample_objective_relative_excess_max=_optional_float(
            comparison.get("sample_objective_relative_excess_max")
        ),
        sample_support_agreement_fraction=_optional_float(
            comparison.get("sample_support_agreement_fraction")
        ),
        sample_topology_agreement_fraction=_optional_float(
            comparison.get("sample_topology_agreement_fraction")
        ),
        reference_unique_supports=_optional_int(
            comparison.get("reference_unique_supports")
        ),
        candidate_unique_supports=_optional_int(
            comparison.get("candidate_unique_supports")
        ),
        reference_unique_topologies=_optional_int(
            comparison.get("reference_unique_topologies")
        ),
        candidate_unique_topologies=_optional_int(
            comparison.get("candidate_unique_topologies")
        ),
        error_type=result.error_type,
        error_message=result.error_message,
    )


def records_as_dicts(
    records: Iterable[ScalingRecord | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        asdict(record) if isinstance(record, ScalingRecord) else dict(record)
        for record in records
    ]


def aggregate_records(
    records: Iterable[ScalingRecord | Mapping[str, Any]],
    *,
    group_fields: Sequence[str] = (
        "scaling_axis",
        "method",
        "requested_samples",
        "requested_internal_nodes",
    ),
) -> list[dict[str, Any]]:
    """Aggregate usable timings while retaining completion/failure counts."""

    rows = records_as_dicts(records)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_fields)
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        members = grouped[key]
        executed = [
            row
            for row in members
            if row.get("outcome") == "complete"
            and row.get("solver_wall_seconds") is not None
        ]
        completed = [
            row
            for row in executed
            if row.get("solver_status", "complete") == "complete"
            and (
                row.get("method") != METHOD_EXACT
                or bool(row.get("exact", True))
            )
        ]
        usable = [
            row
            for row in executed
            if (
                (
                    row.get("method") == METHOD_EXACT
                    and row.get("solver_status", "complete") == "complete"
                    and bool(row.get("exact", True))
                )
                or (
                    row.get("method") == METHOD_APA_WARM
                    and row.get("solver_status", "complete")
                    in {"complete", "partial"}
                )
            )
        ]
        timings = np.asarray(
            [float(row["solver_wall_seconds"]) for row in usable], dtype=float
        )
        summary = dict(zip(group_fields, key))
        summary.update(
            {
                "runs": len(members),
                "executed_runs": len(executed),
                "usable_runs": len(usable),
                "completed_runs": len(completed),
                "incomplete_path_runs": len(executed) - len(completed),
                "iteration_limited_runs": sum(
                    row.get("method") == METHOD_APA_WARM
                    and row.get("all_points_converged") is False
                    for row in executed
                ),
                "timeout_runs": sum(
                    row.get("outcome") == "timeout" for row in members
                ),
                "failed_runs": sum(
                    row.get("outcome") == "failed" for row in members
                ),
                "censored_runs": sum(bool(row.get("censored")) for row in members),
                "wall_seconds_median": (
                    float(np.median(timings)) if timings.size else None
                ),
                "wall_seconds_q1": (
                    float(np.percentile(timings, 25.0)) if timings.size else None
                ),
                "wall_seconds_q3": (
                    float(np.percentile(timings, 75.0)) if timings.size else None
                ),
            }
        )
        aggregate_metrics = {
            "native_points": "native_points_median",
            "n_regions": "n_regions_median",
            "probe_attempts": "probe_attempts_median",
            "coefficient_relative_error_median": (
                "coefficient_relative_error_median"
            ),
            "sample_coefficient_relative_error_median": (
                "sample_coefficient_relative_error_median"
            ),
            "support_agreement_fraction": "support_agreement_fraction_median",
            "sample_support_agreement_fraction": (
                "sample_support_agreement_fraction_median"
            ),
            "topology_agreement_fraction": "topology_agreement_fraction_median",
        }
        for metric, aggregate_name in aggregate_metrics.items():
            values = [
                float(row[metric])
                for row in usable
                if row.get(metric) is not None
            ]
            summary[aggregate_name] = (
                float(np.median(values)) if values else None
            )
        summaries.append(summary)
    return summaries


def source_provenance(repository_root: Path | None = None) -> dict[str, Any]:
    """Identify the working source, including changes not represented by HEAD.

    Fingerprints cover Python source under ``imodels`` and ``benchmarks`` plus
    packaging configuration, not generated results or dataset caches. Git
    fields are unavailable (None) in a source archive without Git metadata.
    Call outside solver timings; this intentionally reads the current files.
    """
    root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
    source_paths = {
        path for directory in (root / "imodels", root / "benchmarks")
        for path in directory.rglob("*.py") if path.is_file()
    }
    source_paths.update(
        path for name in ("pyproject.toml", "setup.py", "setup.cfg")
        if (path := root / name).is_file()
    )
    fingerprints = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_paths)
    }
    source_digest = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    commit, dirty = None, None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        # A source archive is runnable even when Git is absent or unavailable.
        pass
    return {
        "git_commit_sha": commit,
        "git_dirty": dirty,
        "source_sha256": source_digest,
        "source_file_sha256": fingerprints,
    }


def environment_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "source_provenance": source_provenance(),
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_results(
    records: Iterable[ScalingRecord | Mapping[str, Any]],
    *,
    csv_path: str | os.PathLike[str],
    json_path: str | os.PathLike[str],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Atomically write the same ordered records as CSV and structured JSON."""

    field_names = [item.name for item in fields(ScalingRecord)]
    rows = records_as_dicts(records)
    unknown = sorted(set().union(*(row.keys() for row in rows)) - set(field_names))
    if unknown:
        raise ValueError(f"records contain fields outside ScalingRecord: {unknown}")
    missing = [
        (index, sorted(set(field_names) - set(row)))
        for index, row in enumerate(rows)
        if set(field_names) - set(row)
    ]
    if missing:
        index, names = missing[0]
        raise ValueError(
            f"record {index} is missing ScalingRecord fields: {names}"
        )
    mismatched_versions = [
        index
        for index, row in enumerate(rows)
        if row.get("schema_version") != SCHEMA_VERSION
    ]
    if mismatched_versions:
        raise ValueError(
            "record schema_version must equal "
            f"{SCHEMA_VERSION}; mismatches at {mismatched_versions}"
        )
    normalized_rows = [
        {name: _json_compatible(row.get(name)) for name in field_names}
        for row in rows
    ]
    csv_buffer: list[str] = []
    # csv.writer needs a file-like object; StringIO avoids partially written
    # result files and keeps the final rename atomic.
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(normalized_rows)
    csv_buffer.append(stream.getvalue())
    csv_output = Path(csv_path)
    json_output = Path(json_path)
    _atomic_text_write(csv_output, "".join(csv_buffer))
    combined_metadata = environment_metadata()
    if metadata is not None:
        combined_metadata.update(dict(metadata))
    combined_metadata["schema_version"] = SCHEMA_VERSION
    document = {
        "schema_version": SCHEMA_VERSION,
        "metadata": _json_compatible(combined_metadata),
        "records": normalized_rows,
        "aggregates": _json_compatible(aggregate_records(normalized_rows)),
    }
    _atomic_text_write(
        json_output, json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    return csv_output.resolve(), json_output.resolve()


def read_json_results(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported result schema {document.get('schema_version')!r}"
        )
    if not isinstance(document.get("records"), list):
        raise ValueError("result document must contain a records list")
    field_names = {item.name for item in fields(ScalingRecord)}
    for index, row in enumerate(document["records"]):
        if not isinstance(row, Mapping):
            raise ValueError(f"result record {index} must be an object")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"result record {index} has unsupported schema "
                f"{row.get('schema_version')!r}"
            )
        missing = sorted(field_names - set(row))
        unknown = sorted(set(row) - field_names)
        if missing or unknown:
            raise ValueError(
                f"result record {index} fields do not match ScalingRecord; "
                f"missing={missing}, unknown={unknown}"
            )
    return document


__all__ = [
    "SCHEMA_VERSION",
    "METHOD_APA_WARM",
    "METHOD_EXACT",
    "SUPPORTED_METHODS",
    "LambdaGrid",
    "RegressionDataset",
    "ScalingConfig",
    "ScalingRecord",
    "SolverPayload",
    "SolverResult",
    "TreePathProblem",
    "aggregate_records",
    "build_tree_path_problem",
    "compare_paths",
    "descendant_groups",
    "environment_metadata",
    "evaluate_solver_result",
    "group_penalty",
    "load_regression_dataset",
    "make_lambda_grid",
    "make_scaling_record",
    "objective_values",
    "read_json_results",
    "records_as_dicts",
    "run_solver_payload",
    "subset_regression_dataset",
    "terminal_solver_result",
    "topology_signature",
    "write_results",
    "zero_solution_lambda_max",
]
