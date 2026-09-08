"""Conditional generalized cross-validation for node-based regression HS.

The partition and any preceding pruning are held fixed. Its exact linear
smoother trace is ``1 + sum_v n_v / (n_v + rho)`` over retained splits.
GCV is an approximation to prediction risk, not exact leave-one-out refitting:
refitting HS changes node counts, and learning/pruning the tree from the same
responses introduces selection effects not included in this trace.

Only single-output mean-regression trees fitted with uniform observation
weights are supported. Without explicit sample weights, proportional weighted
and raw node counts are checked, but aggregated statistics cannot prove that
unknown individual training weights were uniform.
Stat-only calls also assume that stored CART squared-error impurities retain
sufficient accuracy; supplying the training response checks root variance for
large-offset cancellation and adds a one-time O(n_samples) validation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF
from sklearn.utils.validation import check_is_fitted


def _tree_statistics(estimator, sample_weight):
    if not isinstance(estimator, DecisionTreeRegressor):
        raise TypeError("HS GCV requires a single DecisionTreeRegressor; forests and classifiers are unsupported")
    check_is_fitted(estimator, "tree_")
    if estimator.n_outputs_ != 1:
        raise ValueError("HS GCV requires a single-output regression tree")
    if estimator.criterion not in {"squared_error", "friedman_mse"}:
        raise ValueError("HS GCV requires criterion='squared_error' or 'friedman_mse'")
    monotonic = getattr(estimator, "monotonic_cst", None)
    if monotonic is not None and np.any(np.asarray(monotonic) != 0):
        raise ValueError("HS GCV does not support active monotonic constraints")

    tree = estimator.tree_
    reachable, internal, leaves, seen = [], [], [], set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in seen or node < 0 or node >= tree.node_count:
            raise ValueError("HS GCV requires a valid binary tree without cycles")
        seen.add(node)
        reachable.append(node)
        left, right = int(tree.children_left[node]), int(tree.children_right[node])
        if left == right == TREE_LEAF:
            leaves.append(node)
        elif left >= 0 and right >= 0 and left != right:
            internal.append(node)
            stack.extend((right, left))
        else:
            raise ValueError("HS GCV requires two children at every retained split")

    means = np.asarray(tree.value[:, 0, 0], dtype=float)
    counts = np.asarray(tree.n_node_samples, dtype=float)
    weighted = np.asarray(tree.weighted_n_node_samples, dtype=float)
    impurity = np.asarray(tree.impurity, dtype=float).copy()
    for values in (means, counts, weighted, impurity):
        if not np.all(np.isfinite(values[reachable])):
            raise ValueError("HS GCV requires finite retained-node statistics")
    if np.any(counts[reachable] <= 0) or np.any(weighted[reachable] <= 0):
        raise ValueError("HS GCV requires positive retained-node sample counts")
    n_samples = int(counts[0])
    weight_scale = weighted[0] / counts[0]
    tolerance = 128 * max(1, n_samples) * np.finfo(float).eps
    if not np.isfinite(weight_scale) or weight_scale <= 0:
        raise ValueError("HS GCV requires representable uniform weights")
    if not np.allclose(weighted[reachable] / counts[reachable], weight_scale,
                       rtol=tolerance, atol=0):
        raise ValueError("HS GCV does not support nonuniform weights or bootstrap multiplicities")
    if sample_weight is not None:
        raw = np.asarray(sample_weight)
        if raw.shape != (n_samples,) or np.iscomplexobj(raw):
            raise ValueError("sample_weight must have one real value per fitted observation")
        weights = np.asarray(raw, dtype=float)
        if (np.any(~np.isfinite(weights)) or np.any(weights <= 0)
                or not np.all(weights == weights[0])):
            raise ValueError("HS GCV supports only strictly positive uniform sample_weight")
        if not np.isclose(weights[0], weight_scale, rtol=tolerance, atol=0):
            raise ValueError("sample_weight does not match the fitted tree's node counts")

    # CART impurities are computed by subtracting squared means; tolerate their
    # roundoff at the response scale, including tiny negative singleton values.
    response_scale = max(float(np.max(np.abs(means[reachable]))),
                         float(np.sqrt(np.max(np.abs(impurity[reachable])))),
                         np.sqrt(np.finfo(float).tiny))
    with np.errstate(over="ignore", invalid="ignore"):
        variance_tolerance = tolerance * response_scale * response_scale
    if not np.isfinite(variance_tolerance):
        raise ValueError("HS GCV cannot represent the response-statistic scale")
    if np.any(impurity[reachable] < -variance_tolerance):
        raise ValueError("HS GCV requires nonnegative node squared-error impurities")
    impurity[reachable] = np.maximum(impurity[reachable], 0)
    energy = []
    for node in internal:
        left, right = tree.children_left[node], tree.children_right[node]
        if counts[left] + counts[right] != counts[node]:
            raise ValueError("HS GCV requires child counts to sum to the parent count")
        fraction = counts[left] / counts[node]
        parent_mean = fraction * means[left] + (1 - fraction) * means[right]
        if abs(parent_mean - means[node]) > tolerance * response_scale:
            raise ValueError("HS GCV requires unmodified weighted-mean node values")
        with np.errstate(over="ignore", invalid="ignore"):
            gain = fraction * counts[right] * (means[left] - means[right]) ** 2
            parent_sse = counts[node] * impurity[node]
            child_sse = counts[left] * impurity[left] + counts[right] * impurity[right]
        if not np.isfinite(gain) or not np.isfinite(parent_sse + child_sse):
            raise ValueError("HS GCV cannot represent the node squared-error statistics")
        if abs(parent_sse - child_sse - gain) > counts[node] * variance_tolerance:
            raise ValueError("HS GCV requires unmodified mean/squared-error tree statistics")
        energy.append(float(gain))
    for leaf in leaves:
        if counts[leaf] == 1:
            if impurity[leaf] > variance_tolerance:
                raise ValueError("a singleton leaf must have zero squared-error impurity")
            impurity[leaf] = 0.0
    leaf_rss = float(np.sum(counts[leaves] * impurity[leaves]))
    with np.errstate(over="ignore", invalid="ignore"):
        total_rss = leaf_rss + float(np.sum(energy))
    if not np.isfinite(total_rss):
        raise ValueError("HS GCV cannot represent the leaf squared-error statistics")
    return (weighted[internal] / weighted[0], np.asarray(energy), leaf_rss,
            n_samples, float(weighted[0]), float(weight_scale), len(leaves))


def select_hs_reg_param(estimator, *, sample_weight=None, y=None):
    """Select nonnegative node-based HS shrinkage by conditional tree GCV.

    Returns ``(rho, info)`` without modifying ``estimator``. ``rho=+inf`` is
    supported and denotes the intercept-only limit; callers must implement
    that limit explicitly when applying shrinkage. Uniform fitting weights
    rescale rho but not reported RSS/GCV, which use ordinary response units.

    Each evaluation costs O(retained nodes). A logarithmic grid followed by
    bounded refinement of detected minima is used, including zero and infinity
    endpoints. Successful local refinement does not certify a global optimum.
    At an interpolating zero endpoint, the analytic right-hand GCV limit is
    used instead of interpreting a zero training error as a zero GCV score.
    Optional ``y`` must be the fitted training response. It checks whether
    cancellation corrupted the stored root variance; without it, statistic
    accuracy is assumed and cannot be recovered from the tree alone.
    """
    ratios, energy, leaf_rss, n, root_weight, weight_scale, n_leaves = _tree_statistics(
        estimator, sample_weight
    )
    if y is not None:
        target = np.asarray(y)
        if target.shape != (n,) or np.iscomplexobj(target):
            raise ValueError("y must be a finite real vector with one entry per fitted observation")
        target = target.astype(float)
        if np.any(~np.isfinite(target)):
            raise ValueError("y must contain only finite responses")
        with np.errstate(over="ignore", invalid="ignore"):
            centered = target - target[0]
            centered -= np.mean(centered)
            empirical_rss = float(centered @ centered)
        stored_rss = float(n * max(estimator.tree_.impurity[0], 0.0))
        mean_roundoff = 128 * n * np.finfo(float).eps * abs(float(target[0]))
        constant_roundoff = bool(
            empirical_rss == 0
            and abs(float(estimator.tree_.value[0, 0, 0]) - target[0]) <= mean_roundoff
            and float(np.sum(energy)) <= n * mean_roundoff**2
            and stored_rss <= n * mean_roundoff * abs(float(target[0]))
        )
        if constant_roundoff:
            # CART can split a nonzero constant response because its impurity
            # subtraction leaves roundoff. The supplied y resolves that case.
            leaf_rss = 0.0
            energy = np.zeros_like(energy)
        elif not np.isfinite(empirical_rss) or not np.isclose(
                empirical_rss, stored_rss, rtol=1e-7, atol=0):
            raise ValueError(
                "stored root variance does not match y; center/rescale the "
                "response before fitting to avoid impurity cancellation"
            )
    residual_dimension = n - n_leaves
    interpolating = residual_dimension == 0 and leaf_rss == 0 and bool(len(ratios))

    def evaluate(t):
        if np.isinf(t):
            rss, df = leaf_rss + float(np.sum(energy)), 1.0
            denominator = n - 1.0
        else:
            shrink = t / (ratios + t)
            rss = leaf_rss + float(np.dot(shrink * shrink, energy))
            denominator = residual_dimension + float(np.sum(shrink))
            df = n - denominator
            if interpolating:
                # Cancel t**2 analytically, avoiding both 0/0 and loss of
                # precision in n-df near the interpolation endpoint.
                inverse = 1.0 / (ratios + t)
                normalized = inverse / np.sum(inverse)
                score = n * float(np.dot(normalized * normalized, energy))
                return score, rss, df
        score = (rss / denominator) * (n / denominator) if denominator > 0 else np.inf
        return float(score), float(rss), float(df)

    candidates = [(0.0, "zero")]
    refined_success = []
    if len(ratios) and (leaf_rss > 0 or np.any(energy > 0)):
        grid = np.geomspace(1e-8, 1e8, 97)
        with np.errstate(over="ignore", under="ignore"):
            grid_parameters = grid * root_weight
        grid_parameters = grid_parameters[np.isfinite(grid_parameters) & (grid_parameters > 0)]
        # Quantization matters for subnormal uniform weights: evaluate the
        # parameter that callers can actually apply, not its unrounded target.
        grid = np.unique(grid_parameters / root_weight)
        candidates.extend((float(t), "grid") for t in grid)
        scores = [evaluate(t)[0] for t, _ in candidates]
        # Include the first interval so a minimum below the logarithmic grid
        # can be distinguished from the exact zero endpoint.
        brackets = [(0.0, float(grid[0]))] if len(grid) else []
        for i in range(1, len(candidates)):
            left = scores[i - 1]
            right = scores[i + 1] if i + 1 < len(scores) else evaluate(np.inf)[0]
            if scores[i] <= min(left, right) and scores[i] < max(left, right):
                upper = candidates[min(i + 1, len(candidates) - 1)][0]
                if upper > candidates[i - 1][0]:
                    brackets.append((candidates[i - 1][0], upper))
        for low, high in brackets:
            if low == 0:
                result = minimize_scalar(lambda t: evaluate(t)[0], bounds=(low, high),
                                         method="bounded", options={"xatol": high * 1e-10})
                t = float(result.x)
            else:
                result = minimize_scalar(lambda z: evaluate(np.exp(z))[0],
                                         bounds=(np.log(low), np.log(high)), method="bounded",
                                         options={"xatol": 1e-10})
                t = float(np.exp(result.x))
            accepted = bool(result.success and np.isfinite(result.fun)
                            and np.isfinite(t) and low <= t <= high)
            if accepted:
                with np.errstate(over="ignore", under="ignore"):
                    parameter = t * root_weight
                accepted = bool(np.isfinite(parameter) and parameter > 0)
                if accepted:
                    candidates.append((float(parameter / root_weight), "refined"))
            refined_success.append(accepted)
        candidates.append((np.inf, "root_only_limit"))

    candidates.sort(key=lambda candidate: candidate[0])
    parameters = np.asarray([t * root_weight for t, _ in candidates])
    evaluated = np.asarray([evaluate(t) for t, _ in candidates])
    scores, rss_values, dfs = evaluated.T
    selected = int(np.argmin(scores))
    if n > 1 and not np.isfinite(scores[selected]):
        raise ValueError("HS GCV could not obtain a finite score from the fitted statistics")
    # For a single-node tree every rho has identical predictions. Return zero
    # even for n=1, where no data-driven GCV comparison is defined.
    info = {
        "reg_params": parameters, "gcv_scores": scores,
        "rss": rss_values, "effective_dfs": dfs,
        "selected_index": selected, "gcv_score": float(scores[selected]),
        "effective_df": float(dfs[selected]), "n_samples": n,
        "conditional_on_tree": True,
        "search_converged": all(refined_success),
        "global_optimum_certified": False,
        "candidate_kinds": tuple(kind for _, kind in candidates),
        "zero_score_is_limit": bool(interpolating),
        "gcv_defined": bool(np.isfinite(scores[selected])),
        "n_internal_nodes": len(ratios), "n_leaves": n_leaves,
        "uniform_weight_scale": weight_scale,
        "root_weight": root_weight,
        "selected_root_only_limit": bool(np.isinf(parameters[selected])),
    }
    return float(parameters[selected]), info


def apply_node_based_hs(tree, reg_param):
    """Apply selected HS in place to a validated scalar tree; return the tree.

    Iteration avoids a Python recursion limit on deep retained trees. Original
    parent means are carried separately from their already-shrunken values.
    """
    if reg_param == 0:
        return tree
    root_mean = float(tree.value[0, 0, 0])
    stack = [(0, None, None, root_mean)]
    while stack:
        node, parent_mean, parent_mass, prediction = stack.pop()
        original_mean = float(tree.value[node, 0, 0])
        if parent_mean is not None:
            prediction += (original_mean - parent_mean) / (1 + reg_param / parent_mass)
        tree.value[node, 0, 0] = prediction
        left, right = int(tree.children_left[node]), int(tree.children_right[node])
        if left != TREE_LEAF:
            mass = float(tree.weighted_n_node_samples[node])
            stack.append((right, original_mean, mass, prediction))
            stack.append((left, original_mean, mass, prediction))
    return tree


__all__ = ["select_hs_reg_param", "apply_node_based_hs"]
