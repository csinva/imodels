import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize

from imodels.tree.sparse_pruning.optimization.tree_prox import (
    LaminarGroupLinfProx,
    prox_laminar_group_linf,
)


def _scipy_epigraph_prox(
    value, groups, scale, group_weights, coordinate_weights=None
):
    """Small independent convex-QP oracle for the proximal map."""
    value = np.asarray(value, dtype=float)
    groups = [np.asarray(group, dtype=int) for group in groups]
    weights = np.asarray(group_weights, dtype=float)
    n_features = value.size
    n_groups = len(groups)
    if coordinate_weights is None:
        coordinate_weights = np.ones(n_features)
    coordinate_weights = np.asarray(coordinate_weights, dtype=float)

    # z = (beta, t), with -t_g <= beta_j <= t_g for j in g.
    constraint_rows = []
    for group_number, group in enumerate(groups):
        for feature in group:
            upper = np.zeros(n_features + n_groups)
            upper[feature] = coordinate_weights[feature]
            upper[n_features + group_number] = -1.0
            constraint_rows.append(upper)

            lower = np.zeros(n_features + n_groups)
            lower[feature] = -coordinate_weights[feature]
            lower[n_features + group_number] = -1.0
            constraint_rows.append(lower)
    constraint_matrix = np.asarray(constraint_rows)

    def objective(z):
        delta = z[:n_features] - value
        return 0.5 * float(delta @ delta) + scale * float(
            weights @ z[n_features:]
        )

    def gradient(z):
        return np.concatenate(
            (z[:n_features] - value, scale * weights)
        )

    initial_epigraph = np.array(
        [
            np.max(coordinate_weights[group] * np.abs(value[group]))
            for group in groups
        ]
    )
    initial = np.concatenate((value, initial_epigraph))
    result = minimize(
        objective,
        initial,
        jac=gradient,
        constraints=LinearConstraint(constraint_matrix, -np.inf, 0.0),
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 2_000},
    )
    assert result.success, result.message
    return result.x[:n_features], result.fun


@pytest.mark.parametrize(
    "value,groups,weights,scale",
    [
        (
            np.array([2.5, -0.7, 1.3]),
            [np.array([0, 1, 2]), np.array([0]), np.array([0, 1])],
            np.array([0.7, 1.3, 0.4]),
            0.8,
        ),
        (
            np.array([1.2, -2.1, 0.4, 3.0, -1.7]),
            [
                np.array([0, 1, 2, 3, 4]),
                np.array([3, 4]),
                np.array([0, 1]),
                np.array([4]),
                np.array([0]),
            ],
            np.array([0.2, 1.4, 0.8, 0.35, 1.1]),
            1.25,
        ),
        (
            np.array([-0.9, 2.2, -3.1, 0.3]),
            [np.array([0, 1]), np.array([2]), np.array([3])],
            np.array([1.7, 0.3, 2.0]),
            0.55,
        ),
    ],
)
def test_exact_prox_matches_independent_scipy_epigraph_qp(
    value, groups, weights, scale
):
    expected, expected_objective = _scipy_epigraph_prox(
        value, groups, scale, weights
    )

    operator = LaminarGroupLinfProx(groups, value.size, weights)
    actual, info = operator(value, scale, return_info=True)

    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
    assert info["primal_objective"] == pytest.approx(
        expected_objective, abs=2e-9, rel=2e-9
    )
    assert info["duality_gap"] <= 5e-13
    assert info["relative_duality_gap"] <= 5e-13
    assert abs(info["raw_relative_duality_gap"]) <= 5e-13
    assert info["max_dual_l1_violation"] <= 5e-15
    assert info["max_relative_dual_l1_violation"] <= 5e-13


@pytest.mark.parametrize("seed", range(6))
def test_coordinate_weighted_exact_prox_matches_scipy_qp(seed):
    rng = np.random.default_rng(seed)
    base_groups = [
        np.array([0, 1, 2, 3]),
        np.array([0, 1]),
        np.array([2, 3]),
        np.array([0]),
        np.array([3]),
    ]
    permutation = rng.permutation(len(base_groups))
    groups = [base_groups[index] for index in permutation]
    group_weights = rng.uniform(0.2, 1.5, size=len(groups))
    coordinate_weights = rng.uniform(0.4, 2.0, size=4)
    value = rng.normal(size=4)
    scale = rng.uniform(0.1, 0.8)

    expected, expected_objective = _scipy_epigraph_prox(
        value,
        groups,
        scale,
        group_weights,
        coordinate_weights,
    )
    actual, info = LaminarGroupLinfProx(
        groups,
        4,
        group_weights,
        coordinate_weights,
    )(value, scale, return_info=True)

    np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=5e-7)
    assert info["primal_objective"] == pytest.approx(
        expected_objective, abs=5e-9, rel=5e-9
    )
    assert info["duality_gap"] <= 1e-12
    assert info["max_dual_l1_violation"] <= 1e-14
    assert info["max_relative_dual_l1_violation"] <= 1e-12


def test_compilation_orders_children_before_parents_and_records_hierarchy():
    groups = [
        np.array([0, 1, 2, 3]),
        np.array([0, 1]),
        np.array([0]),
        np.array([2, 3]),
    ]
    operator = LaminarGroupLinfProx(groups, 4)

    np.testing.assert_array_equal(operator.source_order, [2, 1, 3, 0])
    np.testing.assert_array_equal(operator.parents, [1, 3, 3, -1])
    assert [group.tolist() for group in operator.groups] == [
        [0],
        [0, 1],
        [2, 3],
        [0, 1, 2, 3],
    ]


def test_optional_order_check_rejects_ancestor_first_but_auto_order_accepts_it():
    groups = [np.array([0, 1]), np.array([0])]

    automatic = LaminarGroupLinfProx(groups, 2)
    np.testing.assert_array_equal(automatic.source_order, [1, 0])
    with pytest.raises(ValueError, match="leaf-to-root order"):
        LaminarGroupLinfProx(groups, 2, require_leaf_to_root=True)

    already_ordered = LaminarGroupLinfProx(
        groups[::-1], 2, require_leaf_to_root=True
    )
    np.testing.assert_array_equal(already_ordered.source_order, [0, 1])


def test_crossing_groups_are_rejected():
    with pytest.raises(ValueError, match="laminar"):
        LaminarGroupLinfProx(
            [np.array([0, 1]), np.array([1, 2])], n_features=3
        )


@pytest.mark.parametrize(
    "groups,error",
    [
        ([], "at least one"),
        ([np.array([])], "non-empty"),
        ([np.array([0.0])], "integer"),
        ([np.array([0, 0])], "duplicate indices"),
        ([np.array([2])], "outside"),
        ([np.array([0]), np.array([0])], "duplicate groups"),
    ],
)
def test_invalid_groups_are_rejected(groups, error):
    with pytest.raises(ValueError, match=error):
        LaminarGroupLinfProx(groups, n_features=2)


@pytest.mark.parametrize(
    "weights,error",
    [
        ([-1.0, 1.0], "nonnegative"),
        ([np.inf, 1.0], "finite"),
        ([1.0], "shape"),
        ([True, False], "scalars"),
        ([1.0 + 2.0j, 1.0], "shape"),
    ],
)
def test_invalid_group_weights_are_rejected(weights, error):
    groups = [np.array([0]), np.array([0, 1])]
    with pytest.raises(ValueError, match=error):
        LaminarGroupLinfProx(groups, 2, weights)


def test_scalar_group_weight_broadcast_and_convenience_function():
    value = np.array([3.0, -2.0, 0.5])
    groups = [np.array([0]), np.array([0, 1]), np.array([0, 1, 2])]

    compiled = LaminarGroupLinfProx(groups, 3, group_weights=0.75)
    expected = compiled(value, scale=0.6)
    actual = prox_laminar_group_linf(
        value,
        groups,
        scale=0.6,
        group_weights=np.full(3, 0.75),
    )

    np.testing.assert_allclose(actual, expected)


def test_scalar_coordinate_weight_broadcast_and_convenience_function():
    value = np.array([3.0, -2.0, 0.5])
    groups = [np.array([0]), np.array([0, 1]), np.array([0, 1, 2])]

    compiled = LaminarGroupLinfProx(
        groups, 3, coordinate_weights=1.25
    )
    expected = compiled(value, scale=0.6)
    actual = prox_laminar_group_linf(
        value,
        groups,
        scale=0.6,
        coordinate_weights=np.full(3, 1.25),
    )

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(compiled.coordinate_weights, np.full(3, 1.25))


@pytest.mark.parametrize(
    "weights,error",
    [
        ([1.0, 0.0], "positive"),
        ([1.0, -1.0], "positive"),
        ([1.0, np.inf], "finite"),
        ([1.0], "shape"),
        ([True, False], "scalars"),
        ([1.0 + 2.0j, 1.0], "shape"),
    ],
)
def test_invalid_coordinate_weights_are_rejected(weights, error):
    with pytest.raises(ValueError, match=error):
        LaminarGroupLinfProx(
            [np.array([0, 1])], 2, coordinate_weights=weights
        )


def test_compiled_operator_is_reusable_and_does_not_mutate_inputs():
    groups = [np.array([0]), np.array([0, 1])]
    operator = LaminarGroupLinfProx(groups, 3)
    first_input = np.array([2.0, -1.0, 7.0])
    first_copy = first_input.copy()

    first = operator(first_input, 0.5)
    second = operator(np.array([-3.0, 4.0, -2.0]), 1.25)

    np.testing.assert_array_equal(first_input, first_copy)
    np.testing.assert_allclose(first, [1.0, -1.0, 7.0])
    np.testing.assert_allclose(second, [-1.75, 2.75, -2.0])


def test_zero_scale_and_zero_weight_leave_unpenalized_coordinates_unchanged():
    value = np.array([2.0, -3.0, 4.0])
    operator = LaminarGroupLinfProx(
        [np.array([0]), np.array([0, 1])],
        3,
        group_weights=[0.0, 1.0],
    )

    zero, zero_info = operator(value, 0.0, return_info=True)
    positive = operator(value, 0.5)

    np.testing.assert_array_equal(zero, value)
    assert zero_info["duality_gap"] == 0.0
    assert zero_info["raw_relative_duality_gap"] == 0.0
    assert zero_info["max_relative_dual_l1_violation"] == 0.0
    assert zero_info["primal_objective"] == 0.0
    assert positive[0] == pytest.approx(2.0)
    assert positive[1] == pytest.approx(-2.5)
    assert positive[2] == value[2]


def test_weighted_projection_is_stable_at_extreme_finite_scales():
    operator = LaminarGroupLinfProx(
        [np.array([0])], 1, coordinate_weights=[1e-200]
    )

    result, info = operator(np.array([1.0]), 1e199, return_info=True)

    np.testing.assert_allclose(result, [0.9], rtol=2e-15, atol=0.0)
    assert info["relative_duality_gap"] <= 1e-14
    assert info["max_relative_dual_l1_violation"] <= 1e-14


def test_weighted_projection_keeps_the_last_face_under_tiny_radius_cancellation():
    operator = LaminarGroupLinfProx(
        [np.array([0, 1])],
        2,
        coordinate_weights=[1e9, 1.0],
    )

    result, info = operator(
        np.array([5e-9, 1.0]), 1e-17, return_info=True
    )

    np.testing.assert_allclose(result, [1e-9, 1.0], rtol=2e-15, atol=1e-24)
    assert info["max_relative_dual_l1_violation"] == 0.0
    assert abs(info["raw_relative_duality_gap"]) <= 2e-15


def test_weighted_projection_does_not_admit_a_rounded_extra_face():
    operator = LaminarGroupLinfProx(
        [np.array([0, 1, 2])],
        3,
        coordinate_weights=[1.0, 1e10, 5.0],
    )

    result, info = operator(
        np.array([0.1, 0.1, 0.1]),
        1.0000000000000001e-11,
        return_info=True,
    )

    np.testing.assert_allclose(result, [0.1, 5e-11, 0.1], rtol=2e-14)
    assert abs(info["raw_relative_duality_gap"]) <= 2e-14
    assert info["max_relative_dual_l1_violation"] <= 2e-14


def test_weighted_projection_resolves_sub_ulp_prefix_face_comparison():
    operator = LaminarGroupLinfProx(
        [np.array([0, 1])],
        2,
        coordinate_weights=[1.0, 1e17],
    )

    result, info = operator(
        np.array([1.0, 1.0]), 1e-20, return_info=True
    )

    np.testing.assert_allclose(result, [1.0, 0.999], rtol=2e-14)
    assert abs(info["raw_relative_duality_gap"]) <= 2e-14
    assert info["max_relative_dual_l1_violation"] <= 2e-14
    assert info["relative_moreau_residual"] <= 2e-14


def test_weighted_projection_ignores_extreme_inactive_reciprocal_scale():
    value = np.array([1.0, 1.0, np.ldexp(1.0, 200)])
    coordinate_weights = np.array(
        [
            np.ldexp(1.0, 400),
            np.ldexp(1.0, 300),
            np.ldexp(1.0, -400),
        ]
    )
    scale = np.ldexp(1.0, -440)
    operator = LaminarGroupLinfProx(
        [np.arange(3)], 3, coordinate_weights=coordinate_weights
    )

    result, info = operator(value, scale, return_info=True)

    expected = np.array(
        [1.0 - np.ldexp(1.0, -40), 1.0, np.ldexp(1.0, 200)]
    )
    np.testing.assert_array_equal(result, expected)
    assert abs(info["raw_relative_duality_gap"]) <= 2e-14
    assert info["max_relative_dual_l1_violation"] <= 2e-14
    assert info["relative_moreau_residual"] <= 2e-14


def test_weighted_projection_certificate_is_componentwise_at_tiny_scale():
    scale_factor = 1e-180
    value = scale_factor * np.array([1.0, 1.0, 0.0])
    operator = LaminarGroupLinfProx(
        [np.arange(3)],
        3,
        coordinate_weights=[1.0, 1e17, 1e-70],
    )

    result, info = operator(value, scale_factor * 1e-20, return_info=True)

    expected = scale_factor * np.array([1.0, 0.999, 0.0])
    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=0.0)
    assert abs(info["raw_relative_duality_gap"]) <= 2e-14
    assert info["max_relative_dual_l1_violation"] <= 2e-14
    assert info["relative_moreau_residual"] <= 2e-14


def test_unit_weight_projection_handles_a_tiny_positive_radius():
    value = np.array([1e-180, 1e-180])
    operator = LaminarGroupLinfProx([np.arange(2)], 2)

    result, info = operator(value, 1e-200, return_info=True)

    np.testing.assert_array_equal(result, value)
    assert abs(info["raw_relative_duality_gap"]) <= 2e-14
    assert info["max_relative_dual_l1_violation"] <= 2e-14
    assert info["relative_moreau_residual"] <= 2e-14


def test_weighted_projection_accepts_a_positive_radius_below_scaled_resolution():
    operator = LaminarGroupLinfProx(
        [np.array([0])], 1, coordinate_weights=[1e-200]
    )

    result = operator(np.array([1.0]), 1e-200)

    np.testing.assert_array_equal(result, [1.0])


def test_scaled_penalty_certificate_avoids_intermediate_overflow():
    operator = LaminarGroupLinfProx(
        [np.array([0])], 1, coordinate_weights=[1e200]
    )

    result, info = operator(np.array([1e150]), 5e-51, return_info=True)

    np.testing.assert_allclose(result, [5e149], rtol=2e-15)
    assert info["primal_objective"] == pytest.approx(3.75e299, rel=2e-15)
    assert np.isfinite(info["dual_objective"])
    assert abs(info["raw_relative_duality_gap"]) <= 2e-15


def test_scaled_penalty_certificate_avoids_intermediate_underflow():
    operator = LaminarGroupLinfProx(
        [np.array([0])], 1, coordinate_weights=[1e-300]
    )

    result, info = operator(np.array([2e-100]), 1e200, return_info=True)

    np.testing.assert_allclose(result, [1e-100], rtol=2e-15, atol=0.0)
    assert info["primal_objective"] == pytest.approx(1.5e-200, rel=2e-15)
    assert info["dual_objective"] == pytest.approx(1.5e-200, rel=2e-15)
    assert abs(info["raw_relative_duality_gap"]) <= 2e-15


def test_scaled_penalty_certificate_avoids_nonzero_subnormal_intermediate():
    operator = LaminarGroupLinfProx([[0]], 1, coordinate_weights=[1e-200])

    result, info = operator(np.array([2e-120]), 1e80, return_info=True)

    np.testing.assert_allclose(result, [1e-120], rtol=2e-15, atol=0.0)
    # a * beta is nonzero (~1e-320), but rounding it before multiplication by
    # scale produces a spurious relative duality gap of approximately 7e-6.
    assert info["primal_objective"] == pytest.approx(1.5e-240, rel=2e-15, abs=0.0)
    assert info["dual_objective"] == pytest.approx(1.5e-240, rel=2e-15, abs=0.0)
    assert abs(info["raw_relative_duality_gap"]) <= 2e-15


def test_zero_penalties_avoid_overflow_in_certificate_evaluation():
    value = np.array([1e308])
    zero_scale = LaminarGroupLinfProx(
        [np.array([0])], 1, coordinate_weights=[1e308]
    )
    zero_weight = LaminarGroupLinfProx(
        [np.array([0])],
        1,
        group_weights=[0.0],
        coordinate_weights=[1e308],
    )

    for operator, scale in ((zero_scale, 0.0), (zero_weight, 1.0)):
        result, info = operator(value, scale, return_info=True)
        np.testing.assert_array_equal(result, value)
        assert info["raw_duality_gap"] == 0.0
        assert info["relative_duality_gap"] == 0.0
        assert info["max_relative_dual_l1_violation"] == 0.0


@pytest.mark.parametrize(
    "value,scale,error",
    [
        (np.ones((2, 1)), 1.0, "shape"),
        (np.array([np.nan, 1.0]), 1.0, "finite"),
        (np.ones(2), -1.0, "nonnegative"),
        (np.ones(2), np.inf, "finite"),
        (np.ones(2), True, "scalar"),
        (np.ones(2, dtype=complex), 1.0, "real"),
        (np.ones(2), 1.0 + 2.0j, "scalar"),
    ],
)
def test_invalid_evaluation_inputs_are_rejected(value, scale, error):
    operator = LaminarGroupLinfProx([np.array([0, 1])], 2)
    with pytest.raises(ValueError, match=error):
        operator(value, scale)


def test_plain_prox_skips_optional_dual_certificate_work(monkeypatch):
    from imodels.tree.sparse_pruning.optimization import tree_prox

    value = np.array([2.0, -3.0, 1.0])
    operator = LaminarGroupLinfProx(
        [[0], [0, 1], [0, 1, 2]], 3,
        group_weights=[0.7, 0.2, 0.9], coordinate_weights=[1.2, 0.3, 2.0],
    )
    expected, info = operator(value, 0.4, return_info=True)
    assert abs(info["raw_relative_duality_gap"]) < 1e-12

    def unexpected_certificate(*args, **kwargs):
        raise AssertionError("return_info=False must not assemble certificates")

    monkeypatch.setattr(tree_prox, "_weighted_l1_feasibility", unexpected_certificate)
    np.testing.assert_array_equal(operator(value, 0.4), expected)


@pytest.mark.parametrize("coordinate_weight", [0.3, 1.0, 5.0, 1e-200, 1e200])
@pytest.mark.parametrize("coefficient", [-2.0, 0.0, 2.0])
def test_singleton_projection_matches_exact_rational_reference(coordinate_weight, coefficient):
    from imodels.tree.sparse_pruning.optimization.tree_prox import (
        _exact_weighted_l1_projection, _project_weighted_l1_ball,
    )

    value = np.array([coefficient])
    weights = np.array([coordinate_weight])
    for radius in [0.1 / coordinate_weight, 2.0 / coordinate_weight, 3.0 / coordinate_weight]:
        expected_projection, expected_result = _exact_weighted_l1_projection(value, weights, radius)
        projection, result = _project_weighted_l1_ball(value, weights, radius)
        np.testing.assert_allclose(projection, expected_projection, rtol=2e-15, atol=0.0)
        np.testing.assert_allclose(result, expected_result, rtol=2e-15, atol=0.0)


def test_singleton_projection_retains_sub_ulp_primal_near_rounded_boundary():
    from imodels.tree.sparse_pruning.optimization.tree_prox import (
        _exact_weighted_l1_projection, _project_weighted_l1_ball,
    )

    value = np.array([0.1])
    weights = np.array([1.0 / 3.0])
    radius = 0.3
    expected_projection, expected_result = _exact_weighted_l1_projection(value, weights, radius)
    assert expected_result[0] > 0.0
    projection, result = _project_weighted_l1_ball(value, weights, radius)
    np.testing.assert_array_equal(projection, expected_projection)
    np.testing.assert_array_equal(result, expected_result)


def test_singleton_projection_resolves_half_subnormal_product_rounding():
    from imodels.tree.sparse_pruning.optimization.tree_prox import _project_weighted_l1_ball

    smallest = np.nextafter(0.0, 1.0)
    projection, result = _project_weighted_l1_ball(
        np.array([smallest]), np.array([smallest]), 0.5,
    )
    # Both exact values equal half the smallest subnormal and round to zero.
    # Rounding the product first and then subtracting would incorrectly retain
    # the whole input coefficient in the primal result.
    np.testing.assert_array_equal(projection, [0.0])
    np.testing.assert_array_equal(result, [0.0])


@pytest.mark.parametrize("return_info", [False, True])
def test_positive_group_radius_underflow_is_rejected_instead_of_certifying_zero_penalty(return_info):
    operator = LaminarGroupLinfProx(
        [[0]], 1, group_weights=[1e-200], coordinate_weights=[1e300],
    )
    # The mathematical effective scalar penalty is 1e-100. Computing only
    # scale * group_weight first rounds to zero and formerly returned the
    # unchanged 2e-100 coefficient with a zero-gap certificate.
    with pytest.raises(ValueError, match="positive group weight underflowed"):
        operator(np.array([2e-100]), 1e-200, return_info=return_info)
