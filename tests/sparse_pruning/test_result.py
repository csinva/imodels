"""Interpolation conventions and lookup cost of stored regularization paths."""
from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from imodels.tree.sparse_pruning.optimization._result import RegularizationPath


def test_duplicate_knots_keep_last_stored_row_at_and_between_knots():
    path = RegularizationPath(
        lambdas=np.array([5.0, 5.0, 3.0, 3.0, 3.0, 1.0, 1.0]),
        coefficients=np.array([
            [99.0, 99.0], [10.0, -10.0],
            [88.0, 88.0], [77.0, 77.0], [6.0, -6.0],
            [66.0, 66.0], [2.0, -2.0],
        ]),
        intercepts=np.array([99.0, 15.0, 88.0, 77.0, 9.0, 66.0, 3.0]),
    )
    for lam in [1.0, 2.0, 3.0, 4.0, 5.0]:
        beta, intercept = path.at(lam)
        assert_array_equal(beta, [2.0 * lam, -2.0 * lam])
        assert intercept == 3.0 * lam


@pytest.mark.parametrize("with_intercepts", [False, True])
def test_binary_interpolation_matches_previous_unique_interp_convention(with_intercepts):
    rng = np.random.default_rng(23)
    lambdas = np.repeat(np.arange(14.0, 0.0, -1.0), rng.integers(1, 9, size=14))
    coefficients = rng.normal(size=(len(lambdas), 5))
    intercepts = rng.normal(size=len(lambdas)) if with_intercepts else None
    path = RegularizationPath(lambdas, coefficients, intercepts=intercepts)
    ascending, indices = np.unique(lambdas[::-1], return_index=True)
    canonical_coefficients = coefficients[::-1][indices]
    for lam in np.r_[ascending, rng.uniform(1.0, 14.0, size=100)]:
        expected = np.array([
            np.interp(lam, ascending, canonical_coefficients[:, column])
            for column in range(coefficients.shape[1])
        ])
        beta, intercept = path.at(float(lam))
        assert_allclose(beta, expected, rtol=2e-14, atol=2e-15)
        if intercepts is None:
            assert intercept is None
        else:
            assert_allclose(
                intercept, np.interp(lam, ascending, intercepts[::-1][indices]),
                rtol=2e-14, atol=2e-15,
            )


@pytest.mark.parametrize("n_rows", [1, 5])
def test_single_unique_lambda_returns_last_stored_row_and_independent_copy(n_rows):
    path = RegularizationPath(
        np.full(n_rows, 2.0), np.arange(n_rows * 3.0).reshape(n_rows, 3),
        intercepts=np.arange(float(n_rows)),
    )
    expected = path.coefficients[-1].copy()
    for lam in [2.0, np.nextafter(2.0, 0.0), np.nextafter(2.0, np.inf)]:
        beta, intercept = path.at(lam)
        assert_array_equal(beta, expected)
        assert intercept == n_rows - 1
        beta[:] = -1000.0
        assert_array_equal(path.coefficients[-1], expected)


def test_endpoint_tolerance_and_out_of_range_checks_are_preserved():
    path = RegularizationPath(np.array([4.0, 2.0]), np.array([[1.0], [3.0]]))
    assert_array_equal(path.at(np.nextafter(4.0, np.inf))[0], [1.0])
    assert_array_equal(path.at(np.nextafter(2.0, 0.0))[0], [3.0])
    for lam in [4.000001, 1.999999, -1.0, np.inf, np.nan, [3.0]]:
        with pytest.raises(ValueError):
            path.at(lam)


@pytest.mark.parametrize("upper, lower", [(1e-20, 5e-21), (1e20, 1.0)])
def test_incomplete_path_does_not_extrapolate_to_zero(upper, lower):
    path = RegularizationPath(
        [upper, lower], [[0.0], [1.0]], exact=False, status="partial"
    )
    for lam in [0.0, lower / 2, upper * 2]:
        with pytest.raises(ValueError, match="outside the stored path"):
            path.at(lam)
    assert_array_equal(path.at(np.nextafter(upper, np.inf))[0], [0.0])
    assert_array_equal(path.at(np.nextafter(lower, 0.0))[0], [1.0])


def test_subnormal_lambda_range_does_not_expand_by_an_absolute_epsilon():
    smallest = np.nextafter(0.0, 1.0)
    path = RegularizationPath([smallest, 0.0], [[0.0], [1.0]])
    assert_array_equal(path.at(smallest)[0], [0.0])
    assert_array_equal(path.at(0.0)[0], [1.0])
    with pytest.raises(ValueError, match="outside the stored path"):
        path.at(2 * smallest)


@pytest.mark.parametrize("lam", ["1", "bad", 1 + 0j, 1 + 2j, True, None, 10**400])
def test_invalid_query_types_raise_value_error(lam):
    path = RegularizationPath([2.0, 0.0], [[0.0], [1.0]])
    with pytest.raises(ValueError, match="nonnegative finite scalar"):
        path.at(lam)


@pytest.mark.parametrize("field", ["lambdas", "coefficients", "intercepts", "penalties"])
def test_complex_result_arrays_are_rejected_without_discarding_imaginary_parts(field):
    inputs = dict(
        lambdas=np.array([2.0, 0.0]),
        coefficients=np.array([[0.0], [1.0]]),
        intercepts=np.array([0.0, 0.0]),
        penalties=np.array([0.0, 1.0]),
    )
    inputs[field] = inputs[field].astype(complex) + 1j
    with pytest.raises(ValueError, match=field):
        RegularizationPath(**inputs)


def test_result_arrays_are_independent_read_only_snapshots():
    inputs = dict(
        lambdas=np.array([2.0, 0.0]),
        coefficients=np.array([[0.0], [2.0]]),
        intercepts=np.array([3.0, 3.0]),
        penalties=np.array([0.0, 2.0]),
    )
    originals = {name: values.copy() for name, values in inputs.items()}
    path = RegularizationPath(**inputs, exact=True)
    for name, values in inputs.items():
        values[:] = -100.0
        stored = getattr(path, name)
        assert_array_equal(stored, originals[name])
        with pytest.raises(ValueError, match="read-only"):
            stored.flat[0] = 100.0
    assert_array_equal(path.at(1.0)[0], [1.0])
    assert path.at(1.0)[1] == 3.0
    assert path.exact


def test_empty_coefficient_vector_and_noncontiguous_storage():
    empty = RegularizationPath(np.array([2.0, 0.0]), np.empty((2, 0)))
    assert empty.at(1.0)[0].shape == (0,)
    backing = np.arange(24.0).reshape(4, 6)
    path = RegularizationPath(np.array([6.0, 4.0, 2.0, 0.0]), backing[:, ::2])
    assert_allclose(path.at(3.0)[0], (backing[1, ::2] + backing[2, ::2]) / 2)


def test_lookup_does_not_copy_or_deduplicate_all_knots(monkeypatch):
    class AdjacentRowsOnly(np.ndarray):
        def __getitem__(self, item):
            if not isinstance(item, (int, np.integer)):
                raise AssertionError("Interpolation must access individual rows only")
            return super().__getitem__(item)

    path = RegularizationPath(
        np.arange(1000.0, -1.0, -1.0), np.arange(2002.0).reshape(1001, 2),
    )
    object.__setattr__(path, "coefficients", path.coefficients.view(AdjacentRowsOnly))

    def reject_unique(*args, **kwargs):
        raise AssertionError("Interpolation must not scan/deduplicate the knot vector")

    monkeypatch.setattr(np, "unique", reject_unique)
    beta, intercept = path.at(499.5)
    assert_allclose(np.asarray(beta), [1001.0, 1002.0])
    assert intercept is None


def test_multiclass_coefficients_and_intercepts_interpolate_owned_snapshots():
    coefficients = np.arange(18.).reshape(2, 3, 3)
    intercepts = np.array([[1., 2., 3.], [4., 5., 6.]])
    path = RegularizationPath([2., 1.], coefficients, intercepts=intercepts)
    beta, intercept = path.at(1.5)
    assert_allclose(beta, coefficients.mean(axis=0))
    assert_allclose(intercept, intercepts.mean(axis=0))
    beta[:] = -1
    intercept[:] = -1
    assert_array_equal(path.at(2.)[1], intercepts[0])
    path.at(2.)[1][:] = -2
    assert_array_equal(path.intercepts, intercepts)
    assert not path.exact
    with pytest.raises(ValueError, match="class dimensions"):
        RegularizationPath([2., 1.], coefficients, intercepts=[1., 2.])
    with pytest.raises(ValueError, match="at least two classes"):
        RegularizationPath([2., 1.], np.zeros((2, 3, 1)))
