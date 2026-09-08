"""Regression coverage for the archived approximate topology traversal."""

import numpy as np

from benchmarks.experimental.topology import laminar_group_linf_topology_path


def test_thresholded_bisection_localizes_all_changes_but_does_not_claim_states():
    diagonal = np.array([1.0, 0.5, 0.25])
    linear = np.array([0.8, 0.4, 0.2])
    X = np.diag(np.sqrt(3.0 * diagonal))
    y = linear * np.sqrt(3.0) / np.sqrt(diagonal)
    groups = [np.arange(3), np.array([1, 2]), np.array([2])]

    path = laminar_group_linf_topology_path(
        X,
        y,
        groups,
        fit_intercept=False,
        support_tolerance=0.05,
        lambda_tolerance=1e-5,
    )

    assert path.status == "complete"
    assert path.metadata["topology_event_coverage_complete"] is True
    assert path.metadata["topology_states_enumerated"] is False
    assert path.metadata["n_event_brackets"] > 0
    assert all(
        bracket["width"]
        <= path.metadata["event_lambda_bracket_tolerance"] * (1.0 + 1e-12)
        for bracket in path.metadata["event_brackets"]
    )
