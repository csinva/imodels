"""Thin wrappers over private sklearn helpers whose signatures change between versions."""

from inspect import signature

from sklearn.ensemble._forest import (
    _generate_sample_indices,
    _generate_unsampled_indices,
)


def _call_with_optional_sample_weight(func, random_state, n_samples, n_samples_bootstrap):
    """Call one of sklearn's bootstrap-index helpers across sklearn versions.

    sklearn 1.9 added a required `sample_weight` argument to these private helpers.
    """
    if "sample_weight" in signature(func).parameters:
        return func(random_state, n_samples, n_samples_bootstrap, None)
    return func(random_state, n_samples, n_samples_bootstrap)


def generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """Indices of the bootstrap sample a tree was fit on."""
    return _call_with_optional_sample_weight(
        _generate_sample_indices, random_state, n_samples, n_samples_bootstrap
    )


def generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap):
    """Indices left out of the bootstrap sample a tree was fit on (its OOB set)."""
    return _call_with_optional_sample_weight(
        _generate_unsampled_indices, random_state, n_samples, n_samples_bootstrap
    )
