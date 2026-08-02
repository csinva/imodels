import numpy as np
import pytest

import imodels


def test_get_data():
    X, y, feature_names = imodels.get_clean_dataset('friedman1', data_source='synthetic')
    assert X.shape[0] == 200
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(feature_names, list)


def test_shorthand_names_resolve_regardless_of_verbose():
    """verbose controls logging only, never which dataset is fetched

    The lookup that turns a shorthand name like 'compas' into its real dataset
    name and source sat inside `if verbose:`, so verbose=False looked the
    dataset up under the shorthand and failed with a 404.
    """
    from imodels.util.data_util import DSET_KWARGS, get_clean_dataset

    # 'friedman1' is generated, so only its shape and names are comparable
    loud = get_clean_dataset('friedman1', verbose=True)
    quiet = get_clean_dataset('friedman1', verbose=False)

    assert loud[0].shape == quiet[0].shape
    assert list(loud[2]) == list(quiet[2])
    assert DSET_KWARGS['friedman1']['dataset_name'] == 'friedman1'

    # an alias whose real name differs from the shorthand: resolving it must not
    # depend on verbose, which previously decided whether the lookup ran at all
    assert DSET_KWARGS['compas']['dataset_name'] == 'compas_two_year_clean'
    for verbose in (True, False):
        with pytest.raises(Exception, match='404'):
            # a name that is not an alias and does not exist reaches the fetch
            # either way, rather than being silently resolved differently
            get_clean_dataset('definitely_not_a_dataset', verbose=verbose)
