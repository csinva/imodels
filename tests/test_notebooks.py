from unittest.mock import patch

import matplotlib.pyplot

@patch('matplotlib.pyplot')
@patch('sklearn.tree.plot_tree')
@patch('pandas.DataFrame.style')
def test_imodels_demo(mock_pd_style, mock_plot_tree, mock_pyplot):
    from .notebooks import imodels_demo

@patch('matplotlib.pyplot')
def test_imodels_comparisons(mock_pyplot):
    from .notebooks import imodels_comparisons
