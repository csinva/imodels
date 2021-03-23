from unittest.mock import patch, MagicMock

import matplotlib.pyplot
import dvu

@patch('matplotlib.pyplot')
@patch('sklearn.tree.plot_tree')
@patch('pandas.DataFrame.style')
def test_imodels_demo(mock_pd_style, mock_plot_tree, mock_pyplot):
    from .notebooks import imodels_demo

@patch('matplotlib.pyplot')
@patch('dvu.line_legend')
def test_imodels_comparisons(mock_line_legend, mock_pyplot):
    mock_pyplot.subplots.return_value = (MagicMock(), MagicMock())
    from .notebooks import imodels_comparisons
