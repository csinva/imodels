import pytest
from unittest.mock import patch

import matplotlib.pyplot

@patch('matplotlib.pyplot')
@patch('sklearn.tree.plot_tree')
@patch('pandas.DataFrame.style')
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_imodels_demo(mock_pd_style, mock_plot_tree, mock_pyplot):
    from .notebooks import imodels_demo
