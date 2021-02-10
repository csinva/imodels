from unittest.mock import patch

@patch('matplotlib.pyplot')
@patch('sklearn.tree.plot_tree')
@patch('pandas.DataFrame.style')
def test_imodels_demo(mock_pd_style, mock_plot_tree, mock_pyplot):
    import imodels_demo
