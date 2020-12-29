from unittest.mock import patch
import sys
sys.path.append('.')

import pytest
import matplotlib.pyplot
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@patch('matplotlib.pyplot')
@patch('sklearn.tree.plot_tree')
@patch('pandas.DataFrame.style')
@ignore_warnings(category=ConvergenceWarning)
def run_tests(mock_pd_style, mock_plot_tree, mock_pyplot):
    pytest.main(sys.argv[1:] + ['--cov=imodels'])

def main():
    run_tests()

if __name__ == '__main__':
    main()
