# This is just a simple wrapper around sklearn decisiontree
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

from sklearn.tree import DecisionTreeClassifier, export_text, DecisionTreeRegressor
from imodels.util.arguments import check_fit_arguments

from imodels.util.tree import compute_tree_complexity


class GreedyTreeClassifier(DecisionTreeClassifier):
    """Wrapper around sklearn greedy tree classifier
    """

    def fit(self, X, y, feature_names=None, sample_weight=None, check_input=True):
        """Build a decision tree classifier from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.
        feature_names : array-like of shape (n_features)
            The names of the features
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        super().fit(X, y, sample_weight=sample_weight, check_input=check_input)
        self._set_complexity()

    def _set_complexity(self):
        """Set complexity as number of non-leaf nodes
        """
        self.complexity_ = compute_tree_complexity(self.tree_)

    def __str__(self):
        s = '> ------------------------------\n'
        s += '> Greedy CART Tree:\n'
        s += '> \tPrediction is made by looking at the value in the appropriate leaf of the tree\n'
        s += '> ------------------------------' + '\n'
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            return s + export_text(self, feature_names=self.feature_names, show_weights=True)
        else:
            return s + export_text(self, show_weights=True)


class GreedyTreeRegressor(DecisionTreeRegressor):
    """Wrapper around sklearn greedy tree regressor
    """

    def fit(self, X, y, feature_names=None, sample_weight=None, check_input=True):
        """Build a decision tree regressor from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        self : DecisionTreeRegressor
            Fitted estimator.
        """
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = ["X" + str(i + 1) for i in range(X.shape[1])]
        super().fit(X, y, sample_weight=sample_weight, check_input=check_input)
        self._set_complexity()

    def _set_complexity(self):
        """Set complexity as number of non-leaf nodes
        """
        self.complexity_ = compute_tree_complexity(self.tree_)

    def __str__(self):
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            return 'GreedyTree:\n' + export_text(self, feature_names=self.feature_names, show_weights=True)
        else:
            return 'GreedyTree:\n' + export_text(self, show_weights=True)