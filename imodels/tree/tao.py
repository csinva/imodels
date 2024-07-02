import random
from copy import deepcopy
from queue import deque

import numpy as np
from mlxtend.classifier import LogisticRegression
from sklearn import datasets
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.utils import check_X_y

from imodels.util.arguments import check_fit_arguments


class TaoTree(BaseEstimator):

    def __init__(self, model_type: str = 'CART',
                 n_iters: int = 20,
                 model_args: dict = {'max_leaf_nodes': 15},
                 randomize_tree=False,
                 update_scoring='accuracy',
                 min_node_samples_tao=3,
                 min_leaf_samples_tao=2,
                 node_model='stump',
                 node_model_args: dict = {},
                 reg_param: float = 1e-3,
                 weight_errors: bool = False,
                 verbose: int = 0,
                 ):
        """TAO: Alternating optimization of decision trees, with application to learning sparse oblique trees (Neurips 2018)
        https://proceedings.neurips.cc/paper/2018/hash/185c29dc24325934ee377cfda20e414c-Abstract.html
        Note: this implementation learns single-feature splits rather than oblique trees.

        Currently supports
        - given a CART tree, posthoc improve it with TAO
            - also works with HSTreeCV

        Todo
        - update bottom to top otherwise input points don't get updated
        - update leaf nodes
        - support regression
        - support FIGS
        - support error-weighting
        - support oblique trees
            - support generic models at decision node
            - support pruning (e.g. if weights -> 0, then remove a node)
        - support classifiers in leaves

        Parameters
        ----------

        model_type: str
            'CART' or 'FIGS'

        n_iters
            Number of iterations to run TAO

        model_args
            Arguments to pass to the model

        randomize_tree
            Whether to randomize the tree before each iteration

        min_node_samples_tao: int
            Minimum number of samples in a node to apply tao

        min_leaf_samples_tao: int

        node_model: str
            'stump' or 'linear'

        reg_param
            Regularization parameter for node-wise linear model (if node_model is 'linear')

        verbose: int
            Verbosity level
        """
        super().__init__()
        self.model_type = model_type
        self.n_iters = n_iters
        self.model_args = model_args
        self.randomize_tree = randomize_tree
        self.update_scoring = update_scoring
        self.min_node_samples_tao = min_node_samples_tao
        self.min_leaf_samples_tao = min_leaf_samples_tao
        self.node_model = node_model
        self.node_model_args = node_model_args
        self.reg_param = reg_param
        self.weight_errors = weight_errors
        self.verbose = verbose
        self._init_prediction_task()  # decides between regressor and classifier

    def _init_prediction_task(self):
        """
        TaoRegressor and TaoClassifier override this method
        to alter the prediction task. When using this class directly,
        it is equivalent to SuperCARTRegressor
        """
        self.prediction_task = 'classification'

    def fit(self, X, y=None, feature_names=None, sample_weight=None):
        """
        Params
        ------
        _sample_weight: array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Splits that would create child nodes with net zero or negative weight
            are ignored while searching for a split in each node.
        """
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        if isinstance(self, RegressorMixin):
            raise Warning('TAO Regression is not yet tested')
        X, y = check_X_y(X, y)
        y = y.astype(float)
        if feature_names is not None:
            self.feature_names_ = feature_names
        if self.model_type == 'CART':
            if isinstance(self, ClassifierMixin):
                self.model = DecisionTreeClassifier(**self.model_args)
            elif isinstance(self, RegressorMixin):
                self.model = DecisionTreeRegressor(**self.model_args)
            self.model.fit(X, y, sample_weight=sample_weight)
            if self.verbose:
                print(export_text(self.model))
            # plot_tree(self.model)
            # plt.savefig('/Users/chandan/Desktop/tree.png', dpi=300)
            # plt.show()

        if self.randomize_tree:
            # shuffle CART features
            np.random.shuffle(self.model.tree_.feature)
            # np.random.shuffle(self.model.tree_.threshold)
            for i in range(self.model.tree_.node_count):  # split on feature medians
                self.model.tree_.threshold[i] = np.median(
                    X[:, self.model.tree_.feature[i]])
        if self.verbose:
            print('starting score', self.model.score(X, y))
        for i in range(self.n_iters):
            num_updates = self._tao_iter_cart(
                X, y, self.model.tree_, sample_weight=sample_weight)
            if num_updates == 0:
                break

        return self

    def _tao_iter_cart(self, X, y, tree, X_score=None, y_score=None, sample_weight=None):
        """Updates tree by applying the tao algorithm to the tree
        Params
        ------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.
        model: DecisionTreeClassifier.tree_ or DecisionTreeRegressor.tree_
            The model to be post-hoc improved
        """

        # Tree properties
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value

        # For each node, store the path to that node #######################################################
        # data structure with (index, path_to_node_index)
        indexes_with_prefix_paths = []
        # e.g. if if node 3 is the left child of node 1 which is the right child of node 0
        # then we get (3, [(0, R), (1, L)])

        # start with the root node id (0) and its depth (0)
        queue = deque()
        queue.append((0, []))
        while len(queue) > 0:
            node_id, path_to_node_index = queue.popleft()
            indexes_with_prefix_paths.append((node_id, path_to_node_index))

            # If a split node, append left and right children and depth to queue
            if children_left[node_id] != children_right[node_id]:
                queue.append(
                    (children_left[node_id], path_to_node_index + [(node_id, 'L')]))
                queue.append(
                    (children_right[node_id], path_to_node_index + [(node_id, 'R')]))
        # print(indexes_with_prefix_paths)

        num_updates = 0

        # Reversing BFS queue presents nodes bottom -> top one level at a time
        for (node_id, path_to_node_index) in reversed(indexes_with_prefix_paths):
            # For each each node, try a TAO update
            # print('node_id', node_id, path_to_node_index)

            # Compute the points being input to the node ######################################
            def filter_points_by_path(X, y, path_to_node_index):
                """Returns the points in X that are in the path to the node"""
                for node_id, direction in path_to_node_index:
                    idxs = X[:, feature[node_id]] <= threshold[node_id]
                    if direction == 'R':
                        idxs = ~idxs
                    # print('idxs', idxs.size, idxs.sum())
                    X = X[idxs]
                    y = y[idxs]
                return X, y

            X_node, y_node = filter_points_by_path(X, y, path_to_node_index)

            if sample_weight is not None:
                sample_weight_node = filter_points_by_path(
                    X, sample_weight, path_to_node_index)[1]
            else:
                sample_weight_node = np.ones(y_node.size)

            # Skip over leaf nodes and nodes with too few samples ######################################
            if children_left[node_id] == children_right[node_id]:  # is leaf node
                if isinstance(self, RegressorMixin) and X_node.shape[0] >= self.min_leaf_samples_tao:
                    # old_score = self.model.score(X, y)
                    value[node_id] = np.mean(y_node)
                    """
                    new_score = self.model.score(X, y)
                    if new_score > old_score:
                        print(f'\tLeaf improved score from {old_score:0.3f} to {new_score:0.3f}')
                    if new_score < old_score:
                        print(f'\tLeaf reduced score from {old_score:0.3f} to {new_score:0.3f}')
                        # raise ValueError('Leaf update reduced score')
                    """
                # print('\tshapes', X_node.shape, y_node.shape)
                # print('\tvals:', value[node_id][0][0], np.mean(y_node))
                # assert value[node_id][0][0] == np.mean(y_node), 'unless tree changed, vals should be leaf means'
                continue
            elif X_node.shape[0] < self.min_node_samples_tao:
                continue

            # Compute the outputs for these points if they go left or right ######################################
            def predict_from_node(X, node_id):
                """Returns predictions for X starting at node node_id"""

                def predict_from_node(x, node_id):
                    """Returns predictions for x starting at node node_id"""
                    if children_left[node_id] == children_right[node_id]:
                        if isinstance(self, RegressorMixin):
                            return value[node_id]
                        if isinstance(self, ClassifierMixin):
                            # note value stores counts for each class
                            return np.argmax(value[node_id])
                    if x[feature[node_id]] <= threshold[node_id]:
                        return predict_from_node(x, children_left[node_id])
                    else:
                        return predict_from_node(x, children_right[node_id])

                preds = np.zeros(X.shape[0])
                for i in range(X.shape[0]):
                    preds[i] = predict_from_node(X[i], node_id)
                return preds

            y_node_left = predict_from_node(X_node, children_left[node_id])
            y_node_right = predict_from_node(X_node, children_right[node_id])
            if node_id == 0:  # root node
                assert np.all(np.logical_or(self.model.predict(X_node) == y_node_left,
                                            self.model.predict(
                                                X_node) == y_node_right)), \
                    'actual predictions should match either predict_from_node left or right'

            # Decide on prediction target (want to go left (0) / right (1) when advantageous)
            # TAO paper binarizes these (e.g. predict 0 or 1 depending on which of these is better)
            y_node_absolute_errors = np.abs(np.vstack((y_node - y_node_left,
                                                       y_node - y_node_right))).T

            # screen out indexes where going left/right has no effect
            idxs_relevant = y_node_absolute_errors[:,
                                                   0] != y_node_absolute_errors[:, 1]
            if idxs_relevant.sum() <= 1:  # nothing to change
                if self.verbose:
                    print('no errors to change')
                continue
            # assert np.all((self.model.predict(X) != y)[idxs_relevant]), 'relevant indexes should be errors'
            y_node_target = np.argmin(y_node_absolute_errors, axis=1)
            y_node_target = y_node_target[idxs_relevant]

            # here, we optionally weight these errors by the size of the error
            # if we want this to work for classification, must switch to predict_proba
            # if self.prediction_task == 'regression':
            # weight by the difference in error ###############################################################
            if self.weight_errors:
                sample_weight_node *= np.abs(
                    y_node_absolute_errors[:, 1] - y_node_absolute_errors[:, 0])
            sample_weight_node_target = sample_weight_node[idxs_relevant]
            X_node = X_node[idxs_relevant]

            # Fit a 1-variable binary classification model on these outputs ######################################
            # Note: this could be customized (e.g. for sparse oblique trees)
            best_score = -np.inf
            best_feat_num = None
            for feat_num in range(X.shape[1]):
                if isinstance(self, ClassifierMixin):
                    if self.node_model == 'linear':
                        m = LogisticRegression(**self.node_model_args)
                    elif self.node_model == 'stump':
                        m = DecisionTreeClassifier(
                            max_depth=1, **self.node_model_args)
                if isinstance(self, RegressorMixin):
                    if self.node_model == 'linear':
                        m = LinearRegression(**self.node_model_args)
                    elif self.node_model == 'stump':
                        m = DecisionTreeRegressor(
                            max_depth=1, **self.node_model_args)
                X_node_single_feat = X_node[:, feat_num: feat_num + 1]
                m.fit(X_node_single_feat, y_node_target,
                      sample_weight=sample_weight_node_target)
                score = m.score(X_node_single_feat, y_node_target,
                                sample_weight=sample_weight_node_target)
                if score > best_score:
                    best_score = score
                    best_feat_num = feat_num
                    best_model = deepcopy(m)
                    if self.node_model == 'linear':
                        best_threshold = -best_model.intercept_ / \
                            best_model.coef_[0]
                    elif self.node_model == 'stump':
                        best_threshold = best_model.tree_.threshold[0]
            # print((feature[node_id], threshold[node_id]), '\n->',
            #       (best_feat_num, best_threshold))

            # Update the node with the new feature / threshold ######################################
            old_feat_num = feature[node_id]
            old_threshold = threshold[node_id]
            # print(X.sum(), y.sum())

            if X_score is None:
                X_score = X
            if y_score is None:
                y_score = y

            scorer = get_scorer(self.update_scoring)

            old_score = scorer(self.model, X_score, y_score)

            feature[node_id] = best_feat_num
            threshold[node_id] = best_threshold
            new_score = scorer(self.model, X_score, y_score)

            # debugging
            if self.verbose > 1:
                if old_score == new_score:
                    print('\tno change', best_feat_num, old_feat_num)
                print(f'\tscore_total {old_score:0.4f} -> {new_score:0.4f}')
            if old_score >= new_score:
                feature[node_id] = old_feat_num
                threshold[node_id] = old_threshold
            else:
                # (Track if any updates were necessary)
                num_updates += 1
                if self.verbose > 0:
                    print(
                        f'Improved score from {old_score:0.4f} to {new_score:0.4f}')

            # debugging snippet (if score_m_new > score_m_old, then new_score should be > old_score, but it isn't!!!!)
            if self.verbose > 1:
                """
                X_node_single_feat = X_node[:, best_feat_num: best_feat_num + 1]
                score_m_new = best_model.score(X_node_single_feat, y_node_target, sample_weight=sample_weight)
                best_model.tree_.feature[0] = old_feat_num
                best_model.tree_.threshold[0] = old_threshold
                X_node_single_feat = X_node[:, old_feat_num: old_feat_num + 1]
                score_m_old = best_model.score(X_node_single_feat, y_node_target, sample_weight=sample_weight)
                print('\t\t', f'score_local {score_m_old:0.4f} -> {score_m_new:0.4f}')
                """

        return num_updates

    def predict(self, X):
        preds = self.model.predict(X)
        if hasattr(self, "classes_"):
            print("classes_", self.classes_, 'preds', preds)
            return np.array([self.classes_[int(i)] for i in preds])
        else:
            return preds

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    # def score(self, X, y):
        # return self.model.score(X, y)


class TaoTreeRegressor(TaoTree, RegressorMixin):
    pass


class TaoTreeClassifier(TaoTree, ClassifierMixin):
    pass


if __name__ == '__main__':
    np.random.seed(13)
    random.seed(13)
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    print('X.shape', X.shape)
    print('ys', np.unique(y_train), '\n\n')
    m = TaoTreeClassifier(randomize_tree=False, weight_errors=False,
                          node_model='stump', model_args={'max_depth': 3},
                          verbose=1)
    m.fit(X_train, y_train)
    print('Train acc', np.mean(m.predict(X_train) == y_train))
    print('Test acc', np.mean(m.predict(X_test) == y_test))
    # print(m.predict(X_train), m.predict_proba(X_train).shape)
    # print(m.predict_proba(X_train))

    # X, y = datasets.load_diabetes(return_X_y=True)  # regression
    # X = np.random.randn(500, 10)
    # y = (X[:, 0] > 0).astype(float) + (X[:, 1] > 1).astype(float)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.33, random_state=42
    # )
    # m = TaoRegressor()
    # m.fit(X_train, y_train)
    # print('mse', np.mean(np.square(m.predict(X_test) - y_test)),
    #       'baseline', np.mean(np.square(y_test)))
