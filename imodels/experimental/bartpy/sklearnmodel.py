import copy
from copy import deepcopy
from typing import List, Callable, Mapping, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from joblib import Parallel, delayed
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, model_selection

from .data import Data
from .initializers.initializer import Initializer
from .model import Model
from .node import LeafNode, DecisionNode
from .samplers.leafnode import LeafNodeSampler
from .samplers.modelsampler import ModelSampler, Chain
from .samplers.schedule import SampleSchedule
from .samplers.sigma import SigmaSampler
from .samplers.treemutation import TreeMutationSampler
from .samplers.unconstrainedtree.treemutation import get_tree_sampler
from .sigma import Sigma


def run_chain(model: 'SklearnModel', X: np.ndarray, y: np.ndarray):
    """
    Run a single chain for a model
    Primarily used as a building block for constructing a parallel run of multiple chains
    """

    # TODO: support for classification F^{-1} (y) ~ N(G(x), 1)

    # if model.classification:
    #     z = np.random.normal(loc=model.predict(X))
    #     z[y == 1] = np.maximum(z[y == 1], 0)
    #     z[y == 0] = np.minimum(z[y == 0], 0)
    #     y = z

    model.model = model._construct_model(X, y)

    return model.sampler.samples(model.model,
                                 model.n_samples,
                                 model.n_burn,
                                 model.thin,
                                 model.store_in_sample_predictions,
                                 model.store_acceptance_trace)


def delayed_run_chain():
    return run_chain


def get_nodes(root_node):
    decision_nodes = []
    leaf_nodes = []

    def _add_nodes(n):
        if type(n) == LeafNode:
            leaf_nodes.append(n)
        elif type(n) == DecisionNode:
            decision_nodes.append(n)
        if n.left_child:
            _add_nodes(n.left_child)
        if n.right_child:
            _add_nodes(n.right_child)

    _add_nodes(root_node)
    return decision_nodes, leaf_nodes


def get_root_node(tree):
    for n in tree.decision_nodes:
        if n.depth == 0:
            return n


def shrink_tree(tree, reg_param):
    root = get_root_node(tree)
    tree_d_node = shrink_node(root, reg_param)
    d, l = get_nodes(tree_d_node)
    tree._nodes = d + l
    return tree


def expand_node(node):
    mask_int = 1 - node.data.mask.astype(int)
    # y = node.data.y.values
    val = np.sum(node.data.y.values
                 * mask_int) / np.sum(mask_int)
    node.set_value(val)
    return node


def expand_tree(tree):
    decision, leaves = get_nodes(get_root_node(tree))
    leaves_new = [expand_node(n) for n in leaves]
    tree._nodes = decision + leaves_new
    return tree


def shrink_node(node, reg_param, parent_val, parent_num, cum_sum, scheme, constant):
    """Shrink the tree
    """

    # node.set_value(node.mean_response)

    left = node.left_child
    right = node.right_child
    is_leaf = type(node) == LeafNode
    # if self.prediction_task == 'regression':
    val = node.current_value
    is_root = parent_val is None and parent_num is None
    n_samples = node.n_obs if (scheme != "leaf_based" or is_root) else parent_num

    if is_root:
        val_new = val

    else:
        reg_term = reg_param if scheme == "constant" else reg_param / parent_num

        val_new = (val - parent_val) / (1 + reg_term)

    cum_sum += val_new

    if is_leaf:
        if scheme == "leaf_based":
            v = constant + (val - constant) / (1 + reg_param / node.n_obs)
            node.set_value(v)
        else:
            node.set_value(cum_sum)

    else:
        shrink_node(left, reg_param, val, parent_num=n_samples, cum_sum=cum_sum, scheme=scheme, constant=constant)
        shrink_node(right, reg_param, val, parent_num=n_samples, cum_sum=cum_sum, scheme=scheme, constant=constant)

    return node


class SklearnModel(BaseEstimator, RegressorMixin):
    """
    The main access point to building BART models in BartPy

    Parameters
    ----------
    n_trees: int
        the number of trees to use, more trees will make a smoother fit, but slow training and fitting
    n_chains: int
        the number of independent chains to run
        more chains will improve the quality of the samples, but will require more computation
    sigma_a: float
        shape parameter of the prior on sigma
    sigma_b: float
        scale parameter of the prior on sigma
    n_samples: int
        how many recorded samples to take
    n_burn: int
        how many samples to run without recording to reach convergence
    thin: float
        percentage of samples to store.
        use this to save memory when running large models
    p_grow: float
        probability of choosing a grow mutation in tree mutation sampling
    p_prune: float
        probability of choosing a prune mutation in tree mutation sampling
    alpha: float
        prior parameter on tree structure
    beta: float
        prior parameter on tree structure
    store_in_sample_predictions: bool
        whether to store full prediction samples
        set to False if you don't need in sample results - saves a lot of memory
    store_acceptance_trace: bool
        whether to store acceptance rates of the gibbs samples
        unless you're very memory constrained, you wouldn't want to set this to false
        useful for diagnostics
    tree_sampler: TreeMutationSampler
        Method of sampling used on trees
        defaults to `bartpy.samplers.unconstrainedtree`
    initializer: Initializer
        Class that handles the initialization of tree structure and leaf values
    n_jobs: int
        how many cores to use when computing MCMC samples
        set to `-1` to use all cores
    """

    def __init__(self,
                 n_trees: int = 200,
                 n_chains: int = 4,
                 sigma_a: float = 0.001,
                 sigma_b: float = 0.001,
                 n_samples: int = 200,
                 n_burn: int = 200,
                 thin: float = 0.1,
                 alpha: float = 0.95,
                 beta: float = 2.,
                 store_in_sample_predictions: bool = False,
                 store_acceptance_trace: bool = False,
                 tree_sampler: TreeMutationSampler = get_tree_sampler(0.5, 0.5),
                 initializer: Optional[Initializer] = None,
                 n_jobs=-1,
                 classification: bool = False,
                 max_rules=None):
        self.n_trees = n_trees
        self.n_chains = n_chains
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.p_grow = 0.5
        self.p_prune = 0.5
        self.alpha = alpha
        self.beta = beta
        self.thin = thin
        self.n_jobs = n_jobs
        self.store_in_sample_predictions = store_in_sample_predictions
        self.store_acceptance_trace = store_acceptance_trace
        self.columns = None
        self.tree_sampler = tree_sampler
        self.initializer = initializer
        self.schedule = SampleSchedule(self.tree_sampler, LeafNodeSampler(), SigmaSampler())
        self.sampler = ModelSampler(self.schedule)
        self.classification = classification
        self.max_rules = max_rules

        self.sigma, self.data, self.model, self._prediction_samples, self._model_samples, self.extract = [None] * 6

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> 'SklearnModel':
        """
        Learn the model based on training data

        Parameters
        ----------
        X: pd.DataFrame
            training covariates
        y: np.ndarray
            training targets

        Returns
        -------
        SklearnModel
            self with trained parameter values
        """
        self.model = self._construct_model(X, y)
        self.extract = Parallel(n_jobs=self.n_jobs)(self.f_delayed_chains(X, y))
        self.combined_chains = self._combine_chains(self.extract)
        self._model_samples, self._prediction_samples = self.combined_chains["model"], self.combined_chains[
            "in_sample_predictions"]
        self._acceptance_trace = self.combined_chains["acceptance"]
        self._likelihood = self.combined_chains["likelihood"]
        self._probs = self.combined_chains["probs"]

        self.fitted_ = True
        return self

    @property
    def fitted(self):
        if hasattr(self, "fitted_"):
            return self.fitted_
        return False

    @property
    def complexity_(self):
        if hasattr(self.initializer, "_tree"):
            estimator = self.initializer._tree
            if hasattr(estimator, 'complexity_'):
                return estimator.complexity_

    @staticmethod
    def _combine_chains(extract: List[Chain]) -> Chain:
        keys = list(extract[0].keys())
        combined = {}
        for key in keys:
            combined[key] = np.concatenate([chain[key] for chain in extract], axis=0)
        return combined

    @staticmethod
    def _convert_covariates_to_data(X: np.ndarray, y: np.ndarray) -> Data:
        from copy import deepcopy
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        return Data(deepcopy(X), deepcopy(y), normalize=True)

    def _construct_model(self, X: np.ndarray, y: np.ndarray) -> Model:
        if len(X) == 0 or X.shape[1] == 0:
            raise ValueError("Empty covariate matrix passed")
        self.data = self._convert_covariates_to_data(X, y)
        self.sigma = Sigma(self.sigma_a, self.sigma_b, self.data.y.normalizing_scale, self.classification)
        self.model = Model(self.data,
                           self.sigma,
                           n_trees=self.n_trees,
                           alpha=self.alpha,
                           beta=self.beta,
                           initializer=self.initializer,
                           classification=self.classification)
        n_trees = self.n_trees if self.initializer is None else self.initializer.n_trees
        self.n_trees = n_trees
        return self.model

    def f_delayed_chains(self, X: np.ndarray, y: np.ndarray):
        """
        Access point for getting access to delayed methods for running chains
        Useful for when you want to run multiple instances of the model in parallel
        e.g. when calculating a null distribution for feature importance

        Parameters
        ----------
        X: np.ndarray
            Covariate matrix
        y: np.ndarray
            Target array

        Returns
        -------
        List[Callable[[], ChainExtract]]
        """
        return [delayed(x)(self, X, y) for x in self.f_chains()]

    def f_chains(self) -> List[Callable[[], Chain]]:
        """
        List of methods to run MCMC chains
        Useful for running multiple models in parallel

        Returns
        -------
        List[Callable[[], Extract]]
            List of method to run individual chains
            Length of n_chains
        """
        return [delayed_run_chain() for _ in range(self.n_chains)]

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        """
        Predict the target corresponding to the provided covariate matrix
        If X is None, will predict based on training covariates

        Prediction is based on the mean of all samples

        Parameters
        ----------
        X: pd.DataFrame
            covariates to predict from

        Returns
        -------
        np.ndarray
            predictions for the X covariates
        """
        if X is None and self.store_in_sample_predictions:
            return self.data.y.unnormalize_y(np.mean(self._prediction_samples, axis=0))
        elif X is None and not self.store_in_sample_predictions:
            raise ValueError(
                "In sample predictions only possible if model.store_in_sample_predictions is `True`.  Either set the parameter to True or pass a non-None X parameter")
        else:
            predictions = self._out_of_sample_predict(X)
            if self.classification:
                return np.round(predictions, 0)
            return predictions

    def predict_proba(self, X: np.ndarray = None) -> np.ndarray:
        preds = self._out_of_sample_predict(X)
        return np.stack([preds, 1 - preds], axis=1)

    def residuals(self, X=None, y=None) -> np.ndarray:
        """
        Array of error for each observation

        Parameters
        ----------
        X: np.ndarray
            Covariate matrix
        y: np.ndarray
            Target array

        Returns
        -------
        np.ndarray
            Error for each observation
        """
        if y is None:
            return self.model.data.y.unnormalized_y - self.predict(X)
        else:
            return y - self.predict(X)

    def l2_error(self, X=None, y=None) -> np.ndarray:
        """
        Calculate the squared errors for each row in the covariate matrix

        Parameters
        ----------
        X: np.ndarray
            Covariate matrix
        y: np.ndarray
            Target array
        Returns
        -------
        np.ndarray
            Squared error for each observation
        """
        return np.square(self.residuals(X, y))

    def rmse(self, X, y) -> float:
        """
        The total RMSE error of the model
        The sum of squared errors over all observations

        Parameters
        ----------
        X: np.ndarray
            Covariate matrix
        y: np.ndarray
            Target array

        Returns
        -------
        float
            The total summed L2 error for the model
        """
        return np.sqrt(np.sum(self.l2_error(X, y)))

    def _chain_pred_arr(self, X, chain_number):
        chain_len = int(self.n_samples)
        samples_chain = self._model_samples[chain_number * chain_len: (chain_number + 1) * chain_len]
        predictions_transformed = [x.predict(X) for x in samples_chain]
        return predictions_transformed

    def predict_chain(self, X, chain_number):
        predictions_transformed = self._chain_pred_arr(X, chain_number)
        predictions = self.data.y.unnormalize_y(np.mean(predictions_transformed, axis=0))
        if self.classification:
            predictions = scipy.stats.norm.cdf(predictions)
        return predictions

    def chain_mse_std(self, X, y, chain_number):
        predictions_transformed = self._chain_pred_arr(X, chain_number)
        predictions_std = np.std(
            [mean_squared_error(self.data.y.unnormalize_y(preds), y) for preds in predictions_transformed])
        return predictions_std

    def chain_predictions(self, X, chain_number):
        predictions_transformed = self._chain_pred_arr(X, chain_number)
        preds_arr = [self.data.y.unnormalize_y(preds) for preds in predictions_transformed]
        return preds_arr

    def between_chains_var(self, X):
        all_predictions = np.stack([self.data.y.unnormalize_y(x.predict(X)) for x in self._model_samples], axis=1)

        def _get_var(preds_arr):
            mean_pred = preds_arr.mean(axis=1)
            var = np.mean((preds_arr - np.expand_dims(mean_pred, 1)) ** 2)
            return var

        total_var = _get_var(all_predictions)
        within_chain_var = 0
        for c in range(self.n_chains):
            chain_preds = self._chain_pred_arr(X, c)
            within_chain_var += _get_var(np.stack(chain_preds, axis=1))
        return total_var - within_chain_var

    def _out_of_sample_predict(self, X):
        samples = self._model_samples
        predictions_transformed = [x.predict(X) for x in samples]
        predictions = self.data.y.unnormalize_y(np.mean(predictions_transformed, axis=0))
        if self.classification:
            predictions = scipy.stats.norm.cdf(predictions)
        return predictions

    def fit_predict(self, X, y):
        self.fit(X, y)
        if self.store_in_sample_predictions:
            return self.predict()
        else:
            return self.predict(X)

    @property
    def model_samples(self) -> List[Model]:
        """
        Array of the model as it was after each sample.
        Useful for examining for:

         - examining the state of trees, nodes and sigma throughout the sampling
         - out of sample prediction

        Returns None if the model hasn't been fit

        Returns
        -------
        List[Model]
        """
        return self._model_samples

    @property
    def acceptance_trace(self) -> List[Mapping[str, float]]:
        """
        List of Mappings from variable name to acceptance rates

        Each entry is the acceptance rate of the variable in each iteration of the model

        Returns
        -------
        List[Mapping[str, float]]
        """
        return self._acceptance_trace

    @property
    def likelihood(self) -> List:
        """
        List of Mappings from variable name to likelihood

        Each entry is the acceptance rate of the variable in each iteration of the model

        Returns
        -------
        List[Mapping[str, float]]
        """
        return self._likelihood

    @property
    def probs(self) -> List:
        """
        List of Mappings from variable name to likelihood

        Each entry is the acceptance rate of the variable in each iteration of the model

        Returns
        -------
        List[Mapping[str, float]]
        """
        return self._probs

    @property
    def prediction_samples(self) -> np.ndarray:
        """
        Matrix of prediction samples at each point in sampling
        Useful for assessing convergence, calculating point estimates etc.

        Returns
        -------
        np.ndarray
            prediction samples with dimensionality n_samples * n_points
        """
        return self.prediction_samples

    def from_extract(self, extract: List[Chain], X: np.ndarray, y: np.ndarray) -> 'SklearnModel':
        """
        Create a copy of the model using an extract
        Useful for doing operations on extracts created in external processes like feature selection
        Parameters
        ----------
        extract: Extract
            samples produced by delayed chain methods
        X: np.ndarray
            Covariate matrix
        y: np.ndarray
            Target variable

        Returns
        -------
        SklearnModel
            Copy of the current model with samples
        """
        new_model = deepcopy(self)
        combined_chain = self._combine_chains(extract)
        self._model_samples, self._prediction_samples = combined_chain["model"], combined_chain["in_sample_predictions"]
        self._acceptance_trace = combined_chain["acceptance"]
        new_model.data = self._convert_covariates_to_data(X, y)
        return new_model


class BART(SklearnModel):

    @staticmethod
    def _get_n_nodes(trees):
        nodes = 0
        for tree in trees:
            nodes += len(tree.decision_nodes)
        return nodes

    @property
    def sample_complexity(self):
        # samples = self._model_samples
        # trees = [s.trees for s in samples]
        complexities = [self._get_n_nodes(t) for t in self.trees]
        return np.sum(complexities)

    @staticmethod
    def sub_forest(trees, n_nodes):
        nodes = 0
        for i, tree in enumerate(trees):
            nodes += len(tree.decision_nodes)
            if nodes >= n_nodes:
                return trees[0:i + 1]

    @property
    def trees(self):
        trs = [s.trees for s in self._model_samples]
        return trs

    def update_complexity(self, i):
        samples_complexity = [self._get_n_nodes(t) for t in self.trees]

        # complexity_sum = 0
        arg_sort_complexity = np.argsort(samples_complexity)
        self._model_samples = self._model_samples[arg_sort_complexity[:i + 1]]

        return self


class ImputedBART(BaseEstimator):
    def __init__(self, estimator_):
        # super(ShrunkBARTRegressor, self).__init__()
        self.estimator_ = estimator_

    def predict(self, *args, **kwargs):
        return self.estimator_.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(*args, **kwargs)
        else:
            return NotImplemented

    def score(self, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(*args, **kwargs)
        else:
            return NotImplemented


class ShrunkBART(ImputedBART):

    def __init__(self, estimator_, reg_param, scheme):
        super(ShrunkBART, self).__init__(estimator_)
        self.reg_param = reg_param
        self.scheme = scheme

    def shrink_tree(self, tree):
        root = get_root_node(tree)
        tree_d_node = shrink_node(root, self.reg_param, parent_val=None, parent_num=None, cum_sum=0, scheme=self.scheme,
                                  constant=np.mean(self.estimator_.data.y.values))
        d, l = get_nodes(tree_d_node)
        tree._nodes = d + l
        return tree

    def fit(self, *args, **kwargs):
        if not self.estimator_.fitted:
            self.estimator_.fit(*args, **kwargs)
        samples = []
        for s in self.estimator_.model_samples:
            for i, tree in enumerate(s._trees):
                s_tree = self.shrink_tree(expand_tree(copy.deepcopy(tree)))
                s._trees[i] = s_tree
            samples.append(s)
        self.estimator_._model_samples = samples
        self.fitted_ = True


class ExpandedBART(ImputedBART):

    def fit(self, *args, **kwargs):
        if not self.estimator_.fitted:
            self.estimator_.fit(*args, **kwargs)
        samples = []
        for s in self.estimator_.model_samples:
            for i, tree in enumerate(s._trees):
                s_tree = expand_tree(copy.deepcopy(tree))
                s._trees[i] = s_tree
            samples.append(s)
        self.estimator_._model_samples = samples
        self.fitted_ = True


# class ExpandedBARTRegressor(ImputedBARTRegressor):
#
#     def fit(self, *args, **kwargs):
#         if not self.estimator_.fitted:
#             self.estimator_.fit(*args, **kwargs)
#         samples = []
#         for s in self.estimator_.model_samples:
#             for i, tree in enumerate(s._trees):
#                 s_tree = expend_tree(copy.deepcopy(tree), args[1])
#                 s._trees[i] = s_tree
#             samples.append(s)
#         self.estimator_._model_samples = samples
#         self.fitted_ = True


class ShrunkBARTCV(ShrunkBART):
    def __init__(self, estimator_: BaseEstimator, scheme: str,
                 reg_param_list: List[float] = [0.1, 1, 10, 50, 100, 500],
                 cv: int = 3, scoring=None):
        super(ShrunkBARTCV, self).__init__(estimator_, None, scheme)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y, *args, **kwargs):
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = ShrunkBART(deepcopy(self.estimator_), reg_param, self.scheme)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y)


def main():
    # iris = datasets.load_iris()
    # idx = np.logical_or(iris.target == 0, iris.target == 1)
    # X, y = iris.data[idx, ...], iris.target[idx]
    X, y = datasets.load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=1)
    bart = BART(classification=False)
    bart.fit(X_train, y_train)
    preds_org = bart.predict(X_test)
    mse = np.linalg.norm(preds_org - y_test)
    print(mse)
    # tree = DecisionTreeClassifier()
    # tree.fit(X, y)
    # preds_tree = tree.predict_proba(X)
    # bart_s = ShrunkBARTCV(copy.deepcopy(bart), scheme="node_based")
    # bart_s.fit(X, y)
    #
    # # bart_s_c = ShrunkBART(copy.deepcopy(bart), reg_param=2, scheme="constant")
    # # bart_s_c.fit(X, y)
    # #
    # # bart_s_l = ShrunkBART(copy.deepcopy(bart), reg_param=2, scheme="leaf_based")
    # # bart_s_l.fit(X, y)
    # # bart_s_cv = ShrunkBARTRegressorCV(estimator_=copy.deepcopy(bart))
    # # bart_s_cv.fit(X, y)
    # e_bart = ExpandedBART(estimator_=copy.deepcopy(bart))
    # e_bart.fit(X, y)
    #
    # preds = bart_s.predict(X)
    #
    # # preds_c = bart_s_c.predict(X)
    # # preds_l = bart_s_l.predict(X)
    # # # preds_cv = bart_s_cv.predict(X)
    # preds_bart_e = e_bart.predict(X)
    # fig, ax = plt.subplots(1)
    #
    # ax.scatter(np.arange(len(y)), preds_org, c="orange", label="bart")
    # ax.scatter(np.arange(len(y)), preds, c="purple", alpha=0.3, label="shrunk node")
    # # ax.scatter(np.arange(len(y)), preds_c, c="blue", alpha=0.3, label="shrunk constant")
    # # ax.scatter(np.arange(len(y)), preds_l, c="red", alpha=0.3, label="shrunk leaf")
    # ax.scatter(np.arange(len(y)), preds_bart_e, c="green", alpha=0.3, label="average")
    # # preds_all = [preds_org, preds_c, preds_l, preds_bart_e, preds]
    # # shift = 0.5
    # # rng = (np.min([np.min(p) for p in preds_all]) - shift, np.max([np.max(p) for p in preds_all]) + shift)
    # # n_bins = 200
    # # alpha = 0.8
    # # ax.hist(preds_org, color="orange", alpha=alpha, label="bart", bins=n_bins, range=rng)
    # # ax.hist(preds, color="purple", alpha=alpha, label="shrunk node", bins=n_bins, range=rng)
    # # ax.hist(preds_c, color="blue", alpha=alpha, label="shrunk constant", bins=n_bins, range=rng)
    # # ax.hist(preds_l, color="red", alpha=alpha, label="shrunk leaf", bins=n_bins, range=rng)
    # # ax.hist(preds_bart_e, color="green", alpha=alpha, label="average", bins=n_bins, range=rng)
    # #
    # # ax.set_xlabel("Predicted Value")
    # # ax.set_ylabel("Count")
    #
    # plt.title(np.mean(y))
    #
    # plt.legend(loc="upper left")
    # plt.savefig("bart_shrink.png")
    # # plt.show()
    # #
    # # plt.close()
    #


if __name__ == '__main__':
    main()
