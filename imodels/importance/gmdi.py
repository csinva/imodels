import copy

import numpy as np
import pandas as pd
from sklearn.ensemble._forest import _generate_unsampled_indices, \
    _generate_sample_indices
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

from .block_transformers import GmdiDefaultTransformer, TreeTransformer
from .ppms import RidgePPM, LogisticPPM


class GMDI:
    """
    The class object for computing GMDI feature importances. Generalized mean
    decrease in impurity (GMDI) is a flexible framework for computing RF
    feature importances. For more details, refer to [paper].

    Parameters
    ----------
    rf_model: scikit-learn random forest object or None
        The RF model to be used for interpretation. If None, then a new
        RandomForestRegressor or RandomForestClassifier is instantiated with
        **kwargs, depending on the task flag.
    partial_prediction_model: A _PartialPredictionModelBase object or "auto"
        The partial prediction model to be used for computing partial
        predictions.
        If value is set to "auto", then a default is chosen as follows:
         - If task is set to "regression", then RidgePPM is used.
         - If task is set to "classification", then LogisticPPM is used.
    scoring_fns: function or dict with functions as values or "auto"
        The scoring functions used for evaluating the partial predictions.
        If "auto", then a default is chosen as follows:
         - If task is set to "regression", then r2_score is used.
         - If task is set to "classification", then log_loss is used.
    mode: string in {"keep_k", "keep_rest"}
        Mode for the method. "keep_k" imputes the mean of each feature not
        in block k when making a partial model prediction, while "keep_rest"
        imputes the mean of each feature in block k. "keep_k" is strongly
        recommended for computational considerations.
    sample_split: string in {"loo", "oob", "inbag"} or None
        The sample splitting strategy to be used when evaluating the partial
        model predictions. The default "loo" (leave-one-out) is strongly
        recommended for performance and in particular, for overcoming the known
        correlation and entropy biases suffered by MDI. "oob" (out-of-bag) can
        also be used to overcome these biases. "inbag" is the sample splitting
        strategy used by MDI. If None, no sample splitting is performed and the
        full data set is used to evaluate the partial model predictions.
    include_raw: bool
        Flag for whether to augment the local decision stump features extracted
        from the RF model with the original features.
    drop_features: bool
        Flag for whether to use an intercept model for the partial predictions
        on a given feature if a tree does not have any nodes that split on it,
        or to use a model on the raw feature (if include_raw is True).
    refit_rf: bool
        Flag for whether to refit the supplied RF model when fitting the GMDI
        feature importances.
    task: string in {"regression", "classifcation"}
        The supervised learning task for the RF model. Used for choosing
        defaults for partial_prediction_model and scoring_fns. Currently only
        regression and classification are supported.
    """

    def __init__(self,
                 rf_model=None,
                 partial_prediction_model="auto",
                 scoring_fns="auto",
                 mode="keep_k",
                 sample_split="loo",
                 include_raw=True,
                 drop_features=True,
                 refit_rf=True,
                 task="regression",
                 **kwargs):
        assert mode in ["keep_k", "keep_rest"]
        assert sample_split in ["loo", "oob", "inbag", None]
        assert task in ["regression", "classification"]
        if rf_model is not None:
            self.rf_model = copy.deepcopy(rf_model)
        elif task == "regression":
            self.rf_model = RandomForestRegressor(**kwargs)
        elif task == "classification":
            self.rf_model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError("Unsupported task.")
        self.partial_prediction_model = partial_prediction_model
        self.scoring_fns = scoring_fns
        self.mode = mode
        self.sample_split = sample_split
        self.include_raw = include_raw
        self.drop_features = drop_features
        self.refit_rf = refit_rf
        self.task = task
        self.is_fitted = False
        self._scores = pd.DataFrame({})

    def get_scores(self, X=None, y=None):
        """
        Obtain the GMDI feature importances.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix. If a pd.DataFrame object is supplied, then
            the column names are used in the output
        y: ndarray of shape (n_samples, n_targets)
            The observed responses.

        Returns
        -------
        scores: pd.DataFrame of shape (n_features, n_scoring_fns)
            The GMDI feature importances.
        """
        if self.is_fitted:
            pass
        else:
            if X is None or y is None:
                raise ValueError("Not yet fitted. Need X and y as inputs.")
            else:
                self._fit_importance_scores(X, y)
        return self._scores

    def _fit_importance_scores(self, X, y):
        if self.refit_rf:
            self.rf_model.fit(X, y)
        all_scores = []
        if self.task == "regression":
            ppm, scoring_fns = self._get_reg_defaults()
        elif self.task == "classification":
            ppm, scoring_fns = self._get_classification_defaults()
            if len(np.unique(y)) > 2:
                y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        else:
            raise ValueError("Only regression and classification tasks "
                             "currently supported.")
            # ppm, scoring_fns = self.partial_prediction_model, self.scoring_fns
        for tree_model in self.rf_model.estimators_:
            transformer = GmdiDefaultTransformer(
                tree_model, drop_features=self.drop_features) if \
                self.include_raw else TreeTransformer(tree_model)
            if self.sample_split == "loo":
                ppm.loo = True
            else:
                ppm.loo = False
            gmdi_helper = GmdiHelper(transformer, ppm, scoring_fns,
                                     self.mode, self.sample_split)
            scores = gmdi_helper.get_scores(X, y)
            if scores is not None:
                all_scores.append(scores)
        scoring_fns = scoring_fns if isinstance(scoring_fns, dict) \
            else {"importance": scoring_fns}
        for fn_name in scoring_fns.keys():
            self._scores[fn_name] = np.mean([scores[fn_name] for scores
                                             in all_scores], axis=0)
        if isinstance(X, pd.DataFrame):
            self._scores.index = X.columns
        self._scores.index.name = 'var'
        self._scores.reset_index(inplace=True)
        self.is_fitted = True

    def _get_reg_defaults(self):
        if self.partial_prediction_model == "auto":
            ppm = RidgePPM()
        else:
            ppm = self.partial_prediction_model
        if self.scoring_fns == "auto":
            scoring_fns = {"importance": _fast_r2_score}
        else:
            scoring_fns = self.scoring_fns
        return ppm, scoring_fns

    def _get_classification_defaults(self):
        if self.partial_prediction_model == "auto":
            ppm = LogisticPPM()
        else:
            ppm = self.partial_prediction_model
        if self.scoring_fns == "auto":
            scoring_fns = {"importance": log_loss}
        else:
            scoring_fns = self.scoring_fns
        return ppm, scoring_fns


class GmdiHelper:
    """
    A class that is primarily used by GMDI for fitting the GMDI scores for
    a single tree in the RF. This class can also be used for further extensions
    of the GMDI framework by supplying block transformers that provide
    other types of feature engineering.

    Parameters
    ----------
    transformer: A BlockTransformerBase object
        A block feature transformer used to generate blocks of engineered
        features for each original feature. GMDI is computed by evaluating
        partial models on these blocks.
    partial_prediction_model: A _PartialPredictionModelBase object
        The partial prediction model to be used for computing partial
        predictions.
    scoring_fns: function or dict with functions as values
        The scoring functions used for evaluating the partial predictions.
    mode: string in {"keep_k", "keep_rest"}
        Mode for the method. "keep_k" imputes the mean of each feature not
        in block k when making a partial model prediction, while "keep_rest"
        imputes the mean of each feature in block k. "keep_k" is strongly
        recommended for computational considerations.
    sample_split: string in {"loo", "oob", "inbag"} or None
        The sample splitting strategy to be used when evaluating the partial
        model predictions. The default "loo" (leave-one-out) is strongly
        recommended for performance and in particular, for overcoming the known
        correlation and entropy biases suffered by MDI. "oob" (out-of-bag) can
        also be used to overcome these biases. "inbag" is the sample splitting
        strategy used by MDI. If None, no sample splitting is performed and the
        full data set is used to evaluate the partial model predictions.
    center: bool
        Flag for whether to center the engineered features.
    normalize: bool
        Flag for whether to rescale the engineered features to have unit
        variance.
    """

    def __init__(self, transformer, partial_prediction_model, scoring_fns,
                 mode="keep_k", sample_split="loo", center=True, normalize=False):
        assert mode in ["keep_k", "keep_rest"]
        assert sample_split in ["loo", "oob", "inbag", None]
        self.transformer = transformer
        self.partial_prediction_model = copy.deepcopy(partial_prediction_model)
        self.scoring_fns = scoring_fns
        self.mode = mode
        self.sample_split = sample_split
        if sample_split in ["oob", "inbag"] and hasattr(partial_prediction_model, "loo") and \
                partial_prediction_model.loo:
            raise ValueError("Cannot use LOO together with OOB or in-bag sample splitting.")
        self.center = center
        self.normalize = normalize
        self._scores = None
        self.is_fitted = False

    def get_scores(self, X=None, y=None):
        """
        Obtain the GMDI feature importances.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix. If a pd.DataFrame object is supplied, then
            the column names are used in the output
        y: ndarray of shape (n_samples, n_targets)
            The observed responses.

        Returns
        -------
        scores: pd.DataFrame of shape (n_features, n_scoring_fns)
            The GMDI feature importances.
        """
        if X is None or y is None:
            if self.is_fitted:
                pass
            else:
                raise ValueError("Not yet fitted. Need X and y as inputs.")
        else:
            self._fit_importance_scores(X, y)
        return self._scores

    def _fit_importance_scores(self, X, y):
        blocked_data = self.transformer.transform(X, center=self.center,
                                                  normalize=self.normalize)
        self.n_features = blocked_data.n_blocks
        if self.sample_split == "oob":
            train_blocked_data, test_blocked_data, y_train, y_test = \
                self._train_test_split(blocked_data, y)
        elif self.sample_split == "inbag":
            train_blocked_data, _, y_train, _ = \
                self._train_test_split(blocked_data, y)
            test_blocked_data = train_blocked_data
            y_test = y_train
        else:
            train_blocked_data = test_blocked_data = blocked_data
            y_train = y_test = y
        if train_blocked_data.get_all_data().shape[1] == 0:
            self._scores = None
            raise Warning("Transformer representation is empty.")
        else:
            full_preds, partial_preds = self._get_preds(
                train_blocked_data, y_train, test_blocked_data, y_test)
            self._score_partial_predictions(full_preds, partial_preds, y_test)
        self.is_fitted = True

    def _get_preds_one_target(self, train_blocked_data, y_train,
                              test_blocked_data, y_test):
        ppm = copy.deepcopy(self.partial_prediction_model)
        ppm.fit(train_blocked_data, y_train, test_blocked_data,
                y_test, self.mode)
        full_preds = ppm.get_full_predictions()
        partial_preds = ppm.get_all_partial_predictions()
        return full_preds, partial_preds

    def _get_preds(self, train_blocked_data, y_train,
                   test_blocked_data, y_test):
        if y_train.ndim == 1:
            return self._get_preds_one_target(train_blocked_data, y_train,
                                              test_blocked_data, y_test)
        else:
            full_preds_list = []
            partial_preds_list = []
            for j in range(y_train.shape[1]):
                yj_train = y_train[:, j]
                yj_test = y_test[:, j]
                full_preds_j, partial_preds_j = self._get_preds_one_target(
                    train_blocked_data, yj_train, test_blocked_data, yj_test)
                full_preds_list.append(full_preds_j)
                partial_preds_list.append(partial_preds_j)
            full_preds = np.array(full_preds_list).T
            partial_preds = dict()
            for k in range(self.n_features):
                partial_preds[k] = \
                    np.array([partial_preds_j[k] for partial_preds_j in
                              partial_preds_list]).T
            return full_preds, partial_preds

    def _score_partial_predictions(self, full_preds, partial_preds, y_test):
        scoring_fns = self.scoring_fns if isinstance(self.scoring_fns, dict) \
            else {"importance": self.scoring_fns}
        all_scores = pd.DataFrame({})
        for fn_name, scoring_fn in scoring_fns.items():
            scores = _partial_preds_to_scores(partial_preds, y_test,
                                              scoring_fn)
            if self.mode == "keep_rest":
                full_score = scoring_fn(y_test, full_preds)
                scores = full_score - scores
            scores = scores.ravel()
            all_scores[fn_name] = scores
        self._scores = all_scores

    def _train_test_split(self, blocked_data, y):
        n_samples = len(y)
        train_indices = _generate_sample_indices(
            self.transformer.oob_seed, n_samples, n_samples)
        test_indices = _generate_unsampled_indices(
            self.transformer.oob_seed, n_samples, n_samples)
        train_blocked_data, test_blocked_data = \
            blocked_data.train_test_split(train_indices, test_indices)
        if y.ndim > 1:
            y_train = y[train_indices, :]
            y_test = y[test_indices, :]
        else:
            y_train = y[train_indices]
            y_test = y[test_indices]
        return train_blocked_data, test_blocked_data, y_train, y_test


def _partial_preds_to_scores(partial_preds, y_test, scoring_fn):
    scores = []
    for k, y_pred in partial_preds.items():
        if not hasattr(y_pred, "__len__"):  # if constant model
            y_pred = np.ones_like(y_test) * y_pred
        elif y_test.shape != y_pred.shape:
            # if constant model for multitarget prediction
            y_pred = np.array([y_pred] * y_test.shape[0])
        scores.append(scoring_fn(y_test, y_pred))
    return np.vstack(scores)


def _fast_r2_score(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - np.mean(y_true, axis=0)) ** 2). \
        sum(axis=0, dtype=np.float64)
    return 1 - numerator / denominator
