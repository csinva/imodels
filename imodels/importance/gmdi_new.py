import numpy as np
import pandas as pd

from .ppms_new import PartialPredictionModelBase, GenericRegressorPPM, GenericClassifierPPM
from .block_transformers_new import _blocked_train_test_split


class ForestGMDI:
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
    prediction_model: A PartialPredictionModelBase object or "auto"
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
    store_prediction_model: bool
        Flag for whether to store fitted RF+ prediction model
    task: string in {"regression", "classification"}
        The supervised learning task for the RF model. Used for choosing
        defaults for prediction_model and scoring_fns. Currently only
        regression and classification are supported.
    """

    def __init__(self, estimators, transformers, scoring_fns,
                 sample_split="loo", tree_random_states=None, mode="keep_k",
                 task="regression", center=True, normalize=False):
        assert sample_split in ["loo", "oob", "inbag", None]
        assert mode in ["keep_k", "keep_rest"]
        assert task in ["regression", "classification"]
        self.estimators = estimators
        self.transformers = transformers
        self.scoring_fns = scoring_fns
        self.sample_split = sample_split
        self.tree_random_states = tree_random_states
        if self.sample_split in ["oob", "inbag"] and not self.tree_random_states:
            raise ValueError("Must specify tree_random_states to use 'oob' or 'inbag' sample_split.")
        self.mode = mode
        self.task = task
        self.center = center
        self.normalize = normalize
        self.is_fitted = False
        self.prediction_score_ = pd.DataFrame({})
        self.feature_importances_ = pd.DataFrame({})
        self.feature_importances_by_tree_ = {}

    def get_scores(self, X, y):
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
        self._fit_importance_scores(X, y)
        return self.feature_importances_

    def _fit_importance_scores(self, X, y):
        all_scores = []
        all_full_preds = []
        for estimator, transformer, tree_random_state in \
                zip(self.estimators, self.transformers, self.tree_random_states):
            tree_gmdi = TreeGMDI(estimator=estimator,
                                 transformer=transformer,
                                 scoring_fns=self.scoring_fns,
                                 sample_split=self.sample_split,
                                 tree_random_state=tree_random_state,
                                 mode=self.mode,
                                 task=self.task,
                                 center=self.center,
                                 normalize=self.normalize)
            scores = tree_gmdi.get_scores(X, y)
            if scores is not None:
                all_scores.append(scores)
                all_full_preds.append(tree_gmdi._full_preds)
        if len(all_scores) == 0:
            raise ValueError("Transformer representation was empty for all trees.")
        full_preds = np.nanmean(all_full_preds, axis=0)
        self._full_preds = full_preds
        scoring_fns = self.scoring_fns if isinstance(self.scoring_fns, dict) \
            else {"importance": self.scoring_fns}
        for fn_name, scoring_fn in scoring_fns.items():
            self.feature_importances_by_tree_[fn_name] = pd.concat([scores[fn_name] for scores in all_scores], axis=1)
            self.feature_importances_by_tree_[fn_name].columns = np.arange(len(all_scores))
            self.feature_importances_[fn_name] = np.mean(self.feature_importances_by_tree_[fn_name], axis=1)
            self.prediction_score_[fn_name] = [scoring_fn(y[~np.isnan(full_preds)], full_preds[~np.isnan(full_preds)])]
        if list(scoring_fns.keys()) == ["importance"]:
            self.prediction_score_ = self.prediction_score_["importance"]
            self.feature_importances_by_tree_ = self.feature_importances_by_tree_["importance"]
        if isinstance(X, pd.DataFrame):
            self.feature_importances_.index = X.columns
        self.feature_importances_.index.name = 'var'
        self.feature_importances_.reset_index(inplace=True)
        self.is_fitted = True


class TreeGMDI:
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
    prediction_model: A PartialPredictionModelBase object
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
    store_prediction_model: bool
        Flag for whether to store fitted RF+ prediction model
    center: bool
        Flag for whether to center the engineered features.
    normalize: bool
        Flag for whether to rescale the engineered features to have unit
        variance.
    """

    def __init__(self, estimator, transformer, scoring_fns,
                 sample_split="loo", tree_random_state=None, mode="keep_k",
                 task="regression", center=True, normalize=False):
        assert sample_split in ["loo", "oob", "inbag", "auto", None]
        assert mode in ["keep_k", "keep_rest"]
        assert task in ["regression", "classification"]
        self.estimator = estimator
        self.transformer = transformer
        self.scoring_fns = scoring_fns
        self.sample_split = sample_split
        self.tree_random_state = tree_random_state
        _validate_sample_split(self.sample_split, self.estimator, isinstance(self.estimator, PartialPredictionModelBase))
        if self.sample_split in ["oob", "inbag"] and not self.tree_random_state:
            raise ValueError("Must specify tree_random_state to use 'oob' or 'inbag' sample_split.")
        self.mode = mode
        self.task = task
        self.center = center
        self.normalize = normalize
        self.is_fitted = False
        self._full_preds = None
        self.prediction_score_ = None
        self.feature_importances_ = None

    def get_scores(self, X, y):
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
        self._fit_importance_scores(X, y)
        return self.feature_importances_

    def _fit_importance_scores(self, X, y):
        n_samples = y.shape[0]
        blocked_data = self.transformer.transform(X, center=self.center,
                                                  normalize=self.normalize)
        self.n_features = blocked_data.n_blocks
        train_blocked_data, test_blocked_data, y_train, y_test, test_indices = \
            _get_sample_split_data(blocked_data, y, self.tree_random_state, self.sample_split)
        if train_blocked_data.get_all_data().shape[1] != 0:
            if hasattr(self.estimator, "predict_full") and \
                    hasattr(self.estimator, "predict_partial"):
                full_preds = self.estimator.predict_full(test_blocked_data)
                partial_preds = self.estimator.predict_partial(test_blocked_data, mode=self.mode)
            else:
                if self.task == "regression":
                    ppm = GenericRegressorPPM(self.estimator)
                elif self.task == "classification":
                    ppm = GenericClassifierPPM(self.estimator)
                full_preds = ppm.predict_full(test_blocked_data)
                partial_preds = ppm.predict_partial(test_blocked_data, mode=self.mode)
            self._score_full_predictions(y_test, full_preds)
            self._score_partial_predictions(y_test, full_preds, partial_preds)

            full_preds_n = np.empty(n_samples) if full_preds.ndim == 1 \
                else np.empty((n_samples, full_preds.shape[1]))
            full_preds_n[:] = np.nan
            full_preds_n[test_indices] = full_preds
            self._full_preds = full_preds_n
        self.is_fitted = True

    def _score_full_predictions(self, y_test, full_preds):
        scoring_fns = self.scoring_fns if isinstance(self.scoring_fns, dict) \
            else {"score": self.scoring_fns}
        all_prediction_scores = pd.DataFrame({})
        for fn_name, scoring_fn in scoring_fns.items():
            scores = scoring_fn(y_test, full_preds)
            all_prediction_scores[fn_name] = [scores]
        self.prediction_score_ = all_prediction_scores

    def _score_partial_predictions(self, y_test, full_preds, partial_preds):
        scoring_fns = self.scoring_fns if isinstance(self.scoring_fns, dict) \
            else {"importance": self.scoring_fns}
        all_scores = pd.DataFrame({})
        for fn_name, scoring_fn in scoring_fns.items():
            scores = _partial_preds_to_scores(partial_preds, y_test, scoring_fn)
            if self.mode == "keep_rest":
                full_score = scoring_fn(y_test, full_preds)
                scores = full_score - scores
            if len(partial_preds) != scores.size:
                if len(scoring_fns) > 1:
                    msg = "scoring_fn={} should return one value for each feature.".format(fn_name)
                else:
                    msg = "scoring_fns should return one value for each feature.".format(fn_name)
                raise ValueError("Unexpected dimensions. {}".format(msg))
            scores = scores.ravel()
            all_scores[fn_name] = scores
        self.feature_importances_ = all_scores


def _partial_preds_to_scores(partial_preds, y_test, scoring_fn):
    scores = []
    for k, y_pred in partial_preds.items():
        if isinstance(y_pred, tuple):  # if constant model
            y_pred = np.ones_like(y_test) * y_pred[1]
        scores.append(scoring_fn(y_test, y_pred))
    return np.vstack(scores)


def _get_default_sample_split(sample_split, prediction_model, is_ppm):
    if sample_split == "auto":
        sample_split = "oob"
        if is_ppm:
            if prediction_model.loo:
                sample_split = "loo"
    return sample_split


def _validate_sample_split(sample_split, prediction_model, is_ppm):
    if sample_split in ["oob", "inbag"] and is_ppm:
        if prediction_model.loo:
            raise ValueError("Cannot use LOO together with OOB or in-bag sample splitting.")


def _get_sample_split_data(blocked_data, y, random_state, sample_split):
    if sample_split == "oob":
        train_blocked_data, test_blocked_data, y_train, y_test, _, test_indices = \
            _blocked_train_test_split(blocked_data, y, random_state)
    elif sample_split == "inbag":
        train_blocked_data, _, y_train, _, test_indices, _ = \
            _blocked_train_test_split(blocked_data, y, random_state)
        test_blocked_data = train_blocked_data
        y_test = y_train
    else:
        train_blocked_data = test_blocked_data = blocked_data
        y_train = y_test = y
        test_indices = np.arange(y.shape[0])
    return train_blocked_data, test_blocked_data, y_train, y_test, test_indices