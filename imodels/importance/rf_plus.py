import copy, pprint, warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imodels.importance.block_transformers import MDIPlusDefaultTransformer, TreeTransformer, \
    CompositeTransformer, IdentityTransformer
from imodels.importance.ppms import PartialPredictionModelBase, GlmClassifierPPM, \
    RidgeRegressorPPM, LogisticClassifierPPM
from imodels.importance.mdi_plus import ForestMDIPlus, \
    _get_default_sample_split, _validate_sample_split, _get_sample_split_data
import imodels

class _RandomForestPlus(BaseEstimator):
    """
    The class object for the Random Forest Plus (RF+) estimator, which can
    be used as a prediction model or interpreted via generalized
    mean decrease in impurity (MDI+). For more details, refer to [paper].

    Parameters
    ----------
    rf_model: scikit-learn random forest object or None
        The RF model to be used to build RF+. If None, then a new
        RandomForestRegressor() or RandomForestClassifier() is instantiated.
    prediction_model: A PartialPredictionModelBase object, scikit-learn type estimator, or "auto"
        The prediction model to be used for making (full and partial) predictions
        using the block transformed data.
        If value is set to "auto", then a default is chosen as follows:
         - For RandomForestPlusRegressor, RidgeRegressorPPM is used.
         - For RandomForestPlusClassifier, LogisticClassifierPPM is used.
    sample_split: string in {"loo", "oob", "inbag"} or None
        The sample splitting strategy to be used for fitting RF+. If "oob",
        RF+ is trained on the out-of-bag data. If "inbag", RF+ is trained on the
        in-bag data. Otherwise, RF+ is trained on the full training data.
    include_raw: bool
        Flag for whether to augment the local decision stump features extracted
        from the RF model with the original features.
    drop_features: bool
        Flag for whether to use an intercept model for the partial predictions
        on a given feature if a tree does not have any nodes that split on it,
        or to use a model on the raw feature (if include_raw is True).
    add_transformers: list of BlockTransformerBase objects or None
        Additional block transformers (if any) to include in the RF+ model.
    center: bool
        Flag for whether to center the transformed data in the transformers.
    normalize: bool
        Flag for whether to rescale the transformed data to have unit
        variance in the transformers.
    """

    def __init__(self, rf_model=None, prediction_model=None, sample_split="auto",
                 include_raw=True, drop_features=True, add_transformers=None,
                 center=True, normalize=False):
        assert sample_split in ["loo", "oob", "inbag", "auto", None]
        super().__init__()
        if isinstance(self, RegressorMixin):
            self._task = "regression"
        elif isinstance(self, ClassifierMixin):
            self._task = "classification"
        else:
            raise ValueError("Unknown task.")
        if rf_model is None:
            if self._task == "regression":
                rf_model = RandomForestRegressor()
            elif self._task == "classification":
                rf_model = RandomForestClassifier()
        if prediction_model is None:
            if self._task == "regression":
                prediction_model = RidgeRegressorPPM()
            elif self._task == "classification":
                prediction_model = LogisticClassifierPPM()
        self.rf_model = rf_model
        self.prediction_model = prediction_model
        self.include_raw = include_raw
        self.drop_features = drop_features
        self.add_transformers = add_transformers
        self.center = center
        self.normalize = normalize
        self._is_ppm = isinstance(prediction_model, PartialPredictionModelBase)
        self.sample_split = _get_default_sample_split(sample_split, prediction_model, self._is_ppm)
        _validate_sample_split(self.sample_split, prediction_model, self._is_ppm)

    def fit(self, X, y, sample_weight=None, **kwargs):
        """
        Fit (or train) Random Forest Plus (RF+) prediction model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix.
        y: ndarray of shape (n_samples, n_targets)
            The observed responses.
        sample_weight: array-like of shape (n_samples,) or None
            Sample weights to use in random forest fit.
            If None, samples are equally weighted.
        **kwargs:
            Additional arguments to pass to self.prediction_model.fit()

        """
        self.transformers_ = []
        self.estimators_ = []
        self._tree_random_states = []
        self.prediction_score_ = None
        self.mdi_plus_ = None
        self.mdi_plus_scores_ = None
        self.feature_names_ = None
        self._n_samples_train = X.shape[0]

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            raise ValueError("Input X must be a pandas DataFrame or numpy array.")
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        elif not isinstance(y, np.ndarray):
            raise ValueError("Input y must be a pandas DataFrame or numpy array.")

        # fit random forest
        n_samples = X.shape[0]
        self.rf_model.fit(X, y, sample_weight=sample_weight)
        # onehot encode multiclass response for GlmClassiferPPM
        if isinstance(self.prediction_model, GlmClassifierPPM):
            self._multi_class = False
            if len(np.unique(y)) > 2:
                self._multi_class = True
                self._y_encoder = OneHotEncoder()
                y = self._y_encoder.fit_transform(y.reshape(-1, 1)).toarray()
        # fit model for each tree
        all_full_preds = []
        for tree_model in self.rf_model.estimators_:
            # get transformer
            if self.add_transformers is None:
                if self.include_raw:
                    transformer = MDIPlusDefaultTransformer(tree_model, drop_features=self.drop_features)
                else:
                    transformer = TreeTransformer(tree_model)
            else:
                if self.include_raw:
                    base_transformer_list = [TreeTransformer(tree_model), IdentityTransformer()]
                else:
                    base_transformer_list = [TreeTransformer(tree_model)]
                transformer = CompositeTransformer(base_transformer_list + self.add_transformers,
                                                   drop_features=self.drop_features)
            # fit transformer
            blocked_data = transformer.fit_transform(X_array, center=self.center, normalize=self.normalize)
            # do sample split
            train_blocked_data, test_blocked_data, y_train, y_test, test_indices = \
                _get_sample_split_data(blocked_data, y, tree_model.random_state, self.sample_split)
            # fit prediction model
            if train_blocked_data.get_all_data().shape[1] != 0:  # if tree has >= 1 split
                self.prediction_model.fit(train_blocked_data.get_all_data(), y_train, **kwargs)
                self.estimators_.append(copy.deepcopy(self.prediction_model))
                self.transformers_.append(copy.deepcopy(transformer))
                self._tree_random_states.append(tree_model.random_state)

                # get full predictions
                pred_func = self._get_pred_func()
                full_preds = pred_func(test_blocked_data.get_all_data())
                full_preds_n = np.empty(n_samples) if full_preds.ndim == 1 \
                    else np.empty((n_samples, full_preds.shape[1]))
                full_preds_n[:] = np.nan
                full_preds_n[test_indices] = full_preds
                all_full_preds.append(full_preds_n)

        # compute prediction accuracy on internal sample split
        full_preds = np.nanmean(all_full_preds, axis=0)
        if self._task == "regression":
            pred_score = r2_score(y, full_preds)
            pred_score_name = "r2"
        elif self._task == "classification":
            if full_preds.shape[1] == 2:
                pred_score = roc_auc_score(y, full_preds[:, 1], multi_class="ovr")
            else:
                pred_score = roc_auc_score(y, full_preds, multi_class="ovr")
            pred_score_name = "auroc"
        self.prediction_score_ = pd.DataFrame({pred_score_name: [pred_score]})
        self._full_preds = full_preds

    def predict(self, X):
        """
        Make predictions on new data using the fitted model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix, for which to make predictions.

        Returns
        -------
        y: ndarray of shape (n_samples,) or (n_samples, n_targets)
            The predicted values

        """
        X = check_array(X)
        check_is_fitted(self, "estimators_")
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ is not None:
                X_array = X.loc[:, self.feature_names_].values
            else:
                X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            raise ValueError("Input X must be a pandas DataFrame or numpy array.")

        if self._task == "regression":
            predictions = 0
            for estimator, transformer in zip(self.estimators_, self.transformers_):
                blocked_data = transformer.transform(X_array, center=self.center, normalize=self.normalize)
                predictions += estimator.predict(blocked_data.get_all_data())
            predictions = predictions / len(self.estimators_)
        elif self._task == "classification":
            prob_predictions = self.predict_proba(X_array)
            if prob_predictions.ndim == 1:
                prob_predictions = np.stack([1-prob_predictions, prob_predictions], axis=1)
            predictions = self.rf_model.classes_[np.argmax(prob_predictions, axis=1)]
        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities on new data using the fitted
        (classification) model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix, for which to make predictions.

        Returns
        -------
        y: ndarray of shape (n_samples, n_classes)
            The predicted class probabilities.

        """
        X = check_array(X)
        check_is_fitted(self, "estimators_")
        if not hasattr(self.estimators_[0], "predict_proba"):
            raise AttributeError("'{}' object has no attribute 'predict_proba'".format(
                self.estimators_[0].__class__.__name__)
            )
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ is not None:
                X_array = X.loc[:, self.feature_names_].values
            else:
                X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            raise ValueError("Input X must be a pandas DataFrame or numpy array.")

        predictions = 0
        for estimator, transformer in zip(self.estimators_, self.transformers_):
            blocked_data = transformer.transform(X_array, center=self.center, normalize=self.normalize)
            predictions += estimator.predict_proba(blocked_data.get_all_data())
        predictions = predictions / len(self.estimators_)
        return predictions

    def get_mdi_plus_scores(self, X=None, y=None,
                            scoring_fns="auto", sample_split="inherit", mode="keep_k"):
        """
        Obtain MDI+ feature importances. Generalized mean decrease in impurity (MDI+)
        is a flexible framework for computing RF feature importances. For more
        details, refer to [paper].

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix. Generally should be the same X as that used for
            fitting the RF+ prediction model. If a pd.DataFrame object is supplied, then
            the column names are used in the output.
        y: ndarray of shape (n_samples, n_targets)
            The observed responses. Generally should be the same y as that used for
            fitting the RF+ prediction model.
        scoring_fns: a function or dict with functions as value and function name (str) as key or "auto"
            The scoring functions used for evaluating the partial predictions.
            If "auto", then a default is chosen as follows:
             - For RandomForestPlusRegressor, then r-squared (_fast_r2_score) is used.
             - For RandomForestPlusClassifier, then the negative log-loss (_neg_log_loss) is used.
        sample_split: string in {"loo", "oob", "inbag", "inherit"} or None
            The sample splitting strategy to be used when evaluating the partial
            model predictions in MDI+. If "inherit" (default), uses the same sample splitting
            strategy as used when fitting the RF+ prediction model. Assuming None or "loo" were used
            as the sample splitting scheme for fitting the RF+ prediction model,
            "loo" (leave-one-out) is strongly recommended here in MDI+ as it overcomes
            the known correlation and entropy biases suffered by MDI. "oob" (out-of-bag) can
            also be used to overcome these biases. "inbag" is the sample splitting
            strategy used by MDI. If None, no sample splitting is performed and the
            full data set is used to evaluate the partial model predictions.
        mode: string in {"keep_k", "keep_rest"}
            Mode for the method. "keep_k" imputes the mean of each feature not
            in block k when making a partial model prediction, while "keep_rest"
            imputes the mean of each feature in block k. "keep_k" is strongly
            recommended for computational considerations.

        Returns
        -------
        scores: pd.DataFrame of shape (n_features, n_scoring_fns)
            The MDI+ feature importances.
        """
        if X is None or y is None:
            if self.mdi_plus_scores_ is None:
                raise ValueError("Need X and y as inputs.")
        else:
            # convert data frame to array
            if isinstance(X, pd.DataFrame):
                if self.feature_names_ is not None:
                    X_array = X.loc[:, self.feature_names_].values
                else:
                    X_array = X.values
            elif isinstance(X, np.ndarray):
                X_array = X
            else:
                raise ValueError("Input X must be a pandas DataFrame or numpy array.")
            if isinstance(y, pd.DataFrame):
                y = y.values.ravel()
            elif not isinstance(y, np.ndarray):
                raise ValueError("Input y must be a pandas DataFrame or numpy array.")
            # get defaults
            if sample_split == "inherit":
                sample_split = self.sample_split
            if X.shape[0] != self._n_samples_train and sample_split is not None:
                raise ValueError("Set sample_split=None to fit MDI+ on non-training X and y. "
                                 "To use other sample_split schemes, input the training X and y data.")
            if scoring_fns == "auto":
                scoring_fns = {"importance": _fast_r2_score} if self._task == "regression" \
                    else {"importance": _neg_log_loss}
            # onehot encode if multi-class for GlmClassiferPPM
            if isinstance(self.prediction_model, GlmClassifierPPM):
                if self._multi_class:
                    y = self._y_encoder.transform(y.reshape(-1, 1)).toarray()
            # compute MDI+ for forest
            mdi_plus_obj = ForestMDIPlus(estimators=self.estimators_,
                                         transformers=self.transformers_,
                                         scoring_fns=scoring_fns,
                                         sample_split=sample_split,
                                         tree_random_states=self._tree_random_states,
                                         mode=mode,
                                         task=self._task,
                                         center=self.center,
                                         normalize=self.normalize)
            self.mdi_plus_ = mdi_plus_obj
            mdi_plus_scores = mdi_plus_obj.get_scores(X_array, y)
            if self.feature_names_ is not None:
                mdi_plus_scores["var"] = self.feature_names_
                self.mdi_plus_.feature_importances_["var"] = self.feature_names_
            self.mdi_plus_scores_ = mdi_plus_scores
        return self.mdi_plus_scores_

    def get_mdi_plus_stability_scores(self, B=10, metrics="auto"):
        """
        Evaluate the stability of the MDI+ feature importance rankings
        across bootstrapped samples of trees. Can be used to select the GLM
        and scoring metric in a data-driven manner, where the GLM and metric, which
        yields the most stable feature rankings across bootstrapped samples is selected.

        Parameters
        ----------
        B: int
            Number of bootstrap samples.
        metrics: "auto" or a dict with functions as value and function name (str) as key
            Metric(s) used to evaluate the stability between two sets of feature importances.
            If "auto", then the feature importance stability metrics are:
                (1) Rank-based overlap (RBO) with p=0.9 (from "A Similarity Measure for
                Indefinite Rankings" by Webber et al. (2010)). Intuitively, this metric gives
                more weight to features with the largest importances, with most of the weight
                going to the ~1/(1-p) features with the largest importances.
                (2) A weighted kendall tau metric (tauAP_b from "The Treatment of Ties in
                AP Correlation" by Urbano and Marrero (2017)), which also gives more weight
                to the features with the largest importances, but uses a different weighting
                scheme from RBO.
            Note that these default metrics assume that a higher MDI+ score indicates
            greater importance and thus give more weight to these features with high
            importance/ranks. If a lower MDI+ score indicates higher importance, then invert
            either these stability metrics or the MDI+ scores before evaluating the stability.

        Returns
        -------
        stability_results: pd.DataFrame of shape (n_features, n_metrics)
            The stability scores of the MDI+ feature rankings across bootstrapped samples.

        """
        if self.mdi_plus_ is None:
            raise ValueError("Need to compute MDI+ scores first using self.get_mdi_plus_scores(X, y)")
        return self.mdi_plus_.get_stability_scores(B=B, metrics=metrics)

    def _get_pred_func(self):
        if hasattr(self.prediction_model, "predict_proba_loo"):
            pred_func = self.prediction_model.predict_proba_loo
        elif hasattr(self.prediction_model, "predict_loo"):
            pred_func = self.prediction_model.predict_loo
        elif hasattr(self.prediction_model, "predict_proba"):
            pred_func = self.prediction_model.predict_proba
        else:
            pred_func = self.prediction_model.predict
        return pred_func


class RandomForestPlusRegressor(_RandomForestPlus, RegressorMixin):
    """
    The class object for the Random Forest Plus (RF+) regression estimator, which can
    be used as a prediction model or interpreted via generalized
    mean decrease in impurity (MDI+). For more details, refer to [paper].
    """
    ...


class RandomForestPlusClassifier(_RandomForestPlus, ClassifierMixin):
    """
    The class object for the Random Forest Plus (RF+) classification estimator, which can
    be used as a prediction model or interpreted via generalized
    mean decrease in impurity (MDI+). For more details, refer to [paper].
    """
    ...


def _fast_r2_score(y_true, y_pred, multiclass=False):
    """
    Evaluates the r-squared value between the observed and estimated responses.
    Equivalent to sklearn.metrics.r2_score but without the robust error
    checking, thus leading to a much faster implementation (at the cost of
    this error checking). For multi-class responses, returns the mean
    r-squared value across each column in the response matrix.

    Parameters
    ----------
    y_true: array-like of shape (n_samples, n_targets)
        Observed responses.
    y_pred: array-like of shape (n_samples, n_targets)
        Predicted responses.
    multiclass: bool
        Whether or not the responses are multi-class.

    Returns
    -------
    Scalar quantity, measuring the r-squared value.
    """
    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - np.mean(y_true, axis=0)) ** 2). \
        sum(axis=0, dtype=np.float64)
    if multiclass:
        return np.mean(1 - numerator / denominator)
    else:
        return 1 - numerator / denominator


def _neg_log_loss(y_true, y_pred):
    """
    Evaluates the negative log-loss between the observed and
    predicted responses.

    Parameters
    ----------
    y_true: array-like of shape (n_samples, n_targets)
        Observed responses.
    y_pred: array-like of shape (n_samples, n_targets)
        Predicted probabilies.

    Returns
    -------
    Scalar quantity, measuring the negative log-loss value.
    """
    return -log_loss(y_true, y_pred)


if __name__ == "__main__":
    

    #Suppress specific warning
    # warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    
    X, y,f = imodels.get_clean_dataset("california_housing")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=1)

    pprint.pprint(f"Shape: {X_train.shape}")

    rf = RandomForestRegressor(n_estimators=3,min_samples_leaf=5,max_features=0.33,random_state=1)
    rf.fit(X_train, y_train)
    
    rf_plus = RandomForestPlusRegressor(rf_model=copy.deepcopy(rf))
    rf_plus.fit(X_train, y_train)

    pprint.pprint(f"RF r2_score: {r2_score(y_test,rf.predict(X_test))}")

    pprint.pprint(f"RF+ r2_score: {r2_score(y_test,rf_plus.predict(X_test))}")

   