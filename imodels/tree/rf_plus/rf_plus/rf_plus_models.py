# generic imports
import numpy as np
import copy
from joblib import Parallel, delayed

# sklearn imports
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices

# rf+ prediction imports
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_regression import AloElasticNetRegressorCV
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_classification import AloLogisticElasticNetClassifierCV
from imodels.tree.rf_plus.rf_plus.rf_plus_utils import _check_X, _check_Xy
from imodels.tree.rf_plus.data_transformations.block_transformers import MDIPlusDefaultTransformer, TreeTransformer, CompositeTransformer, IdentityTransformer

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

    def __init__(self, rf_model=None, prediction_model=None, include_raw=True,
                 drop_features=True, add_transformers=None, center=True, 
                 normalize=False, fit_on = "all",verbose = True, 
                 warm_start = False):

        
        super().__init__()
        assert fit_on in ["inbag","oob","all","uniform"]
        self.rf_model = copy.deepcopy(rf_model)
        self.prediction_model = copy.deepcopy(prediction_model)
        self._initial_prediction_model = copy.deepcopy(prediction_model)
        self.include_raw = include_raw
        self.drop_features = drop_features
        self.add_transformers = add_transformers
        self.center = center
        self.normalize = normalize
        self.fit_on = fit_on
        self.transformers_ = []
        self.estimators_ = []
        self._tree_random_states = []
        self.verbose = verbose
        self._oob_indices = {}
        self._is_gb = isinstance(rf_model, (GradientBoostingClassifier, GradientBoostingRegressor))
        self._resid_diffs = None
        self.warm_start = warm_start
  
    def fit(self, X, y, sample_weight=None, n_jobs = -1, **kwargs):
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
       
        self.prediction_score_ = None
        self.mdi_plus_ = None
        self.mdi_plus_scores_ = None
        self.feature_names_ = None
        self._n_samples_train = X.shape[0]
        self._loo_preds = None
        if self._is_gb:
            if self.rf_model.get_params()['loss'] == "squared_error":
                self._y_avg = np.mean(y)
                self._learning_rate = self.rf_model.learning_rate
                self._loss = "squared_error"
            elif self.rf_model.get_params()['loss'] == "log_loss":
                if self.rf_model.init == "zero":
                    self._y_avg = 0
                elif self.rf_model.init == None:
                    # get the most common class
                    self._y_avg = float(round(np.mean(y)))
                else:
                    raise ValueError("Only zero initialization is supported for gradient boosting")
                # self._y_avg = np.log(np.mean(y) / (1 - np.mean(y)))
                self._learning_rate = self.rf_model.learning_rate
                self._loss = "log_loss"
            else:
                raise ValueError("Only squared_error and log_loss are supported for gradient boosting")

        X_array, y = _check_Xy(copy.deepcopy(X),y)
        
        # check if self.rf_model has already been fit
        if not hasattr(self.rf_model, "estimators_"):
            self.rf_model.fit(X, y, sample_weight=sample_weight)
                                            
        ### GRADIENT BOOSTING CASE
        if self._is_gb:
            
            # make a copy of y, has to be float32 for gradient function
            copy_y = copy.deepcopy(y).astype(np.float32)
            # initialize predictions
            curr_pred = np.full(y.shape[0], self._y_avg)
            
            # if the loss is squared error, residuals are just the difference
            # between the true and predicted values
            if self._loss == "squared_error":
                resid = copy.deepcopy(y) - curr_pred
            # if the loss is log-loss, sklearn actually sneakily uses
            # half-binomial loss. get around this by using the gradient method.
            elif self._loss == "log_loss":
                resid = -self.rf_model._loss.gradient(copy_y,
                                    # the times four took me forever to figure
                                    # out, still don't fully understand why its
                                    # needed to make preds equal that of each
                                    # tree. there is still numerical stability
                                    # issues, since i think their implementation
                                    # keeps choosing numbers very very close to
                                    # four, but not exactly four, which
                                    # accumulates over time.
                                    np.zeros(y.shape[0], dtype=np.float32)) * 4
            # if the loss is something else, we should have already thrown an
            # error, but we do it again just in case.
            else:
                raise ValueError("Only squared_error and log_loss are " + \
                    "supported for gradient boosting")
            past_resid = None
            resid_diff_arr = np.full(len(self.rf_model.estimators_), np.NaN)
            # now we go through each tree and grow them to the residuals
            for i,tree_model in enumerate(self.rf_model.estimators_):
                tree_model = tree_model[0] # trees are in an array in sklearn
                result = self._fit_ith_tree(tree_model, i, X_array, resid,
                                            sample_weight,**kwargs)
                curr_pred = curr_pred + tree_model.predict(X_array) * \
                    self._learning_rate
                if self._loss == "squared_error":
                    past_resid = copy.deepcopy(resid)
                    resid = copy_y - curr_pred
                else: # log-loss case
                    past_resid = copy.deepcopy(resid)
                    resid = -self.rf_model._loss.gradient(copy_y,
                                            # need to convert to np.float32
                                            # to prevent error from gradient
                                            curr_pred.astype(np.float32)) * 4
                resid_diff = np.sum(np.abs(np.abs(resid) - np.abs(past_resid)))
                resid_diff_arr[i] = resid_diff
                self.estimators_.append(result[0])
                self.transformers_.append(result[1])
                self._tree_random_states.append(result[2])
                self._oob_indices[result[3][0]] = result[3][1]
            self._resid_diffs = resid_diff_arr
                
        ### RANDOM FOREST CASE
        elif n_jobs is None:
            for i,tree_model in enumerate(self.rf_model.estimators_):
                result = self._fit_ith_tree(tree_model, i, X_array, y,
                                            sample_weight,**kwargs)
                self.estimators_.append(result[0])
                self.transformers_.append(result[1])
                self._tree_random_states.append(result[2])
                self._oob_indices[result[3][0]] = result[3][1]
        else:
            results = Parallel(n_jobs=n_jobs,verbose=int(self.verbose))(delayed(self._fit_ith_tree)(tree_model,i,X_array,y,sample_weight,**kwargs) for i,tree_model in enumerate(self.rf_model.estimators_))
            for result in results:
                self.estimators_.append(result[0])
                self.transformers_.append(result[1])
                self._tree_random_states.append(result[2])
                self._oob_indices[result[3][0]] = result[3][1]

    def _fit_ith_tree(self, tree_model,i, X, y, sample_weight=None, **kwargs):
        """
        Fit the ith tree in the random forest to the data.
        
        Parameters
        ----------
        tree_model: scikit-learn tree model
            The tree model to be fitted.
        i: int
            The index of the tree in the random forest.
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix.
        y: ndarray of shape (n_samples, n_targets)
            The observed responses.
        sample_weight: array-like of shape (n_samples,) or None
            Sample weights to use in random forest fit.
            If None, samples are equally weighted.
        **kwargs:
            Additional arguments to pass to self.prediction_model.fit()
            
        Returns
        -------
        prediction_model: scikit-learn model
            The fitted prediction model.
        transformer: BlockTransformerBase object
            The fitted transformer.
        random_state: int
            The random state of the tree model.
        i: int
            The index of the tree in the random forest.
        oob_indices: ndarray of shape (n_samples,)
            The out-of-bag indices for the tree model.
        """                

        n_samples = X.shape[0]  
        
        if self.add_transformers is None:
            if self.include_raw:
                transformer = MDIPlusDefaultTransformer(tree_model,drop_features=self.drop_features)
            else:
                transformer = TreeTransformer(tree_model)
        else:
            if self.include_raw:
                base_transformer_list = [TreeTransformer(tree_model), IdentityTransformer()]
            else:
                base_transformer_list = [TreeTransformer(tree_model)]
            transformer = CompositeTransformer(base_transformer_list + self.add_transformers,
                                            drop_features=self.drop_features)
                    
        prediction_model = copy.deepcopy(self._initial_prediction_model)
        
        # get in-bag, oob sample indices 
        oob_indices = _generate_unsampled_indices(tree_model.random_state, n_samples, n_samples)
        inbag_indices = _generate_sample_indices(tree_model.random_state,n_samples,n_samples)
        unique_inbag_indices, counts_elements = np.unique(inbag_indices, return_counts=True)    
        
        # get transformed data
        X_inbag = copy.deepcopy(X)[inbag_indices]
        tree_inbag_blocked_data = transformer.fit_transform(X_inbag, center=self.center, normalize=self.normalize)            
        tree_blocked_data = transformer.transform(X, center=self.center, normalize=self.normalize)
        tree_X_train = tree_blocked_data.get_all_data()
        tree_y_train = copy.deepcopy(y)
        
        # get sample weights depending on fit on strategy
        tree_sample_weight = np.ones(len(tree_y_train))

        if "evaluate_on" in kwargs:
            assert kwargs["evaluate_on"] in ["inbag","oob", "all",None]
            if (kwargs["evaluate_on"] == "inbag"):
                evaluate_on = inbag_indices
            elif (kwargs["evaluate_on"] == "oob"):
                evaluate_on = oob_indices
            else:
                evaluate_on = np.arange(len(tree_y_train))
        else:
            evaluate_on = None
    
        if self.fit_on == "inbag": #only use in-bag samples
            tree_sample_weight = np.zeros(len(tree_y_train))
            tree_sample_weight[unique_inbag_indices] = counts_elements
            tree_sample_weight[oob_indices] = 0 # 1e-12
        
        elif self.fit_on == "oob": #only use oob samples
            tree_sample_weight = np.ones(len(tree_y_train))*1e-12
            tree_sample_weight[oob_indices] = 1
        
        elif self.fit_on == "all": #use all samples
            tree_sample_weight = np.ones(len(tree_y_train))
            tree_sample_weight[unique_inbag_indices] = counts_elements
        
        # boosting doesn't bootstrap
        elif self.fit_on == "uniform":
            tree_sample_weight = None
            
        if self._is_gb:
            tree_sample_weight = None

        if tree_inbag_blocked_data.get_all_data().shape[1] != 0:
            if evaluate_on is None:
                prediction_model.fit(tree_X_train, tree_y_train, sample_weight=tree_sample_weight)
            else:
                prediction_model.fit(tree_X_train, tree_y_train, sample_weight=tree_sample_weight,evaluate_on = evaluate_on)

        return prediction_model, copy.deepcopy(transformer), tree_model.random_state, (i,oob_indices)
        
    def predict(self, X, tree_weights='uniform', n_jobs=1):
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
        X_array = _check_X(X)   

        def parallel_predict_helper(estimator, transformer):
            blocked_data = transformer.transform(X_array, center=self.center, normalize=self.normalize)
            return estimator.predict(blocked_data.get_all_data())
        
        predictions = Parallel(n_jobs=n_jobs)(delayed(parallel_predict_helper)(estimator, transformer) for estimator, transformer in zip(self.estimators_, self.transformers_))
        if self._is_gb:
            final_predictions = np.full(predictions[0].shape, self._y_avg)
            for i in range(len(predictions)):
                final_predictions += self._learning_rate * predictions[i]
            return final_predictions
        
        if tree_weights == 'uniform':
            return np.mean(np.vstack(predictions), axis=0)
        
        else:
            assert len(tree_weights) == len(predictions)
            tree_weights = tree_weights/np.sum(tree_weights)
            return np.average(np.vstack(predictions), axis=0,weights=tree_weights)



class RandomForestPlusRegressor(_RandomForestPlus, RegressorMixin):
    """
    The class object for the Random Forest Plus (RF+) regression estimator, which can
    be used as a prediction model or interpreted via generalized
    mean decrease in impurity (MDI+). For more details, refer to [paper].
    """
    def __init__(self, rf_model=None, prediction_model=None, include_raw=True, drop_features=True, 
                 add_transformers=None, center=True, normalize=False, fit_on="all", 
                 verbose=True, warm_start=False):
        
        if rf_model is None:
            rf_model = RandomForestRegressor(max_features=0.33,min_samples_leaf=5) #R package default values 
        if prediction_model is None:
            prediction_model = AloElasticNetRegressorCV()
        self._task = "regression"
        super().__init__(rf_model, prediction_model, include_raw, drop_features, add_transformers, 
                         center, normalize, fit_on, verbose, warm_start)


class RandomForestPlusClassifier(_RandomForestPlus, ClassifierMixin):
    """
    The class object for the Random Forest Plus (RF+) classification estimator, which can
    be used as a prediction model or interpreted via generalized
    mean decrease in impurity (MDI+). For more details, refer to [paper].
    """
    def __init__(self, rf_model=None, prediction_model=None, include_raw=True, drop_features=True, 
                 add_transformers=None, center=True, normalize=False, fit_on="all", 
                 verbose=True, warm_start=False):
        
        if rf_model is None:
            rf_model = RandomForestClassifier(max_features="sqrt",min_samples_leaf=2) #R package default values 
        if prediction_model is None:
            prediction_model = AloLogisticElasticNetClassifierCV()
        self._task = "classification"
        super().__init__(rf_model, prediction_model, include_raw, drop_features, add_transformers, 
                         center, normalize, fit_on, verbose, warm_start)
    
    def predict(self, X, tree_weights = 'uniform', n_jobs=1):
        
        def parallel_predict_helper(estimator, transformer):
            blocked_data = transformer.transform(X, center=self.center, normalize=self.normalize)
            return estimator.predict(blocked_data.get_all_data())
        if self._is_gb and self._loss == "log_loss":
            predictions = Parallel(n_jobs=n_jobs)(delayed(parallel_predict_helper)(estimator, transformer) for estimator, transformer in zip(self.estimators_, self.transformers_))
            final_predictions = np.full(predictions[0].shape, self._y_avg, dtype=np.float64)
            for i in range(len(predictions)):
                final_predictions += self._learning_rate * predictions[i]
            return final_predictions > 0
        else:
            return np.argmax(self.predict_proba(X, tree_weights,n_jobs), axis=1)

    def predict_proba(self, X, tree_weights = 'uniform', n_jobs=1):
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
            raise AttributeError("'{}' object has no attribute 'predict_proba'".format(self.estimators_[0].__class__.__name__))
        X_array = _check_X(X)

        def parallel_predict_proba_helper(estimator, transformer):
            blocked_data = transformer.transform(X_array, center=self.center, normalize=self.normalize)
            return estimator.predict_proba(blocked_data.get_all_data())
        
        predictions = Parallel(n_jobs=n_jobs)(delayed(parallel_predict_proba_helper)(estimator, transformer) for estimator, 
                                              transformer in zip(self.estimators_, self.transformers_))
        
        predictions = [predictions[i][:,1] for i in range(len(predictions))]
        if tree_weights == 'uniform':
            predictions = np.mean(np.vstack(predictions), axis=0)
        else:
            assert len(tree_weights) == len(predictions)
            tree_weights = tree_weights/np.sum(tree_weights)
            predictions = np.average(np.vstack(predictions), axis=0,weights=tree_weights)
        return np.stack([1 - predictions, predictions]).T
